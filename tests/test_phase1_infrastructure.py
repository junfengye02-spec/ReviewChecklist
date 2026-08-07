from __future__ import annotations

import contextlib
import hashlib
import io
import logging
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from alembic import command
from alembic.autogenerate import compare_metadata
from alembic.config import Config
from alembic.migration import MigrationContext
from sqlalchemy import (
    JSON,
    Column,
    Integer,
    MetaData,
    Table,
    create_engine,
    func,
    select,
)
from sqlalchemy.exc import IntegrityError
from sqlalchemy.inspection import inspect

from tender_review.config import PROJECT_DIR
from tender_review.documents import ArtifactStore
from tender_review.documents.models import ArtifactWrite
from tender_review.infrastructure.database import (
    Base,
    DatabaseHealthAdapter,
    create_session_factory,
    session_scope,
)
from tender_review.infrastructure.database.langgraph_checkpoints import (
    LANGGRAPH_CHECKPOINT_METADATA,
)
from tender_review.infrastructure.object_storage import (
    ArtifactIntegrityError,
    MinioArtifactStore,
)
from tender_review.shared.health import ReadinessCheck


EXPECTED_TABLES = {
    "dataset_annotation_samples",
    "dataset_versions",
    "document_artifacts",
    "document_snapshots",
    "evaluation_cases",
    "evaluation_runs",
    "evaluation_threshold_policies",
    "evidence_references",
    "findings",
    "human_decisions",
    "idempotency_records",
    "job_checkpoints",
    "model_configs",
    "optimization_attempts",
    "optimization_jobs",
    "review_execution_specs",
    "review_jobs",
    "rule_sets",
    "rule_versions",
}
EXPECTED_CHECKPOINT_TABLES = {
    "langgraph_checkpoint_writes",
    "langgraph_checkpoints",
}


class ModelMetadataTests(unittest.TestCase):
    def test_core_plan_entities_and_schema_versions_are_present(self):
        self.assertEqual(set(Base.metadata.tables), EXPECTED_TABLES)
        self.assertEqual(
            set(LANGGRAPH_CHECKPOINT_METADATA.tables), EXPECTED_CHECKPOINT_TABLES
        )
        self.assertTrue(
            set(Base.metadata.tables).isdisjoint(LANGGRAPH_CHECKPOINT_METADATA.tables)
        )
        for table in Base.metadata.tables.values():
            self.assertIn("schema_version", table.c, table.name)
            self.assertFalse(table.c.schema_version.nullable, table.name)

        json_tables = {
            table.name
            for table in Base.metadata.tables.values()
            if any(isinstance(column.type, JSON) for column in table.columns)
        }
        self.assertTrue(json_tables)
        self.assertTrue(
            all(
                "schema_version" in Base.metadata.tables[name].c for name in json_tables
            )
        )

    def test_required_unique_constraints_and_foreign_keys_are_named(self):
        rule_version_names = {
            constraint.name
            for constraint in Base.metadata.tables["rule_versions"].constraints
            if constraint.__class__.__name__ == "UniqueConstraint"
        }
        self.assertEqual(
            rule_version_names,
            {"uq_rule_versions_set_content_hash", "uq_rule_versions_set_version"},
        )

        idempotency_names = {
            constraint.name
            for constraint in Base.metadata.tables["idempotency_records"].constraints
            if constraint.__class__.__name__ == "UniqueConstraint"
        }
        self.assertIn("uq_idempotency_caller_scope_key", idempotency_names)

        rule_set_foreign_keys = {
            foreign_key.constraint.name: foreign_key.target_fullname
            for foreign_key in Base.metadata.tables["rule_sets"].foreign_keys
        }
        self.assertEqual(
            rule_set_foreign_keys["fk_rule_sets_current_version_id_rule_versions"],
            "rule_versions.id",
        )

    def test_database_session_scope_commits_rolls_back_and_reports_health(self):
        engine = create_engine("sqlite:///:memory:")
        metadata = MetaData()
        values = Table(
            "values_for_test",
            metadata,
            Column("id", Integer, primary_key=True),
        )
        metadata.create_all(engine)
        factory = create_session_factory(engine)

        with session_scope(factory) as session:
            session.execute(values.insert().values(id=1))
        with engine.connect() as connection:
            self.assertEqual(
                connection.scalar(select(func.count()).select_from(values)), 1
            )

        with self.assertRaises(IntegrityError):
            with session_scope(factory) as session:
                session.execute(values.insert().values(id=1))
        with engine.connect() as connection:
            self.assertEqual(
                connection.scalar(select(func.count()).select_from(values)), 1
            )

        adapter = DatabaseHealthAdapter(engine)
        self.assertIsInstance(adapter, ReadinessCheck)
        health = adapter.check()
        self.assertTrue(health.healthy)
        self.assertEqual(health.schema_version, "1")
        engine.dispose()


class AlembicMigrationTests(unittest.TestCase):
    def _config(
        self, database_url: str, *, stdout: io.StringIO | None = None
    ) -> Config:
        config = Config(str(PROJECT_DIR / "alembic.ini"), stdout=stdout)
        config.set_main_option("script_location", str(PROJECT_DIR / "alembic"))
        config.set_main_option("sqlalchemy.url", database_url.replace("%", "%%"))
        return config

    def test_initial_migration_matches_metadata_without_external_services(self):
        with tempfile.TemporaryDirectory() as directory:
            database_path = Path(directory) / "schema.db"
            url = f"sqlite:///{database_path.as_posix()}"
            config = self._config(url)
            command.upgrade(config, "head")

            engine = create_engine(url)
            try:
                self.assertEqual(
                    set(inspect(engine).get_table_names()) - {"alembic_version"},
                    EXPECTED_TABLES | EXPECTED_CHECKPOINT_TABLES,
                )
                combined_metadata = MetaData(
                    naming_convention=Base.metadata.naming_convention
                )
                for metadata in (Base.metadata, LANGGRAPH_CHECKPOINT_METADATA):
                    for table in metadata.sorted_tables:
                        table.to_metadata(combined_metadata)
                with engine.connect() as connection:
                    differences = compare_metadata(
                        MigrationContext.configure(connection), combined_metadata
                    )
                self.assertEqual(differences, [])
            finally:
                engine.dispose()

            command.downgrade(config, "base")
            engine = create_engine(url)
            try:
                self.assertEqual(inspect(engine).get_table_names(), ["alembic_version"])
            finally:
                engine.dispose()

    def test_migration_logging_preserves_existing_application_logger(self):
        logger = logging.getLogger("tender_review")
        previous_handlers = tuple(logger.handlers)
        previous_level = logger.level
        previous_disabled = logger.disabled
        previous_propagate = logger.propagate
        stream = io.StringIO()
        handler = logging.StreamHandler(stream)
        for existing_handler in previous_handlers:
            logger.removeHandler(existing_handler)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.disabled = False
        logger.propagate = False
        try:
            with tempfile.TemporaryDirectory() as directory:
                database_path = Path(directory) / "logging.db"
                config = self._config(f"sqlite:///{database_path.as_posix()}")
                command.upgrade(config, "head")

            logger.info("migration logging remained active")
            self.assertFalse(logger.disabled)
            self.assertIn("migration logging remained active", stream.getvalue())
        finally:
            logger.removeHandler(handler)
            handler.close()
            for existing_handler in previous_handlers:
                logger.addHandler(existing_handler)
            logger.setLevel(previous_level)
            logger.disabled = previous_disabled
            logger.propagate = previous_propagate

    def test_mysql_offline_sql_contains_json_constraints_and_delayed_fk(self):
        output = io.StringIO()
        config = self._config(
            "mysql+pymysql://user:pass@localhost/tender_review?charset=utf8mb4",
            stdout=output,
        )
        with contextlib.redirect_stdout(output):
            command.upgrade(config, "head", sql=True)
        sql = output.getvalue()
        self.assertIn("CREATE TABLE review_jobs", sql)
        self.assertIn("metrics_json JSON", sql)
        self.assertIn("uq_rule_versions_set_content_hash", sql)
        self.assertIn("uq_rule_versions_set_version", sql)
        self.assertIn(
            "ALTER TABLE rule_sets ADD CONSTRAINT "
            "fk_rule_sets_current_version_id_rule_versions",
            " ".join(sql.split()),
        )


class FakeS3Error(Exception):
    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


class FakeResponse(io.BytesIO):
    def __init__(self, value: bytes):
        super().__init__(value)
        self.connection_released = False

    def release_conn(self) -> None:
        self.connection_released = True


class FakeMinioClient:
    def __init__(self):
        self.buckets: set[str] = set()
        self.objects: dict[tuple[str, str], dict[str, object]] = {}
        self.put_count = 0

    def bucket_exists(self, bucket: str) -> bool:
        return bucket in self.buckets

    def make_bucket(self, bucket: str) -> None:
        self.buckets.add(bucket)

    def stat_object(self, bucket: str, key: str):
        try:
            value = self.objects[(bucket, key)]
        except KeyError as exc:
            raise FakeS3Error("NoSuchKey") from exc
        return SimpleNamespace(
            size=len(value["content"]),
            content_type=value["content_type"],
            metadata=value["metadata"],
        )

    def put_object(
        self,
        bucket: str,
        key: str,
        stream,
        length: int,
        *,
        content_type: str,
        metadata: dict[str, str],
    ) -> None:
        content = stream.read()
        if len(content) != length:
            raise AssertionError("length mismatch")
        self.put_count += 1
        self.objects[(bucket, key)] = {
            "content": content,
            "content_type": content_type,
            "metadata": metadata,
        }

    def get_object(self, bucket: str, key: str) -> FakeResponse:
        try:
            value = self.objects[(bucket, key)]
        except KeyError as exc:
            raise FakeS3Error("NoSuchKey") from exc
        return FakeResponse(value["content"])

    def presigned_get_object(self, bucket: str, key: str, *, expires) -> str:
        return f"http://minio.invalid/{bucket}/{key}?seconds={int(expires.total_seconds())}"

    def remove_object(self, bucket: str, key: str) -> None:
        self.objects.pop((bucket, key), None)


class MinioArtifactStoreTests(unittest.TestCase):
    def setUp(self):
        self.client = FakeMinioClient()
        self.store = MinioArtifactStore(
            endpoint="minio.invalid:9000",
            access_key="access",
            secret_key="secret",
            bucket="artifacts",
            client=self.client,
        )
        self.store.ensure_bucket()

    def test_content_addressed_put_is_immutable_and_deduplicated(self):
        payload = b"same content"
        expected_hash = hashlib.sha256(payload).hexdigest()

        first = self.store.put_bytes(payload, call_id="call-1")
        second = self.store.put_bytes(payload, call_id="call-2")

        self.assertEqual(first, second)
        self.assertEqual(
            first.object_key, f"sha256/{expected_hash[:2]}/{expected_hash}"
        )
        self.assertEqual(self.client.put_count, 1)
        self.assertEqual(self.store.get_bytes(first), payload)
        self.assertTrue(self.store.check().healthy)

    def test_existing_artifact_and_readiness_protocols_are_implemented(self):
        self.assertIsInstance(self.store, ArtifactStore)
        self.assertIsInstance(self.store, ReadinessCheck)

        reference = self.store.put(
            ArtifactWrite(key="logical-name", content=b"contract payload")
        )
        self.assertTrue(reference.key.startswith("sha256/"))
        self.assertTrue(self.store.exists(reference.key))
        self.assertEqual(self.store.get(reference.key), b"contract payload")
        self.store.delete(reference.key)
        self.assertFalse(self.store.exists(reference.key))

    def test_json_artifacts_require_and_preserve_schema_version(self):
        with self.assertRaisesRegex(ValueError, "schema_version"):
            self.store.put_json({"value": 1})

        reference = self.store.put_json({"schema_version": "2", "value": "中文"})
        self.assertEqual(reference.schema_version, "2")
        self.assertEqual(
            self.store.get_bytes(reference).decode("utf-8"),
            '{"schema_version":"2","value":"中文"}',
        )

    def test_download_detects_corrupted_content(self):
        reference = self.store.put_bytes(b"original")
        stored = self.client.objects[(reference.bucket, reference.object_key)]
        stored["content"] = b"corrupt!"
        with self.assertRaises(ArtifactIntegrityError):
            self.store.get_bytes(reference)


class ComposeDefinitionTests(unittest.TestCase):
    def test_full_stage1_stack_has_health_checks_and_persistent_volumes(self):
        compose = (PROJECT_DIR / "docker-compose.yml").read_text(encoding="utf-8")
        self.assertIn("image: mysql:8.4", compose)
        self.assertIn("mysqladmin ping", compose)
        self.assertIn("minio/health/live", compose)
        self.assertIn("condition: service_healthy", compose)
        self.assertIn("mysql_data:/var/lib/mysql", compose)
        self.assertIn("minio_data:/data", compose)
        self.assertIn("  migrate:", compose)
        self.assertIn("  api:", compose)
        self.assertIn("  worker:", compose)
        self.assertIn("tender_review.api.main:app", compose)
        self.assertIn("tender_review.worker", compose)
        self.assertIn("--check-readiness", compose)
        self.assertTrue((PROJECT_DIR / "Dockerfile").is_file())
        self.assertTrue((PROJECT_DIR / ".dockerignore").is_file())


if __name__ == "__main__":
    unittest.main()
