from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from alembic import command
from alembic.config import Config
from alembic.script import ScriptDirectory
from fastapi.testclient import TestClient
from pydantic import ValidationError
from sqlalchemy import create_engine, func, inspect, select, update

from tender_review.api import create_app
from tender_review.bootstrap import build_container
from tender_review.infrastructure.database import Base, create_session_factory
from tender_review.infrastructure.database.models import (
    DatasetVersion,
    DocumentArtifact,
    DocumentSnapshot,
    IdempotencyRecord as DbIdempotencyRecord,
    ModelConfig,
    ReviewExecutionSpec as DbReviewExecutionSpec,
    ReviewJob as DbReviewJob,
    RuleSet,
    RuleVersion,
)
from tender_review.jobs.adapters import MySqlJobRepository
from tender_review.jobs.public import (
    CreateReviewJobCommand,
    ExecutionArtifactReference,
    IdempotencyRecord,
    ReviewExecutionSpec,
    ReviewExecutionSpecDraft,
    ReviewExecutionSpecParser,
    ReviewJob,
    ReviewJobService,
    build_review_execution_spec,
)
from tender_review.review.public import TextPresenceRule
from tender_review.shared.clock import FixedClock
from tender_review.shared.config import AppSettings
from tender_review.shared.contracts import CallContext
from tender_review.shared.errors import ConflictError, NotFoundError
from tender_review.shared.ids import SequentialIdGenerator


NOW = datetime(2026, 8, 6, 8, 30, tzinfo=timezone.utc)
DOCUMENT_HASH = "1" * 64
RULE_HASH = "2" * 64
DATASET_HASH = "3" * 64
MODEL_HASH = "4" * 64
RETRIEVER_HASH = "5" * 64
INDEX_HASH = "6" * 64
CHUNK_HASH = "7" * 64
RESULTS_HASH = "8" * 64


def artifact_reference(role: str, digest: str) -> ExecutionArtifactReference:
    return ExecutionArtifactReference(
        artifact_id=f"{role}-artifact",
        bucket="review-artifacts",
        object_key=f"sha256/{digest[:2]}/{digest}",
        sha256=digest,
    )


def execution_draft(**updates: object) -> ReviewExecutionSpecDraft:
    payload: dict[str, object] = {
        "document_snapshot_id": "document-1",
        "document_sha256": DOCUMENT_HASH,
        "rule_version_id": "rule-version-1",
        "rule_version_hash": RULE_HASH,
        "dataset_version_id": "dataset-version-1",
        "dataset_version_hash": DATASET_HASH,
        "model_config_id": "model-1",
        "model_config_hash": MODEL_HASH,
        "query": "Does the tender contain the required signed authorization?",
        "retrieval_variant": "hybrid-rrf-v1",
        "retriever_artifact": artifact_reference("retriever", RETRIEVER_HASH),
        "index_artifact": artifact_reference("index", INDEX_HASH),
        "chunk_artifact": artifact_reference("chunk", CHUNK_HASH),
    }
    payload.update(updates)
    return ReviewExecutionSpecDraft.model_validate(payload)


def create_command(**draft_updates: object) -> CreateReviewJobCommand:
    draft = execution_draft(**draft_updates)
    return CreateReviewJobCommand(
        document_snapshot_id=draft.document_snapshot_id,
        document_sha256=draft.document_sha256,
        rule_version_id=draft.rule_version_id,
        rule_version_hash=draft.rule_version_hash,
        model_config_id=draft.model_config_id,
        model_config_hash=draft.model_config_hash,
        execution_spec=draft,
    )


def api_body(**draft_updates: object) -> dict[str, object]:
    command = create_command(**draft_updates)
    return command.model_dump(mode="json")


class ReviewExecutionSpecContractTests(unittest.TestCase):
    def test_hash_covers_job_and_every_input_and_model_is_frozen(self) -> None:
        first = build_review_execution_spec("job-1", execution_draft())
        second = build_review_execution_spec("job-2", execution_draft())
        changed = build_review_execution_spec(
            "job-1", execution_draft(query="A different review question")
        )

        self.assertNotEqual(first.input_sha256, second.input_sha256)
        self.assertNotEqual(first.input_sha256, changed.input_sha256)
        with self.assertRaises(ValidationError):
            first.query = "tampered"  # type: ignore[misc]

        payload = first.model_dump(mode="json")
        payload["query"] = "tampered"
        with self.assertRaisesRegex(ValidationError, "input_sha256"):
            ReviewExecutionSpec.model_validate(payload)

    def test_contract_has_no_secret_or_large_content_escape_hatch(self) -> None:
        payload = execution_draft().model_dump(mode="json")
        payload["api_key"] = "must-not-be-stored"

        with self.assertRaisesRegex(ValidationError, "Extra inputs"):
            ReviewExecutionSpecDraft.model_validate(payload)
        serialized = build_review_execution_spec(
            "job-1", execution_draft()
        ).model_dump_json()
        self.assertNotIn("api_key", serialized)
        self.assertNotIn("raw_text", serialized)
        self.assertLess(len(serialized), 10_000)

    def test_parser_builds_review_request_and_rejects_resolution_conflicts(self) -> None:
        spec = build_review_execution_spec("job-1", execution_draft())
        parser = ReviewExecutionSpecParser()
        rule = TextPresenceRule(
            review_item_id="authorization",
            field_name="authorization_text",
            required_terms=("signed", "authorization"),
        )
        call = CallContext(call_id="worker-call")

        request = parser.parse(
            spec,
            rule=rule,
            resolved_rule_version_id=spec.rule_version_id,
            resolved_rule_version_hash=spec.rule_version_hash,
            resolved_dataset_version_id=spec.dataset_version_id,
            resolved_dataset_version_hash=spec.dataset_version_hash,
            provenance_status="provisional",
            claims_allowed=False,
            retrieval_results_sha256=RESULTS_HASH,
            call=call,
        )

        self.assertEqual(request.review_job_id, spec.job_id)
        self.assertEqual(request.query, spec.query)
        self.assertEqual(request.document_ids, (spec.document_snapshot_id,))
        self.assertEqual(request.provenance.input_sha256, spec.input_sha256)
        self.assertEqual(request.provenance.variant, spec.retrieval_variant)

        with self.assertRaisesRegex(ConflictError, "dataset version"):
            parser.parse(
                spec,
                rule=rule,
                resolved_rule_version_id=spec.rule_version_id,
                resolved_rule_version_hash=spec.rule_version_hash,
                resolved_dataset_version_id="another-dataset",
                resolved_dataset_version_hash=spec.dataset_version_hash,
                provenance_status="provisional",
                claims_allowed=False,
                retrieval_results_sha256=RESULTS_HASH,
                call=call,
            )


class ReviewExecutionSpecApiTests(unittest.TestCase):
    def setUp(self) -> None:
        container = build_container(
            AppSettings(environment="test", adapter_mode="fake", log_json=False)
        )
        service = ReviewJobService(
            repository=container.job_repository,
            ids=SequentialIdGenerator(prefix="a2-api"),
            clock=FixedClock(NOW),
        )
        self.client = TestClient(
            create_app(container.with_overrides(review_jobs=service))
        )

    def test_explicit_spec_round_trips_while_legacy_creation_stays_unprovenanced(self) -> None:
        created = self.client.post(
            "/api/v1/review-jobs",
            json=api_body(),
            headers={"Idempotency-Key": "with-spec"},
        )
        self.assertEqual(created.status_code, 201, created.text)
        job = created.json()
        spec_response = self.client.get(
            f"/api/v1/review-jobs/{job['id']}/execution-spec"
        )
        self.assertEqual(spec_response.status_code, 200, spec_response.text)
        self.assertEqual(
            job["execution_spec_sha256"], spec_response.json()["input_sha256"]
        )

        legacy_body = {
            key: value
            for key, value in api_body().items()
            if key != "execution_spec"
        }
        legacy = self.client.post(
            "/api/v1/review-jobs",
            json=legacy_body,
            headers={"Idempotency-Key": "legacy"},
        )
        self.assertEqual(legacy.status_code, 201, legacy.text)
        self.assertIsNone(legacy.json()["execution_spec_sha256"])
        missing = self.client.get(
            f"/api/v1/review-jobs/{legacy.json()['id']}/execution-spec"
        )
        self.assertEqual(missing.status_code, 404)
        self.assertEqual(
            missing.json()["error"]["code"], "review_execution_spec_not_found"
        )

    def test_api_rejects_secret_fields_and_identity_mismatch(self) -> None:
        secret = api_body()
        secret["execution_spec"]["api_key"] = "secret"  # type: ignore[index]
        rejected_secret = self.client.post(
            "/api/v1/review-jobs",
            json=secret,
            headers={"Idempotency-Key": "secret"},
        )
        mismatch = api_body()
        mismatch["execution_spec"]["document_sha256"] = "9" * 64  # type: ignore[index]
        rejected_mismatch = self.client.post(
            "/api/v1/review-jobs",
            json=mismatch,
            headers={"Idempotency-Key": "mismatch"},
        )

        self.assertEqual(rejected_secret.status_code, 422)
        self.assertEqual(rejected_mismatch.status_code, 422)


class SqlExecutionSpecFixture:
    def __init__(self) -> None:
        self.directory = TemporaryDirectory()
        path = Path(self.directory.name) / "execution-specs.sqlite3"
        self.engine = create_engine(f"sqlite:///{path.as_posix()}")
        Base.metadata.create_all(self.engine)
        self.sessions = create_session_factory(self.engine)
        self.repository = MySqlJobRepository(self.sessions, now_provider=lambda: NOW)
        self.service = ReviewJobService(
            repository=self.repository,
            ids=SequentialIdGenerator(prefix="a2-sql"),
            clock=FixedClock(NOW),
        )
        self._seed_references()

    def close(self) -> None:
        self.engine.dispose()
        self.directory.cleanup()

    def _seed_references(self) -> None:
        with self.sessions.begin() as session:
            session.add(
                DocumentSnapshot(
                    id="document-1",
                    sha256=DOCUMENT_HASH,
                    object_key="sha256/document",
                    source_system="test",
                    source_document_id="source-1",
                    file_name="tender.pdf",
                    size_bytes=1,
                )
            )
            session.add(
                ModelConfig(
                    id="model-1",
                    provider="openai-compatible",
                    model_name="configured-outside-spec",
                    prompt_version="v1",
                    config_hash=MODEL_HASH,
                    parameters_json={},
                )
            )
            session.add(RuleSet(id="rules-1", rule_key="qualification", name="Rules"))
            session.add(
                RuleVersion(
                    id="rule-version-1",
                    rule_set_id="rules-1",
                    version_number=1,
                    content_hash=RULE_HASH,
                    content_json={},
                    execution_config_json={},
                )
            )
            session.add(
                DatasetVersion(
                    id="dataset-version-1",
                    dataset_name="a2-provisional",
                    version_number=1,
                    manifest_hash=DATASET_HASH,
                    source_type="SYNTHETIC",
                    status="PROVISIONAL",
                    split_strategy_json={},
                )
            )
            for role, digest, artifact_type in (
                ("retriever", RETRIEVER_HASH, "index"),
                ("index", INDEX_HASH, "index"),
                ("chunk", CHUNK_HASH, "parsed_json"),
            ):
                reference = artifact_reference(role, digest)
                session.add(
                    DocumentArtifact(
                        id=reference.artifact_id,
                        document_snapshot_id="document-1",
                        artifact_type=artifact_type,
                        bucket=reference.bucket,
                        object_key=reference.object_key,
                        sha256=reference.sha256,
                        size_bytes=1,
                        media_type="application/json",
                        metadata_json={},
                    )
                )


class SqlAlchemyReviewExecutionSpecTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture = SqlExecutionSpecFixture()

    def tearDown(self) -> None:
        self.fixture.close()

    def test_atomic_create_idempotent_replay_and_database_round_trip(self) -> None:
        first = self.fixture.service.create(
            create_command(),
            caller_id="caller",
            scope="POST:/api/v1/review-jobs",
            idempotency_key="stable-key",
        )
        replay = self.fixture.service.create(
            create_command(),
            caller_id="caller",
            scope="POST:/api/v1/review-jobs",
            idempotency_key="stable-key",
        )
        restored = self.fixture.service.get_execution_spec(first.job.id)

        self.assertTrue(first.created)
        self.assertFalse(replay.created)
        self.assertEqual(replay.job.id, first.job.id)
        self.assertEqual(restored.input_sha256, first.job.execution_spec_sha256)
        self.assertEqual(restored, self.fixture.repository.get_review_execution_spec(first.job.id))
        with self.fixture.sessions() as session:
            self.assertEqual(session.scalar(select(func.count(DbReviewJob.id))), 1)
            self.assertEqual(
                session.scalar(select(func.count(DbReviewExecutionSpec.job_id))), 1
            )
            self.assertEqual(
                session.scalar(select(func.count(DbIdempotencyRecord.id))), 1
            )

        with self.assertRaisesRegex(ConflictError, "different request"):
            self.fixture.service.create(
                create_command(query="changed query"),
                caller_id="caller",
                scope="POST:/api/v1/review-jobs",
                idempotency_key="stable-key",
            )

    def test_missing_reference_rolls_back_job_and_idempotency(self) -> None:
        missing = artifact_reference("missing", "9" * 64)
        with self.assertRaises(NotFoundError) as raised:
            self.fixture.service.create(
                create_command(index_artifact=missing),
                caller_id="caller",
                scope="POST:/api/v1/review-jobs",
                idempotency_key="missing-reference",
            )

        self.assertEqual(raised.exception.code, "review_execution_reference_missing")
        with self.fixture.sessions() as session:
            self.assertEqual(session.scalar(select(func.count(DbReviewJob.id))), 0)
            self.assertEqual(
                session.scalar(select(func.count(DbReviewExecutionSpec.job_id))), 0
            )
            self.assertEqual(
                session.scalar(select(func.count(DbIdempotencyRecord.id))), 0
            )

    def test_repository_rejects_same_request_hash_with_changed_replay_spec(self) -> None:
        created = self.fixture.service.create(
            create_command(),
            caller_id="caller",
            scope="POST:/api/v1/review-jobs",
            idempotency_key="replay-conflict",
        )
        record = self.fixture.repository.get_idempotency_record(
            "caller", "POST:/api/v1/review-jobs", "replay-conflict"
        )
        assert record is not None
        replay_spec = build_review_execution_spec(
            "changed-replay-job", execution_draft(query="changed replay query")
        )
        replay_job = ReviewJob(
            id=replay_spec.job_id,
            document_snapshot_id=replay_spec.document_snapshot_id,
            rule_version_id=replay_spec.rule_version_id,
            model_config_id=replay_spec.model_config_id,
            input_fingerprint="9" * 64,
            execution_spec_sha256=replay_spec.input_sha256,
            available_at=NOW,
            created_at=NOW,
            updated_at=NOW,
        )
        replay_record = IdempotencyRecord(
            id="changed-replay-record",
            caller_id=record.caller_id,
            scope=record.scope,
            idempotency_key=record.idempotency_key,
            request_hash=record.request_hash,
            resource_id=replay_job.id,
            created_at=NOW,
        )

        with self.assertRaises(ConflictError) as raised:
            self.fixture.repository.create_review_job(
                replay_job, replay_record, replay_spec
            )
        self.assertEqual(
            raised.exception.code, "review_execution_spec_replay_conflict"
        )
        self.assertEqual(
            self.fixture.service.get_execution_spec(created.job.id).job_id,
            created.job.id,
        )

    def test_explicit_rerun_clones_inputs_and_rebinds_the_spec_hash(self) -> None:
        created = self.fixture.service.create(
            create_command(),
            caller_id="caller",
            scope="POST:/api/v1/review-jobs",
            idempotency_key="rerun-source",
        ).job
        source_spec = self.fixture.service.get_execution_spec(created.id)
        self.fixture.service.cancel(created.id)

        rerun = self.fixture.service.rerun(created.id)
        rerun_spec = self.fixture.service.get_execution_spec(rerun.id)

        self.assertEqual(rerun.rerun_of, created.id)
        self.assertEqual(rerun.input_fingerprint, created.input_fingerprint)
        self.assertEqual(rerun_spec.job_id, rerun.id)
        self.assertNotEqual(rerun_spec.input_sha256, source_spec.input_sha256)
        self.assertEqual(
            rerun_spec.model_dump(exclude={"job_id", "input_sha256"}),
            source_spec.model_dump(exclude={"job_id", "input_sha256"}),
        )

    def test_tampered_payload_and_changed_referenced_hash_are_rejected(self) -> None:
        created = self.fixture.service.create(
            create_command(),
            caller_id="caller",
            scope="POST:/api/v1/review-jobs",
            idempotency_key="tamper",
        )
        with self.fixture.sessions.begin() as session:
            row = session.get(DbReviewExecutionSpec, created.job.id)
            assert row is not None
            payload = dict(row.spec_json)
            payload["query"] = "tampered query"
            row.spec_json = payload

        with self.assertRaises(ConflictError) as tampered:
            self.fixture.service.get_execution_spec(created.job.id)
        self.assertEqual(tampered.exception.code, "review_execution_spec_tampered")

        with self.fixture.sessions.begin() as session:
            row = session.get(DbReviewExecutionSpec, created.job.id)
            assert row is not None
            row.spec_json = build_review_execution_spec(
                created.job.id, execution_draft()
            ).model_dump(mode="json")
            session.execute(
                update(DocumentArtifact)
                .where(DocumentArtifact.id == "index-artifact")
                .values(sha256="a" * 64)
            )

        with self.assertRaises(ConflictError) as changed_reference:
            self.fixture.service.get_execution_spec(created.job.id)
        self.assertEqual(
            changed_reference.exception.code, "review_execution_reference_conflict"
        )


class ReviewExecutionSpecMigrationTests(unittest.TestCase):
    def test_single_head_upgrade_and_downgrade(self) -> None:
        config = Config(str(Path(__file__).parents[1] / "alembic.ini"))
        self.assertEqual(
            ScriptDirectory.from_config(config).get_heads(), ["d5b0f6a8c214"]
        )
        with TemporaryDirectory() as directory:
            database_path = Path(directory) / "migration.sqlite3"
            database_url = f"sqlite:///{database_path.as_posix()}"
            config.set_main_option("sqlalchemy.url", database_url)

            command.upgrade(config, "head")
            engine = create_engine(database_url)
            inspector = inspect(engine)
            self.assertIn("review_execution_specs", inspector.get_table_names())
            review_job_columns = {
                item["name"] for item in inspector.get_columns("review_jobs")
            }
            self.assertIn("execution_spec_sha256", review_job_columns)
            spec_columns = {
                item["name"]
                for item in inspector.get_columns("review_execution_specs")
            }
            self.assertTrue(
                {
                    "job_id",
                    "input_sha256",
                    "spec_json",
                    "dataset_version_id",
                    "retriever_artifact_id",
                    "index_artifact_id",
                    "chunk_artifact_id",
                }.issubset(spec_columns)
            )
            engine.dispose()

            command.downgrade(config, "f6a1b2c3d4e5")
            engine = create_engine(database_url)
            inspector = inspect(engine)
            self.assertNotIn("review_execution_specs", inspector.get_table_names())
            review_job_columns = {
                item["name"] for item in inspector.get_columns("review_jobs")
            }
            self.assertNotIn("execution_spec_sha256", review_job_columns)
            engine.dispose()

            command.upgrade(config, "head")
            engine = create_engine(database_url)
            inspector = inspect(engine)
            self.assertIn("review_execution_specs", inspector.get_table_names())
            self.assertIn(
                "execution_spec_sha256",
                {item["name"] for item in inspector.get_columns("review_jobs")},
            )
            engine.dispose()


if __name__ == "__main__":
    unittest.main()
