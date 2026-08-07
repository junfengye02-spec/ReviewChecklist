from __future__ import annotations

import unittest
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

from sqlalchemy import create_engine, event, func, select

from tender_review.documents.lifecycle import (
    ArtifactSubmission,
    ArtifactType,
    ArtifactValidationError,
    DocumentLifecycleService,
    OrphanCleanupService,
    SourceDocument,
)
from tender_review.documents.storage import InMemoryContentAddressedStore
from tender_review.infrastructure.database import Base, create_session_factory
from tender_review.infrastructure.database.document_lifecycle import (
    SqlAlchemyDocumentLifecycleRepository,
)
from tender_review.infrastructure.database.models import DocumentArtifact, DocumentSnapshot
from tender_review.shared.errors import ConflictError


NOW = datetime(2026, 7, 27, 12, 0, tzinfo=timezone.utc)


class LifecycleFixture:
    def __init__(self) -> None:
        self.temp = TemporaryDirectory()
        self.engine = create_engine(
            f"sqlite:///{Path(self.temp.name) / 'lifecycle.db'}",
            connect_args={"check_same_thread": False},
        )
        Base.metadata.create_all(self.engine)
        self.sessions = create_session_factory(self.engine)
        self.store = InMemoryContentAddressedStore(bucket="artifacts", now_provider=lambda: NOW)
        self.repository = SqlAlchemyDocumentLifecycleRepository(
            self.sessions, snapshot_bucket="artifacts"
        )
        self.service = DocumentLifecycleService(storage=self.store, repository=self.repository)

    def close(self) -> None:
        self.engine.dispose()
        self.temp.cleanup()


class LifecycleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture = LifecycleFixture()

    def tearDown(self) -> None:
        self.fixture.close()

    def test_upload_is_idempotent_and_same_source_hash_conflict_is_explicit(self):
        document = SourceDocument("source", "doc-1", "one.pdf", b"payload")
        first = self.fixture.service.upload_document(document)
        replay = self.fixture.service.upload_document(document)
        self.assertTrue(first.created)
        self.assertFalse(replay.created)
        self.assertEqual(first.snapshot.id, replay.snapshot.id)
        with self.assertRaisesRegex(ConflictError, "different immutable content"):
            self.fixture.service.upload_document(
                SourceDocument("source", "doc-1", "one.pdf", b"different")
            )
        with self.assertRaisesRegex(ConflictError, "different source"):
            self.fixture.service.upload_document(
                SourceDocument("other", "doc-2", "two.pdf", b"payload")
            )

    def test_artifact_schema_media_contract_and_new_object_versions(self):
        snapshot = self.fixture.service.upload_document(
            SourceDocument("source", "doc-1", "one.pdf", b"payload")
        ).snapshot
        with self.assertRaises(ArtifactValidationError):
            ArtifactSubmission(
                snapshot.id,
                ArtifactType.PARSED_JSON,
                b'{"value": 1}',
                "application/json",
                "1",
            )
        first = self.fixture.service.write_artifact(
            ArtifactSubmission(
                snapshot.id,
                ArtifactType.REPORT,
                b'{"schema_version":"1","value":"old"}',
                "application/json",
                "1",
            )
        )
        second = self.fixture.service.write_artifact(
            ArtifactSubmission(
                snapshot.id,
                ArtifactType.REPORT,
                b'{"schema_version":"1","value":"new"}',
                "application/json",
                "1",
            )
        )
        self.assertTrue(first.created)
        self.assertTrue(second.created)
        self.assertNotEqual(first.artifact.id, second.artifact.id)
        self.assertEqual(len(self.fixture.repository.list_artifacts(snapshot.id)), 2)
        self.assertEqual(self.fixture.service.read_artifact(second.artifact.id), b'{"schema_version":"1","value":"new"}')

    def test_database_commit_failure_leaves_a_scannable_orphan(self):
        def fail_commit(session) -> None:  # type: ignore[no-untyped-def]
            del session
            raise RuntimeError("forced database commit failure")

        session_class = self.fixture.sessions.class_
        event.listen(session_class, "before_commit", fail_commit)
        try:
            with self.assertRaisesRegex(RuntimeError, "forced database commit failure"):
                self.fixture.service.upload_document(
                    SourceDocument("source", "doc-1", "one.pdf", b"orphan")
                )
        finally:
            event.remove(session_class, "before_commit", fail_commit)
        with self.fixture.sessions() as session:
            self.assertEqual(session.scalar(select(func.count(DocumentSnapshot.id))), 0)
        self.assertEqual(len(self.fixture.store.list_content_addressed(limit=10).objects), 1)
        cleanup = OrphanCleanupService(
            storage=self.fixture.store,
            repository=self.fixture.repository,
            now_provider=lambda: NOW + timedelta(days=2),
        )
        result = cleanup.clean_page(retention=timedelta(days=1), page_size=10, delete_limit=10)
        self.assertEqual((result.scanned, result.deleted), (1, 1))
        self.assertFalse(self.fixture.store.list_content_addressed(limit=10).objects)

    def test_database_failure_and_orphan_cleanup_share_a_safe_decision_trace(self):
        records: list[logging.LogRecord] = []

        class Capture(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                records.append(record)

        handler = Capture()
        loggers = tuple(
            logging.getLogger(name)
            for name in (
                "tender_review.documents.lifecycle",
                "tender_review.documents.cleanup",
            )
        )
        for logger in loggers:
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)

        def fail_commit(session) -> None:  # type: ignore[no-untyped-def]
            del session
            raise RuntimeError("database unavailable with sensitive row text")

        session_class = self.fixture.sessions.class_
        event.listen(session_class, "before_commit", fail_commit)
        try:
            with self.assertRaises(RuntimeError):
                self.fixture.service.upload_document(
                    SourceDocument("source", "trace", "trace.pdf", b"trace-orphan")
                )
        finally:
            event.remove(session_class, "before_commit", fail_commit)
        try:
            cleanup = OrphanCleanupService(
                storage=self.fixture.store,
                repository=self.fixture.repository,
                now_provider=lambda: NOW + timedelta(days=2),
            )
            result = cleanup.clean_page(retention=timedelta(days=1))
        finally:
            for logger in loggers:
                logger.removeHandler(handler)

        failure = next(
            record
            for record in records
            if getattr(record, "event", None) == "object.database_commit_failed"
        )
        deletion = next(
            record
            for record in records
            if getattr(record, "cleanup_decision", None) == "deleted_orphan"
        )
        self.assertEqual(result.deleted, 1)
        self.assertEqual(failure.call_id, deletion.call_id)
        self.assertTrue(failure.call_id.startswith("object:"))
        self.assertEqual(failure.cleanup_decision, "deferred_retention_cleanup")
        self.assertEqual(failure.error_type, "RuntimeError")
        self.assertNotIn("sensitive row text", failure.getMessage())

    def test_cleanup_rechecks_a_candidate_before_delete(self):
        reference = self.fixture.store.put_content(
            b"race", media_type="application/pdf", schema_version="1", created_at=NOW - timedelta(days=2)
        )

        class ReferencedAfterFirstCheck:
            calls = 0

            def is_object_referenced(self, candidate) -> bool:  # type: ignore[no-untyped-def]
                del candidate
                self.calls += 1
                return self.calls > 1

        cleanup = OrphanCleanupService(
            storage=self.fixture.store,
            repository=ReferencedAfterFirstCheck(),  # type: ignore[arg-type]
            now_provider=lambda: NOW,
        )
        result = cleanup.clean_page(retention=timedelta(days=1), page_size=10)
        self.assertEqual((result.deleted, result.retained), (0, 1))
        self.assertEqual(self.fixture.store.read_content(reference), b"race")

    def test_cleanup_retains_recent_and_referenced_objects_and_honors_batch_limit(self):
        old = self.fixture.store.put_content(b"old", media_type="application/pdf", schema_version="1", created_at=NOW - timedelta(days=2))
        recent = self.fixture.store.put_content(b"recent", media_type="application/pdf", schema_version="1", created_at=NOW)
        snapshot = self.fixture.service.upload_document(SourceDocument("source", "doc", "doc.pdf", b"referenced")).snapshot
        referenced = snapshot.object
        cleanup = OrphanCleanupService(storage=self.fixture.store, repository=self.fixture.repository, now_provider=lambda: NOW)
        result = cleanup.clean_page(retention=timedelta(days=1), page_size=10, delete_limit=1)
        self.assertEqual(result.deleted, 1)
        self.assertTrue(self.fixture.store.read_content(recent))
        self.assertTrue(self.fixture.store.read_content(referenced))
        self.assertFalse(self.fixture.store.list_content_addressed(limit=10).objects[0].object_key == old.object_key)

    def test_metadata_and_foreign_keys_are_persisted(self):
        snapshot = self.fixture.service.upload_document(SourceDocument("source", "doc", "doc.pdf", b"x")).snapshot
        artifact = self.fixture.service.write_artifact(
            ArtifactSubmission(snapshot.id, ArtifactType.TABLES, b"a,b\n1,2\n", "text/csv", "3", {"columns": ["a", "b"]})
        ).artifact
        with self.fixture.sessions() as session:
            row = session.scalar(select(DocumentArtifact).where(DocumentArtifact.id == artifact.id))
            self.assertEqual(row.schema_version, "3")
            self.assertEqual(row.metadata_json["columns"], ["a", "b"])
            self.assertIsNotNone(session.get(DocumentSnapshot, snapshot.id))


if __name__ == "__main__":
    unittest.main()
