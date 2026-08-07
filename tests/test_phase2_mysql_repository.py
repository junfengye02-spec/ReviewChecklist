from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, func, inspect, select
from sqlalchemy.dialects import mysql

from tender_review.infrastructure.database import Base, create_session_factory
from tender_review.infrastructure.database.models import (
    DocumentSnapshot,
    IdempotencyRecord as DbIdempotencyRecord,
    ModelConfig,
    ReviewJob as DbReviewJob,
    RuleSet,
    RuleVersion,
)
from tender_review.jobs.adapters import MySqlJobRepository
from tender_review.jobs.models import (
    CheckpointState,
    CheckpointValue,
    IdempotencyRecord,
    JobCheckpoint,
    JobFailure,
    JobLifecycle,
    JobResult,
    ReviewJob,
    ReviewStage,
)
from tender_review.shared.errors import ConflictError, ErrorCategory


NOW = datetime(2026, 7, 27, 10, 0, tzinfo=timezone.utc)


class MutableClock:
    def __init__(self, value: datetime) -> None:
        self.value = value

    def now(self) -> datetime:
        return self.value

    def advance(self, seconds: int) -> None:
        self.value += timedelta(seconds=seconds)


class RepositoryFixture:
    def __init__(self) -> None:
        self.temporary_directory = TemporaryDirectory()
        database_path = Path(self.temporary_directory.name) / "jobs.sqlite3"
        self.engine = create_engine(
            f"sqlite:///{database_path}",
            connect_args={"check_same_thread": False, "timeout": 5.0},
        )
        Base.metadata.create_all(self.engine)
        self.sessions = create_session_factory(self.engine)
        self.clock = MutableClock(NOW)
        self.repository = MySqlJobRepository(
            self.sessions,
            retry_base_seconds=2,
            retry_max_seconds=8,
            now_provider=self.clock.now,
        )
        self._seed_dependencies()

    def close(self) -> None:
        self.engine.dispose()
        self.temporary_directory.cleanup()

    def _seed_dependencies(self) -> None:
        with self.sessions.begin() as session:
            session.add(
                DocumentSnapshot(
                    id="document-1",
                    sha256="d" * 64,
                    object_key="sha256/document",
                    source_system="test",
                    source_document_id="source-document-1",
                    file_name="document.pdf",
                    size_bytes=1,
                )
            )
            session.add(
                ModelConfig(
                    id="model-1",
                    provider="fake",
                    model_name="fake-model",
                    prompt_version="v1",
                    config_hash="m" * 64,
                    parameters_json={},
                )
            )
            session.add(RuleSet(id="rules-1", rule_key="rules", name="Rules"))
            session.add(
                RuleVersion(
                    id="rule-version-1",
                    rule_set_id="rules-1",
                    version_number=1,
                    content_hash="r" * 64,
                    content_json={},
                    execution_config_json={},
                )
            )

    def job(self, job_id: str, *, max_attempts: int = 3) -> ReviewJob:
        return ReviewJob(
            id=job_id,
            document_snapshot_id="document-1",
            rule_version_id="rule-version-1",
            model_config_id="model-1",
            input_fingerprint="f" * 64,
            max_attempts=max_attempts,
            available_at=self.clock.now(),
            created_at=self.clock.now(),
            updated_at=self.clock.now(),
        )

    def idempotency(
        self, job_id: str, *, record_id: str, request_hash: str = "a" * 64
    ) -> IdempotencyRecord:
        return IdempotencyRecord(
            id=record_id,
            caller_id="caller",
            scope="POST:/api/v1/review-jobs",
            idempotency_key="request-key",
            request_hash=request_hash,
            resource_id=job_id,
            created_at=self.clock.now(),
        )


class MySqlJobRepositoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture = RepositoryFixture()

    def tearDown(self) -> None:
        self.fixture.close()

    def test_mysql_claim_statement_uses_skip_locked_and_all_due_states(self) -> None:
        statement = MySqlJobRepository.claim_candidate_statement(NOW)
        sql = str(statement.compile(dialect=mysql.dialect())).upper()

        self.assertIn("FOR UPDATE SKIP LOCKED", sql)
        self.assertIn("REVIEW_JOBS.STATUS IN", sql)
        self.assertIn("REVIEW_JOBS.STATUS =", sql)
        self.assertIn("REVIEW_JOBS.LEASE_UNTIL <=", sql)

    def test_phase2_alembic_revision_upgrades_the_preserved_initial_schema(self) -> None:
        with TemporaryDirectory() as directory:
            database_path = Path(directory) / "migration.sqlite3"
            config = Config(str(Path(__file__).parents[1] / "alembic.ini"))
            config.set_main_option("sqlalchemy.url", f"sqlite:///{database_path}")
            command.upgrade(config, "head")

            engine = create_engine(f"sqlite:///{database_path}")
            try:
                review_job_columns = {
                    column["name"] for column in inspect(engine).get_columns("review_jobs")
                }
                checkpoint_columns = {
                    column["name"]
                    for column in inspect(engine).get_columns("job_checkpoints")
                }
                self.assertTrue(
                    {
                        "job_type",
                        "input_reference",
                        "checkpoint_sequence",
                        "error_code",
                        "output_reference",
                        "output_summary",
                    }.issubset(review_job_columns)
                )
                self.assertIn("sequence", checkpoint_columns)
            finally:
                engine.dispose()

    def test_concurrent_idempotent_create_commits_one_business_job(self) -> None:
        repository = self.fixture.repository

        def create(index: int):
            job_id = f"job-{index}"
            return repository.create_review_job(
                self.fixture.job(job_id),
                self.fixture.idempotency(job_id, record_id=f"idem-{index}"),
            )

        with ThreadPoolExecutor(max_workers=6) as executor:
            results = tuple(executor.map(create, range(6)))

        self.assertEqual({result.job.id for result in results}, {results[0].job.id})
        self.assertEqual(sum(result.created for result in results), 1)
        with self.fixture.sessions() as session:
            self.assertEqual(session.scalar(select(func.count(DbReviewJob.id))), 1)
            self.assertEqual(
                session.scalar(select(func.count(DbIdempotencyRecord.id))), 1
            )

        replay_id = results[0].job.id
        replay = repository.create_review_job(
            self.fixture.job("unused-replay-id"),
            self.fixture.idempotency(
                "unused-replay-id", record_id="unused-replay-record"
            ),
        )
        self.assertFalse(replay.created)
        self.assertEqual(replay.job.id, replay_id)

        with self.assertRaisesRegex(ConflictError, "different request"):
            repository.create_review_job(
                self.fixture.job("different-job"),
                self.fixture.idempotency(
                    "different-job",
                    record_id="different-record",
                    request_hash="b" * 64,
                ),
            )

    def test_expired_job_is_reclaimed_and_old_lease_is_fenced(self) -> None:
        repository = self.fixture.repository
        created = repository.create_review_job(
            self.fixture.job("job-fence"),
            self.fixture.idempotency("job-fence", record_id="idem-fence"),
        )
        first = repository.claim_next(
            "worker-a", now=self.fixture.clock.now(), lease_seconds=10
        )
        self.assertIsNotNone(first)
        assert first is not None
        repository.save_checkpoint(
            created.job.id,
            first.lease,
            node_name="parse",
            stage="PARSING",
            state_json={"schema_version": 1, "values": []},
        )

        self.fixture.clock.advance(11)
        second = repository.claim_next(
            "worker-b", now=self.fixture.clock.now(), lease_seconds=10
        )
        self.assertIsNotNone(second)
        assert second is not None
        self.assertGreater(second.lease.token, first.lease.token)
        self.assertEqual(second.message.attempt, 2)

        with self.assertRaisesRegex(ConflictError, "fenced"):
            repository.mark_completed(
                created.job.id, first.lease.token, JobResult(summary="stale")
            )
        with self.assertRaisesRegex(ConflictError, "fenced"):
            repository.save_checkpoint(
                created.job.id,
                first.lease,
                node_name="parse",
                stage="PARSING",
                state_json={"schema_version": 1, "values": []},
            )

        latest = repository.save_checkpoint(
            created.job.id,
            second.lease,
            node_name="index",
            stage="INDEXING",
            state_json={"schema_version": 1, "values": []},
        )
        self.assertEqual(repository.load_latest_checkpoint(created.job.id), latest)
        repository.mark_completed(
            created.job.id,
            second.lease.token,
            JobResult(output_reference="artifact://report", summary="done"),
        )
        completed = repository.get_job(created.job.id)
        self.assertEqual(completed.status, "COMPLETED")
        self.assertEqual(completed.output_reference, "artifact://report")

    def test_cancel_fences_an_active_lease(self) -> None:
        repository = self.fixture.repository
        created = repository.create_review_job(
            self.fixture.job("job-cancel-fence")
        ).job
        claimed = repository.claim_next(
            "worker-a", now=self.fixture.clock.now(), lease_seconds=30
        )
        assert claimed is not None

        cancelled = repository.cancel(created.id)
        self.assertEqual(cancelled.status, "CANCELLED")
        self.assertGreater(cancelled.lease_token, claimed.lease.token)
        with self.assertRaisesRegex(ConflictError, "fenced"):
            repository.mark_completed(
                created.id, claimed.lease.token, JobResult(summary="stale")
            )

    def test_public_checkpoint_port_upserts_and_orders_completed_nodes(self) -> None:
        repository = self.fixture.repository
        created = repository.create_review_job(self.fixture.job("job-checkpoints"))
        claimed = repository.claim_next(
            "worker", now=self.fixture.clock.now(), lease_seconds=30
        )
        assert claimed is not None
        first = JobCheckpoint(
            job_id=created.job.id,
            node_name="parse",
            stage=ReviewStage.PARSING,
            lease_token=claimed.lease.token,
            state=CheckpointState(
                values=(CheckpointValue(key="cursor", value="page-1"),)
            ),
            completed_at=self.fixture.clock.now(),
        )
        repository.save_checkpoint(first)
        self.fixture.clock.advance(1)
        repository.save_checkpoint(
            first.model_copy(
                update={
                    "state": CheckpointState(
                        values=(CheckpointValue(key="cursor", value="page-2"),)
                    ),
                    "completed_at": self.fixture.clock.now(),
                }
            )
        )
        second = JobCheckpoint(
            job_id=created.job.id,
            node_name="index",
            stage=ReviewStage.INDEXING,
            lease_token=claimed.lease.token,
            state=CheckpointState(),
            completed_at=self.fixture.clock.now(),
        )
        repository.save_checkpoint(second)

        checkpoints = repository.list_checkpoints(created.job.id)
        self.assertEqual(tuple(item.node_name for item in checkpoints), ("parse", "index"))
        self.assertEqual(checkpoints[0].state.values[0].value, "page-2")

    def test_retry_backoff_dead_cancel_and_explicit_rerun_are_persisted(self) -> None:
        repository = self.fixture.repository
        retry_job = repository.create_review_job(
            self.fixture.job("job-retry", max_attempts=2)
        ).job
        first = repository.claim_next(
            "worker", now=self.fixture.clock.now(), lease_seconds=30
        )
        assert first is not None
        failure = JobFailure(
            code="model_timeout",
            message="model timed out",
            category=ErrorCategory.RETRYABLE,
            retryable=True,
            stage=ReviewStage.PARSING,
        )
        repository.mark_failed(retry_job.id, first.lease.token, failure)
        delayed = repository.get_review_job(retry_job.id)
        self.assertEqual(delayed.status, JobLifecycle.RETRY_WAIT)
        self.assertEqual(
            delayed.available_at.replace(tzinfo=timezone.utc), NOW + timedelta(seconds=2)
        )

        self.fixture.clock.advance(2)
        second = repository.claim_next(
            "worker", now=self.fixture.clock.now(), lease_seconds=30
        )
        assert second is not None
        repository.mark_failed(retry_job.id, second.lease.token, failure)
        dead = repository.get_review_job(retry_job.id)
        self.assertEqual(dead.status, JobLifecycle.DEAD)
        self.assertEqual(dead.failure.code, "model_timeout")

        rerun = repository.rerun(retry_job.id)
        self.assertEqual(rerun.status, "QUEUED")
        self.assertEqual(rerun.rerun_of_id, retry_job.id)
        self.assertNotEqual(rerun.id, retry_job.id)

        cancelled_job = repository.create_review_job(
            self.fixture.job("job-cancel")
        ).job
        cancelled = repository.cancel(cancelled_job.id)
        self.assertEqual(cancelled.status, "CANCELLED")
        queued = repository.next_queued()
        self.assertIsNotNone(queued)
        assert queued is not None
        self.assertEqual(queued.job_id, rerun.id)


if __name__ == "__main__":
    unittest.main()
