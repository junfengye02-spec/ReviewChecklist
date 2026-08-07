from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
import time
import unittest

from sqlalchemy import create_engine, select

from tender_review.infrastructure.database import Base, create_session_factory
from tender_review.infrastructure.database.models import (
    DocumentSnapshot,
    ModelConfig,
    ReviewJob as DbReviewJob,
    RuleSet,
    RuleVersion,
)
from tender_review.jobs.adapters import MySqlJobRepository
from tender_review.jobs.fakes import FakeLeaseManager, InMemoryJobRepository
from tender_review.jobs.models import JobMessage, JobResult, ReviewJob
from tender_review.shared.clock import SystemClock
from tender_review.shared.errors import ConflictError
from tender_review.worker import Worker


NOW = datetime(2026, 7, 27, 12, 0, tzinfo=timezone.utc)


class MutableClock:
    def __init__(self, value: datetime) -> None:
        self.value = value

    def now(self) -> datetime:
        return self.value

    def advance(self, seconds: int) -> None:
        self.value += timedelta(seconds=seconds)


class DurableWorkerFixture:
    def __init__(self) -> None:
        self.temporary_directory = TemporaryDirectory()
        database_path = Path(self.temporary_directory.name) / "worker.sqlite3"
        self.engine = create_engine(
            f"sqlite:///{database_path}",
            connect_args={"check_same_thread": False, "timeout": 5.0},
        )
        Base.metadata.create_all(self.engine)
        self.sessions = create_session_factory(self.engine)
        self.clock = MutableClock(NOW)
        self.repository = MySqlJobRepository(
            self.sessions, now_provider=self.clock.now
        )
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
                    model_name="fake",
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

    def close(self) -> None:
        self.engine.dispose()
        self.temporary_directory.cleanup()

    def create_job(self, job_id: str) -> ReviewJob:
        return self.repository.create_review_job(
            ReviewJob(
                id=job_id,
                document_snapshot_id="document-1",
                rule_version_id="rule-version-1",
                model_config_id="model-1",
                input_fingerprint="f" * 64,
                available_at=self.clock.now(),
                created_at=self.clock.now(),
                updated_at=self.clock.now(),
            )
        ).job


class CountingLeaseManager(FakeLeaseManager):
    def __init__(self) -> None:
        super().__init__()
        self.renewal_count = 0

    def renew(self, *args, **kwargs):
        renewed = super().renew(*args, **kwargs)
        self.renewal_count += 1
        return renewed


class ReliableWorkerTests(unittest.TestCase):
    def test_single_process_sqlite_expired_and_reclaimed_leases_fence_old_writes(
        self,
    ) -> None:
        fixture = DurableWorkerFixture()
        try:
            job = fixture.create_job("job-fence-expired")
            old_claim = fixture.repository.claim_next(
                "worker-old", now=fixture.clock.now(), lease_seconds=5
            )
            assert old_claim is not None
            fixture.clock.advance(5)

            with self.assertRaises(ConflictError) as expired_checkpoint:
                fixture.repository.save_checkpoint(
                    job.id,
                    old_claim.lease,
                    node_name="expired",
                    stage="RETRIEVING",
                    state_json={"schema_version": 1, "values": []},
                )
            self.assertEqual(expired_checkpoint.exception.code, "stale_lease")
            with self.assertRaises(ConflictError) as expired_final:
                fixture.repository.mark_completed(
                    job.id, old_claim.lease.token, JobResult(summary="stale")
                )
            self.assertEqual(expired_final.exception.code, "stale_lease")

            new_claim = fixture.repository.claim_next(
                "worker-new", now=fixture.clock.now(), lease_seconds=5
            )
            assert new_claim is not None
            self.assertGreater(new_claim.lease.token, old_claim.lease.token)
            with self.assertRaises(ConflictError) as reclaimed_final:
                fixture.repository.mark_completed(
                    job.id, old_claim.lease.token, JobResult(summary="older token")
                )
            self.assertEqual(reclaimed_final.exception.code, "stale_lease")
            fixture.repository.mark_completed(
                job.id, new_claim.lease.token, JobResult(summary="current owner")
            )
            self.assertEqual(fixture.repository.get_job(job.id).status, "COMPLETED")
        finally:
            fixture.close()

    def test_worker_reclaims_from_latest_checkpoint_and_executes_outside_claim(self) -> None:
        fixture = DurableWorkerFixture()
        try:
            job = fixture.create_job("job-recovery")
            abandoned = fixture.repository.claim_next(
                "worker-a", now=fixture.clock.now(), lease_seconds=5
            )
            assert abandoned is not None
            fixture.repository.save_checkpoint(
                job.id,
                abandoned.lease,
                node_name="parse",
                stage="PARSING",
                state_json={
                    "schema_version": 1,
                    "values": [
                        {"schema_version": 1, "key": "cursor", "value": "page-4"}
                    ],
                },
            )
            fixture.clock.advance(6)
            observed: dict[str, object] = {}

            def handler(message, context):
                observed["checkpoint"] = context.latest_checkpoint().node_name
                with fixture.sessions() as session:
                    observed["status_during_handler"] = session.scalar(
                        select(DbReviewJob.status).where(DbReviewJob.id == message.job_id)
                    )
                context.save_checkpoint(
                    node_name="index",
                    stage="INDEXING",
                    state_json={"schema_version": 1, "values": []},
                )
                return JobResult(
                    output_reference="artifact://report", summary="recovered"
                )

            worker = Worker(
                worker_id="worker-b",
                repository=fixture.repository,
                leases=fixture.repository,
                handlers={"review": handler},
                clock=fixture.clock,
                lease_seconds=5,
                poll_interval_seconds=0,
            )

            self.assertTrue(worker.run_once())
            self.assertEqual(observed["checkpoint"], "parse")
            self.assertEqual(observed["status_during_handler"], "RUNNING")
            durable = fixture.repository.get_job(job.id)
            self.assertEqual(durable.status, "COMPLETED")
            self.assertEqual(durable.attempt_count, 2)
            self.assertEqual(
                fixture.repository.load_latest_checkpoint(job.id).node_name, "index"
            )
        finally:
            fixture.close()

    def test_worker_reclaims_interruptions_in_each_review_processing_stage(self) -> None:
        drill_stages = (
            ("PARSING", "PARSING"),
            ("RETRIEVING", "RETRIEVING"),
            # The durable job state machine names the first review substage
            # EXTRACTING; REVIEWING is the fault-drill label, not an enum value.
            ("REVIEWING", "EXTRACTING"),
        )
        for drill_stage, persisted_stage in drill_stages:
            with self.subTest(stage=drill_stage):
                fixture = DurableWorkerFixture()
                try:
                    job = fixture.create_job(f"job-recovery-{drill_stage.lower()}")
                    abandoned = fixture.repository.claim_next(
                        "worker-a", now=fixture.clock.now(), lease_seconds=5
                    )
                    assert abandoned is not None
                    checkpoint_name = f"{drill_stage.lower()}-checkpoint"
                    fixture.repository.save_checkpoint(
                        job.id,
                        abandoned.lease,
                        node_name=checkpoint_name,
                        stage=persisted_stage,
                        state_json={
                            "schema_version": 1,
                            "values": [
                                {
                                    "schema_version": 1,
                                    "key": "resume-stage",
                                    "value": drill_stage,
                                }
                            ],
                        },
                    )
                    fixture.clock.advance(6)
                    observed: dict[str, object] = {}

                    def handler(message, context):
                        observed["attempt"] = message.attempt
                        observed["checkpoint"] = context.latest_checkpoint().node_name
                        observed["stage"] = context.latest_checkpoint().stage
                        return JobResult(
                            output_reference=f"artifact://recovered/{drill_stage.lower()}",
                            summary=f"recovered from {drill_stage}",
                        )

                    worker = Worker(
                        worker_id="worker-b",
                        repository=fixture.repository,
                        leases=fixture.repository,
                        handlers={"review": handler},
                        clock=fixture.clock,
                        lease_seconds=5,
                        poll_interval_seconds=0,
                    )

                    self.assertTrue(worker.run_once())
                    self.assertEqual(observed["attempt"], 2)
                    self.assertEqual(observed["checkpoint"], checkpoint_name)
                    self.assertEqual(observed["stage"], persisted_stage)
                    recovered = fixture.repository.get_job(job.id)
                    self.assertEqual(recovered.status, "COMPLETED")
                    self.assertEqual(recovered.attempt_count, 2)
                    self.assertEqual(
                        recovered.output_reference,
                        f"artifact://recovered/{drill_stage.lower()}",
                    )
                finally:
                    fixture.close()

    def test_background_heartbeat_renews_long_running_legacy_handler(self) -> None:
        repository = InMemoryJobRepository()
        leases = CountingLeaseManager()
        repository.enqueue(
            JobMessage(
                job_id="legacy-job",
                job_type="review",
                input_reference="memory://document",
            )
        )

        def handler(job):
            del job
            time.sleep(0.45)
            return JobResult(summary="done")

        worker = Worker(
            worker_id="worker-heartbeat",
            repository=repository,
            leases=leases,
            handlers={"review": handler},
            clock=SystemClock(),
            lease_seconds=1,
            poll_interval_seconds=0,
        )

        self.assertTrue(worker.run_once())
        self.assertGreaterEqual(leases.renewal_count, 1)
        self.assertEqual(repository.completed["legacy-job"].summary, "done")


if __name__ == "__main__":
    unittest.main()
