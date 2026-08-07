from __future__ import annotations

import unittest
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

from tender_review.jobs.fakes import InMemoryJobRepository
from tender_review.jobs.domain import transition_lifecycle
from tender_review.jobs.public import (
    CheckpointState,
    CheckpointValue,
    CreateReviewJobCommand,
    JobCheckpoint,
    JobFailure,
    JobLifecycle,
    ReviewJobService,
    ReviewStage,
    execution_fingerprint,
    normalized_request_hash,
)
from tender_review.shared.clock import FixedClock
from tender_review.shared.errors import ConflictError, ErrorCategory
from tender_review.shared.ids import SequentialIdGenerator, UuidGenerator


NOW = datetime(2026, 7, 27, 8, 0, tzinfo=timezone.utc)


def command(*, max_attempts: int = 3, model_hash: str = "c" * 64):
    return CreateReviewJobCommand(
        document_snapshot_id=" document-1 ",
        document_sha256="A" * 64,
        rule_version_id=" rule-1 ",
        rule_version_hash="B" * 64,
        model_config_id=" model-1 ",
        model_config_hash=model_hash,
        max_attempts=max_attempts,
    )


def service(
    repository: InMemoryJobRepository | None = None,
    *,
    ids=None,
) -> tuple[ReviewJobService, InMemoryJobRepository]:
    resolved_repository = repository or InMemoryJobRepository()
    return (
        ReviewJobService(
            repository=resolved_repository,
            ids=ids or SequentialIdGenerator(prefix="resource"),
            clock=FixedClock(NOW),
        ),
        resolved_repository,
    )


def create_job(review_jobs: ReviewJobService, **overrides):
    values = {
        "caller_id": "caller-1",
        "scope": "POST:/api/v1/review-jobs",
        "idempotency_key": "request-1",
    }
    values.update(overrides)
    return review_jobs.create(command(), **values).job


class ReviewJobDomainTests(unittest.TestCase):
    def test_lifecycle_state_graph_accepts_only_declared_edges(self):
        review_jobs, _ = service()
        base = create_job(review_jobs)
        legal = {
            JobLifecycle.QUEUED: {JobLifecycle.RUNNING, JobLifecycle.CANCELLED},
            JobLifecycle.RUNNING: {
                JobLifecycle.WAITING_HUMAN,
                JobLifecycle.COMPLETED,
                JobLifecycle.RETRY_WAIT,
                JobLifecycle.FAILED,
                JobLifecycle.DEAD,
                JobLifecycle.CANCELLED,
            },
            JobLifecycle.WAITING_HUMAN: {JobLifecycle.COMPLETED},
            JobLifecycle.RETRY_WAIT: {JobLifecycle.QUEUED},
            JobLifecycle.COMPLETED: set(),
            JobLifecycle.FAILED: set(),
            JobLifecycle.DEAD: set(),
            JobLifecycle.CANCELLED: set(),
        }

        for source in JobLifecycle:
            source_job = base.model_copy(update={"status": source})
            for target in JobLifecycle:
                with self.subTest(source=source, target=target):
                    if target in legal[source]:
                        transitioned = transition_lifecycle(
                            source_job, target, now=NOW
                        )
                        self.assertEqual(transitioned.status, target)
                    else:
                        with self.assertRaises(ConflictError):
                            transition_lifecycle(source_job, target, now=NOW)

    def test_lifecycle_and_processing_stage_are_independent_and_ordered(self):
        review_jobs, _ = service()
        created = create_job(review_jobs)

        self.assertEqual(created.status, JobLifecycle.QUEUED)
        self.assertIsNone(created.stage)
        with self.assertRaisesRegex(ConflictError, "Cannot transition"):
            review_jobs.complete(created.id)

        running = review_jobs.start(created.id)
        parsing = review_jobs.advance_stage(running.id, ReviewStage.PARSING)

        self.assertEqual(parsing.status, JobLifecycle.RUNNING)
        self.assertEqual(parsing.stage, ReviewStage.PARSING)
        self.assertEqual(parsing.attempt_count, 1)
        with self.assertRaisesRegex(ConflictError, "progress in order"):
            review_jobs.advance_stage(parsing.id, ReviewStage.RETRIEVING)

        for stage in tuple(ReviewStage)[1:]:
            parsing = review_jobs.advance_stage(parsing.id, stage)
        completed = review_jobs.complete(parsing.id)
        self.assertEqual(completed.status, JobLifecycle.COMPLETED)
        self.assertEqual(completed.stage, ReviewStage.REPORTING)
        self.assertEqual(completed.completed_at, NOW)

    def test_failure_classification_covers_retry_wait_dead_failed_and_cancelled(self):
        review_jobs, _ = service()
        retry_job = review_jobs.create(
            command(max_attempts=2),
            caller_id="caller",
            scope="create",
            idempotency_key="retry",
        ).job
        review_jobs.start(retry_job.id)
        review_jobs.advance_stage(retry_job.id, ReviewStage.PARSING)
        transient = JobFailure(
            code="model_timeout",
            message="model timed out",
            category=ErrorCategory.RETRYABLE,
            retryable=True,
        )

        waiting = review_jobs.fail(retry_job.id, transient)
        self.assertEqual(waiting.status, JobLifecycle.RETRY_WAIT)
        self.assertEqual(waiting.failure_stage, ReviewStage.PARSING)
        review_jobs.retry(retry_job.id)
        review_jobs.start(retry_job.id)
        dead = review_jobs.fail(retry_job.id, transient)
        self.assertEqual(dead.status, JobLifecycle.DEAD)

        permanent = review_jobs.create(
            command(),
            caller_id="caller",
            scope="create",
            idempotency_key="permanent",
        ).job
        review_jobs.start(permanent.id)
        review_jobs.advance_stage(permanent.id, ReviewStage.PARSING)
        failed = review_jobs.fail(
            permanent.id,
            JobFailure(
                code="document_corrupt",
                message="document cannot be parsed",
                category=ErrorCategory.PERMANENT,
                retryable=False,
            ),
        )
        self.assertEqual(failed.status, JobLifecycle.FAILED)

        cancelled = review_jobs.create(
            command(),
            caller_id="caller",
            scope="create",
            idempotency_key="cancelled",
        ).job
        review_jobs.start(cancelled.id)
        review_jobs.advance_stage(cancelled.id, ReviewStage.PARSING)
        cancelled = review_jobs.fail(
            cancelled.id,
            JobFailure(
                code="call_cancelled",
                message="caller cancelled",
                category=ErrorCategory.CANCELLED,
                retryable=False,
            ),
        )
        self.assertEqual(cancelled.status, JobLifecycle.CANCELLED)

    def test_request_hash_is_normalized_and_execution_fingerprint_is_stable(self):
        first = command()
        second = CreateReviewJobCommand(
            document_snapshot_id="document-1",
            document_sha256="a" * 64,
            rule_version_id="rule-1",
            rule_version_hash="b" * 64,
            model_config_id="model-1",
            model_config_hash="c" * 64,
        )

        self.assertEqual(first, second)
        self.assertEqual(normalized_request_hash(first), normalized_request_hash(second))
        self.assertEqual(execution_fingerprint(first), execution_fingerprint(second))
        self.assertEqual(len(execution_fingerprint(first)), 64)

    def test_idempotency_is_scoped_by_caller_and_detects_request_conflicts(self):
        review_jobs, repository = service()
        first = create_job(review_jobs)
        replay = create_job(review_jobs)

        self.assertEqual(replay.id, first.id)
        self.assertEqual(repository.review_job_count, 1)
        with self.assertRaises(ConflictError) as raised:
            review_jobs.create(
                command(model_hash="d" * 64),
                caller_id="caller-1",
                scope="POST:/api/v1/review-jobs",
                idempotency_key="request-1",
            )
        self.assertEqual(raised.exception.code, "idempotency_key_reused")

        other_caller = create_job(review_jobs, caller_id="caller-2")
        other_scope = create_job(review_jobs, scope="review-jobs:import")
        self.assertNotEqual(other_caller.id, first.id)
        self.assertNotEqual(other_scope.id, first.id)
        self.assertEqual(repository.review_job_count, 3)

    def test_concurrent_duplicate_calls_create_one_business_job(self):
        review_jobs, repository = service(ids=UuidGenerator())

        def submit(_index: int):
            return create_job(review_jobs).id

        with ThreadPoolExecutor(max_workers=16) as executor:
            job_ids = set(executor.map(submit, range(64)))

        self.assertEqual(len(job_ids), 1)
        self.assertEqual(repository.review_job_count, 1)

    def test_cancel_rerun_and_checkpoint_preserve_provenance(self):
        review_jobs, _ = service()
        source = create_job(review_jobs)
        cancelled = review_jobs.cancel(source.id)
        self.assertEqual(cancelled.status, JobLifecycle.CANCELLED)
        self.assertEqual(review_jobs.cancel(source.id), cancelled)

        rerun = review_jobs.rerun(source.id)
        self.assertEqual(rerun.status, JobLifecycle.QUEUED)
        self.assertEqual(rerun.rerun_of, source.id)
        self.assertEqual(rerun.rerun_of_id, source.id)
        self.assertNotIn("rerun_of_id", rerun.model_dump(mode="json"))
        self.assertEqual(rerun.input_fingerprint, source.input_fingerprint)
        self.assertNotEqual(rerun.id, source.id)

        checkpoint = JobCheckpoint(
            job_id=rerun.id,
            node_name="parse-document",
            stage=ReviewStage.PARSING,
            lease_token=0,
            state=CheckpointState(
                values=(CheckpointValue(key="artifact", value="memory://parsed"),)
            ),
            completed_at=NOW,
        )
        review_jobs.save_checkpoint(checkpoint)
        self.assertEqual(review_jobs.list_checkpoints(rerun.id), (checkpoint,))
        self.assertEqual(checkpoint.schema_version, 1)
        self.assertEqual(checkpoint.state.schema_version, 1)


if __name__ == "__main__":
    unittest.main()
