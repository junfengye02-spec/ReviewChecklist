from __future__ import annotations

from datetime import datetime
from typing import Protocol, runtime_checkable

from .models import (
    IdempotencyRecord,
    IdempotentReviewJob,
    JobCheckpoint,
    JobFailure,
    JobLease,
    JobMessage,
    JobResult,
    ReviewExecutionSpec,
    ReviewJob,
)


@runtime_checkable
class JobRepository(Protocol):
    """Stage 1 worker queue boundary retained for compatibility."""

    def enqueue(self, job: JobMessage) -> None: ...

    def next_queued(self) -> JobMessage | None: ...

    def mark_completed(
        self, job_id: str, lease_token: int, result: JobResult
    ) -> None: ...

    def mark_waiting_human(
        self, job_id: str, lease_token: int, result: JobResult
    ) -> None: ...

    def mark_failed(
        self, job_id: str, lease_token: int, failure: JobFailure
    ) -> None: ...


@runtime_checkable
class ReviewJobRepository(Protocol):
    """Application persistence boundary using only stable, versioned DTOs."""

    def create_review_job(
        self,
        job: ReviewJob,
        idempotency: IdempotencyRecord | None = None,
        execution_spec: ReviewExecutionSpec | None = None,
    ) -> IdempotentReviewJob: ...

    def get_review_job(self, job_id: str) -> ReviewJob: ...

    def get_review_execution_spec(self, job_id: str) -> ReviewExecutionSpec: ...

    def verify_review_execution_spec(self, spec: ReviewExecutionSpec) -> None: ...

    def save_review_job(self, job: ReviewJob) -> ReviewJob: ...

    def get_idempotency_record(
        self, caller_id: str, scope: str, idempotency_key: str
    ) -> IdempotencyRecord | None: ...

    def save_checkpoint(self, checkpoint: JobCheckpoint) -> JobCheckpoint: ...

    def list_checkpoints(self, job_id: str) -> tuple[JobCheckpoint, ...]: ...


@runtime_checkable
class LeaseManager(Protocol):
    def acquire(
        self,
        job_id: str,
        owner: str,
        *,
        now: datetime,
        lease_seconds: int,
    ) -> JobLease | None: ...

    def renew(
        self,
        lease: JobLease,
        *,
        now: datetime,
        lease_seconds: int,
    ) -> JobLease: ...

    def release(self, lease: JobLease) -> None: ...
