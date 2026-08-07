from __future__ import annotations

from collections import deque
from datetime import datetime, timedelta
from threading import Lock

from tender_review.shared.errors import ConflictError, NotFoundError

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


class InMemoryJobRepository:
    """Thread-safe Fake for both legacy worker messages and Stage 2 review jobs."""

    def __init__(self) -> None:
        self._queue: deque[str] = deque()
        self._jobs: dict[str, JobMessage] = {}
        self.completed: dict[str, JobResult] = {}
        self.waiting_human: dict[str, JobResult] = {}
        self.failed: dict[str, JobFailure] = {}
        self._review_jobs: dict[str, ReviewJob] = {}
        self._execution_specs: dict[str, ReviewExecutionSpec] = {}
        self._idempotency: dict[tuple[str, str, str], IdempotencyRecord] = {}
        self._checkpoints: dict[tuple[str, str], JobCheckpoint] = {}
        self._lock = Lock()

    def enqueue(self, job: JobMessage) -> None:
        with self._lock:
            existing = self._jobs.get(job.job_id)
            if existing is not None:
                if existing == job:
                    return
                raise ConflictError(
                    f"Job {job.job_id!r} already exists with different input",
                    code="job_conflict",
                )
            self._jobs[job.job_id] = job
            self._queue.append(job.job_id)

    def next_queued(self) -> JobMessage | None:
        with self._lock:
            while self._queue:
                job_id = self._queue[0]
                if (
                    job_id in self.completed
                    or job_id in self.waiting_human
                    or job_id in self.failed
                ):
                    self._queue.popleft()
                    continue
                return self._jobs[job_id]
            return None

    def mark_completed(self, job_id: str, lease_token: int, result: JobResult) -> None:
        del lease_token
        with self._lock:
            self._require_legacy_job(job_id)
            self.completed[job_id] = result
            self.waiting_human.pop(job_id, None)
            self.failed.pop(job_id, None)
            self._remove_from_queue(job_id)

    def mark_waiting_human(
        self, job_id: str, lease_token: int, result: JobResult
    ) -> None:
        del lease_token
        with self._lock:
            self._require_legacy_job(job_id)
            self.waiting_human[job_id] = result
            self.completed.pop(job_id, None)
            self._remove_from_queue(job_id)

    def mark_failed(self, job_id: str, lease_token: int, failure: JobFailure) -> None:
        del lease_token
        with self._lock:
            self._require_legacy_job(job_id)
            self.failed[job_id] = failure
            self.waiting_human.pop(job_id, None)
            self._remove_from_queue(job_id)

    def create_review_job(
        self,
        job: ReviewJob,
        idempotency: IdempotencyRecord | None = None,
        execution_spec: ReviewExecutionSpec | None = None,
    ) -> IdempotentReviewJob:
        """Atomically create the resource and its caller/scope/key record."""

        with self._lock:
            if idempotency is not None:
                key = (
                    idempotency.caller_id,
                    idempotency.scope,
                    idempotency.idempotency_key,
                )
                existing_record = self._idempotency.get(key)
                if existing_record is not None:
                    if existing_record.request_hash != idempotency.request_hash:
                        raise ConflictError(
                            "Idempotency key was already used with a different request",
                            code="idempotency_key_reused",
                            details={
                                "caller_id": idempotency.caller_id,
                                "scope": idempotency.scope,
                            },
                        )
                    existing_job = self._require_review_job(existing_record.resource_id)
                    self._assert_replayed_spec(existing_job, execution_spec)
                    return IdempotentReviewJob(job=existing_job, created=False)
                if idempotency.resource_id != job.id:
                    raise ConflictError(
                        "Idempotency record must reference the job being created",
                        code="idempotency_resource_mismatch",
                    )
            if job.id in self._review_jobs:
                raise ConflictError(
                    f"Review job {job.id!r} already exists",
                    code="review_job_conflict",
                )
            self._validate_execution_spec(job, execution_spec)
            self._review_jobs[job.id] = job
            if execution_spec is not None:
                self._execution_specs[job.id] = execution_spec
            if idempotency is not None:
                self._idempotency[key] = idempotency
            return IdempotentReviewJob(job=job, created=True)

    def get_review_job(self, job_id: str) -> ReviewJob:
        with self._lock:
            return self._require_review_job(job_id)

    def get_review_execution_spec(self, job_id: str) -> ReviewExecutionSpec:
        with self._lock:
            job = self._require_review_job(job_id)
            try:
                spec = self._execution_specs[job_id]
            except KeyError as exc:
                raise NotFoundError(
                    f"Review job {job_id!r} has no execution spec",
                    code="review_execution_spec_not_found",
                ) from exc
            if job.execution_spec_sha256 != spec.input_sha256:
                raise ConflictError(
                    "Review execution spec hash differs from its job",
                    code="review_execution_spec_tampered",
                )
            return spec

    def verify_review_execution_spec(self, spec: ReviewExecutionSpec) -> None:
        with self._lock:
            job = self._require_review_job(spec.job_id)
            stored = self._execution_specs.get(spec.job_id)
            if stored != spec or job.execution_spec_sha256 != spec.input_sha256:
                raise ConflictError(
                    "Stored review execution spec failed integrity validation",
                    code="review_execution_spec_tampered",
                )

    def save_review_job(self, job: ReviewJob) -> ReviewJob:
        with self._lock:
            current = self._require_review_job(job.id)
            if current.execution_spec_sha256 != job.execution_spec_sha256:
                raise ConflictError(
                    "Immutable review-job fields cannot be changed",
                    code="review_job_immutable_field_changed",
                )
            self._review_jobs[job.id] = job
            return job

    def get_idempotency_record(
        self, caller_id: str, scope: str, idempotency_key: str
    ) -> IdempotencyRecord | None:
        with self._lock:
            return self._idempotency.get((caller_id, scope, idempotency_key))

    def save_checkpoint(self, checkpoint: JobCheckpoint) -> JobCheckpoint:
        with self._lock:
            self._require_review_job(checkpoint.job_id)
            self._checkpoints[(checkpoint.job_id, checkpoint.node_name)] = checkpoint
            return checkpoint

    def list_checkpoints(self, job_id: str) -> tuple[JobCheckpoint, ...]:
        with self._lock:
            self._require_review_job(job_id)
            return tuple(
                sorted(
                    (
                        checkpoint
                        for checkpoint in self._checkpoints.values()
                        if checkpoint.job_id == job_id
                    ),
                    key=lambda checkpoint: (checkpoint.sequence, checkpoint.node_name),
                )
            )

    @property
    def review_job_count(self) -> int:
        with self._lock:
            return len(self._review_jobs)

    def _require_legacy_job(self, job_id: str) -> None:
        if job_id not in self._jobs:
            raise NotFoundError(f"Job {job_id!r} does not exist", code="job_not_found")

    def _require_review_job(self, job_id: str) -> ReviewJob:
        try:
            return self._review_jobs[job_id]
        except KeyError as exc:
            raise NotFoundError(
                f"Review job {job_id!r} does not exist", code="review_job_not_found"
            ) from exc

    @staticmethod
    def _validate_execution_spec(
        job: ReviewJob, execution_spec: ReviewExecutionSpec | None
    ) -> None:
        if execution_spec is None:
            if job.execution_spec_sha256 is not None:
                raise ConflictError(
                    "Review job declares an execution spec that was not supplied",
                    code="review_execution_spec_missing",
                )
            return
        if execution_spec.job_id != job.id:
            raise ConflictError(
                "Review execution spec belongs to another job",
                code="review_execution_spec_job_mismatch",
            )
        if execution_spec.input_sha256 != job.execution_spec_sha256:
            raise ConflictError(
                "Review execution spec hash differs from its job",
                code="review_execution_spec_hash_mismatch",
            )

    def _assert_replayed_spec(
        self, job: ReviewJob, execution_spec: ReviewExecutionSpec | None
    ) -> None:
        stored = self._execution_specs.get(job.id)
        if stored is None and execution_spec is None:
            return
        if stored is None or execution_spec is None:
            raise ConflictError(
                "Idempotent replay changed review execution provenance",
                code="review_execution_spec_replay_conflict",
            )
        stored_inputs = stored.model_dump(
            mode="json", exclude={"job_id", "input_sha256"}
        )
        replayed_inputs = execution_spec.model_dump(
            mode="json", exclude={"job_id", "input_sha256"}
        )
        if stored_inputs != replayed_inputs:
            raise ConflictError(
                "Idempotent replay changed review execution inputs",
                code="review_execution_spec_replay_conflict",
            )

    def _remove_from_queue(self, job_id: str) -> None:
        try:
            self._queue.remove(job_id)
        except ValueError:
            pass


class FakeLeaseManager:
    def __init__(self) -> None:
        self._leases: dict[str, JobLease] = {}
        self._tokens: dict[str, int] = {}
        self._lock = Lock()

    def acquire(
        self,
        job_id: str,
        owner: str,
        *,
        now: datetime,
        lease_seconds: int,
    ) -> JobLease | None:
        with self._lock:
            current = self._leases.get(job_id)
            if current is not None and current.expires_at > now:
                return None
            token = self._tokens.get(job_id, 0) + 1
            lease = JobLease(
                job_id=job_id,
                owner=owner,
                token=token,
                expires_at=now + timedelta(seconds=lease_seconds),
            )
            self._tokens[job_id] = token
            self._leases[job_id] = lease
            return lease

    def renew(
        self,
        lease: JobLease,
        *,
        now: datetime,
        lease_seconds: int,
    ) -> JobLease:
        with self._lock:
            current = self._leases.get(lease.job_id)
            if current != lease or current.expires_at <= now:
                raise ConflictError(
                    "Lease is missing, expired, or fenced", code="stale_lease"
                )
            renewed = lease.model_copy(
                update={"expires_at": now + timedelta(seconds=lease_seconds)}
            )
            self._leases[lease.job_id] = renewed
            return renewed

    def release(self, lease: JobLease) -> None:
        with self._lock:
            if self._leases.get(lease.job_id) == lease:
                del self._leases[lease.job_id]
