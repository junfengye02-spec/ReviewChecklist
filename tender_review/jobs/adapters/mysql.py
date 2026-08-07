from __future__ import annotations

from collections.abc import Callable, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import time
from typing import Any, Iterator

from pydantic import ValidationError
from sqlalchemy import Select, and_, or_, select, update
from sqlalchemy.exc import IntegrityError, OperationalError
from sqlalchemy.orm import Session, sessionmaker

from tender_review.infrastructure.database.models import (
    SCHEMA_VERSION,
    DatasetVersion,
    DocumentArtifact,
    DocumentSnapshot,
    IdempotencyRecord,
    JobCheckpoint,
    ModelConfig,
    ReviewExecutionSpec as DbReviewExecutionSpec,
    ReviewJob,
    RuleVersion,
)
from tender_review.jobs.models import (
    CheckpointState,
    IdempotencyRecord as IdempotencyRecordDto,
    IdempotentReviewJob,
    JobCheckpoint as JobCheckpointDto,
    JobFailure,
    JobLease,
    JobLifecycle,
    JobMessage,
    JobResult,
    ReviewExecutionSpec,
    ReviewJob as ReviewJobDto,
    ReviewStage,
    clone_review_execution_spec,
)
from tender_review.shared.errors import ConflictError, ErrorCategory, NotFoundError


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _require_aware(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("lease times must be timezone-aware")
    return value


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


@dataclass(frozen=True, slots=True)
class ReviewJobCreate:
    """Input for an idempotent review-job create transaction."""

    document_snapshot_id: str
    rule_version_id: str
    model_config_id: str
    input_fingerprint: str
    caller_id: str
    scope: str
    idempotency_key: str
    request_hash: str
    job_type: str = "review"
    input_reference: str | None = None
    max_attempts: int = 3
    rerun_of_id: str | None = None
    available_at: datetime | None = None

    def __post_init__(self) -> None:
        for field_name in (
            "document_snapshot_id",
            "rule_version_id",
            "model_config_id",
            "input_fingerprint",
            "caller_id",
            "scope",
            "idempotency_key",
            "request_hash",
            "job_type",
        ):
            if not getattr(self, field_name).strip():
                raise ValueError(f"{field_name} must not be empty")
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be positive")
        if self.available_at is not None:
            _require_aware(self.available_at)


@dataclass(frozen=True, slots=True)
class ReviewJobSnapshot:
    id: str
    status: str
    job_type: str
    input_reference: str
    input_fingerprint: str
    attempt_count: int
    max_attempts: int
    available_at: datetime
    lease_token: int
    stage: str | None
    rerun_of_id: str | None
    output_reference: str | None
    output_summary: str | None


@dataclass(frozen=True, slots=True)
class ClaimedJob:
    message: JobMessage
    lease: JobLease


@dataclass(frozen=True, slots=True)
class CheckpointSnapshot:
    job_id: str
    node_name: str
    stage: str
    lease_token: int
    sequence: int
    state_json: Mapping[str, Any]
    output_artifact_id: str | None
    completed_at: datetime


class MySqlJobRepository:
    """MySQL 8 review-job repository and lease manager.

    Every method opens and closes its own SQLAlchemy session.  In particular,
    ``claim_next`` commits its row lock before it returns work to a Worker.
    Callers must use the same instance for the repository and lease-manager
    ports so final writes are fenced by the lease token it issued.
    """

    _RESOURCE_TYPE = "review_job"
    _TERMINAL_STATUSES = frozenset({"COMPLETED", "FAILED", "DEAD", "CANCELLED"})

    def __init__(
        self,
        sessions: sessionmaker[Session],
        *,
        retry_base_seconds: int = 5,
        retry_max_seconds: int = 300,
        now_provider: Callable[[], datetime] = _utc_now,
    ) -> None:
        if retry_base_seconds < 1:
            raise ValueError("retry_base_seconds must be positive")
        if retry_max_seconds < retry_base_seconds:
            raise ValueError("retry_max_seconds must be at least retry_base_seconds")
        self._sessions = sessions
        self._retry_base_seconds = retry_base_seconds
        self._retry_max_seconds = retry_max_seconds
        self._now_provider = now_provider

    @contextmanager
    def _transaction(self) -> Iterator[Session]:
        session = self._sessions()
        try:
            with session.begin():
                yield session
        finally:
            session.close()

    def _now(self) -> datetime:
        return _require_aware(self._now_provider())

    @staticmethod
    def _eligible_clause(now: datetime):
        return or_(
            and_(
                ReviewJob.status.in_(("QUEUED", "RETRY_WAIT")),
                ReviewJob.available_at <= now,
            ),
            and_(
                ReviewJob.status == "RUNNING",
                ReviewJob.lease_until.is_not(None),
                ReviewJob.lease_until <= now,
            ),
        )

    @classmethod
    def claim_candidate_statement(cls, now: datetime) -> Select[tuple[ReviewJob]]:
        """Return the MySQL-compiled statement used for ``SKIP LOCKED`` claims."""

        _require_aware(now)
        return (
            select(ReviewJob)
            .where(cls._eligible_clause(now))
            .order_by(ReviewJob.available_at, ReviewJob.created_at, ReviewJob.id)
            .limit(1)
            .with_for_update(skip_locked=True)
        )

    def create_idempotent(self, request: ReviewJobCreate) -> ReviewJobSnapshot:
        """Create one business job, replaying only an identical idempotent request."""

        for attempt in range(5):
            session = self._sessions()
            try:
                existing = session.execute(
                    select(IdempotencyRecord)
                    .where(
                        IdempotencyRecord.caller_id == request.caller_id,
                        IdempotencyRecord.scope == request.scope,
                        IdempotencyRecord.idempotency_key == request.idempotency_key,
                    )
                    .with_for_update()
                ).scalar_one_or_none()
                if existing is not None:
                    return self._idempotent_result(session, existing, request.request_hash)

                job = ReviewJob(
                    document_snapshot_id=request.document_snapshot_id,
                    rule_version_id=request.rule_version_id,
                    model_config_id=request.model_config_id,
                    rerun_of_id=request.rerun_of_id,
                    status="QUEUED",
                    job_type=request.job_type,
                    input_reference=request.input_reference or request.document_snapshot_id,
                    input_fingerprint=request.input_fingerprint,
                    max_attempts=request.max_attempts,
                    available_at=request.available_at or self._now(),
                )
                session.add(job)
                session.flush()
                session.add(
                    IdempotencyRecord(
                        caller_id=request.caller_id,
                        scope=request.scope,
                        idempotency_key=request.idempotency_key,
                        request_hash=request.request_hash,
                        resource_type=self._RESOURCE_TYPE,
                        resource_id=job.id,
                    )
                )
                session.commit()
                return self._snapshot(job)
            except IntegrityError:
                session.rollback()
                winner = session.execute(
                    select(IdempotencyRecord).where(
                        IdempotencyRecord.caller_id == request.caller_id,
                        IdempotencyRecord.scope == request.scope,
                        IdempotencyRecord.idempotency_key == request.idempotency_key,
                    )
                ).scalar_one_or_none()
                if winner is not None:
                    return self._idempotent_result(session, winner, request.request_hash)
                if attempt == 4:
                    raise
            except OperationalError as exc:
                session.rollback()
                if not self._is_concurrency_retryable(exc) or attempt == 4:
                    raise
                time.sleep(0.01 * (attempt + 1))
            finally:
                session.close()
        raise AssertionError("unreachable")

    def create_review_job(
        self,
        job: ReviewJobDto,
        idempotency: IdempotencyRecordDto | None = None,
        execution_spec: ReviewExecutionSpec | None = None,
    ) -> IdempotentReviewJob:
        """Implement the public port with one job/idempotency transaction."""

        for attempt in range(5):
            session = self._sessions()
            try:
                if idempotency is not None:
                    existing = session.execute(
                        select(IdempotencyRecord)
                        .where(
                            IdempotencyRecord.caller_id == idempotency.caller_id,
                            IdempotencyRecord.scope == idempotency.scope,
                            IdempotencyRecord.idempotency_key
                            == idempotency.idempotency_key,
                        )
                        .with_for_update()
                    ).scalar_one_or_none()
                    if existing is not None:
                        return self._public_idempotent_result(
                            session,
                            existing,
                            idempotency.request_hash,
                            execution_spec,
                        )
                    if idempotency.resource_id != job.id:
                        raise ConflictError(
                            "Idempotency record must reference the job being created",
                            code="idempotency_resource_mismatch",
                        )

                if session.get(ReviewJob, job.id) is not None:
                    raise ConflictError(
                        f"Review job {job.id!r} already exists",
                        code="review_job_conflict",
                    )
                self._validate_spec_for_job(job, execution_spec)
                if execution_spec is not None:
                    self._validate_execution_references(session, execution_spec)
                durable_job = self._new_durable_job(job)
                session.add(durable_job)
                if execution_spec is not None:
                    session.add(self._new_execution_spec(execution_spec))
                if idempotency is not None:
                    session.add(
                        IdempotencyRecord(
                            id=idempotency.id,
                            caller_id=idempotency.caller_id,
                            scope=idempotency.scope,
                            idempotency_key=idempotency.idempotency_key,
                            request_hash=idempotency.request_hash,
                            resource_type=idempotency.resource_type,
                            resource_id=idempotency.resource_id,
                            expires_at=idempotency.expires_at,
                            created_at=idempotency.created_at,
                            updated_at=idempotency.created_at,
                        )
                    )
                session.commit()
                return IdempotentReviewJob(
                    job=self._review_job_dto(durable_job), created=True
                )
            except IntegrityError as exc:
                session.rollback()
                if idempotency is not None:
                    winner = session.execute(
                        select(IdempotencyRecord).where(
                            IdempotencyRecord.caller_id == idempotency.caller_id,
                            IdempotencyRecord.scope == idempotency.scope,
                            IdempotencyRecord.idempotency_key
                            == idempotency.idempotency_key,
                        )
                    ).scalar_one_or_none()
                    if winner is not None:
                        return self._public_idempotent_result(
                            session,
                            winner,
                            idempotency.request_hash,
                            execution_spec,
                        )
                if attempt == 4:
                    raise ConflictError(
                        "Review job could not be created because a unique resource exists",
                        code="review_job_conflict",
                    ) from exc
            except OperationalError as exc:
                session.rollback()
                if not self._is_concurrency_retryable(exc) or attempt == 4:
                    raise
                time.sleep(0.01 * (attempt + 1))
            finally:
                session.close()
        raise AssertionError("unreachable")

    def get_review_job(self, job_id: str) -> ReviewJobDto:
        session = self._sessions()
        try:
            job = session.get(ReviewJob, job_id)
            if job is None:
                raise NotFoundError(
                    f"Review job {job_id!r} does not exist",
                    code="review_job_not_found",
                )
            return self._review_job_dto(job)
        finally:
            session.close()

    def get_review_execution_spec(self, job_id: str) -> ReviewExecutionSpec:
        session = self._sessions()
        try:
            job = session.get(ReviewJob, job_id)
            if job is None:
                raise NotFoundError(
                    f"Review job {job_id!r} does not exist",
                    code="review_job_not_found",
                )
            row = session.get(DbReviewExecutionSpec, job_id)
            if row is None:
                raise NotFoundError(
                    f"Review job {job_id!r} has no execution spec",
                    code="review_execution_spec_not_found",
                )
            spec = self._validated_stored_spec(job, row)
            self._validate_execution_references(session, spec)
            return spec
        finally:
            session.close()

    def save_review_job(self, job: ReviewJobDto) -> ReviewJobDto:
        """Persist application state without allowing a stale DTO to change a fence."""

        with self._transaction() as session:
            durable_job = session.execute(
                select(ReviewJob).where(ReviewJob.id == job.id).with_for_update()
            ).scalar_one_or_none()
            if durable_job is None:
                raise NotFoundError(
                    f"Review job {job.id!r} does not exist",
                    code="review_job_not_found",
                )
            if durable_job.lease_token != job.lease_token:
                raise ConflictError(
                    "Review job was changed by a newer lease",
                    code="stale_lease",
                )
            self._copy_public_state(durable_job, job)
            session.flush()
            return self._review_job_dto(durable_job)

    def get_idempotency_record(
        self, caller_id: str, scope: str, idempotency_key: str
    ) -> IdempotencyRecordDto | None:
        session = self._sessions()
        try:
            record = session.execute(
                select(IdempotencyRecord).where(
                    IdempotencyRecord.caller_id == caller_id,
                    IdempotencyRecord.scope == scope,
                    IdempotencyRecord.idempotency_key == idempotency_key,
                )
            ).scalar_one_or_none()
            return None if record is None else self._idempotency_dto(record)
        finally:
            session.close()

    def _idempotent_result(
        self,
        session: Session,
        record: IdempotencyRecord,
        request_hash: str,
    ) -> ReviewJobSnapshot:
        if record.request_hash != request_hash:
            raise ConflictError(
                "Idempotency key was already used for a different request",
                code="idempotency_key_reused",
            )
        if record.resource_type != self._RESOURCE_TYPE:
            raise ConflictError(
                "Idempotency key refers to a different resource type",
                code="idempotency_resource_conflict",
            )
        job = session.get(ReviewJob, record.resource_id)
        if job is None:
            raise ConflictError(
                "Idempotency record refers to a missing review job",
                code="idempotency_resource_missing",
            )
        return self._snapshot(job)

    def _public_idempotent_result(
        self,
        session: Session,
        record: IdempotencyRecord,
        request_hash: str,
        execution_spec: ReviewExecutionSpec | None,
    ) -> IdempotentReviewJob:
        if record.request_hash != request_hash:
            raise ConflictError(
                "Idempotency key was already used with a different request",
                code="idempotency_key_reused",
                details={"caller_id": record.caller_id, "scope": record.scope},
            )
        if record.resource_type != self._RESOURCE_TYPE:
            raise ConflictError(
                "Idempotency key refers to a different resource type",
                code="idempotency_resource_conflict",
            )
        job = session.get(ReviewJob, record.resource_id)
        if job is None:
            raise ConflictError(
                "Idempotency record refers to a missing review job",
                code="idempotency_resource_missing",
            )
        self._validate_replayed_spec(session, job, execution_spec)
        return IdempotentReviewJob(job=self._review_job_dto(job), created=False)

    def enqueue(self, job: JobMessage) -> None:
        """Validate a previously-created durable review job for the stable queue port."""

        with self._transaction() as session:
            durable_job = session.get(ReviewJob, job.job_id)
            if durable_job is None:
                raise NotFoundError(
                    f"Review job {job.job_id!r} does not exist",
                    code="job_not_found",
                )
            if (
                durable_job.job_type != job.job_type
                or durable_job.input_reference != job.input_reference
            ):
                raise ConflictError(
                    "Durable review job payload differs from the queue message",
                    code="job_payload_conflict",
                )

    def get_job(self, job_id: str) -> ReviewJobSnapshot:
        session = self._sessions()
        try:
            job = session.get(ReviewJob, job_id)
            if job is None:
                raise NotFoundError(
                    f"Review job {job_id!r} does not exist",
                    code="job_not_found",
                )
            return self._snapshot(job)
        finally:
            session.close()

    def next_queued(self) -> JobMessage | None:
        now = self._now()
        session = self._sessions()
        try:
            job = session.execute(
                select(ReviewJob)
                .where(self._eligible_clause(now))
                .order_by(ReviewJob.available_at, ReviewJob.created_at, ReviewJob.id)
                .limit(1)
            ).scalar_one_or_none()
            return None if job is None else self._message(job)
        finally:
            session.close()

    def claim_next(
        self,
        owner: str,
        *,
        now: datetime,
        lease_seconds: int,
    ) -> ClaimedJob | None:
        """Atomically claim one due job using ``SELECT .. FOR UPDATE SKIP LOCKED``."""

        if not owner.strip():
            raise ValueError("owner must not be empty")
        if lease_seconds < 1:
            raise ValueError("lease_seconds must be positive")
        _require_aware(now)
        with self._transaction() as session:
            job = session.execute(self.claim_candidate_statement(now)).scalar_one_or_none()
            if job is None:
                return None
            if job.attempt_count >= job.max_attempts:
                self._mark_expired_job_dead(job, now)
                return None
            lease = self._assign_lease(job, owner, now, lease_seconds)
            session.flush()
            return ClaimedJob(message=self._message(job), lease=lease)

    def acquire(
        self,
        job_id: str,
        owner: str,
        *,
        now: datetime,
        lease_seconds: int,
    ) -> JobLease | None:
        """Compatibility path for the stage-1 separate repository/lease ports."""

        if not owner.strip():
            raise ValueError("owner must not be empty")
        if lease_seconds < 1:
            raise ValueError("lease_seconds must be positive")
        _require_aware(now)
        with self._transaction() as session:
            job = session.execute(
                select(ReviewJob).where(ReviewJob.id == job_id).with_for_update()
            ).scalar_one_or_none()
            if job is None:
                raise NotFoundError(
                    f"Review job {job_id!r} does not exist",
                    code="job_not_found",
                )
            if not self._is_eligible(job, now):
                return None
            if job.attempt_count >= job.max_attempts:
                self._mark_expired_job_dead(job, now)
                return None
            lease = self._assign_lease(job, owner, now, lease_seconds)
            session.flush()
            return lease

    def renew(
        self,
        lease: JobLease,
        *,
        now: datetime,
        lease_seconds: int,
    ) -> JobLease:
        if lease_seconds < 1:
            raise ValueError("lease_seconds must be positive")
        _require_aware(now)
        expires_at = now + timedelta(seconds=lease_seconds)
        with self._transaction() as session:
            result = session.execute(
                update(ReviewJob)
                .where(
                    ReviewJob.id == lease.job_id,
                    ReviewJob.status == "RUNNING",
                    ReviewJob.lease_token == lease.token,
                    ReviewJob.lease_owner == lease.owner,
                    ReviewJob.lease_until.is_not(None),
                    ReviewJob.lease_until > now,
                )
                .values(lease_until=expires_at, heartbeat_at=now)
            )
            if result.rowcount != 1:
                raise ConflictError(
                    "Lease is missing, expired, or fenced",
                    code="stale_lease",
                )
        return lease.model_copy(update={"expires_at": expires_at})

    def release(self, lease: JobLease) -> None:
        """Make an unfinished lease immediately eligible without changing its token."""

        now = self._now()
        with self._transaction() as session:
            session.execute(
                update(ReviewJob)
                .where(
                    ReviewJob.id == lease.job_id,
                    ReviewJob.status == "RUNNING",
                    ReviewJob.lease_token == lease.token,
                )
                .values(lease_owner=None, lease_until=now, heartbeat_at=now)
            )

    def mark_completed(self, job_id: str, lease_token: int, result: JobResult) -> None:
        now = self._now()
        with self._transaction() as session:
            updated = session.execute(
                update(ReviewJob)
                .where(
                    ReviewJob.id == job_id,
                    ReviewJob.status == "RUNNING",
                    ReviewJob.lease_token == lease_token,
                    ReviewJob.lease_until.is_not(None),
                    ReviewJob.lease_until > now,
                )
                .values(
                    status="COMPLETED",
                    completed_at=now,
                    lease_owner=None,
                    lease_until=None,
                    heartbeat_at=now,
                    failure_stage=None,
                    error_code=None,
                    error_category=None,
                    error_message=None,
                    error_retryable=None,
                    output_reference=result.output_reference,
                    output_summary=result.summary,
                )
            )
            self._require_fenced_update(session, job_id, updated.rowcount)

    def mark_waiting_human(
        self, job_id: str, lease_token: int, result: JobResult
    ) -> None:
        now = self._now()
        with self._transaction() as session:
            updated = session.execute(
                update(ReviewJob)
                .where(
                    ReviewJob.id == job_id,
                    ReviewJob.status == "RUNNING",
                    ReviewJob.lease_token == lease_token,
                    ReviewJob.lease_until.is_not(None),
                    ReviewJob.lease_until > now,
                )
                .values(
                    status="WAITING_HUMAN",
                    lease_owner=None,
                    lease_until=None,
                    heartbeat_at=now,
                    failure_stage=None,
                    error_code=None,
                    error_category=None,
                    error_message=None,
                    error_retryable=None,
                    output_reference=result.output_reference,
                    output_summary=result.summary,
                )
            )
            self._require_fenced_update(session, job_id, updated.rowcount)

    def mark_failed(self, job_id: str, lease_token: int, failure: JobFailure) -> None:
        now = self._now()
        with self._transaction() as session:
            job = session.execute(
                select(ReviewJob)
                .where(
                    ReviewJob.id == job_id,
                    ReviewJob.status == "RUNNING",
                    ReviewJob.lease_token == lease_token,
                    ReviewJob.lease_until.is_not(None),
                    ReviewJob.lease_until > now,
                )
                .with_for_update()
            ).scalar_one_or_none()
            if job is None:
                self._raise_missing_or_stale(session, job_id)
            assert job is not None
            job.failure_stage = (
                failure.stage.value if failure.stage is not None else job.stage
            )
            job.error_code = failure.code
            job.error_category = failure.category.value
            job.error_message = failure.message
            job.error_retryable = failure.retryable
            job.lease_owner = None
            job.lease_until = None
            job.heartbeat_at = now
            if failure.category is ErrorCategory.CANCELLED:
                job.status = "CANCELLED"
            elif failure.retryable and job.attempt_count < job.max_attempts:
                job.status = "RETRY_WAIT"
                job.available_at = now + timedelta(
                    seconds=self._retry_delay_seconds(job.attempt_count)
                )
            elif failure.retryable:
                job.status = "DEAD"
            else:
                job.status = "FAILED"
            session.flush()

    def cancel(self, job_id: str) -> ReviewJobSnapshot:
        """Cancel a queued, delayed, or running job and fence a live worker."""

        with self._transaction() as session:
            job = session.execute(
                select(ReviewJob).where(ReviewJob.id == job_id).with_for_update()
            ).scalar_one_or_none()
            if job is None:
                raise NotFoundError(
                    f"Review job {job_id!r} does not exist",
                    code="job_not_found",
                )
            if job.status in self._TERMINAL_STATUSES or job.status == "WAITING_HUMAN":
                return self._snapshot(job)
            job.status = "CANCELLED"
            job.lease_owner = None
            job.lease_until = None
            job.heartbeat_at = self._now()
            job.lease_token += 1
            session.flush()
            return self._snapshot(job)

    def cancel_review_job(self, job_id: str) -> ReviewJobDto:
        """Cancel through the durable row lock, including a live lease fence."""

        self.cancel(job_id)
        return self.get_review_job(job_id)

    def rerun(self, job_id: str) -> ReviewJobSnapshot:
        """Create an explicit fresh job rather than mutating retry state in place."""

        now = self._now()
        with self._transaction() as session:
            source = session.get(ReviewJob, job_id)
            if source is None:
                raise NotFoundError(
                    f"Review job {job_id!r} does not exist",
                    code="job_not_found",
                )
            if source.status not in self._TERMINAL_STATUSES:
                raise ConflictError(
                    "Only terminal review jobs can be explicitly rerun",
                    code="review_job_rerun_not_allowed",
                    details={"status": source.status},
                )
            rerun = ReviewJob(
                document_snapshot_id=source.document_snapshot_id,
                rule_version_id=source.rule_version_id,
                model_config_id=source.model_config_id,
                rerun_of_id=source.id,
                status="QUEUED",
                job_type=source.job_type,
                input_reference=source.input_reference,
                input_fingerprint=source.input_fingerprint,
                max_attempts=source.max_attempts,
                available_at=now,
            )
            session.add(rerun)
            session.flush()
            source_spec_row = session.get(DbReviewExecutionSpec, source.id)
            if source.execution_spec_sha256 is not None:
                if source_spec_row is None:
                    raise ConflictError(
                        "Review job declares a missing execution spec",
                        code="review_execution_spec_missing",
                    )
                source_spec = self._validated_stored_spec(source, source_spec_row)
                rerun_spec = clone_review_execution_spec(source_spec, rerun.id)
                rerun.execution_spec_sha256 = rerun_spec.input_sha256
                session.add(self._new_execution_spec(rerun_spec))
                session.flush()
            elif source_spec_row is not None:
                raise ConflictError(
                    "Review job has an undeclared execution spec",
                    code="review_execution_spec_tampered",
                )
            return self._snapshot(rerun)

    def save_checkpoint(
        self,
        checkpoint_or_job_id: JobCheckpointDto | str,
        lease: JobLease | None = None,
        *,
        node_name: str | None = None,
        stage: str | None = None,
        state_json: Mapping[str, Any] | None = None,
        output_artifact_id: str | None = None,
    ) -> JobCheckpointDto | CheckpointSnapshot:
        """Support both the public DTO port and the leased Worker checkpoint API."""

        if isinstance(checkpoint_or_job_id, JobCheckpointDto):
            if lease is not None:
                raise TypeError("lease is not accepted with a JobCheckpoint DTO")
            return self._save_public_checkpoint(checkpoint_or_job_id)
        if lease is None or node_name is None or stage is None or state_json is None:
            raise TypeError(
                "leased checkpoint writes require lease, node_name, stage, and state_json"
            )
        return self._save_leased_checkpoint(
            checkpoint_or_job_id,
            lease,
            node_name=node_name,
            stage=stage,
            state_json=state_json,
            output_artifact_id=output_artifact_id,
        )

    def _save_leased_checkpoint(
        self,
        job_id: str,
        lease: JobLease,
        *,
        node_name: str,
        stage: str,
        state_json: Mapping[str, Any],
        output_artifact_id: str | None,
    ) -> CheckpointSnapshot:
        """Idempotently upsert a completed node after checking the active fence."""

        if lease.job_id != job_id:
            raise ValueError("checkpoint job_id must match the lease")
        if not node_name.strip() or not stage.strip():
            raise ValueError("checkpoint node_name and stage must not be empty")
        state = dict(state_json)
        state.setdefault("schema_version", SCHEMA_VERSION)
        now = self._now()
        with self._transaction() as session:
            job = session.execute(
                select(ReviewJob)
                .where(
                    ReviewJob.id == job_id,
                    ReviewJob.status == "RUNNING",
                    ReviewJob.lease_token == lease.token,
                    ReviewJob.lease_owner == lease.owner,
                    ReviewJob.lease_until.is_not(None),
                    ReviewJob.lease_until > now,
                )
                .with_for_update()
            ).scalar_one_or_none()
            if job is None:
                self._raise_missing_or_stale(session, job_id)
            assert job is not None
            checkpoint = session.execute(
                select(JobCheckpoint)
                .where(
                    JobCheckpoint.review_job_id == job_id,
                    JobCheckpoint.node_name == node_name,
                )
                .with_for_update()
            ).scalar_one_or_none()
            if checkpoint is not None and checkpoint.lease_token > lease.token:
                raise ConflictError(
                    "Checkpoint was written by a newer lease",
                    code="stale_lease",
                )
            job.checkpoint_sequence += 1
            if checkpoint is None:
                checkpoint = JobCheckpoint(
                    review_job_id=job_id,
                    node_name=node_name,
                    stage=stage,
                    lease_token=lease.token,
                    sequence=job.checkpoint_sequence,
                    output_artifact_id=output_artifact_id,
                    state_json=state,
                    completed_at=now,
                )
                session.add(checkpoint)
            else:
                checkpoint.stage = stage
                checkpoint.lease_token = lease.token
                checkpoint.sequence = job.checkpoint_sequence
                checkpoint.output_artifact_id = output_artifact_id
                checkpoint.state_json = state
                checkpoint.completed_at = now
            session.flush()
            return self._checkpoint_snapshot(checkpoint)

    def _save_public_checkpoint(
        self, checkpoint_dto: JobCheckpointDto
    ) -> JobCheckpointDto:
        state_json = checkpoint_dto.state.model_dump(mode="json")
        now = self._now()
        with self._transaction() as session:
            job = session.execute(
                select(ReviewJob)
                .where(
                    ReviewJob.id == checkpoint_dto.job_id,
                    ReviewJob.status == "RUNNING",
                    ReviewJob.lease_token == checkpoint_dto.lease_token,
                    ReviewJob.lease_until.is_not(None),
                    ReviewJob.lease_until > now,
                )
                .with_for_update()
            ).scalar_one_or_none()
            if job is None:
                self._raise_missing_or_stale(session, checkpoint_dto.job_id)
            assert job is not None
            checkpoint = session.execute(
                select(JobCheckpoint)
                .where(
                    JobCheckpoint.review_job_id == checkpoint_dto.job_id,
                    JobCheckpoint.node_name == checkpoint_dto.node_name,
                )
                .with_for_update()
            ).scalar_one_or_none()
            if (
                checkpoint is not None
                and checkpoint.lease_token > checkpoint_dto.lease_token
            ):
                raise ConflictError(
                    "Checkpoint was written by a newer lease",
                    code="stale_lease",
                )
            job.checkpoint_sequence += 1
            if checkpoint is None:
                checkpoint = JobCheckpoint(
                    review_job_id=checkpoint_dto.job_id,
                    node_name=checkpoint_dto.node_name,
                    stage=checkpoint_dto.stage.value,
                    lease_token=checkpoint_dto.lease_token,
                    sequence=job.checkpoint_sequence,
                    output_artifact_id=checkpoint_dto.output_artifact_id,
                    state_json=state_json,
                    completed_at=checkpoint_dto.completed_at,
                )
                session.add(checkpoint)
            else:
                checkpoint.stage = checkpoint_dto.stage.value
                checkpoint.lease_token = checkpoint_dto.lease_token
                checkpoint.sequence = job.checkpoint_sequence
                checkpoint.output_artifact_id = checkpoint_dto.output_artifact_id
                checkpoint.state_json = state_json
                checkpoint.completed_at = checkpoint_dto.completed_at
            session.flush()
            return self._job_checkpoint_dto(checkpoint)

    def load_latest_checkpoint(self, job_id: str) -> CheckpointSnapshot | None:
        session = self._sessions()
        try:
            checkpoint = session.execute(
                select(JobCheckpoint)
                .where(JobCheckpoint.review_job_id == job_id)
                .order_by(
                    JobCheckpoint.sequence.desc(),
                    JobCheckpoint.completed_at.desc(),
                    JobCheckpoint.id.desc(),
                )
                .limit(1)
            ).scalar_one_or_none()
            return None if checkpoint is None else self._checkpoint_snapshot(checkpoint)
        finally:
            session.close()

    def list_checkpoints(self, job_id: str) -> tuple[JobCheckpointDto, ...]:
        session = self._sessions()
        try:
            if session.get(ReviewJob, job_id) is None:
                raise NotFoundError(
                    f"Review job {job_id!r} does not exist",
                    code="review_job_not_found",
                )
            checkpoints = session.execute(
                select(JobCheckpoint)
                .where(JobCheckpoint.review_job_id == job_id)
                .order_by(JobCheckpoint.sequence, JobCheckpoint.completed_at)
            ).scalars()
            return tuple(self._job_checkpoint_dto(item) for item in checkpoints)
        finally:
            session.close()

    @staticmethod
    def _validate_spec_for_job(
        job: ReviewJobDto, execution_spec: ReviewExecutionSpec | None
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
        identities = (
            (execution_spec.document_snapshot_id, job.document_snapshot_id),
            (execution_spec.rule_version_id, job.rule_version_id),
            (execution_spec.model_config_id, job.model_config_id),
        )
        if any(spec_value != job_value for spec_value, job_value in identities):
            raise ConflictError(
                "Review execution spec identities differ from its job",
                code="review_execution_spec_job_mismatch",
            )

    def verify_review_execution_spec(self, spec: ReviewExecutionSpec) -> None:
        with self._sessions() as session:
            job = session.get(ReviewJob, spec.job_id)
            if job is None:
                raise NotFoundError(
                    f"Review job {spec.job_id!r} does not exist",
                    code="review_job_not_found",
                )
            row = session.get(DbReviewExecutionSpec, spec.job_id)
            if row is None:
                raise NotFoundError(
                    f"Review job {spec.job_id!r} has no execution spec",
                    code="review_execution_spec_not_found",
                )
            if self._validated_stored_spec(job, row) != spec:
                raise ConflictError(
                    "Stored review execution spec changed during execution",
                    code="review_execution_spec_tampered",
                )
            self._validate_execution_references(session, spec)

    @classmethod
    def _validate_execution_references(
        cls, session: Session, spec: ReviewExecutionSpec
    ) -> None:
        cls._validate_hashed_reference(
            session.get(DocumentSnapshot, spec.document_snapshot_id),
            reference_type="document_snapshot",
            reference_id=spec.document_snapshot_id,
            actual_hash_attribute="sha256",
            expected_hash=spec.document_sha256,
        )
        cls._validate_hashed_reference(
            session.get(RuleVersion, spec.rule_version_id),
            reference_type="rule_version",
            reference_id=spec.rule_version_id,
            actual_hash_attribute="content_hash",
            expected_hash=spec.rule_version_hash,
        )
        cls._validate_hashed_reference(
            session.get(DatasetVersion, spec.dataset_version_id),
            reference_type="dataset_version",
            reference_id=spec.dataset_version_id,
            actual_hash_attribute="manifest_hash",
            expected_hash=spec.dataset_version_hash,
        )
        cls._validate_hashed_reference(
            session.get(ModelConfig, spec.model_config_id),
            reference_type="model_config",
            reference_id=spec.model_config_id,
            actual_hash_attribute="config_hash",
            expected_hash=spec.model_config_hash,
        )
        for role, reference in (
            ("retriever_artifact", spec.retriever_artifact),
            ("index_artifact", spec.index_artifact),
            ("chunk_artifact", spec.chunk_artifact),
        ):
            artifact = session.get(DocumentArtifact, reference.artifact_id)
            cls._validate_hashed_reference(
                artifact,
                reference_type=role,
                reference_id=reference.artifact_id,
                actual_hash_attribute="sha256",
                expected_hash=reference.sha256,
            )
            assert artifact is not None
            if (
                artifact.document_snapshot_id != spec.document_snapshot_id
                or artifact.bucket != reference.bucket
                or artifact.object_key != reference.object_key
            ):
                raise ConflictError(
                    f"{role} metadata conflicts with the execution spec",
                    code="review_execution_reference_conflict",
                    details={"reference_type": role, "reference_id": reference.artifact_id},
                )

    @staticmethod
    def _validate_hashed_reference(
        row: object | None,
        *,
        reference_type: str,
        reference_id: str,
        actual_hash_attribute: str,
        expected_hash: str,
    ) -> None:
        if row is None:
            raise NotFoundError(
                f"{reference_type} {reference_id!r} does not exist",
                code="review_execution_reference_missing",
                details={
                    "reference_type": reference_type,
                    "reference_id": reference_id,
                },
            )
        if getattr(row, actual_hash_attribute) != expected_hash:
            raise ConflictError(
                f"{reference_type} hash conflicts with the execution spec",
                code="review_execution_reference_conflict",
                details={
                    "reference_type": reference_type,
                    "reference_id": reference_id,
                },
            )

    @staticmethod
    def _new_execution_spec(spec: ReviewExecutionSpec) -> DbReviewExecutionSpec:
        return DbReviewExecutionSpec(
            job_id=spec.job_id,
            input_sha256=spec.input_sha256,
            spec_json=spec.model_dump(mode="json"),
            document_snapshot_id=spec.document_snapshot_id,
            rule_version_id=spec.rule_version_id,
            dataset_version_id=spec.dataset_version_id,
            model_config_id=spec.model_config_id,
            retriever_artifact_id=spec.retriever_artifact.artifact_id,
            index_artifact_id=spec.index_artifact.artifact_id,
            chunk_artifact_id=spec.chunk_artifact.artifact_id,
        )

    @classmethod
    def _validated_stored_spec(
        cls, job: ReviewJob, row: DbReviewExecutionSpec
    ) -> ReviewExecutionSpec:
        try:
            spec = ReviewExecutionSpec.model_validate(row.spec_json)
        except ValidationError as exc:
            raise ConflictError(
                "Stored review execution spec failed integrity validation",
                code="review_execution_spec_tampered",
            ) from exc
        if (
            spec.job_id != job.id
            or spec.input_sha256 != job.execution_spec_sha256
            or spec.input_sha256 != row.input_sha256
            or spec.document_snapshot_id != row.document_snapshot_id
            or spec.rule_version_id != row.rule_version_id
            or spec.dataset_version_id != row.dataset_version_id
            or spec.model_config_id != row.model_config_id
            or spec.retriever_artifact.artifact_id != row.retriever_artifact_id
            or spec.index_artifact.artifact_id != row.index_artifact_id
            or spec.chunk_artifact.artifact_id != row.chunk_artifact_id
        ):
            raise ConflictError(
                "Stored review execution spec differs from its bound columns",
                code="review_execution_spec_tampered",
            )
        return spec

    @classmethod
    def _validate_replayed_spec(
        cls,
        session: Session,
        job: ReviewJob,
        expected: ReviewExecutionSpec | None,
    ) -> None:
        row = session.get(DbReviewExecutionSpec, job.id)
        if row is None and expected is None:
            return
        if row is None or expected is None:
            raise ConflictError(
                "Idempotent replay changed review execution provenance",
                code="review_execution_spec_replay_conflict",
            )
        stored = cls._validated_stored_spec(job, row)
        stored_inputs = stored.model_dump(
            mode="json", exclude={"job_id", "input_sha256"}
        )
        replayed_inputs = expected.model_dump(
            mode="json", exclude={"job_id", "input_sha256"}
        )
        if stored_inputs != replayed_inputs:
            raise ConflictError(
                "Idempotent replay changed review execution inputs",
                code="review_execution_spec_replay_conflict",
            )

    @staticmethod
    def _new_durable_job(job: ReviewJobDto) -> ReviewJob:
        failure = job.failure
        return ReviewJob(
            id=job.id,
            document_snapshot_id=job.document_snapshot_id,
            rule_version_id=job.rule_version_id,
            model_config_id=job.model_config_id,
            rerun_of_id=job.rerun_of_id,
            status=job.status.value,
            job_type="review",
            input_reference=job.document_snapshot_id,
            stage=job.stage.value if job.stage is not None else None,
            input_fingerprint=job.input_fingerprint,
            execution_spec_sha256=job.execution_spec_sha256,
            attempt_count=job.attempt_count,
            max_attempts=job.max_attempts,
            available_at=job.available_at,
            lease_owner=job.lease_owner,
            lease_until=job.lease_until,
            heartbeat_at=job.heartbeat_at,
            lease_token=job.lease_token,
            failure_stage=(
                job.failure_stage.value if job.failure_stage is not None else None
            ),
            error_code=failure.code if failure is not None else None,
            error_category=failure.category.value if failure is not None else None,
            error_message=failure.message if failure is not None else None,
            error_retryable=failure.retryable if failure is not None else None,
            completed_at=job.completed_at,
            created_at=job.created_at,
            updated_at=job.updated_at,
        )

    @staticmethod
    def _copy_public_state(durable_job: ReviewJob, job: ReviewJobDto) -> None:
        immutable_values = (
            (durable_job.document_snapshot_id, job.document_snapshot_id),
            (durable_job.rule_version_id, job.rule_version_id),
            (durable_job.model_config_id, job.model_config_id),
            (durable_job.input_fingerprint, job.input_fingerprint),
            (durable_job.execution_spec_sha256, job.execution_spec_sha256),
            (durable_job.rerun_of_id, job.rerun_of_id),
            (durable_job.max_attempts, job.max_attempts),
            (_as_utc(durable_job.created_at), _as_utc(job.created_at)),
        )
        if any(current != proposed for current, proposed in immutable_values):
            raise ConflictError(
                "Immutable review-job fields cannot be changed",
                code="review_job_immutable_field_changed",
            )
        durable_job.status = job.status.value
        durable_job.stage = job.stage.value if job.stage is not None else None
        durable_job.attempt_count = job.attempt_count
        durable_job.available_at = job.available_at
        durable_job.lease_owner = job.lease_owner
        durable_job.lease_until = job.lease_until
        durable_job.heartbeat_at = job.heartbeat_at
        durable_job.failure_stage = (
            job.failure_stage.value if job.failure_stage is not None else None
        )
        durable_job.completed_at = job.completed_at
        durable_job.updated_at = job.updated_at
        failure = job.failure
        durable_job.error_code = failure.code if failure is not None else None
        durable_job.error_category = (
            failure.category.value if failure is not None else None
        )
        durable_job.error_message = failure.message if failure is not None else None
        durable_job.error_retryable = (
            failure.retryable if failure is not None else None
        )

    @staticmethod
    def _review_job_dto(job: ReviewJob) -> ReviewJobDto:
        failure_stage = (
            ReviewStage(job.failure_stage) if job.failure_stage is not None else None
        )
        failure = None
        if job.error_message is not None:
            failure = JobFailure(
                code=job.error_code or "job_failed",
                message=job.error_message,
                category=ErrorCategory(job.error_category or ErrorCategory.INTERNAL.value),
                retryable=bool(job.error_retryable),
                stage=failure_stage,
            )
        return ReviewJobDto(
            id=job.id,
            document_snapshot_id=job.document_snapshot_id,
            rule_version_id=job.rule_version_id,
            model_config_id=job.model_config_id,
            input_fingerprint=job.input_fingerprint,
            execution_spec_sha256=job.execution_spec_sha256,
            status=JobLifecycle(job.status),
            stage=ReviewStage(job.stage) if job.stage is not None else None,
            rerun_of_id=job.rerun_of_id,
            attempt_count=job.attempt_count,
            max_attempts=job.max_attempts,
            available_at=_as_utc(job.available_at),
            lease_owner=job.lease_owner,
            lease_until=_as_utc(job.lease_until) if job.lease_until else None,
            heartbeat_at=_as_utc(job.heartbeat_at) if job.heartbeat_at else None,
            lease_token=job.lease_token,
            failure_stage=failure_stage,
            failure=failure,
            completed_at=_as_utc(job.completed_at) if job.completed_at else None,
            created_at=_as_utc(job.created_at),
            updated_at=_as_utc(job.updated_at),
        )

    @staticmethod
    def _idempotency_dto(record: IdempotencyRecord) -> IdempotencyRecordDto:
        return IdempotencyRecordDto(
            id=record.id,
            caller_id=record.caller_id,
            scope=record.scope,
            idempotency_key=record.idempotency_key,
            request_hash=record.request_hash,
            resource_type=record.resource_type,
            resource_id=record.resource_id,
            created_at=_as_utc(record.created_at),
            expires_at=_as_utc(record.expires_at) if record.expires_at else None,
        )

    @staticmethod
    def _job_checkpoint_dto(checkpoint: JobCheckpoint) -> JobCheckpointDto:
        return JobCheckpointDto(
            job_id=checkpoint.review_job_id,
            node_name=checkpoint.node_name,
            stage=ReviewStage(checkpoint.stage),
            lease_token=checkpoint.lease_token,
            sequence=checkpoint.sequence,
            state=CheckpointState.model_validate(checkpoint.state_json),
            output_artifact_id=checkpoint.output_artifact_id,
            completed_at=_as_utc(checkpoint.completed_at),
        )

    def _assign_lease(
        self,
        job: ReviewJob,
        owner: str,
        now: datetime,
        lease_seconds: int,
    ) -> JobLease:
        job.status = "RUNNING"
        job.lease_owner = owner
        job.lease_until = now + timedelta(seconds=lease_seconds)
        job.heartbeat_at = now
        job.lease_token += 1
        job.attempt_count += 1
        return JobLease(
            job_id=job.id,
            owner=owner,
            token=job.lease_token,
            expires_at=job.lease_until,
        )

    def _mark_expired_job_dead(self, job: ReviewJob, now: datetime) -> None:
        job.status = "DEAD"
        job.lease_owner = None
        job.lease_until = None
        job.heartbeat_at = now
        job.error_category = ErrorCategory.RETRYABLE.value
        job.error_message = "Lease expired after the maximum number of attempts"
        job.error_retryable = True

    def _is_eligible(self, job: ReviewJob, now: datetime) -> bool:
        if job.status in {"QUEUED", "RETRY_WAIT"}:
            return _as_utc(job.available_at) <= now
        return (
            job.status == "RUNNING"
            and job.lease_until is not None
            and _as_utc(job.lease_until) <= now
        )

    def _retry_delay_seconds(self, attempt_count: int) -> int:
        exponent = max(0, attempt_count - 1)
        return min(self._retry_max_seconds, self._retry_base_seconds * (2**exponent))

    @staticmethod
    def _is_concurrency_retryable(exc: OperationalError) -> bool:
        original = exc.orig
        args = getattr(original, "args", ())
        mysql_code = args[0] if args else None
        return mysql_code in {1205, 1213} or "database is locked" in str(original).lower()

    def _require_fenced_update(
        self, session: Session, job_id: str, rowcount: int
    ) -> None:
        if rowcount == 1:
            return
        self._raise_missing_or_stale(session, job_id)

    @staticmethod
    def _raise_missing_or_stale(session: Session, job_id: str) -> None:
        if session.get(ReviewJob, job_id) is None:
            raise NotFoundError(
                f"Review job {job_id!r} does not exist",
                code="job_not_found",
            )
        raise ConflictError(
            "Lease is missing, expired, or fenced",
            code="stale_lease",
        )

    @staticmethod
    def _message(job: ReviewJob) -> JobMessage:
        return JobMessage(
            job_id=job.id,
            job_type=job.job_type,
            input_reference=job.input_reference,
            attempt=job.attempt_count,
            enqueued_at=_as_utc(job.created_at),
        )

    @staticmethod
    def _snapshot(job: ReviewJob) -> ReviewJobSnapshot:
        return ReviewJobSnapshot(
            id=job.id,
            status=job.status,
            job_type=job.job_type,
            input_reference=job.input_reference,
            input_fingerprint=job.input_fingerprint,
            attempt_count=job.attempt_count,
            max_attempts=job.max_attempts,
            available_at=_as_utc(job.available_at),
            lease_token=job.lease_token,
            stage=job.stage,
            rerun_of_id=job.rerun_of_id,
            output_reference=job.output_reference,
            output_summary=job.output_summary,
        )

    @staticmethod
    def _checkpoint_snapshot(checkpoint: JobCheckpoint) -> CheckpointSnapshot:
        return CheckpointSnapshot(
            job_id=checkpoint.review_job_id,
            node_name=checkpoint.node_name,
            stage=checkpoint.stage,
            lease_token=checkpoint.lease_token,
            sequence=checkpoint.sequence,
            state_json=dict(checkpoint.state_json),
            output_artifact_id=checkpoint.output_artifact_id,
            completed_at=_as_utc(checkpoint.completed_at),
        )


MySQLJobRepository = MySqlJobRepository
