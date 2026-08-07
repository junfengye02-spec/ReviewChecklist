from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime

from tender_review.shared.errors import ConflictError, ErrorCategory

from .models import JobFailure, JobLifecycle, ReviewJob, ReviewStage


_LEGAL_LIFECYCLE_TRANSITIONS: Mapping[JobLifecycle, frozenset[JobLifecycle]] = {
    JobLifecycle.QUEUED: frozenset({JobLifecycle.RUNNING, JobLifecycle.CANCELLED}),
    JobLifecycle.RUNNING: frozenset(
        {
            JobLifecycle.WAITING_HUMAN,
            JobLifecycle.COMPLETED,
            JobLifecycle.RETRY_WAIT,
            JobLifecycle.FAILED,
            JobLifecycle.DEAD,
            JobLifecycle.CANCELLED,
        }
    ),
    JobLifecycle.WAITING_HUMAN: frozenset({JobLifecycle.COMPLETED}),
    JobLifecycle.RETRY_WAIT: frozenset({JobLifecycle.QUEUED}),
    JobLifecycle.COMPLETED: frozenset(),
    JobLifecycle.FAILED: frozenset(),
    JobLifecycle.DEAD: frozenset(),
    JobLifecycle.CANCELLED: frozenset(),
}

_STAGE_ORDER = tuple(ReviewStage)


def transition_lifecycle(
    job: ReviewJob, target: JobLifecycle, *, now: datetime
) -> ReviewJob:
    """Move a job through one declared lifecycle edge."""

    if target not in _LEGAL_LIFECYCLE_TRANSITIONS[job.status]:
        raise ConflictError(
            f"Cannot transition review job from {job.status.value} to {target.value}",
            code="invalid_job_transition",
            details={"from_status": job.status.value, "to_status": target.value},
        )
    updates: dict[str, object] = {"status": target, "updated_at": now}
    if target is JobLifecycle.RUNNING:
        next_attempt = job.attempt_count + 1
        if next_attempt > job.max_attempts:
            raise ConflictError(
                "A job cannot start after its maximum attempts",
                code="job_attempt_limit_reached",
            )
        updates["attempt_count"] = next_attempt
    elif job.status is JobLifecycle.RUNNING:
        updates.update(
            {
                "lease_owner": None,
                "lease_until": None,
                "heartbeat_at": now,
                # Any terminal, retry, or human handoff from RUNNING fences
                # the worker that held the previous lease token.
                "lease_token": job.lease_token + 1,
            }
        )
    if target is JobLifecycle.COMPLETED:
        updates["completed_at"] = now
    return job.model_copy(update=updates)


def start_job(job: ReviewJob, *, now: datetime) -> ReviewJob:
    return transition_lifecycle(job, JobLifecycle.RUNNING, now=now)


def advance_stage(job: ReviewJob, stage: ReviewStage, *, now: datetime) -> ReviewJob:
    """Advance exactly one processing stage while the lifecycle remains RUNNING."""

    if job.status is not JobLifecycle.RUNNING:
        raise ConflictError(
            "A processing stage can only advance while a job is running",
            code="stage_transition_requires_running_job",
        )
    expected_index = 0 if job.stage is None else _STAGE_ORDER.index(job.stage) + 1
    if expected_index >= len(_STAGE_ORDER) or _STAGE_ORDER[expected_index] is not stage:
        expected = None if expected_index >= len(_STAGE_ORDER) else _STAGE_ORDER[expected_index]
        raise ConflictError(
            "Review stages must progress in order",
            code="invalid_stage_transition",
            details={
                "current_stage": job.stage.value if job.stage else None,
                "expected_stage": expected.value if expected else None,
                "requested_stage": stage.value,
            },
        )
    return job.model_copy(update={"stage": stage, "updated_at": now})


def wait_for_human(job: ReviewJob, *, now: datetime) -> ReviewJob:
    return transition_lifecycle(job, JobLifecycle.WAITING_HUMAN, now=now)


def complete_job(job: ReviewJob, *, now: datetime) -> ReviewJob:
    return transition_lifecycle(job, JobLifecycle.COMPLETED, now=now)


def cancel_job(job: ReviewJob, *, now: datetime) -> ReviewJob:
    return transition_lifecycle(job, JobLifecycle.CANCELLED, now=now)


def queue_retry(job: ReviewJob, *, now: datetime) -> ReviewJob:
    retried = transition_lifecycle(job, JobLifecycle.QUEUED, now=now)
    return retried.model_copy(update={"available_at": now})


def record_failure(job: ReviewJob, failure: JobFailure, *, now: datetime) -> ReviewJob:
    """Classify a failure as retry-wait, terminal failure, or dead-lettered."""

    if job.status is not JobLifecycle.RUNNING:
        raise ConflictError(
            "Failures can only be recorded for running jobs",
            code="failure_requires_running_job",
        )
    failure_stage = failure.stage or job.stage
    if failure_stage is None:
        raise ConflictError(
            "A review job failure must identify its processing stage",
            code="failure_stage_required",
        )
    staged_failure = failure.model_copy(update={"stage": failure_stage})
    if staged_failure.category is ErrorCategory.CANCELLED:
        target = JobLifecycle.CANCELLED
    elif not staged_failure.retryable:
        target = JobLifecycle.FAILED
    elif job.attempt_count >= job.max_attempts:
        target = JobLifecycle.DEAD
    else:
        target = JobLifecycle.RETRY_WAIT
    transitioned = transition_lifecycle(job, target, now=now)
    return transitioned.model_copy(
        update={
            "failure_stage": failure_stage,
            "failure": staged_failure,
            "updated_at": now,
        }
    )


def can_rerun(job: ReviewJob) -> bool:
    return job.status in {
        JobLifecycle.COMPLETED,
        JobLifecycle.FAILED,
        JobLifecycle.DEAD,
        JobLifecycle.CANCELLED,
    }
