from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from tender_review.shared.errors import PermanentError, ServiceError

from .models import (
    AttemptFailure,
    AttemptStatus,
    JointRegressionResult,
    OptimizationAttempt,
    OptimizationCandidate,
    OptimizationJob,
    OptimizationReadinessStatus,
    SampleRole,
    stable_sha256,
)


def new_attempt(
    *, attempt_id: str, job_id: str, attempt_number: int, now: datetime
) -> OptimizationAttempt:
    payload = {
        "schema_version": 1,
        "attempt_id": attempt_id,
        "optimization_job_id": job_id,
        "attempt_number": attempt_number,
        "status": AttemptStatus.STARTED,
        "root_cause": None,
        "candidates": (),
        "evaluations": (),
        "selected_candidate_id": None,
        "candidate_rule_version_id": None,
        "failure": None,
        "started_at": now,
        "updated_at": now,
        "completed_at": None,
    }
    return OptimizationAttempt(**payload, checkpoint_sha256=stable_sha256(payload))


def replace_attempt(
    attempt: OptimizationAttempt, now: datetime, **updates: Any
) -> OptimizationAttempt:
    payload = attempt.model_dump(mode="json", exclude={"checkpoint_sha256"})
    payload.update(updates)
    payload["updated_at"] = now
    return OptimizationAttempt(**payload, checkpoint_sha256=stable_sha256(payload))


def failure_from_exception(phase: str, exc: Exception) -> AttemptFailure:
    if isinstance(exc, ServiceError):
        return AttemptFailure(
            phase=phase,
            code=exc.code,
            message=exc.message,
            retryable=exc.retryable,
            call_id=(
                str(exc.details["call_id"])
                if exc.details.get("call_id") is not None
                else None
            ),
        )
    return AttemptFailure(
        phase=phase,
        code="optimization_round_failed",
        message=str(exc)[:4000] or type(exc).__name__,
        retryable=False,
        call_id=None,
    )


def validate_result_boundary(
    job: OptimizationJob, result: JointRegressionResult
) -> None:
    if job.readiness.status is not OptimizationReadinessStatus.READY:
        raise PermanentError(
            "non-ready optimization cannot evaluate candidates",
            code="optimization_regression_not_ready",
        )
    if result.provisional or (
        result.accepted_for_manual_review and not result.claims_allowed
    ) or (result.claims_allowed and not job.provenance.claims_allowed):
        raise PermanentError(
            "regression result does not preserve the A5 claim boundary",
            code="optimization_regression_boundary",
        )


def validate_candidate_boundary(
    job: OptimizationJob,
    attempt_number: int,
    base_content_json: str,
    base_execution_config_json: str,
    candidate: OptimizationCandidate,
) -> None:
    provenance = candidate.provenance
    if (
        provenance.optimization_job_id != job.optimization_job_id
        or provenance.attempt_number != attempt_number
        or provenance.base_rule_version_id != job.base_rule_version_id
        or provenance.dataset_version_id != job.dataset_version_id
        or provenance.hashes != job.hashes
        or provenance.status != job.provenance.status
        or provenance.claims_allowed != job.provenance.claims_allowed
    ):
        raise PermanentError(
            "candidate provenance does not match the optimization checkpoint",
            code="optimization_candidate_provenance_mismatch",
        )
    target_ids = {
        item.sample_id for item in job.samples if item.role is SampleRole.TARGET
    }
    protection_ids = {
        item.sample_id for item in job.samples if item.role is SampleRole.PROTECTION
    }
    if not set(candidate.target_sample_ids).issubset(target_ids) or not set(
        candidate.affected_protection_sample_ids
    ).issubset(protection_ids):
        raise PermanentError(
            "candidate sample scope exceeds the optimization dataset",
            code="optimization_candidate_sample_scope",
        )
    content = json.loads(base_content_json)
    execution = json.loads(base_execution_config_json)
    change = candidate.change
    target = content if change.scope == "content" else execution
    before = _value_at_path(target, change.path)
    expected_before = (
        json.loads(change.before_json) if change.before_json is not None else None
    )
    if before != expected_before:
        raise PermanentError(
            "candidate before value differs from the immutable base",
            code="optimization_candidate_before_mismatch",
        )
    _set_path(target, change.path, json.loads(change.after_json))
    if (
        candidate.content_json
        != json.dumps(content, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        or candidate.execution_config_json
        != json.dumps(execution, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    ):
        raise PermanentError(
            "candidate content contains changes outside its declared minimal change",
            code="optimization_candidate_not_minimal",
        )


def _path_parts(path: str) -> tuple[str, ...]:
    if not path.startswith("$.") or "[" in path or "]" in path:
        raise PermanentError(
            "candidate change path is unsupported",
            code="optimization_candidate_path_unsupported",
        )
    parts = tuple(path[2:].split("."))
    if not parts or any(not item for item in parts):
        raise PermanentError(
            "candidate change path is invalid",
            code="optimization_candidate_path_invalid",
        )
    return parts


def _value_at_path(value: dict[str, Any], path: str) -> Any:
    current: Any = value
    for part in _path_parts(path):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def _set_path(value: dict[str, Any], path: str, after: Any) -> None:
    parts = _path_parts(path)
    current = value
    for part in parts[:-1]:
        child = current.setdefault(part, {})
        if not isinstance(child, dict):
            raise PermanentError(
                "candidate change traverses a non-object value",
                code="optimization_candidate_path_conflict",
            )
        current = child
    current[parts[-1]] = after
