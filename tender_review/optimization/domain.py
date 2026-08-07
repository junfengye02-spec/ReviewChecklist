from __future__ import annotations

from datetime import datetime

from tender_review.shared.errors import ConflictError

from .models import (
    OptimizationJob,
    OptimizationStatus,
    RootCause,
    RootCauseDecision,
    SampleRole,
)


_ALLOWED_TRANSITIONS = {
    OptimizationStatus.PENDING: {
        OptimizationStatus.RUNNING,
        OptimizationStatus.CANCELLED,
    },
    OptimizationStatus.RUNNING: {
        OptimizationStatus.RUNNING,
        OptimizationStatus.WAITING_APPROVAL,
        OptimizationStatus.WAITING_HUMAN,
        OptimizationStatus.OPTIMIZATION_FAILED,
        OptimizationStatus.CANCELLED,
    },
}


def transition_job(
    job: OptimizationJob,
    status: OptimizationStatus,
    *,
    now: datetime,
    **updates: object,
) -> OptimizationJob:
    if status is job.status:
        return job
    if status not in _ALLOWED_TRANSITIONS.get(job.status, set()):
        raise ConflictError(
            "invalid optimization job transition",
            code="optimization_transition_invalid",
            details={"from": job.status.value, "to": status.value},
        )
    values = {"status": status, "updated_at": now, **updates}
    if status in {
        OptimizationStatus.WAITING_APPROVAL,
        OptimizationStatus.WAITING_HUMAN,
        OptimizationStatus.OPTIMIZATION_FAILED,
        OptimizationStatus.CANCELLED,
    }:
        values["completed_at"] = now
    payload = job.model_dump(mode="json")
    payload.update(values)
    return OptimizationJob.model_validate(payload)


def deterministic_root_cause(job: OptimizationJob) -> RootCauseDecision | None:
    targets = tuple(item for item in job.samples if item.role is SampleRole.TARGET)
    target_ids = tuple(item.sample_id for item in targets)
    signals = tuple(item.signals for item in targets if item.signals is not None)
    if any(item.label_conflict is True or item.evidence_conflict is True for item in signals):
        return RootCauseDecision(
            root_cause=RootCause.LABEL_UNCERTAIN,
            classifier="deterministic",
            rationale="A target contains conflicting label or evidence provenance.",
            target_sample_ids=target_ids,
        )
    if any(item.evidence_in_top_k is False for item in signals):
        return RootCauseDecision(
            root_cause=RootCause.RETRIEVAL_MISS,
            classifier="deterministic",
            rationale="Confirmed target evidence did not enter Top-K retrieval.",
            target_sample_ids=target_ids,
        )
    if any(
        item.evidence_in_top_k is True
        and item.extraction_matches_expected is False
        for item in signals
    ):
        return RootCauseDecision(
            root_cause=RootCause.EXTRACTION_ERROR,
            classifier="deterministic",
            rationale="Evidence was retrieved but structured extraction differed.",
            target_sample_ids=target_ids,
        )
    if any(
        item.extraction_matches_expected is True
        and item.tool_matches_expected is False
        for item in signals
    ):
        return RootCauseDecision(
            root_cause=RootCause.TOOL_ERROR,
            classifier="deterministic",
            rationale="Structured fields were correct but the deterministic tool result differed.",
            target_sample_ids=target_ids,
        )
    if any(item.repeated_outputs_consistent is False for item in signals):
        return RootCauseDecision(
            root_cause=RootCause.MODEL_INSTABILITY,
            classifier="deterministic",
            rationale="Repeated runs for an identical input produced inconsistent outputs.",
            target_sample_ids=target_ids,
        )
    if signals and all(
        item.evidence_in_top_k is True
        and item.extraction_matches_expected is True
        and item.tool_matches_expected is True
        and item.repeated_outputs_consistent is True
        for item in signals
    ):
        return RootCauseDecision(
            root_cause=RootCause.RULE_GAP,
            classifier="deterministic",
            rationale="Retrieval, extraction, tool, and stability checks passed; the rule lacks coverage.",
            target_sample_ids=target_ids,
        )
    return None
