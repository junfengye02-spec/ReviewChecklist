from __future__ import annotations

from datetime import datetime

from tender_review.findings.public import (
    DocumentIdentity,
    Finding,
    FindingProvenance,
    FindingWorkflowState,
    build_finding,
)
from tender_review.shared.errors import PermanentError

from .models import ReviewGraphNode, ReviewGraphState, ReviewLifecycle


def approval_finding_from_review_state(
    state: ReviewGraphState,
    *,
    rule_version_id: str,
    documents: tuple[DocumentIdentity, ...],
    created_at: datetime,
) -> Finding:
    """Project a Stage 5 terminal state without weakening its provenance."""

    provenance = FindingProvenance(
        source_kind=state.provenance.source_kind,
        status=state.provenance.status,
        claims_allowed=state.provenance.claims_allowed,
        dataset_version_id=state.provenance.dataset_version_id,
        review_input_sha256=state.provenance.input_sha256,
        retrieval_results_sha256=state.provenance.results_sha256,
        retrieval_variant=state.provenance.variant,
    )
    if state.node is ReviewGraphNode.DONE:
        if state.lifecycle is not ReviewLifecycle.COMPLETED or state.finding is None:
            raise PermanentError(
                "DONE review state is incomplete",
                code="review_finding_projection_invalid",
            )
        return build_finding(
            finding_id=state.finding.finding_id,
            review_job_id=state.review_job_id,
            rule_version_id=rule_version_id,
            review_item_id=state.rule.review_item_id,
            workflow_state=FindingWorkflowState.DONE,
            message=state.finding.message,
            documents=documents,
            provenance=provenance,
            created_at=created_at,
            conclusion=state.finding.conclusion,
            evidence=state.finding.evidence,
        )
    work_item_state = {
        ReviewGraphNode.NEED_MORE_EVIDENCE: FindingWorkflowState.NEED_MORE_EVIDENCE,
        ReviewGraphNode.HUMAN_HANDOFF: FindingWorkflowState.WAITING_HUMAN,
    }.get(state.node)
    if work_item_state is None:
        raise PermanentError(
            "only completed or explicit handoff states can enter approval storage",
            code="review_state_not_approvable",
            details={"node": state.node.value},
        )
    return build_finding(
        finding_id=f"work-item:{state.review_job_id}:{state.rule.review_item_id}",
        review_job_id=state.review_job_id,
        rule_version_id=rule_version_id,
        review_item_id=state.rule.review_item_id,
        workflow_state=work_item_state,
        message=state.reason or "review requires explicit human work",
        documents=documents,
        provenance=provenance,
        created_at=created_at,
    )

