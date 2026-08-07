from __future__ import annotations

from tender_review.shared.clock import Clock
from tender_review.shared.errors import ConflictError, PermanentError
from tender_review.shared.ids import IdGenerator

from .models import (
    DecisionOutcome,
    Finding,
    FindingStatus,
    FindingWorkflowState,
    HumanDecision,
    HumanDecisionType,
    SubmitHumanDecision,
    stable_sha256,
)
from .ports import FindingRepository


class FindingDecisionService:
    def __init__(self, repository: FindingRepository, ids: IdGenerator, clock: Clock) -> None:
        self._repository = repository
        self._ids = ids
        self._clock = clock

    def submit(self, command: SubmitHumanDecision) -> DecisionOutcome:
        finding = self._repository.get_finding(command.finding_id)
        prior = self._repository.list_decisions(command.finding_id)
        latest = prior[-1] if prior else None
        if latest is None and command.supersedes_decision_id is not None:
            raise ConflictError(
                "there is no prior decision to supersede",
                code="decision_supersedes_missing",
            )
        if latest is not None and command.supersedes_decision_id != latest.decision_id:
            raise ConflictError(
                "a correction must explicitly supersede the latest decision",
                code="decision_supersedes_latest_required",
                details={"latest_decision_id": latest.decision_id},
            )
        self._validate_action(finding, command.decision)
        decided_at = self._clock.now()
        evidence_sha256 = stable_sha256(
            [item.model_dump(mode="json") for item in finding.evidence]
        )
        payload = {
            "schema_version": 1,
            "decision_id": self._ids.new(),
            "finding_id": finding.finding_id,
            "reviewer_kind": command.reviewer_kind,
            "reviewer_id": command.reviewer_id,
            "decision": command.decision,
            "reason": command.reason,
            "revision": (
                command.revision.model_dump(mode="json") if command.revision else None
            ),
            "supersedes_decision_id": command.supersedes_decision_id,
            "decided_at": decided_at,
            "review_input_sha256": finding.provenance.review_input_sha256,
            "finding_content_sha256": finding.finding_content_sha256,
            "evidence_sha256": evidence_sha256,
        }
        decision = HumanDecision(**payload, decision_sha256=stable_sha256(payload))
        updated = finding.model_copy(update={"status": _status_for(command.decision)})
        self._repository.append_decision(finding=updated, decision=decision)
        return DecisionOutcome(finding=updated, decision=decision)

    @staticmethod
    def _validate_action(finding: Finding, decision: HumanDecisionType) -> None:
        if finding.workflow_state is not FindingWorkflowState.DONE:
            if decision not in {
                HumanDecisionType.MODIFY,
                HumanDecisionType.INSUFFICIENT_EVIDENCE,
            }:
                raise PermanentError(
                    "handoff branches must be resolved as explicit work items",
                    code="work_item_decision_invalid",
                    details={"workflow_state": finding.workflow_state.value},
                )
            return
        if (
            not finding.human_approval_allowed
            and decision is HumanDecisionType.APPROVE
        ):
            raise PermanentError(
                "This demonstration finding cannot be recorded as human-approved truth",
                code="provisional_finding_approval_forbidden",
                details={"finding_id": finding.finding_id},
            )


def _status_for(decision: HumanDecisionType) -> FindingStatus:
    return {
        HumanDecisionType.APPROVE: FindingStatus.APPROVED,
        HumanDecisionType.REJECT: FindingStatus.REJECTED,
        HumanDecisionType.MODIFY: FindingStatus.MODIFIED,
        HumanDecisionType.INSUFFICIENT_EVIDENCE: FindingStatus.INSUFFICIENT_EVIDENCE,
    }[decision]
