from __future__ import annotations

import hashlib
import json
from datetime import datetime
from enum import Enum
from typing import Any, Literal, Self

from pydantic import Field, field_validator, model_validator
from pydantic_core import to_jsonable_python

from tender_review.shared.contracts import ContractModel


SHA256_PATTERN = r"^[0-9a-f]{64}$"


def stable_sha256(value: Any) -> str:
    encoded = json.dumps(
        to_jsonable_python(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _not_blank(value: str) -> str:
    if not value.strip():
        raise ValueError("value must not be blank")
    return value


def _named_human(value: str) -> str:
    normalized = value.strip()
    first_token = normalized.casefold()
    for separator in (":", "/", "_", "-"):
        first_token = first_token.split(separator, 1)[0]
    if not normalized or first_token in {
        "ai", "assistant", "anonymous", "bot", "fake", "model",
        "provisional", "service", "synthetic", "system",
    }:
        raise ValueError("reviewer_id must identify a named human reviewer")
    return normalized


class EvidenceReference(ContractModel):
    document_id: str = Field(min_length=1, max_length=256)
    chunk_id: str = Field(min_length=1, max_length=256)
    page_number: int = Field(ge=1)
    page_end: int | None = Field(default=None, ge=1)
    section_path: tuple[str, ...] = Field(default=(), max_length=32)
    excerpt: str = Field(min_length=1, max_length=8000)
    text_sha256: str = Field(pattern=SHA256_PATTERN)

    @model_validator(mode="before")
    @classmethod
    def normalize_page_end(cls, value: Any) -> Any:
        if isinstance(value, dict) and value.get("page_end") is None:
            value = dict(value)
            value["page_end"] = value.get("page_number")
        return value

    @field_validator("document_id", "chunk_id", "excerpt")
    @classmethod
    def values_are_not_blank(cls, value: str) -> str:
        return _not_blank(value)

    @model_validator(mode="after")
    def is_locatable_and_untampered(self) -> Self:
        if self.page_end is not None and self.page_end < self.page_number:
            raise ValueError("page_end must not be less than page_number")
        if self.text_sha256 != hashlib.sha256(self.excerpt.encode("utf-8")).hexdigest():
            raise ValueError("text_sha256 does not match excerpt")
        if any(not part.strip() for part in self.section_path):
            raise ValueError("section_path must not contain blank values")
        return self


class FindingSummary(ContractModel):
    finding_id: str = Field(min_length=1, max_length=128)
    review_job_id: str = Field(min_length=1, max_length=128)
    conclusion: Literal["compliant", "noncompliant", "needs_more_evidence"]
    message: str = Field(min_length=1, max_length=8000)
    evidence: tuple[EvidenceReference, ...] = ()

    @field_validator("finding_id", "review_job_id", "message")
    @classmethod
    def values_are_not_blank(cls, value: str) -> str:
        return _not_blank(value)


class FindingWorkflowState(str, Enum):
    DONE = "DONE"
    NEED_MORE_EVIDENCE = "NEED_MORE_EVIDENCE"
    WAITING_HUMAN = "WAITING_HUMAN"


class FindingStatus(str, Enum):
    PENDING_DECISION = "PENDING_DECISION"
    WORK_ITEM_OPEN = "WORK_ITEM_OPEN"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    MODIFIED = "MODIFIED"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


class FindingProvenance(ContractModel):
    source_kind: Literal["provisional_retrieval", "verified_retrieval"]
    status: Literal["provisional", "verified"]
    claims_allowed: bool
    dataset_version_id: str = Field(min_length=1, max_length=128)
    review_input_sha256: str = Field(pattern=SHA256_PATTERN)
    retrieval_results_sha256: str = Field(pattern=SHA256_PATTERN)
    retrieval_variant: str = Field(min_length=1, max_length=128)

    @model_validator(mode="after")
    def claim_boundary_is_preserved(self) -> Self:
        if self.source_kind == "provisional_retrieval":
            if self.status != "provisional" or self.claims_allowed:
                raise ValueError("provisional provenance cannot allow claims")
        elif self.status != "verified":
            raise ValueError("verified retrieval must retain verified status")
        return self


class DocumentIdentity(ContractModel):
    document_id: str = Field(min_length=1, max_length=256)
    document_sha256: str = Field(pattern=SHA256_PATTERN)


class Finding(ContractModel):
    finding_id: str = Field(min_length=1, max_length=128)
    review_job_id: str = Field(min_length=1, max_length=128)
    rule_version_id: str = Field(min_length=1, max_length=128)
    review_item_id: str = Field(min_length=1, max_length=128)
    workflow_state: FindingWorkflowState
    status: FindingStatus
    conclusion: Literal["compliant", "noncompliant"] | None = None
    message: str = Field(min_length=1, max_length=8000)
    evidence: tuple[EvidenceReference, ...] = ()
    documents: tuple[DocumentIdentity, ...] = Field(min_length=1, max_length=100)
    provenance: FindingProvenance
    human_approval_allowed: bool = True
    created_at: datetime
    finding_content_sha256: str = Field(pattern=SHA256_PATTERN)

    @field_validator("finding_id", "review_job_id", "rule_version_id", "review_item_id", "message")
    @classmethod
    def values_are_not_blank(cls, value: str) -> str:
        return _not_blank(value)

    @model_validator(mode="after")
    def state_and_hash_are_consistent(self) -> Self:
        if self.created_at.tzinfo is None:
            raise ValueError("created_at must be timezone-aware")
        document_ids = tuple(item.document_id for item in self.documents)
        if len(document_ids) != len(set(document_ids)):
            raise ValueError("finding documents must be unique")
        if self.workflow_state is FindingWorkflowState.DONE:
            if self.status not in {
                FindingStatus.PENDING_DECISION,
                FindingStatus.APPROVED,
                FindingStatus.REJECTED,
                FindingStatus.MODIFIED,
                FindingStatus.INSUFFICIENT_EVIDENCE,
            }:
                raise ValueError("DONE finding has an invalid status")
            if self.conclusion is None or not self.evidence:
                raise ValueError("DONE finding needs a conclusion and evidence")
        else:
            if self.status not in {
                FindingStatus.WORK_ITEM_OPEN,
                FindingStatus.MODIFIED,
                FindingStatus.INSUFFICIENT_EVIDENCE,
            }:
                raise ValueError("handoff branches must remain explicit work items")
            if self.conclusion is not None or self.evidence:
                raise ValueError("open work items cannot expose a determined finding")
        evidence_documents = {item.document_id for item in self.evidence}
        if not evidence_documents.issubset(set(document_ids)):
            raise ValueError("evidence references an unknown finding document")
        payload = self.model_dump(
            mode="json", exclude={"status", "finding_content_sha256"}
        )
        if self.finding_content_sha256 != stable_sha256(payload):
            raise ValueError("finding_content_sha256 does not match finding content")
        return self


class HumanDecisionType(str, Enum):
    APPROVE = "APPROVE"
    REJECT = "REJECT"
    MODIFY = "MODIFY"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


class FindingRevision(ContractModel):
    conclusion: Literal["compliant", "noncompliant"]
    message: str = Field(min_length=1, max_length=8000)
    evidence: tuple[EvidenceReference, ...] = Field(min_length=1)

    @field_validator("message")
    @classmethod
    def message_is_not_blank(cls, value: str) -> str:
        return _not_blank(value)


class HumanDecision(ContractModel):
    decision_id: str = Field(min_length=1, max_length=128)
    finding_id: str = Field(min_length=1, max_length=128)
    reviewer_kind: Literal["human"] = "human"
    reviewer_id: str = Field(min_length=1, max_length=255)
    decision: HumanDecisionType
    reason: str = Field(min_length=1, max_length=8000)
    revision: FindingRevision | None = None
    supersedes_decision_id: str | None = Field(default=None, min_length=1, max_length=128)
    decided_at: datetime
    review_input_sha256: str = Field(pattern=SHA256_PATTERN)
    finding_content_sha256: str = Field(pattern=SHA256_PATTERN)
    evidence_sha256: str = Field(pattern=SHA256_PATTERN)
    decision_sha256: str = Field(pattern=SHA256_PATTERN)

    @field_validator("reviewer_id")
    @classmethod
    def reviewer_is_named_human(cls, value: str) -> str:
        return _named_human(value)

    @field_validator("reason")
    @classmethod
    def reason_is_not_blank(cls, value: str) -> str:
        return _not_blank(value)

    @model_validator(mode="after")
    def revision_and_hash_are_consistent(self) -> Self:
        if self.decided_at.tzinfo is None:
            raise ValueError("decided_at must be timezone-aware")
        if (self.decision is HumanDecisionType.MODIFY) != (self.revision is not None):
            raise ValueError("MODIFY alone requires a revision")
        payload = self.model_dump(mode="json", exclude={"decision_sha256"})
        if self.decision_sha256 != stable_sha256(payload):
            raise ValueError("decision_sha256 does not match decision content")
        return self


class SubmitHumanDecision(ContractModel):
    finding_id: str = Field(min_length=1, max_length=128)
    reviewer_kind: Literal["human"] = "human"
    reviewer_id: str = Field(min_length=1, max_length=255)
    decision: HumanDecisionType
    reason: str = Field(min_length=1, max_length=8000)
    revision: FindingRevision | None = None
    supersedes_decision_id: str | None = Field(default=None, min_length=1, max_length=128)

    @field_validator("reviewer_id")
    @classmethod
    def reviewer_is_named_human(cls, value: str) -> str:
        return _named_human(value)

    @field_validator("reason")
    @classmethod
    def reason_is_not_blank(cls, value: str) -> str:
        return _not_blank(value)

    @model_validator(mode="after")
    def revision_matches_decision(self) -> Self:
        if (self.decision is HumanDecisionType.MODIFY) != (self.revision is not None):
            raise ValueError("MODIFY alone requires a revision")
        return self


class DecisionOutcome(ContractModel):
    finding: Finding
    decision: HumanDecision


def build_finding(
    *,
    finding_id: str,
    review_job_id: str,
    rule_version_id: str,
    review_item_id: str,
    workflow_state: FindingWorkflowState,
    message: str,
    documents: tuple[DocumentIdentity, ...],
    provenance: FindingProvenance,
    created_at: datetime,
    conclusion: Literal["compliant", "noncompliant"] | None = None,
    evidence: tuple[EvidenceReference, ...] = (),
    human_approval_allowed: bool = True,
) -> Finding:
    status = (
        FindingStatus.PENDING_DECISION
        if workflow_state is FindingWorkflowState.DONE
        else FindingStatus.WORK_ITEM_OPEN
    )
    payload = {
        "schema_version": 1,
        "finding_id": finding_id,
        "review_job_id": review_job_id,
        "rule_version_id": rule_version_id,
        "review_item_id": review_item_id,
        "workflow_state": workflow_state,
        "status": status,
        "conclusion": conclusion,
        "message": message,
        "evidence": [item.model_dump(mode="json") for item in evidence],
        "documents": [item.model_dump(mode="json") for item in documents],
        "provenance": provenance.model_dump(mode="json"),
        "human_approval_allowed": human_approval_allowed,
        "created_at": created_at,
    }
    hash_payload = dict(payload)
    hash_payload.pop("status")
    return Finding(**payload, finding_content_sha256=stable_sha256(hash_payload))
