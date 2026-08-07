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


def canonical_json(value: Any) -> str:
    return json.dumps(
        to_jsonable_python(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def stable_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _parse_object(value: str, field_name: str) -> dict[str, Any]:
    try:
        parsed = json.loads(value)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{field_name} must be valid JSON") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"{field_name} must contain a JSON object")
    if value != canonical_json(parsed):
        raise ValueError(f"{field_name} must use canonical JSON")
    return parsed


class RuleVersionStatus(str, Enum):
    DRAFT = "DRAFT"
    OPTIMIZING = "OPTIMIZING"
    EVALUATING = "EVALUATING"
    WAITING_APPROVAL = "WAITING_APPROVAL"
    PUBLISHED = "PUBLISHED"
    REJECTED = "REJECTED"
    ROLLED_BACK = "ROLLED_BACK"


class RuleProvenance(ContractModel):
    source_type: Literal[
        "manual", "human_approved", "optimization", "provisional", "synthetic"
    ]
    status: Literal["provisional", "verified"]
    claims_allowed: bool
    source_finding_ids: tuple[str, ...] = ()
    source_decision_ids: tuple[str, ...] = ()
    review_input_sha256s: tuple[str, ...] = ()
    evidence_sha256s: tuple[str, ...] = ()
    dataset_version_id: str | None = Field(default=None, min_length=1, max_length=128)

    @model_validator(mode="after")
    def is_consistent(self) -> Self:
        if self.status == "provisional" and self.claims_allowed:
            raise ValueError("provisional rule provenance cannot allow claims")
        if self.source_type == "human_approved" and not self.source_decision_ids:
            raise ValueError("human-approved provenance needs a source decision")
        if self.source_type in {"human_approved", "optimization"} and (
            not self.source_finding_ids
            or not self.review_input_sha256s
            or not self.evidence_sha256s
        ):
            raise ValueError(
                "rule candidates require finding, evidence, and review-input provenance"
            )
        for values, name in (
            (self.source_finding_ids, "source_finding_ids"),
            (self.source_decision_ids, "source_decision_ids"),
            (self.review_input_sha256s, "review_input_sha256s"),
            (self.evidence_sha256s, "evidence_sha256s"),
        ):
            if len(values) != len(set(values)):
                raise ValueError(f"{name} must contain unique values")
        if any(
            len(value) != 64 or any(char not in "0123456789abcdef" for char in value)
            for value in self.review_input_sha256s
        ):
            raise ValueError("review_input_sha256s must contain SHA-256 digests")
        if any(
            len(value) != 64 or any(char not in "0123456789abcdef" for char in value)
            for value in self.evidence_sha256s
        ):
            raise ValueError("evidence_sha256s must contain SHA-256 digests")
        return self


class EvaluationGateStatus(str, Enum):
    PENDING = "PENDING"
    PASSED = "PASSED"
    FAILED = "FAILED"
    PROVISIONAL = "PROVISIONAL"


class EvaluationGate(ContractModel):
    gate_id: str = Field(min_length=1, max_length=128)
    rule_version_id: str = Field(min_length=1, max_length=128)
    dataset_version_id: str = Field(min_length=1, max_length=128)
    status: EvaluationGateStatus
    provisional: bool
    claims_allowed: bool
    evaluation_run_id: str | None = Field(default=None, min_length=1, max_length=128)
    report_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)
    requested_at: datetime
    completed_at: datetime | None = None
    gate_sha256: str = Field(pattern=SHA256_PATTERN)

    @model_validator(mode="after")
    def state_and_hash_are_consistent(self) -> Self:
        if self.requested_at.tzinfo is None:
            raise ValueError("requested_at must be timezone-aware")
        if self.completed_at is not None and self.completed_at.tzinfo is None:
            raise ValueError("completed_at must be timezone-aware")
        if self.status is EvaluationGateStatus.PENDING:
            if self.completed_at is not None or self.evaluation_run_id is not None:
                raise ValueError("pending gate cannot contain completed result fields")
        else:
            if self.completed_at is None or self.evaluation_run_id is None or self.report_sha256 is None:
                raise ValueError("completed gate needs run, report, and completion time")
        if self.status is EvaluationGateStatus.PASSED and (
            self.provisional or not self.claims_allowed
        ):
            raise ValueError("a provisional or non-claimable gate cannot pass")
        if self.provisional and self.status not in {
            EvaluationGateStatus.PENDING,
            EvaluationGateStatus.PROVISIONAL,
        }:
            raise ValueError("provisional gate must remain provisional")
        payload = self.model_dump(mode="json", exclude={"gate_sha256"})
        if self.gate_sha256 != stable_sha256(payload):
            raise ValueError("gate_sha256 does not match evaluation gate")
        return self


class RuleSet(ContractModel):
    rule_set_id: str = Field(min_length=1, max_length=128)
    rule_key: str = Field(min_length=1, max_length=128)
    name: str = Field(min_length=1, max_length=255)
    description: str | None = Field(default=None, max_length=8000)
    current_version_id: str | None = Field(default=None, min_length=1, max_length=128)
    created_at: datetime


class RuleVersion(ContractModel):
    rule_version_id: str = Field(min_length=1, max_length=128)
    rule_set_id: str = Field(min_length=1, max_length=128)
    version_number: int = Field(ge=1)
    parent_version_id: str | None = Field(default=None, min_length=1, max_length=128)
    status: RuleVersionStatus
    content_json: str = Field(min_length=2)
    execution_config_json: str = Field(min_length=2)
    content_sha256: str = Field(pattern=SHA256_PATTERN)
    change_summary: str = Field(min_length=1, max_length=8000)
    provenance: RuleProvenance
    evaluation_gate: EvaluationGate | None = None
    created_at: datetime
    published_at: datetime | None = None
    published_by: str | None = Field(default=None, min_length=1, max_length=255)

    @field_validator("change_summary")
    @classmethod
    def change_summary_is_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("change_summary must not be blank")
        return value

    @model_validator(mode="after")
    def immutable_content_is_valid(self) -> Self:
        content = _parse_object(self.content_json, "content_json")
        execution = _parse_object(self.execution_config_json, "execution_config_json")
        expected = stable_sha256({"content": content, "execution_config": execution})
        if self.content_sha256 != expected:
            raise ValueError("content_sha256 does not match rule content")
        if self.created_at.tzinfo is None:
            raise ValueError("created_at must be timezone-aware")
        if self.version_number == 1 and self.parent_version_id is not None:
            raise ValueError("first rule version cannot have a parent")
        if self.version_number > 1 and self.parent_version_id is None:
            raise ValueError("later rule versions require a parent")
        if self.status is RuleVersionStatus.PUBLISHED:
            if self.published_at is None or self.published_by is None:
                raise ValueError("published rule needs publication audit fields")
        return self


class CreateRuleVersion(ContractModel):
    rule_set_id: str = Field(min_length=1, max_length=128)
    rule_key: str = Field(min_length=1, max_length=128)
    rule_set_name: str = Field(min_length=1, max_length=255)
    rule_set_description: str | None = Field(default=None, max_length=8000)
    parent_version_id: str | None = Field(default=None, min_length=1, max_length=128)
    content_json: str = Field(min_length=2)
    execution_config_json: str = Field(default="{}", min_length=2)
    change_summary: str = Field(min_length=1, max_length=8000)
    provenance: RuleProvenance

    @field_validator("change_summary")
    @classmethod
    def change_summary_is_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("change_summary must not be blank")
        return value

    @model_validator(mode="after")
    def json_is_canonical(self) -> Self:
        _parse_object(self.content_json, "content_json")
        _parse_object(self.execution_config_json, "execution_config_json")
        return self


class RuleDiffChange(ContractModel):
    path: str
    operation: Literal["add", "remove", "replace"]
    before_json: str | None = None
    after_json: str | None = None


class RuleVersionDiff(ContractModel):
    from_version_id: str
    to_version_id: str
    changes: tuple[RuleDiffChange, ...]


class CompleteEvaluationGate(ContractModel):
    rule_version_id: str
    gate_id: str
    evaluation_run_id: str
    status: Literal["PASSED", "FAILED", "PROVISIONAL"]
    provisional: bool
    claims_allowed: bool
    report_sha256: str = Field(pattern=SHA256_PATTERN)


class PublishRuleVersion(ContractModel):
    rule_version_id: str
    approver_kind: Literal["human"] = "human"
    approver_id: str = Field(min_length=1, max_length=255)

    @field_validator("approver_id")
    @classmethod
    def approver_is_named_human(cls, value: str) -> str:
        return require_named_human(value)


class RollbackRuleSet(ContractModel):
    rule_set_id: str
    target_version_id: str
    approver_kind: Literal["human"] = "human"
    approver_id: str = Field(min_length=1, max_length=255)
    reason: str = Field(min_length=1, max_length=8000)

    @field_validator("approver_id")
    @classmethod
    def approver_is_named_human(cls, value: str) -> str:
        return require_named_human(value)

    @field_validator("reason")
    @classmethod
    def reason_is_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("reason must not be blank")
        return value


def require_named_human(value: str) -> str:
    normalized = value.strip()
    first_token = normalized.casefold()
    for separator in (":", "/", "_", "-"):
        first_token = first_token.split(separator, 1)[0]
    if first_token in {
        "", "ai", "assistant", "anonymous", "bot", "fake", "model",
        "provisional", "service", "synthetic", "system",
    }:
        raise ValueError("a named human identity is required")
    return normalized
