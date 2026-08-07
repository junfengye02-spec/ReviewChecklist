from __future__ import annotations

import hashlib
import json
from datetime import datetime
from enum import Enum
from typing import Any, Literal, Self

from pydantic import AliasChoices, Field, model_validator
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


class ReportSourceType(str, Enum):
    REAL = "real"
    PROVISIONAL = "provisional"
    SYNTHETIC = "synthetic"
    EXTERNAL_PLATFORM = "external-platform"


class RunStatus(str, Enum):
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    UNKNOWN = "unknown"


class MetricStatus(str, Enum):
    MEASURED = "measured"
    PROVISIONAL = "provisional"
    UNKNOWN = "unknown"


class EvaluationRunHashes(ContractModel):
    dataset_sha256: str = Field(pattern=SHA256_PATTERN)
    input_sha256: str = Field(pattern=SHA256_PATTERN)
    results_sha256: str = Field(pattern=SHA256_PATTERN)
    config_sha256: str = Field(pattern=SHA256_PATTERN)
    code_sha256: str = Field(pattern=SHA256_PATTERN)


class ReportMetric(ContractModel):
    metric_id: str = Field(min_length=1, max_length=128)
    label: str = Field(min_length=1, max_length=255)
    value: int | float | str | None = None
    unit: str | None = Field(default=None, max_length=64)
    source_type: ReportSourceType
    status: MetricStatus
    claims_allowed: bool
    collected: bool
    interpretation: str = Field(min_length=1, max_length=2000)

    @model_validator(mode="after")
    def claim_boundary_is_consistent(self) -> Self:
        if not self.collected and self.value is not None:
            raise ValueError("uncollected metrics cannot contain a value")
        if self.status is MetricStatus.UNKNOWN and (
            self.collected or self.value is not None or self.claims_allowed
        ):
            raise ValueError("unknown metrics must remain uncollected and non-claimable")
        if self.source_type is not ReportSourceType.REAL and self.claims_allowed:
            raise ValueError("non-real metrics cannot allow claims")
        if self.status is MetricStatus.PROVISIONAL and self.claims_allowed:
            raise ValueError("provisional metrics cannot allow claims")
        return self


class ReportSection(ContractModel):
    section_id: Literal["conclusion", "evidence", "engineering", "cost"]
    title: str = Field(min_length=1, max_length=128)
    metrics: tuple[ReportMetric, ...]


class EvaluationRun(ContractModel):
    run_id: str = Field(min_length=1, max_length=128)
    name: str = Field(min_length=1, max_length=255)
    source_type: ReportSourceType
    status: RunStatus
    provenance_status: Literal["verified", "provisional", "unknown"]
    claims_allowed: bool
    dataset_version_id: str = Field(min_length=1, max_length=128)
    input_artifact_id: str = Field(min_length=1, max_length=512)
    results_artifact_id: str = Field(min_length=1, max_length=512)
    config_artifact_id: str = Field(min_length=1, max_length=512)
    code_version_id: str = Field(min_length=1, max_length=255)
    hashes: EvaluationRunHashes
    call_id: str = Field(min_length=1, max_length=128)
    request_id: str = Field(min_length=1, max_length=128)
    started_at: datetime
    completed_at: datetime | None = None
    report_sha256: str = Field(pattern=SHA256_PATTERN)

    @model_validator(mode="after")
    def run_boundary_is_consistent(self) -> Self:
        if self.started_at.tzinfo is None:
            raise ValueError("started_at must be timezone-aware")
        if self.completed_at is not None and self.completed_at.tzinfo is None:
            raise ValueError("completed_at must be timezone-aware")
        if self.status is RunStatus.COMPLETED and self.completed_at is None:
            raise ValueError("completed runs require completed_at")
        if self.provenance_status != "verified" and self.claims_allowed:
            raise ValueError("unverified runs cannot allow claims")
        if self.source_type is not ReportSourceType.REAL and self.claims_allowed:
            raise ValueError("non-real runs cannot allow claims")
        return self


class EvaluationReport(ContractModel):
    run_id: str = Field(min_length=1, max_length=128)
    source_type: ReportSourceType
    status: Literal["verified", "provisional", "unknown"]
    claims_allowed: bool
    human_annotation_cases: int = Field(ge=0)
    required_human_cases: int = Field(ge=0)
    sections: tuple[ReportSection, ...]
    limitations: tuple[str, ...] = Field(min_length=1)
    generated_at: datetime
    report_sha256: str = Field(pattern=SHA256_PATTERN)

    @model_validator(mode="after")
    def report_is_traceable_and_bounded(self) -> Self:
        if self.generated_at.tzinfo is None:
            raise ValueError("generated_at must be timezone-aware")
        if self.status != "verified" and self.claims_allowed:
            raise ValueError("unverified reports cannot allow claims")
        if self.source_type is not ReportSourceType.REAL and self.claims_allowed:
            raise ValueError("non-real reports cannot allow claims")
        if any(
            metric.claims_allowed and not self.claims_allowed
            for section in self.sections
            for metric in section.metrics
        ):
            raise ValueError("a non-claimable report cannot contain claimable metrics")
        payload = self.model_dump(mode="json", exclude={"report_sha256"})
        if self.report_sha256 != stable_sha256(payload):
            raise ValueError("report_sha256 does not match report content")
        return self


class WorkbenchResourceIndex(ContractModel):
    demo_mode: bool
    environment: str = Field(min_length=1, max_length=64)
    source_type: ReportSourceType
    status: Literal["verified", "provisional", "unknown"]
    claims_allowed: bool
    human_annotation_cases: int = Field(ge=0)
    required_human_cases: int = Field(ge=0)
    review_job_ids: tuple[str, ...] = ()
    finding_ids: tuple[str, ...] = ()
    rule_set_ids: tuple[str, ...] = ()
    optimization_job_ids: tuple[str, ...] = ()
    evaluation_run_ids: tuple[str, ...] = ()
    generated_at: datetime

    @model_validator(mode="after")
    def demo_boundary_is_visible(self) -> Self:
        if self.generated_at.tzinfo is None:
            raise ValueError("generated_at must be timezone-aware")
        if self.demo_mode and (
            self.status != "provisional"
            or self.claims_allowed
            or self.source_type is ReportSourceType.REAL
        ):
            raise ValueError("demo workbench data must be visibly provisional")
        return self


class ActorKind(str, Enum):
    HUMAN = "human"
    SYSTEM = "system"
    AI = "ai"


class AuditResult(str, Enum):
    SUCCEEDED = "succeeded"
    REJECTED = "rejected"
    FAILED = "failed"


class AuditActor(ContractModel):
    kind: ActorKind
    actor_id: str = Field(min_length=1, max_length=255)


class AuditResource(ContractModel):
    resource_type: str = Field(min_length=1, max_length=128)
    resource_id: str = Field(min_length=1, max_length=256)


class AuditProvenance(ContractModel):
    source_type: ReportSourceType
    status: Literal["verified", "provisional", "unknown"]
    claims_allowed: bool
    artifact_sha256s: tuple[str, ...] = ()

    @model_validator(mode="after")
    def provenance_boundary_is_consistent(self) -> Self:
        if self.status != "verified" and self.claims_allowed:
            raise ValueError("unverified audit provenance cannot allow claims")
        return self


class AuditEvent(ContractModel):
    event_id: str = Field(min_length=1, max_length=128)
    actor: AuditActor
    action: str = Field(min_length=1, max_length=255)
    resource: AuditResource
    before_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)
    after_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)
    provenance: AuditProvenance
    call_id: str = Field(min_length=1, max_length=128)
    request_id: str = Field(min_length=1, max_length=128)
    job_id: str | None = Field(default=None, min_length=1, max_length=128)
    thread_id: str | None = Field(default=None, min_length=1, max_length=128)
    checkpoint_id: str | None = Field(default=None, min_length=1, max_length=128)
    rule_version: str | None = Field(default=None, min_length=1, max_length=128)
    dataset_version: str | None = Field(default=None, min_length=1, max_length=128)
    model_configuration: str | None = Field(
        default=None,
        min_length=1,
        max_length=128,
        validation_alias=AliasChoices("model_configuration", "model_config"),
        serialization_alias="model_config",
    )
    occurred_at: datetime
    result: AuditResult

    @model_validator(mode="after")
    def timestamp_is_aware(self) -> Self:
        if self.occurred_at.tzinfo is None:
            raise ValueError("occurred_at must be timezone-aware")
        return self
