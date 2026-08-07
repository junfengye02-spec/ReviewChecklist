from __future__ import annotations

from datetime import datetime
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from tender_review.jobs.public import (
    CreateReviewJobCommand,
    JobFailure,
    JobLifecycle,
    ReviewExecutionSpecDraft,
    ReviewJob,
    ReviewStage,
)
from tender_review.documents.public import SnapshotRecord
from tender_review.findings.public import (
    FindingRevision,
    HumanDecisionType,
    SubmitHumanDecision,
)
from tender_review.rule_management.public import (
    CreateRuleVersion,
    PublishRuleVersion,
    RollbackRuleSet,
    RuleProvenance,
    canonical_json,
)
from tender_review.optimization.public import (
    CreateOptimizationJob,
    OptimizationProvenance,
    OptimizationSample,
)
from tender_review.evaluation.public import (
    AnnotationSampleInput,
    ChunkRelevanceLabel,
    CreateAnnotationDatasetRevision,
    CreateAnnotationDatasetVersion,
    SubmitAnnotationLabel,
    CreateEvaluationRun,
    DatasetSplit,
    EvaluationPurpose,
)


class ApiModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int = Field(default=1, ge=1)


class CreateEvaluationRunRequest(ApiModel):
    dataset_version_id: str = Field(min_length=1, max_length=128)
    purpose: EvaluationPurpose
    dataset_split: DatasetSplit
    model_config_id: str = Field(min_length=1, max_length=128)
    retriever_version: str = Field(min_length=1, max_length=128)
    evaluator_version: str = Field(min_length=1, max_length=128)
    input_sha256: str = Field(min_length=64, max_length=64)
    config_sha256: str = Field(min_length=64, max_length=64)
    code_sha256: str = Field(min_length=64, max_length=64)
    model_sha256: str = Field(min_length=64, max_length=64)
    prompt_sha256: str = Field(min_length=64, max_length=64)
    reproducibility_command: str = Field(min_length=1, max_length=4000)

    def to_command(self, rule_version_id: str) -> CreateEvaluationRun:
        return CreateEvaluationRun(
            rule_version_id=rule_version_id,
            **self.model_dump(mode="python", exclude={"schema_version"}),
        )


class ApiIndexResponse(ApiModel):
    service: str
    version: str
    api_version: Literal["v1"] = "v1"


class LivenessResponse(ApiModel):
    status: Literal["ok"] = "ok"
    service: str
    version: str


class ComponentCheck(ApiModel):
    status: Literal["ready", "not_ready"]
    detail: str = ""


class ReadinessResponse(ApiModel):
    status: Literal["ready", "not_ready"]
    service: str
    version: str
    checks: dict[str, ComponentCheck]


class CreateReviewJobRequest(ApiModel):
    document_snapshot_id: str = Field(min_length=1, max_length=128)
    document_sha256: str = Field(min_length=64, max_length=64)
    rule_version_id: str = Field(min_length=1, max_length=128)
    rule_version_hash: str = Field(min_length=64, max_length=64)
    model_config_id: str = Field(min_length=1, max_length=128)
    model_config_hash: str = Field(min_length=64, max_length=64)
    execution_spec: ReviewExecutionSpecDraft | None = None
    max_attempts: int = Field(default=3, ge=1, le=100)

    @field_validator(
        "document_snapshot_id", "rule_version_id", "model_config_id", mode="before"
    )
    @classmethod
    def normalize_identifiers(cls, value: str) -> str:
        normalized = str(value).strip()
        if not normalized:
            raise ValueError("value must not be blank")
        return normalized

    @field_validator(
        "document_sha256", "rule_version_hash", "model_config_hash", mode="before"
    )
    @classmethod
    def normalize_hashes(cls, value: str) -> str:
        normalized = str(value).strip().lower()
        if len(normalized) != 64 or any(
            character not in "0123456789abcdef" for character in normalized
        ):
            raise ValueError("value must be a SHA-256 hexadecimal digest")
        return normalized

    @model_validator(mode="after")
    def execution_spec_matches_job_identity(self) -> Self:
        if self.execution_spec is None:
            return self
        expected = (
            self.document_snapshot_id,
            self.document_sha256,
            self.rule_version_id,
            self.rule_version_hash,
            self.model_config_id,
            self.model_config_hash,
        )
        supplied = (
            self.execution_spec.document_snapshot_id,
            self.execution_spec.document_sha256,
            self.execution_spec.rule_version_id,
            self.execution_spec.rule_version_hash,
            self.execution_spec.model_config_id,
            self.execution_spec.model_config_hash,
        )
        if supplied != expected:
            raise ValueError(
                "execution_spec identities and hashes must match the review job request"
            )
        return self

    def to_command(self) -> CreateReviewJobCommand:
        return CreateReviewJobCommand(
            schema_version=self.schema_version,
            document_snapshot_id=self.document_snapshot_id,
            document_sha256=self.document_sha256,
            rule_version_id=self.rule_version_id,
            rule_version_hash=self.rule_version_hash,
            model_config_id=self.model_config_id,
            model_config_hash=self.model_config_hash,
            execution_spec=self.execution_spec,
            max_attempts=self.max_attempts,
        )


class ReviewJobResponse(ApiModel):
    id: str
    document_snapshot_id: str
    rule_version_id: str
    model_config_id: str
    input_fingerprint: str
    execution_spec_sha256: str | None = None
    status: JobLifecycle
    stage: ReviewStage | None = None
    rerun_of: str | None = None
    attempt_count: int
    recovery_count: int
    recovery_metric_source: Literal["review_jobs.attempt_count"] = (
        "review_jobs.attempt_count"
    )
    max_attempts: int
    available_at: datetime
    lease_token: int
    failure_stage: ReviewStage | None = None
    failure: JobFailure | None = None
    safe_failure_code: str | None = None
    safe_failure_category: str | None = None
    safe_failure_retryable: bool | None = None
    completed_at: datetime | None = None
    created_at: datetime
    updated_at: datetime

    @classmethod
    def from_job(cls, job: ReviewJob) -> "ReviewJobResponse":
        return cls(
            schema_version=job.schema_version,
            id=job.id,
            document_snapshot_id=job.document_snapshot_id,
            rule_version_id=job.rule_version_id,
            model_config_id=job.model_config_id,
            input_fingerprint=job.input_fingerprint,
            execution_spec_sha256=job.execution_spec_sha256,
            status=job.status,
            stage=job.stage,
            rerun_of=job.rerun_of,
            attempt_count=job.attempt_count,
            recovery_count=max(0, job.attempt_count - 1),
            max_attempts=job.max_attempts,
            available_at=job.available_at,
            lease_token=job.lease_token,
            failure_stage=job.failure_stage,
            failure=job.failure,
            safe_failure_code=job.failure.code if job.failure else None,
            safe_failure_category=(
                job.failure.category.value if job.failure else None
            ),
            safe_failure_retryable=(
                job.failure.retryable if job.failure else None
            ),
            completed_at=job.completed_at,
            created_at=job.created_at,
            updated_at=job.updated_at,
        )


class DocumentSnapshotResponse(ApiModel):
    id: str
    source_system: str
    source_document_id: str
    file_name: str
    sha256: str
    size_bytes: int
    media_type: str
    parse_status: str
    parser_name: str | None = None
    parser_version: str | None = None
    created: bool

    @classmethod
    def from_snapshot(
        cls, snapshot: SnapshotRecord, *, created: bool
    ) -> "DocumentSnapshotResponse":
        return cls(
            id=snapshot.id,
            source_system=snapshot.source_system,
            source_document_id=snapshot.source_document_id,
            file_name=snapshot.file_name,
            sha256=snapshot.object.sha256,
            size_bytes=snapshot.object.size_bytes,
            media_type=snapshot.object.media_type,
            parse_status=snapshot.parse_status,
            parser_name=snapshot.parser_name,
            parser_version=snapshot.parser_version,
            created=created,
        )


class HumanDecisionRequest(ApiModel):
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
        if not value.strip():
            raise ValueError("reason must not be blank")
        return value

    @model_validator(mode="after")
    def revision_matches_decision(self) -> Self:
        if (self.decision is HumanDecisionType.MODIFY) != (self.revision is not None):
            raise ValueError("MODIFY alone requires a revision")
        return self

    def to_command(self, finding_id: str) -> SubmitHumanDecision:
        return SubmitHumanDecision(
            finding_id=finding_id,
            reviewer_kind=self.reviewer_kind,
            reviewer_id=self.reviewer_id,
            decision=self.decision,
            reason=self.reason,
            revision=self.revision,
            supersedes_decision_id=self.supersedes_decision_id,
        )


class CreateRuleVersionRequest(ApiModel):
    rule_key: str = Field(min_length=1, max_length=128)
    rule_set_name: str = Field(min_length=1, max_length=255)
    rule_set_description: str | None = Field(default=None, max_length=8000)
    parent_version_id: str | None = Field(default=None, min_length=1, max_length=128)
    content: dict[str, Any]
    execution_config: dict[str, Any] = Field(default_factory=dict)
    change_summary: str = Field(min_length=1, max_length=8000)
    provenance: RuleProvenance

    def to_command(self, rule_set_id: str) -> CreateRuleVersion:
        return CreateRuleVersion(
            rule_set_id=rule_set_id,
            rule_key=self.rule_key,
            rule_set_name=self.rule_set_name,
            rule_set_description=self.rule_set_description,
            parent_version_id=self.parent_version_id,
            content_json=canonical_json(self.content),
            execution_config_json=canonical_json(self.execution_config),
            change_summary=self.change_summary,
            provenance=self.provenance,
        )


class EvaluateRuleVersionRequest(ApiModel):
    dataset_version_id: str = Field(min_length=1, max_length=128)


class OptimizeRuleVersionRequest(ApiModel):
    dataset_version_id: str = Field(min_length=1, max_length=128)
    max_rounds: int = Field(default=3, ge=1, le=20)
    candidates_per_round: int = Field(default=2, ge=1, le=10)
    required_stability_runs: int = Field(default=2, ge=2, le=20)
    model_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    prompt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    retriever_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    tool_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    samples: tuple[OptimizationSample, ...] = Field(min_length=2)
    provenance: OptimizationProvenance

    def to_command(self, rule_version_id: str) -> CreateOptimizationJob:
        return CreateOptimizationJob(
            base_rule_version_id=rule_version_id,
            dataset_version_id=self.dataset_version_id,
            max_rounds=self.max_rounds,
            candidates_per_round=self.candidates_per_round,
            required_stability_runs=self.required_stability_runs,
            model_sha256=self.model_sha256,
            prompt_sha256=self.prompt_sha256,
            retriever_sha256=self.retriever_sha256,
            tool_sha256=self.tool_sha256,
            samples=self.samples,
            provenance=self.provenance,
        )


class A5OptimizeRuleVersionRequest(OptimizeRuleVersionRequest):
    a4_evaluation_run_id: str = Field(min_length=1, max_length=128)
    a4_report_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    def to_command(self, rule_version_id: str) -> CreateOptimizationJob:
        return super().to_command(rule_version_id).model_copy(
            update={
                "a4_evaluation_run_id": self.a4_evaluation_run_id,
                "a4_report_sha256": self.a4_report_sha256,
            }
        )


class CreateAnnotationDatasetRequest(ApiModel):
    dataset_name: str = Field(min_length=1, max_length=255)
    parent_version_id: str | None = Field(default=None, min_length=1, max_length=128)
    change_summary: str = Field(min_length=1, max_length=8000)
    source_description: str = Field(min_length=1, max_length=8000)
    source_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_work_package_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    samples: tuple[AnnotationSampleInput, ...] = Field(min_length=1)

    def to_command(self) -> CreateAnnotationDatasetVersion:
        return CreateAnnotationDatasetVersion(
            dataset_name=self.dataset_name,
            parent_version_id=self.parent_version_id,
            change_summary=self.change_summary,
            source_description=self.source_description,
            source_manifest_sha256=self.source_manifest_sha256,
            source_work_package_sha256=self.source_work_package_sha256,
            samples=self.samples,
        )


class SubmitAnnotationLabelRequest(ApiModel):
    actor_id: str = Field(min_length=1, max_length=255)
    human_decision_id: str = Field(min_length=1, max_length=128)
    label: ChunkRelevanceLabel

    @field_validator("actor_id")
    @classmethod
    def actor_is_named_human(cls, value: str) -> str:
        return _named_human(value)

    def to_command(
        self, dataset_version_id: str, sample_id: str
    ) -> SubmitAnnotationLabel:
        return SubmitAnnotationLabel(
            dataset_version_id=dataset_version_id,
            sample_id=sample_id,
            actor_id=self.actor_id,
            human_decision_id=self.human_decision_id,
            label=self.label,
        )


class CreateAnnotationDatasetRevisionRequest(ApiModel):
    change_summary: str = Field(min_length=1, max_length=8000)
    reset_sample_ids: tuple[str, ...] = Field(min_length=1)

    def to_command(self, parent_version_id: str) -> CreateAnnotationDatasetRevision:
        return CreateAnnotationDatasetRevision(
            parent_version_id=parent_version_id,
            change_summary=self.change_summary,
            reset_sample_ids=self.reset_sample_ids,
        )


class PublishRuleVersionRequest(ApiModel):
    approver_kind: Literal["human"] = "human"
    approver_id: str = Field(min_length=1, max_length=255)

    @field_validator("approver_id")
    @classmethod
    def approver_is_named_human(cls, value: str) -> str:
        return _named_human(value)

    def to_command(self, rule_version_id: str) -> PublishRuleVersion:
        return PublishRuleVersion(
            rule_version_id=rule_version_id,
            approver_kind=self.approver_kind,
            approver_id=self.approver_id,
        )


class RollbackRuleSetRequest(ApiModel):
    target_version_id: str = Field(min_length=1, max_length=128)
    approver_kind: Literal["human"] = "human"
    approver_id: str = Field(min_length=1, max_length=255)
    reason: str = Field(min_length=1, max_length=8000)

    @field_validator("approver_id")
    @classmethod
    def approver_is_named_human(cls, value: str) -> str:
        return _named_human(value)

    def to_command(self, rule_set_id: str) -> RollbackRuleSet:
        return RollbackRuleSet(
            rule_set_id=rule_set_id,
            target_version_id=self.target_version_id,
            approver_kind=self.approver_kind,
            approver_id=self.approver_id,
            reason=self.reason,
        )


def _named_human(value: str) -> str:
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


class ErrorDetail(BaseModel):
    model_config = ConfigDict(extra="allow")

    location: tuple[str | int, ...] = ()
    message: str
    type: str = ""


class ApiError(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: str
    message: str
    category: str
    retryable: bool
    request_id: str
    details: dict[str, Any] = Field(default_factory=dict)
    validation_errors: tuple[ErrorDetail, ...] = ()


class ErrorResponse(ApiModel):
    error: ApiError
