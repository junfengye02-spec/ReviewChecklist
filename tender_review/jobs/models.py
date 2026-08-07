from __future__ import annotations

import hashlib
import json
from datetime import datetime
from enum import Enum
from typing import Any, Literal, Self

from pydantic import AliasChoices, Field, field_validator, model_validator
from pydantic_core import to_jsonable_python

from tender_review.shared.contracts import ContractModel
from tender_review.shared.errors import ErrorCategory


class JobLifecycle(str, Enum):
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    WAITING_HUMAN = "WAITING_HUMAN"
    COMPLETED = "COMPLETED"
    RETRY_WAIT = "RETRY_WAIT"
    FAILED = "FAILED"
    DEAD = "DEAD"
    CANCELLED = "CANCELLED"


JobStatus = JobLifecycle


class ReviewStage(str, Enum):
    PARSING = "PARSING"
    INDEXING = "INDEXING"
    RETRIEVING = "RETRIEVING"
    EXTRACTING = "EXTRACTING"
    COMPARING = "COMPARING"
    VERIFYING = "VERIFYING"
    REPORTING = "REPORTING"


def _normalize_identifier(value: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError("value must not be blank")
    return normalized


def _normalize_sha256(value: str) -> str:
    normalized = value.strip().lower()
    if len(normalized) != 64 or any(character not in "0123456789abcdef" for character in normalized):
        raise ValueError("value must be a lowercase SHA-256 hexadecimal digest")
    return normalized


def _canonical_json(value: Any) -> str:
    return json.dumps(
        to_jsonable_python(value),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


class ExecutionArtifactReference(ContractModel):
    """Bounded pointer to immutable object storage content."""

    schema_version: Literal[1] = 1
    artifact_id: str = Field(min_length=1, max_length=36)
    bucket: str = Field(min_length=1, max_length=255)
    object_key: str = Field(min_length=1, max_length=512)
    sha256: str = Field(min_length=64, max_length=64)

    @field_validator("artifact_id", "bucket", "object_key", mode="before")
    @classmethod
    def normalize_references(cls, value: str) -> str:
        return _normalize_identifier(str(value))

    @field_validator("sha256", mode="before")
    @classmethod
    def normalize_artifact_hash(cls, value: str) -> str:
        return _normalize_sha256(str(value))


class ReviewExecutionSpecDraft(ContractModel):
    """Versioned execution inputs supplied before a job ID has been allocated."""

    schema_version: Literal[1] = 1
    document_snapshot_id: str = Field(min_length=1, max_length=36)
    document_sha256: str = Field(min_length=64, max_length=64)
    rule_version_id: str = Field(min_length=1, max_length=36)
    rule_version_hash: str = Field(min_length=64, max_length=64)
    dataset_version_id: str = Field(min_length=1, max_length=36)
    dataset_version_hash: str = Field(min_length=64, max_length=64)
    model_config_id: str = Field(min_length=1, max_length=36)
    model_config_hash: str = Field(min_length=64, max_length=64)
    query: str = Field(min_length=1, max_length=4000)
    retrieval_variant: str = Field(min_length=1, max_length=128)
    retriever_artifact: ExecutionArtifactReference
    index_artifact: ExecutionArtifactReference
    chunk_artifact: ExecutionArtifactReference

    @field_validator(
        "document_snapshot_id",
        "rule_version_id",
        "dataset_version_id",
        "model_config_id",
        "query",
        "retrieval_variant",
        mode="before",
    )
    @classmethod
    def normalize_execution_values(cls, value: str) -> str:
        return _normalize_identifier(str(value))

    @field_validator(
        "document_sha256",
        "rule_version_hash",
        "dataset_version_hash",
        "model_config_hash",
        mode="before",
    )
    @classmethod
    def normalize_execution_hashes(cls, value: str) -> str:
        return _normalize_sha256(str(value))

    @model_validator(mode="after")
    def artifact_roles_are_distinct(self) -> Self:
        artifact_ids = {
            self.retriever_artifact.artifact_id,
            self.index_artifact.artifact_id,
            self.chunk_artifact.artifact_id,
        }
        if len(artifact_ids) != 3:
            raise ValueError("execution artifact references must be distinct")
        return self


class ReviewExecutionSpec(ReviewExecutionSpecDraft):
    """Immutable, self-verifying inputs required to rebuild one review execution."""

    job_id: str = Field(min_length=1, max_length=36)
    input_sha256: str = Field(min_length=64, max_length=64)

    @field_validator("job_id", mode="before")
    @classmethod
    def normalize_job_id(cls, value: str) -> str:
        return _normalize_identifier(str(value))

    @field_validator("input_sha256", mode="before")
    @classmethod
    def normalize_input_hash(cls, value: str) -> str:
        return _normalize_sha256(str(value))

    @model_validator(mode="after")
    def input_hash_matches_payload(self) -> Self:
        expected = execution_spec_sha256(
            self.model_dump(mode="json", exclude={"input_sha256"})
        )
        if self.input_sha256 != expected:
            raise ValueError("input_sha256 does not match review execution spec")
        return self


def execution_spec_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def build_review_execution_spec(
    job_id: str, draft: ReviewExecutionSpecDraft
) -> ReviewExecutionSpec:
    payload = {"job_id": _normalize_identifier(job_id), **draft.model_dump(mode="json")}
    return ReviewExecutionSpec(
        **payload,
        input_sha256=execution_spec_sha256(payload),
    )


def clone_review_execution_spec(
    spec: ReviewExecutionSpec, job_id: str
) -> ReviewExecutionSpec:
    draft = ReviewExecutionSpecDraft.model_validate(
        spec.model_dump(mode="json", exclude={"job_id", "input_sha256"})
    )
    return build_review_execution_spec(job_id, draft)


class JobMessage(ContractModel):
    """Legacy worker queue message retained for the Stage 1 worker contract."""

    job_id: str = Field(min_length=1, max_length=128)
    job_type: str = Field(min_length=1, max_length=128)
    input_reference: str = Field(min_length=1, max_length=1024)
    attempt: int = Field(default=0, ge=0)
    enqueued_at: datetime | None = None


class JobLease(ContractModel):
    job_id: str
    owner: str
    token: int = Field(ge=1)
    expires_at: datetime


class JobResult(ContractModel):
    output_reference: str | None = None
    summary: str = ""


class JobHandlerStatus(str, Enum):
    COMPLETED = "COMPLETED"
    WAITING_HUMAN = "WAITING_HUMAN"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class JobFailure(ContractModel):
    code: str = Field(min_length=1, max_length=128)
    message: str = Field(min_length=1)
    category: ErrorCategory
    retryable: bool
    stage: ReviewStage | None = None


class JobHandlerOutcome(ContractModel):
    """Explicit handler result for durable states beyond simple completion."""

    status: JobHandlerStatus
    result: JobResult = Field(default_factory=JobResult)
    failure: JobFailure | None = None

    @model_validator(mode="after")
    def status_and_failure_match(self) -> Self:
        needs_failure = self.status in {
            JobHandlerStatus.FAILED,
            JobHandlerStatus.CANCELLED,
        }
        if needs_failure != (self.failure is not None):
            raise ValueError("failed and cancelled outcomes require exactly one failure")
        if (
            self.status is JobHandlerStatus.CANCELLED
            and self.failure is not None
            and self.failure.category is not ErrorCategory.CANCELLED
        ):
            raise ValueError("cancelled outcomes require a cancelled failure")
        return self


class CreateReviewJobCommand(ContractModel):
    """Immutable, normalized input for creating a review job."""

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
        return _normalize_identifier(str(value))

    @field_validator(
        "document_sha256", "rule_version_hash", "model_config_hash", mode="before"
    )
    @classmethod
    def normalize_hashes(cls, value: str) -> str:
        return _normalize_sha256(str(value))

    @model_validator(mode="after")
    def execution_spec_matches_legacy_identity(self) -> Self:
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


class ReviewJob(ContractModel):
    """Durable review-job DTO. Lifecycle and stage intentionally remain separate."""

    id: str = Field(min_length=1, max_length=128)
    document_snapshot_id: str = Field(min_length=1, max_length=128)
    rule_version_id: str = Field(min_length=1, max_length=128)
    model_config_id: str = Field(min_length=1, max_length=128)
    input_fingerprint: str = Field(min_length=64, max_length=64)
    execution_spec_sha256: str | None = Field(
        default=None, min_length=64, max_length=64
    )
    status: JobLifecycle = JobLifecycle.QUEUED
    stage: ReviewStage | None = None
    rerun_of: str | None = Field(
        default=None,
        min_length=1,
        max_length=128,
        validation_alias=AliasChoices("rerun_of", "rerun_of_id"),
    )
    attempt_count: int = Field(default=0, ge=0)
    max_attempts: int = Field(default=3, ge=1, le=100)
    available_at: datetime
    lease_owner: str | None = Field(default=None, max_length=255)
    lease_until: datetime | None = None
    heartbeat_at: datetime | None = None
    lease_token: int = Field(default=0, ge=0)
    failure_stage: ReviewStage | None = None
    failure: JobFailure | None = None
    completed_at: datetime | None = None
    created_at: datetime
    updated_at: datetime

    @field_validator(
        "id", "document_snapshot_id", "rule_version_id", "model_config_id", mode="before"
    )
    @classmethod
    def normalize_job_identifiers(cls, value: str) -> str:
        return _normalize_identifier(str(value))

    @field_validator("input_fingerprint", mode="before")
    @classmethod
    def normalize_fingerprint(cls, value: str) -> str:
        return _normalize_sha256(str(value))

    @field_validator("execution_spec_sha256", mode="before")
    @classmethod
    def normalize_execution_spec_hash(cls, value: str | None) -> str | None:
        return None if value is None else _normalize_sha256(str(value))

    @property
    def rerun_of_id(self) -> str | None:
        """Persistence-name compatibility without exposing it as a public field."""

        return self.rerun_of


class CheckpointValue(ContractModel):
    key: str = Field(min_length=1, max_length=128)
    value: str = Field(max_length=10000)


class CheckpointState(ContractModel):
    """Structured checkpoint state that avoids leaking unbounded JSON dictionaries."""

    values: tuple[CheckpointValue, ...] = ()


class JobCheckpoint(ContractModel):
    job_id: str = Field(min_length=1, max_length=128)
    node_name: str = Field(min_length=1, max_length=128)
    stage: ReviewStage
    lease_token: int = Field(ge=0)
    sequence: int = Field(default=0, ge=0)
    state: CheckpointState
    output_artifact_id: str | None = Field(default=None, min_length=1, max_length=128)
    completed_at: datetime


class IdempotencyRecord(ContractModel):
    id: str = Field(min_length=1, max_length=128)
    caller_id: str = Field(min_length=1, max_length=255)
    scope: str = Field(min_length=1, max_length=255)
    idempotency_key: str = Field(min_length=1, max_length=255)
    request_hash: str = Field(min_length=64, max_length=64)
    resource_type: str = Field(default="review_job", min_length=1, max_length=64)
    resource_id: str = Field(min_length=1, max_length=128)
    created_at: datetime
    expires_at: datetime | None = None

    @field_validator("request_hash", mode="before")
    @classmethod
    def normalize_request_hash(cls, value: str) -> str:
        return _normalize_sha256(str(value))


class IdempotentReviewJob(ContractModel):
    job: ReviewJob
    created: bool
