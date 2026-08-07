from __future__ import annotations

import uuid
from datetime import datetime
from decimal import Decimal
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base


SCHEMA_VERSION = "1"


def _new_id() -> str:
    return str(uuid.uuid4())


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )


class SchemaVersionMixin:
    schema_version: Mapped[str] = mapped_column(
        String(16),
        nullable=False,
        default=SCHEMA_VERSION,
        server_default=SCHEMA_VERSION,
    )


class DocumentSnapshot(TimestampMixin, SchemaVersionMixin, Base):
    __tablename__ = "document_snapshots"
    __table_args__ = (
        UniqueConstraint("sha256"),
        UniqueConstraint("source_system", "source_document_id"),
        CheckConstraint("size_bytes >= 0", name="size_nonnegative"),
        CheckConstraint(
            "parse_status IN ('UPLOADED', 'PARSING', 'PARSED', 'FAILED')",
            name="parse_status",
        ),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_new_id)
    sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    object_key: Mapped[str] = mapped_column(String(512), nullable=False)
    source_system: Mapped[str] = mapped_column(String(64), nullable=False)
    source_document_id: Mapped[str] = mapped_column(String(255), nullable=False)
    file_name: Mapped[str] = mapped_column(String(512), nullable=False)
    media_type: Mapped[str] = mapped_column(
        String(255), nullable=False, server_default="application/pdf"
    )
    size_bytes: Mapped[int] = mapped_column(Integer, nullable=False)
    parse_status: Mapped[str] = mapped_column(
        String(32), nullable=False, server_default="UPLOADED"
    )
    parser_name: Mapped[str | None] = mapped_column(String(128))
    parser_version: Mapped[str | None] = mapped_column(String(128))


class DocumentArtifact(TimestampMixin, SchemaVersionMixin, Base):
    __tablename__ = "document_artifacts"
    __table_args__ = (
        UniqueConstraint("document_snapshot_id", "artifact_type", "object_key"),
        CheckConstraint("size_bytes >= 0", name="size_nonnegative"),
        Index("ix_document_artifacts_object", "bucket", "object_key"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_new_id)
    document_snapshot_id: Mapped[str] = mapped_column(
        ForeignKey("document_snapshots.id", ondelete="CASCADE"), nullable=False
    )
    artifact_type: Mapped[str] = mapped_column(String(64), nullable=False)
    bucket: Mapped[str] = mapped_column(String(255), nullable=False)
    object_key: Mapped[str] = mapped_column(String(512), nullable=False)
    sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    size_bytes: Mapped[int] = mapped_column(Integer, nullable=False)
    media_type: Mapped[str] = mapped_column(String(255), nullable=False)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(
        JSON, nullable=False, default=dict
    )


class ModelConfig(TimestampMixin, SchemaVersionMixin, Base):
    __tablename__ = "model_configs"
    __table_args__ = (UniqueConstraint("config_hash"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_new_id)
    provider: Mapped[str] = mapped_column(String(128), nullable=False)
    model_name: Mapped[str] = mapped_column(String(255), nullable=False)
    prompt_version: Mapped[str] = mapped_column(String(128), nullable=False)
    config_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    parameters_json: Mapped[dict[str, Any]] = mapped_column(
        JSON, nullable=False, default=dict
    )


class RuleSet(TimestampMixin, SchemaVersionMixin, Base):
    __tablename__ = "rule_sets"
    __table_args__ = (UniqueConstraint("rule_key"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_new_id)
    rule_key: Mapped[str] = mapped_column(String(128), nullable=False)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str | None] = mapped_column(Text)
    current_version_id: Mapped[str | None] = mapped_column(
        ForeignKey(
            "rule_versions.id",
            name="fk_rule_sets_current_version_id_rule_versions",
            use_alter=True,
            ondelete="SET NULL",
        )
    )


class RuleVersion(TimestampMixin, SchemaVersionMixin, Base):
    __tablename__ = "rule_versions"
    __table_args__ = (
        UniqueConstraint(
            "rule_set_id", "version_number", name="uq_rule_versions_set_version"
        ),
        UniqueConstraint(
            "rule_set_id", "content_hash", name="uq_rule_versions_set_content_hash"
        ),
        CheckConstraint("version_number > 0", name="version_positive"),
        CheckConstraint(
            "status IN ('DRAFT', 'OPTIMIZING', 'EVALUATING', "
            "'WAITING_APPROVAL', 'PUBLISHED', 'REJECTED', 'ROLLED_BACK')",
            name="status",
        ),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_new_id)
    rule_set_id: Mapped[str] = mapped_column(
        ForeignKey("rule_sets.id", ondelete="CASCADE"), nullable=False
    )
    parent_version_id: Mapped[str | None] = mapped_column(
        ForeignKey("rule_versions.id", ondelete="SET NULL")
    )
    version_number: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[str] = mapped_column(
        String(32), nullable=False, server_default="DRAFT"
    )
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    content_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    execution_config_json: Mapped[dict[str, Any]] = mapped_column(
        JSON, nullable=False, default=dict
    )
    change_summary: Mapped[str | None] = mapped_column(Text)
    provenance_json: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    evaluation_gate_json: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    published_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    published_by: Mapped[str | None] = mapped_column(String(255))


class ReviewJob(TimestampMixin, SchemaVersionMixin, Base):
    __tablename__ = "review_jobs"
    __table_args__ = (
        CheckConstraint(
            "status IN ('QUEUED', 'RUNNING', 'WAITING_HUMAN', 'COMPLETED', "
            "'RETRY_WAIT', 'FAILED', 'DEAD', 'CANCELLED')",
            name="status",
        ),
        CheckConstraint(
            "stage IS NULL OR stage IN ('PARSING', 'INDEXING', 'RETRIEVING', "
            "'EXTRACTING', 'COMPARING', 'VERIFYING', 'REPORTING')",
            name="stage",
        ),
        CheckConstraint(
            "attempt_count >= 0 AND max_attempts > 0 AND attempt_count <= max_attempts",
            name="attempt_counts",
        ),
        CheckConstraint("lease_token >= 0", name="lease_token_nonnegative"),
        Index("ix_review_jobs_queue", "status", "available_at", "created_at"),
        Index(
            "ix_review_jobs_queue_created",
            "status",
            "available_at",
            "created_at",
            "id",
        ),
        Index("ix_review_jobs_input_fingerprint", "input_fingerprint"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_new_id)
    document_snapshot_id: Mapped[str] = mapped_column(
        ForeignKey("document_snapshots.id", ondelete="RESTRICT"), nullable=False
    )
    rule_version_id: Mapped[str] = mapped_column(
        ForeignKey("rule_versions.id", ondelete="RESTRICT"), nullable=False
    )
    model_config_id: Mapped[str] = mapped_column(
        ForeignKey("model_configs.id", ondelete="RESTRICT"), nullable=False
    )
    rerun_of_id: Mapped[str | None] = mapped_column(
        ForeignKey("review_jobs.id", ondelete="SET NULL")
    )
    status: Mapped[str] = mapped_column(
        String(32), nullable=False, server_default="QUEUED"
    )
    job_type: Mapped[str] = mapped_column(
        String(128), nullable=False, server_default="review"
    )
    input_reference: Mapped[str] = mapped_column(
        String(1024), nullable=False, server_default=""
    )
    stage: Mapped[str | None] = mapped_column(String(32))
    input_fingerprint: Mapped[str] = mapped_column(String(64), nullable=False)
    execution_spec_sha256: Mapped[str | None] = mapped_column(String(64))
    attempt_count: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default="0"
    )
    max_attempts: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default="3"
    )
    available_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    lease_owner: Mapped[str | None] = mapped_column(String(255))
    lease_until: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    heartbeat_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    lease_token: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default="0"
    )
    checkpoint_sequence: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default="0"
    )
    failure_stage: Mapped[str | None] = mapped_column(String(32))
    error_code: Mapped[str | None] = mapped_column(String(128))
    error_category: Mapped[str | None] = mapped_column(String(64))
    error_message: Mapped[str | None] = mapped_column(Text)
    error_retryable: Mapped[bool | None] = mapped_column(Boolean)
    output_reference: Mapped[str | None] = mapped_column(String(1024))
    output_summary: Mapped[str | None] = mapped_column(Text)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class ReviewExecutionSpec(TimestampMixin, SchemaVersionMixin, Base):
    __tablename__ = "review_execution_specs"
    __table_args__ = (
        Index("ix_review_execution_specs_input_sha256", "input_sha256"),
    )

    job_id: Mapped[str] = mapped_column(
        ForeignKey("review_jobs.id", ondelete="CASCADE"), primary_key=True
    )
    input_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    spec_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    document_snapshot_id: Mapped[str] = mapped_column(
        ForeignKey("document_snapshots.id", ondelete="RESTRICT"), nullable=False
    )
    rule_version_id: Mapped[str] = mapped_column(
        ForeignKey("rule_versions.id", ondelete="RESTRICT"), nullable=False
    )
    dataset_version_id: Mapped[str] = mapped_column(
        ForeignKey("dataset_versions.id", ondelete="RESTRICT"), nullable=False
    )
    model_config_id: Mapped[str] = mapped_column(
        ForeignKey("model_configs.id", ondelete="RESTRICT"), nullable=False
    )
    retriever_artifact_id: Mapped[str] = mapped_column(
        ForeignKey("document_artifacts.id", ondelete="RESTRICT"), nullable=False
    )
    index_artifact_id: Mapped[str] = mapped_column(
        ForeignKey("document_artifacts.id", ondelete="RESTRICT"), nullable=False
    )
    chunk_artifact_id: Mapped[str] = mapped_column(
        ForeignKey("document_artifacts.id", ondelete="RESTRICT"), nullable=False
    )


class JobCheckpoint(TimestampMixin, SchemaVersionMixin, Base):
    __tablename__ = "job_checkpoints"
    __table_args__ = (
        UniqueConstraint(
            "review_job_id", "node_name", name="uq_job_checkpoints_job_node"
        ),
        CheckConstraint("lease_token >= 0", name="lease_token_nonnegative"),
        Index("ix_job_checkpoints_job_sequence", "review_job_id", "sequence"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_new_id)
    review_job_id: Mapped[str] = mapped_column(
        ForeignKey("review_jobs.id", ondelete="CASCADE"), nullable=False
    )
    node_name: Mapped[str] = mapped_column(String(128), nullable=False)
    stage: Mapped[str] = mapped_column(String(32), nullable=False)
    lease_token: Mapped[int] = mapped_column(Integer, nullable=False)
    sequence: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default="0"
    )
    output_artifact_id: Mapped[str | None] = mapped_column(
        ForeignKey("document_artifacts.id", ondelete="SET NULL")
    )
    state_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    completed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class IdempotencyRecord(TimestampMixin, SchemaVersionMixin, Base):
    __tablename__ = "idempotency_records"
    __table_args__ = (
        UniqueConstraint(
            "caller_id",
            "scope",
            "idempotency_key",
            name="uq_idempotency_caller_scope_key",
        ),
        Index("ix_idempotency_records_resource", "resource_type", "resource_id"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_new_id)
    caller_id: Mapped[str] = mapped_column(String(255), nullable=False)
    scope: Mapped[str] = mapped_column(String(255), nullable=False)
    idempotency_key: Mapped[str] = mapped_column(String(255), nullable=False)
    request_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    resource_type: Mapped[str] = mapped_column(String(64), nullable=False)
    resource_id: Mapped[str] = mapped_column(String(36), nullable=False)
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class Finding(TimestampMixin, SchemaVersionMixin, Base):
    __tablename__ = "findings"
    __table_args__ = (
        CheckConstraint(
            "confidence IS NULL OR (confidence >= 0 AND confidence <= 1)",
            name="confidence",
        ),
        CheckConstraint(
            "status IN ('OPEN', 'NEED_MORE_EVIDENCE', 'WAITING_HUMAN', "
            "'PENDING_DECISION', 'WORK_ITEM_OPEN', 'APPROVED', 'REJECTED', "
            "'MODIFIED', 'INSUFFICIENT_EVIDENCE')",
            name="status",
        ),
        Index("ix_findings_job_item", "review_job_id", "review_item"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_new_id)
    review_job_id: Mapped[str] = mapped_column(
        ForeignKey("review_jobs.id", ondelete="CASCADE"), nullable=False
    )
    rule_version_id: Mapped[str] = mapped_column(
        ForeignKey("rule_versions.id", ondelete="RESTRICT"), nullable=False
    )
    review_item: Mapped[str] = mapped_column(String(128), nullable=False)
    status: Mapped[str] = mapped_column(
        String(32), nullable=False, server_default="OPEN"
    )
    compliant: Mapped[bool | None] = mapped_column(Boolean)
    severity: Mapped[str | None] = mapped_column(String(32))
    title: Mapped[str] = mapped_column(String(512), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    confidence: Mapped[Decimal | None] = mapped_column(Numeric(5, 4))
    explanation_json: Mapped[dict[str, Any]] = mapped_column(
        JSON, nullable=False, default=dict
    )
    workflow_state: Mapped[str | None] = mapped_column(String(32))
    review_input_sha256: Mapped[str | None] = mapped_column(String(64))
    finding_content_sha256: Mapped[str | None] = mapped_column(String(64), unique=True)
    provenance_json: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    documents_json: Mapped[list[dict[str, Any]] | None] = mapped_column(JSON)


class EvidenceReference(TimestampMixin, SchemaVersionMixin, Base):
    __tablename__ = "evidence_references"
    __table_args__ = (
        CheckConstraint("page_start > 0", name="page_start_positive"),
        CheckConstraint("page_end >= page_start", name="page_range"),
        Index(
            "ix_evidence_references_document_chunk", "document_snapshot_id", "chunk_id"
        ),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_new_id)
    finding_id: Mapped[str] = mapped_column(
        ForeignKey("findings.id", ondelete="CASCADE"), nullable=False
    )
    document_snapshot_id: Mapped[str] = mapped_column(
        ForeignKey("document_snapshots.id", ondelete="RESTRICT"), nullable=False
    )
    chunk_id: Mapped[str] = mapped_column(String(128), nullable=False)
    page_start: Mapped[int] = mapped_column(Integer, nullable=False)
    page_end: Mapped[int] = mapped_column(Integer, nullable=False)
    section_path: Mapped[str | None] = mapped_column(String(1024))
    excerpt: Mapped[str] = mapped_column(Text, nullable=False)
    bbox_json: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    text_sha256: Mapped[str] = mapped_column(String(64), nullable=False)


class HumanDecision(SchemaVersionMixin, Base):
    __tablename__ = "human_decisions"
    __table_args__ = (
        CheckConstraint(
            "decision IN ('APPROVE', 'REJECT', 'MODIFY', 'INSUFFICIENT_EVIDENCE')",
            name="decision",
        ),
        Index("ix_human_decisions_finding_created", "finding_id", "created_at"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_new_id)
    finding_id: Mapped[str] = mapped_column(
        ForeignKey("findings.id", ondelete="CASCADE"), nullable=False
    )
    supersedes_decision_id: Mapped[str | None] = mapped_column(
        ForeignKey("human_decisions.id", ondelete="SET NULL")
    )
    reviewer_id: Mapped[str] = mapped_column(String(255), nullable=False)
    reviewer_kind: Mapped[str] = mapped_column(
        String(32), nullable=False, server_default="human"
    )
    decision: Mapped[str] = mapped_column(String(32), nullable=False)
    comment: Mapped[str | None] = mapped_column(Text)
    modified_finding_json: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    review_input_sha256: Mapped[str | None] = mapped_column(String(64))
    finding_content_sha256: Mapped[str | None] = mapped_column(String(64))
    evidence_sha256: Mapped[str | None] = mapped_column(String(64))
    decision_sha256: Mapped[str | None] = mapped_column(String(64), unique=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class DatasetVersion(TimestampMixin, SchemaVersionMixin, Base):
    __tablename__ = "dataset_versions"
    __table_args__ = (
        UniqueConstraint(
            "dataset_name", "version_number", name="uq_dataset_versions_name_version"
        ),
        UniqueConstraint("manifest_hash"),
        CheckConstraint("version_number > 0", name="version_positive"),
        CheckConstraint(
            "source_type IN ('REAL', 'SYNTHETIC', 'MIXED')", name="source_type"
        ),
        CheckConstraint(
            "status IN ('DRAFT', 'PROVISIONAL', 'VERIFIED', 'FROZEN')",
            name="status",
        ),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_new_id)
    dataset_name: Mapped[str] = mapped_column(String(255), nullable=False)
    version_number: Mapped[int] = mapped_column(Integer, nullable=False)
    manifest_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    source_type: Mapped[str] = mapped_column(String(32), nullable=False)
    parent_version_id: Mapped[str | None] = mapped_column(
        ForeignKey("dataset_versions.id", ondelete="SET NULL")
    )
    status: Mapped[str] = mapped_column(
        String(32), nullable=False, server_default="DRAFT"
    )
    change_summary: Mapped[str | None] = mapped_column(Text)
    provenance_json: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    manifest_json: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    split_strategy_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    manifest_artifact_id: Mapped[str | None] = mapped_column(
        ForeignKey("document_artifacts.id", ondelete="RESTRICT"), nullable=True
    )
    frozen_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class EvaluationCase(TimestampMixin, SchemaVersionMixin, Base):
    __tablename__ = "evaluation_cases"
    __table_args__ = (
        UniqueConstraint(
            "dataset_version_id", "case_key", name="uq_evaluation_cases_dataset_case"
        ),
        CheckConstraint(
            "split IN ('OPTIMIZATION', 'VALIDATION', 'FROZEN_TEST')", name="split"
        ),
        CheckConstraint(
            "source_type IN ('REAL', 'SYNTHETIC', 'EXTERNAL_PLATFORM')",
            name="source_type",
        ),
        Index("ix_evaluation_cases_dataset_split", "dataset_version_id", "split"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_new_id)
    dataset_version_id: Mapped[str] = mapped_column(
        ForeignKey("dataset_versions.id", ondelete="CASCADE"), nullable=False
    )
    document_snapshot_id: Mapped[str] = mapped_column(
        ForeignKey("document_snapshots.id", ondelete="RESTRICT"), nullable=False
    )
    case_key: Mapped[str] = mapped_column(String(255), nullable=False)
    review_item: Mapped[str] = mapped_column(String(128), nullable=False)
    split: Mapped[str] = mapped_column(String(32), nullable=False)
    source_type: Mapped[str] = mapped_column(String(32), nullable=False)
    expected_compliant: Mapped[bool | None] = mapped_column(Boolean)
    expected_finding_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    expected_evidence_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    finding_id: Mapped[str | None] = mapped_column(
        ForeignKey("findings.id", ondelete="RESTRICT")
    )
    human_decision_id: Mapped[str | None] = mapped_column(
        ForeignKey("human_decisions.id", ondelete="RESTRICT")
    )
    document_sha256: Mapped[str | None] = mapped_column(String(64))
    label_version: Mapped[str | None] = mapped_column(String(128))
    label_status: Mapped[str | None] = mapped_column(String(32))
    review_input_sha256: Mapped[str | None] = mapped_column(String(64))
    evidence_sha256: Mapped[str | None] = mapped_column(String(64))
    sample_sha256: Mapped[str | None] = mapped_column(String(64))


class DatasetAnnotationSampleRecord(TimestampMixin, SchemaVersionMixin, Base):
    __tablename__ = "dataset_annotation_samples"
    __table_args__ = (
        UniqueConstraint(
            "dataset_version_id",
            "sample_key",
            name="uq_dataset_annotation_samples_dataset_sample",
        ),
        CheckConstraint(
            "split IN ('OPTIMIZATION', 'VALIDATION', 'FROZEN_TEST')",
            name="split",
        ),
        CheckConstraint(
            "status IN ('PENDING_ANNOTATION', 'PENDING_REVIEW', 'CONFLICT', "
            "'VERIFIED', 'FROZEN')",
            name="status",
        ),
        Index(
            "ix_dataset_annotation_samples_dataset_status",
            "dataset_version_id",
            "status",
        ),
        Index(
            "ix_dataset_annotation_samples_document_split",
            "document_sha256",
            "split",
        ),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_new_id)
    dataset_version_id: Mapped[str] = mapped_column(
        ForeignKey("dataset_versions.id", ondelete="CASCADE"), nullable=False
    )
    sample_key: Mapped[str] = mapped_column(String(128), nullable=False)
    finding_id: Mapped[str] = mapped_column(
        ForeignKey("findings.id", ondelete="RESTRICT"), nullable=False
    )
    document_snapshot_id: Mapped[str] = mapped_column(
        ForeignKey("document_snapshots.id", ondelete="RESTRICT"), nullable=False
    )
    source_pdf_reference: Mapped[str] = mapped_column(String(2048), nullable=False)
    document_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    source_case_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    rule_version_id: Mapped[str] = mapped_column(
        ForeignKey("rule_versions.id", ondelete="RESTRICT"), nullable=False
    )
    rule_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    query_id: Mapped[str] = mapped_column(String(128), nullable=False)
    question_label: Mapped[str] = mapped_column(String(512), nullable=False)
    split: Mapped[str] = mapped_column(String(32), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    annotation_human_decision_id: Mapped[str | None] = mapped_column(
        ForeignKey("human_decisions.id", ondelete="RESTRICT")
    )
    review_human_decision_id: Mapped[str | None] = mapped_column(
        ForeignKey("human_decisions.id", ondelete="RESTRICT")
    )
    adjudication_human_decision_id: Mapped[str | None] = mapped_column(
        ForeignKey("human_decisions.id", ondelete="RESTRICT")
    )
    annotator_id: Mapped[str | None] = mapped_column(String(255))
    annotated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    reviewer_id: Mapped[str | None] = mapped_column(String(255))
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    adjudicator_id: Mapped[str | None] = mapped_column(String(255))
    adjudicated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    evidence_catalog_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    final_label_sha256: Mapped[str | None] = mapped_column(String(64))
    label_version: Mapped[str | None] = mapped_column(String(128))
    sample_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    sample_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)


class EvaluationRun(TimestampMixin, SchemaVersionMixin, Base):
    __tablename__ = "evaluation_runs"
    __table_args__ = (
        CheckConstraint(
            "status IN ('NOT_READY', 'PENDING', 'RUNNING', 'COMPLETED', 'FAILED')",
            name="status",
        ),
        CheckConstraint(
            "purpose IN ('REAL_BASELINE', 'RELEASE_GATE', 'CANDIDATE_DIAGNOSTIC')",
            name="purpose",
        ),
        CheckConstraint(
            "source_type IN ('real', 'provisional', 'synthetic', 'external-platform')",
            name="source_type",
        ),
        Index("ix_evaluation_runs_dataset_created", "dataset_version_id", "created_at"),
        Index("ix_evaluation_runs_rule_purpose", "rule_version_id", "purpose"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_new_id)
    dataset_version_id: Mapped[str] = mapped_column(
        ForeignKey("dataset_versions.id", ondelete="RESTRICT"), nullable=False
    )
    rule_version_id: Mapped[str] = mapped_column(
        ForeignKey("rule_versions.id", ondelete="RESTRICT"), nullable=False
    )
    model_config_id: Mapped[str] = mapped_column(
        ForeignKey("model_configs.id", ondelete="RESTRICT"), nullable=False
    )
    report_artifact_id: Mapped[str | None] = mapped_column(
        ForeignKey("document_artifacts.id", ondelete="SET NULL")
    )
    status: Mapped[str] = mapped_column(
        String(32), nullable=False, server_default="PENDING"
    )
    retriever_version: Mapped[str] = mapped_column(String(128), nullable=False)
    code_version: Mapped[str] = mapped_column(String(128), nullable=False)
    metrics_json: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    purpose: Mapped[str] = mapped_column(String(32), nullable=False)
    dataset_split: Mapped[str] = mapped_column(String(32), nullable=False)
    source_type: Mapped[str] = mapped_column(String(32), nullable=False)
    provenance_status: Mapped[str] = mapped_column(String(32), nullable=False)
    claims_allowed: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default="0"
    )
    dataset_manifest_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    rule_version_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    input_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    config_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    code_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    model_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    prompt_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    binding_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    result_sha256: Mapped[str | None] = mapped_column(String(64))
    report_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    run_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    evaluator_version: Mapped[str] = mapped_column(String(128), nullable=False)
    reproducibility_command: Mapped[str] = mapped_column(Text, nullable=False)
    blockers_json: Mapped[list[str]] = mapped_column(JSON, nullable=False)
    dataset_snapshot_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    binding_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    run_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    report_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class EvaluationThresholdPolicy(TimestampMixin, SchemaVersionMixin, Base):
    __tablename__ = "evaluation_threshold_policies"
    __table_args__ = (
        UniqueConstraint("baseline_run_id"),
        UniqueConstraint("policy_sha256"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    baseline_run_id: Mapped[str] = mapped_column(
        ForeignKey("evaluation_runs.id", ondelete="RESTRICT"), nullable=False
    )
    baseline_report_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    approved_by: Mapped[str] = mapped_column(String(255), nullable=False)
    frozen_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    policy_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    policy_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)


class OptimizationJob(TimestampMixin, SchemaVersionMixin, Base):
    __tablename__ = "optimization_jobs"
    __table_args__ = (
        CheckConstraint(
            "status IN ('NOT_READY', 'BLOCKED', 'PENDING', 'RUNNING', "
            "'WAITING_APPROVAL', 'WAITING_HUMAN', 'OPTIMIZATION_FAILED', 'CANCELLED')",
            name="status",
        ),
        CheckConstraint(
            "max_rounds > 0 AND current_round >= 0 AND current_round <= max_rounds",
            name="rounds",
        ),
        CheckConstraint(
            "candidates_per_round > 0 AND candidates_per_round <= 10",
            name="candidate_limit",
        ),
        CheckConstraint(
            "required_stability_runs >= 2 AND required_stability_runs <= 20",
            name="stability_runs",
        ),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_new_id)
    base_rule_version_id: Mapped[str] = mapped_column(
        ForeignKey("rule_versions.id", ondelete="RESTRICT"), nullable=False
    )
    dataset_version_id: Mapped[str] = mapped_column(
        ForeignKey("dataset_versions.id", ondelete="RESTRICT"), nullable=False
    )
    candidate_rule_version_id: Mapped[str | None] = mapped_column(
        ForeignKey("rule_versions.id", ondelete="SET NULL")
    )
    status: Mapped[str] = mapped_column(
        String(32), nullable=False, server_default="PENDING"
    )
    max_rounds: Mapped[int] = mapped_column(Integer, nullable=False)
    candidates_per_round: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default="1"
    )
    required_stability_runs: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default="2"
    )
    current_round: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default="0"
    )
    root_cause: Mapped[str | None] = mapped_column(String(64))
    error_json: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    hashes_json: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    provenance_json: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    samples_json: Mapped[list[dict[str, Any]] | None] = mapped_column(JSON)
    failure_trajectory_json: Mapped[list[dict[str, Any]] | None] = mapped_column(JSON)
    readiness_json: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    graph_trace_json: Mapped[list[dict[str, Any]] | None] = mapped_column(JSON)
    last_checkpoint_sha256: Mapped[str | None] = mapped_column(String(64))
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class OptimizationAttempt(TimestampMixin, SchemaVersionMixin, Base):
    __tablename__ = "optimization_attempts"
    __table_args__ = (
        UniqueConstraint(
            "optimization_job_id",
            "attempt_number",
            name="uq_optimization_attempts_job_number",
        ),
        CheckConstraint("attempt_number > 0", name="attempt_positive"),
        CheckConstraint(
            "status IN ('STARTED', 'EVALUATING', 'COMPLETED', 'FAILED', 'WAITING_HUMAN')",
            name="status",
        ),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_new_id)
    optimization_job_id: Mapped[str] = mapped_column(
        ForeignKey("optimization_jobs.id", ondelete="CASCADE"), nullable=False
    )
    candidate_rule_version_id: Mapped[str | None] = mapped_column(
        ForeignKey("rule_versions.id", ondelete="SET NULL")
    )
    attempt_number: Mapped[int] = mapped_column(Integer, nullable=False)
    root_cause: Mapped[str] = mapped_column(String(64), nullable=False)
    rationale: Mapped[str] = mapped_column(Text, nullable=False)
    evaluation_result_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    accepted: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default="0")
    rejection_reason: Mapped[str | None] = mapped_column(Text)
    status: Mapped[str] = mapped_column(
        String(32), nullable=False, server_default="STARTED"
    )
    root_cause_json: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    candidates_json: Mapped[list[dict[str, Any]] | None] = mapped_column(JSON)
    evaluations_json: Mapped[list[dict[str, Any]] | None] = mapped_column(JSON)
    failure_json: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    checkpoint_sha256: Mapped[str | None] = mapped_column(String(64))
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


CORE_MODEL_TYPES = (
    DocumentSnapshot,
    DocumentArtifact,
    ReviewJob,
    ReviewExecutionSpec,
    JobCheckpoint,
    IdempotencyRecord,
    RuleSet,
    RuleVersion,
    Finding,
    EvidenceReference,
    HumanDecision,
    DatasetVersion,
    EvaluationCase,
    DatasetAnnotationSampleRecord,
    EvaluationRun,
    EvaluationThresholdPolicy,
    OptimizationJob,
    OptimizationAttempt,
    ModelConfig,
)
