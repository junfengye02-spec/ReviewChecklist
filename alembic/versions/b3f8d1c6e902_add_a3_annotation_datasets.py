"""add A3 annotation dataset sample workflow

Revision ID: b3f8d1c6e902
Revises: a2e7c4d9b801
Create Date: 2026-08-06 20:30:00
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "b3f8d1c6e902"
down_revision: Union[str, Sequence[str], None] = "a2e7c4d9b801"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "dataset_annotation_samples",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("dataset_version_id", sa.String(length=36), nullable=False),
        sa.Column("sample_key", sa.String(length=128), nullable=False),
        sa.Column("finding_id", sa.String(length=36), nullable=False),
        sa.Column("document_snapshot_id", sa.String(length=36), nullable=False),
        sa.Column("source_pdf_reference", sa.String(length=2048), nullable=False),
        sa.Column("document_sha256", sa.String(length=64), nullable=False),
        sa.Column("source_case_sha256", sa.String(length=64), nullable=False),
        sa.Column("rule_version_id", sa.String(length=36), nullable=False),
        sa.Column("rule_sha256", sa.String(length=64), nullable=False),
        sa.Column("query_id", sa.String(length=128), nullable=False),
        sa.Column("question_label", sa.String(length=512), nullable=False),
        sa.Column("split", sa.String(length=32), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("annotation_human_decision_id", sa.String(length=36), nullable=True),
        sa.Column("review_human_decision_id", sa.String(length=36), nullable=True),
        sa.Column("adjudication_human_decision_id", sa.String(length=36), nullable=True),
        sa.Column("annotator_id", sa.String(length=255), nullable=True),
        sa.Column("annotated_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("reviewer_id", sa.String(length=255), nullable=True),
        sa.Column("reviewed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("adjudicator_id", sa.String(length=255), nullable=True),
        sa.Column("adjudicated_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("evidence_catalog_sha256", sa.String(length=64), nullable=False),
        sa.Column("final_label_sha256", sa.String(length=64), nullable=True),
        sa.Column("label_version", sa.String(length=128), nullable=True),
        sa.Column("sample_sha256", sa.String(length=64), nullable=False),
        sa.Column("sample_json", sa.JSON(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("(CURRENT_TIMESTAMP)"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("(CURRENT_TIMESTAMP)"),
            nullable=False,
        ),
        sa.Column(
            "schema_version", sa.String(length=16), server_default="1", nullable=False
        ),
        sa.CheckConstraint(
            "split IN ('OPTIMIZATION', 'VALIDATION', 'FROZEN_TEST')",
            name=op.f("ck_dataset_annotation_samples_split"),
        ),
        sa.CheckConstraint(
            "status IN ('PENDING_ANNOTATION', 'PENDING_REVIEW', 'CONFLICT', "
            "'VERIFIED', 'FROZEN')",
            name=op.f("ck_dataset_annotation_samples_status"),
        ),
        sa.ForeignKeyConstraint(
            ["dataset_version_id"],
            ["dataset_versions.id"],
            name=op.f(
                "fk_dataset_annotation_samples_dataset_version_id_dataset_versions"
            ),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["finding_id"],
            ["findings.id"],
            name=op.f("fk_dataset_annotation_samples_finding_id_findings"),
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["document_snapshot_id"],
            ["document_snapshots.id"],
            name=op.f(
                "fk_dataset_annotation_samples_document_snapshot_id_document_snapshots"
            ),
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["rule_version_id"],
            ["rule_versions.id"],
            name=op.f("fk_dataset_annotation_samples_rule_version_id_rule_versions"),
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["annotation_human_decision_id"],
            ["human_decisions.id"],
            name=op.f(
                "fk_dataset_annotation_samples_annotation_human_decision_id_human_decisions"
            ),
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["review_human_decision_id"],
            ["human_decisions.id"],
            name=op.f(
                "fk_dataset_annotation_samples_review_human_decision_id_human_decisions"
            ),
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["adjudication_human_decision_id"],
            ["human_decisions.id"],
            name=op.f(
                "fk_dataset_annotation_samples_adjudication_human_decision_id_human_decisions"
            ),
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_dataset_annotation_samples")),
        sa.UniqueConstraint(
            "dataset_version_id",
            "sample_key",
            name="uq_dataset_annotation_samples_dataset_sample",
        ),
    )
    op.create_index(
        "ix_dataset_annotation_samples_dataset_status",
        "dataset_annotation_samples",
        ["dataset_version_id", "status"],
        unique=False,
    )
    op.create_index(
        "ix_dataset_annotation_samples_document_split",
        "dataset_annotation_samples",
        ["document_sha256", "split"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(
        "ix_dataset_annotation_samples_document_split",
        table_name="dataset_annotation_samples",
    )
    op.drop_index(
        "ix_dataset_annotation_samples_dataset_status",
        table_name="dataset_annotation_samples",
    )
    op.drop_table("dataset_annotation_samples")
