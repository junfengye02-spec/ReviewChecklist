"""add phase 6 governance and immutable version provenance

Revision ID: d4b7c9e2f601
Revises: a3f9d8c7b6e5
Create Date: 2026-07-28 13:00:00
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "d4b7c9e2f601"
down_revision: Union[str, Sequence[str], None] = "a3f9d8c7b6e5"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table("rule_versions") as batch:
        batch.add_column(sa.Column("provenance_json", sa.JSON(), nullable=True))
        batch.add_column(sa.Column("evaluation_gate_json", sa.JSON(), nullable=True))
        batch.add_column(sa.Column("published_by", sa.String(length=255), nullable=True))

    with op.batch_alter_table("findings") as batch:
        batch.drop_constraint(op.f("ck_findings_status"), type_="check")
        batch.add_column(sa.Column("workflow_state", sa.String(length=32), nullable=True))
        batch.add_column(sa.Column("review_input_sha256", sa.String(length=64), nullable=True))
        batch.add_column(sa.Column("finding_content_sha256", sa.String(length=64), nullable=True))
        batch.add_column(sa.Column("provenance_json", sa.JSON(), nullable=True))
        batch.add_column(sa.Column("documents_json", sa.JSON(), nullable=True))
        batch.create_check_constraint(
            op.f("ck_findings_status"),
            "status IN ('OPEN', 'NEED_MORE_EVIDENCE', 'WAITING_HUMAN', "
            "'PENDING_DECISION', 'WORK_ITEM_OPEN', 'APPROVED', 'REJECTED', "
            "'MODIFIED', 'INSUFFICIENT_EVIDENCE')",
        )
        batch.create_unique_constraint(
            op.f("uq_findings_finding_content_sha256"), ["finding_content_sha256"]
        )

    with op.batch_alter_table("human_decisions") as batch:
        batch.add_column(
            sa.Column(
                "reviewer_kind",
                sa.String(length=32),
                server_default="human",
                nullable=False,
            )
        )
        batch.add_column(sa.Column("review_input_sha256", sa.String(length=64), nullable=True))
        batch.add_column(sa.Column("finding_content_sha256", sa.String(length=64), nullable=True))
        batch.add_column(sa.Column("evidence_sha256", sa.String(length=64), nullable=True))
        batch.add_column(sa.Column("decision_sha256", sa.String(length=64), nullable=True))
        batch.create_unique_constraint(
            op.f("uq_human_decisions_decision_sha256"), ["decision_sha256"]
        )

    with op.batch_alter_table("dataset_versions") as batch:
        batch.add_column(sa.Column("parent_version_id", sa.String(length=36), nullable=True))
        batch.add_column(
            sa.Column("status", sa.String(length=32), server_default="DRAFT", nullable=False)
        )
        batch.add_column(sa.Column("change_summary", sa.Text(), nullable=True))
        batch.add_column(sa.Column("provenance_json", sa.JSON(), nullable=True))
        batch.add_column(sa.Column("manifest_json", sa.JSON(), nullable=True))
        batch.alter_column("manifest_artifact_id", existing_type=sa.String(length=36), nullable=True)
        batch.create_check_constraint(
            op.f("ck_dataset_versions_status"),
            "status IN ('DRAFT', 'PROVISIONAL', 'VERIFIED', 'FROZEN')",
        )
        batch.create_foreign_key(
            op.f("fk_dataset_versions_parent_version_id_dataset_versions"),
            "dataset_versions",
            ["parent_version_id"],
            ["id"],
            ondelete="SET NULL",
        )

    with op.batch_alter_table("evaluation_cases") as batch:
        batch.add_column(sa.Column("finding_id", sa.String(length=36), nullable=True))
        batch.add_column(sa.Column("human_decision_id", sa.String(length=36), nullable=True))
        batch.add_column(sa.Column("document_sha256", sa.String(length=64), nullable=True))
        batch.add_column(sa.Column("label_version", sa.String(length=128), nullable=True))
        batch.add_column(sa.Column("label_status", sa.String(length=32), nullable=True))
        batch.add_column(sa.Column("review_input_sha256", sa.String(length=64), nullable=True))
        batch.add_column(sa.Column("evidence_sha256", sa.String(length=64), nullable=True))
        batch.add_column(sa.Column("sample_sha256", sa.String(length=64), nullable=True))
        batch.create_foreign_key(
            op.f("fk_evaluation_cases_finding_id_findings"),
            "findings",
            ["finding_id"],
            ["id"],
            ondelete="RESTRICT",
        )
        batch.create_foreign_key(
            op.f("fk_evaluation_cases_human_decision_id_human_decisions"),
            "human_decisions",
            ["human_decision_id"],
            ["id"],
            ondelete="RESTRICT",
        )


def downgrade() -> None:
    with op.batch_alter_table("evaluation_cases") as batch:
        batch.drop_constraint(
            op.f("fk_evaluation_cases_human_decision_id_human_decisions"), type_="foreignkey"
        )
        batch.drop_constraint(op.f("fk_evaluation_cases_finding_id_findings"), type_="foreignkey")
        for name in (
            "sample_sha256", "evidence_sha256", "review_input_sha256",
            "label_status", "label_version", "document_sha256", "human_decision_id", "finding_id",
        ):
            batch.drop_column(name)

    with op.batch_alter_table("dataset_versions") as batch:
        batch.drop_constraint(
            op.f("fk_dataset_versions_parent_version_id_dataset_versions"), type_="foreignkey"
        )
        batch.drop_constraint(op.f("ck_dataset_versions_status"), type_="check")
        batch.alter_column("manifest_artifact_id", existing_type=sa.String(length=36), nullable=False)
        for name in (
            "manifest_json", "provenance_json", "change_summary", "status", "parent_version_id",
        ):
            batch.drop_column(name)

    with op.batch_alter_table("human_decisions") as batch:
        batch.drop_constraint(op.f("uq_human_decisions_decision_sha256"), type_="unique")
        for name in (
            "decision_sha256", "evidence_sha256", "finding_content_sha256",
            "review_input_sha256", "reviewer_kind",
        ):
            batch.drop_column(name)

    with op.batch_alter_table("findings") as batch:
        batch.drop_constraint(op.f("uq_findings_finding_content_sha256"), type_="unique")
        batch.drop_constraint(op.f("ck_findings_status"), type_="check")
        batch.create_check_constraint(
            op.f("ck_findings_status"),
            "status IN ('OPEN', 'NEED_MORE_EVIDENCE', 'WAITING_HUMAN', "
            "'APPROVED', 'REJECTED', 'MODIFIED')",
        )
        for name in (
            "documents_json", "provenance_json", "finding_content_sha256",
            "review_input_sha256", "workflow_state",
        ):
            batch.drop_column(name)

    with op.batch_alter_table("rule_versions") as batch:
        batch.drop_column("published_by")
        batch.drop_column("evaluation_gate_json")
        batch.drop_column("provenance_json")
