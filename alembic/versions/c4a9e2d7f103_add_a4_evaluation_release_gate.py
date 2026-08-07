"""add A4 reproducible evaluation runs and release threshold policies

Revision ID: c4a9e2d7f103
Revises: b3f8d1c6e902
Create Date: 2026-08-06 22:00:00
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "c4a9e2d7f103"
down_revision: Union[str, Sequence[str], None] = "b3f8d1c6e902"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


_ZERO_HASH = "0" * 64


def upgrade() -> None:
    with op.batch_alter_table("evaluation_runs") as batch:
        batch.drop_constraint(op.f("ck_evaluation_runs_status"), type_="check")
        batch.create_check_constraint(
            op.f("ck_evaluation_runs_status"),
            "status IN ('NOT_READY', 'PENDING', 'RUNNING', 'COMPLETED', 'FAILED')",
        )
        batch.add_column(sa.Column("purpose", sa.String(32), server_default="CANDIDATE_DIAGNOSTIC", nullable=False))
        batch.add_column(sa.Column("dataset_split", sa.String(32), server_default="VALIDATION", nullable=False))
        batch.add_column(sa.Column("source_type", sa.String(32), server_default="provisional", nullable=False))
        batch.add_column(sa.Column("provenance_status", sa.String(32), server_default="provisional", nullable=False))
        batch.add_column(sa.Column("claims_allowed", sa.Boolean(), server_default=sa.false(), nullable=False))
        batch.add_column(sa.Column("dataset_manifest_sha256", sa.String(64), server_default=_ZERO_HASH, nullable=False))
        batch.add_column(sa.Column("rule_version_sha256", sa.String(64), server_default=_ZERO_HASH, nullable=False))
        batch.add_column(sa.Column("input_sha256", sa.String(64), server_default=_ZERO_HASH, nullable=False))
        batch.add_column(sa.Column("config_sha256", sa.String(64), server_default=_ZERO_HASH, nullable=False))
        batch.add_column(sa.Column("code_sha256", sa.String(64), server_default=_ZERO_HASH, nullable=False))
        batch.add_column(sa.Column("model_sha256", sa.String(64), server_default=_ZERO_HASH, nullable=False))
        batch.add_column(sa.Column("prompt_sha256", sa.String(64), server_default=_ZERO_HASH, nullable=False))
        batch.add_column(sa.Column("binding_sha256", sa.String(64), server_default=_ZERO_HASH, nullable=False))
        batch.add_column(sa.Column("result_sha256", sa.String(64), nullable=True))
        batch.add_column(sa.Column("report_sha256", sa.String(64), server_default=_ZERO_HASH, nullable=False))
        batch.add_column(sa.Column("run_sha256", sa.String(64), server_default=_ZERO_HASH, nullable=False))
        batch.add_column(sa.Column("evaluator_version", sa.String(128), server_default="legacy-unverified", nullable=False))
        batch.add_column(sa.Column("reproducibility_command", sa.Text(), server_default="unavailable for legacy run", nullable=False))
        batch.add_column(sa.Column("blockers_json", sa.JSON(), nullable=False, server_default="[]"))
        batch.add_column(sa.Column("dataset_snapshot_json", sa.JSON(), nullable=False, server_default="{}"))
        batch.add_column(sa.Column("binding_json", sa.JSON(), nullable=False, server_default="{}"))
        batch.add_column(sa.Column("run_json", sa.JSON(), nullable=False, server_default="{}"))
        batch.add_column(sa.Column("report_json", sa.JSON(), nullable=False, server_default="{}"))
        batch.create_check_constraint(
            op.f("ck_evaluation_runs_purpose"),
            "purpose IN ('REAL_BASELINE', 'RELEASE_GATE', 'CANDIDATE_DIAGNOSTIC')",
        )
        batch.create_check_constraint(
            op.f("ck_evaluation_runs_source_type"),
            "source_type IN ('real', 'provisional', 'synthetic', 'external-platform')",
        )
        batch.create_index("ix_evaluation_runs_rule_purpose", ["rule_version_id", "purpose"], unique=False)

    with op.batch_alter_table("evaluation_runs") as batch:
        for name in (
            "purpose",
            "dataset_split",
            "source_type",
            "provenance_status",
            "dataset_manifest_sha256",
            "rule_version_sha256",
            "input_sha256",
            "config_sha256",
            "code_sha256",
            "model_sha256",
            "prompt_sha256",
            "binding_sha256",
            "report_sha256",
            "run_sha256",
            "evaluator_version",
            "reproducibility_command",
            "blockers_json",
            "dataset_snapshot_json",
            "binding_json",
            "run_json",
            "report_json",
        ):
            batch.alter_column(name, server_default=None)

    op.create_table(
        "evaluation_threshold_policies",
        sa.Column("id", sa.String(36), nullable=False),
        sa.Column("baseline_run_id", sa.String(36), nullable=False),
        sa.Column("baseline_report_sha256", sa.String(64), nullable=False),
        sa.Column("approved_by", sa.String(255), nullable=False),
        sa.Column("frozen_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("policy_sha256", sa.String(64), nullable=False),
        sa.Column("policy_json", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("(CURRENT_TIMESTAMP)"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("(CURRENT_TIMESTAMP)"), nullable=False),
        sa.Column("schema_version", sa.String(16), server_default="1", nullable=False),
        sa.ForeignKeyConstraint(
            ["baseline_run_id"],
            ["evaluation_runs.id"],
            name=op.f("fk_evaluation_threshold_policies_baseline_run_id_evaluation_runs"),
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_evaluation_threshold_policies")),
        sa.UniqueConstraint("baseline_run_id", name=op.f("uq_evaluation_threshold_policies_baseline_run_id")),
        sa.UniqueConstraint("policy_sha256", name=op.f("uq_evaluation_threshold_policies_policy_sha256")),
    )


def downgrade() -> None:
    op.drop_table("evaluation_threshold_policies")
    with op.batch_alter_table("evaluation_runs") as batch:
        batch.drop_index("ix_evaluation_runs_rule_purpose")
        batch.drop_constraint(op.f("ck_evaluation_runs_source_type"), type_="check")
        batch.drop_constraint(op.f("ck_evaluation_runs_purpose"), type_="check")
        for name in (
            "report_json",
            "run_json",
            "binding_json",
            "dataset_snapshot_json",
            "blockers_json",
            "reproducibility_command",
            "evaluator_version",
            "run_sha256",
            "report_sha256",
            "result_sha256",
            "binding_sha256",
            "prompt_sha256",
            "model_sha256",
            "code_sha256",
            "config_sha256",
            "input_sha256",
            "rule_version_sha256",
            "dataset_manifest_sha256",
            "claims_allowed",
            "provenance_status",
            "source_type",
            "dataset_split",
            "purpose",
        ):
            batch.drop_column(name)
        batch.drop_constraint(op.f("ck_evaluation_runs_status"), type_="check")
        batch.create_check_constraint(
            op.f("ck_evaluation_runs_status"),
            "status IN ('PENDING', 'RUNNING', 'COMPLETED', 'FAILED')",
        )
