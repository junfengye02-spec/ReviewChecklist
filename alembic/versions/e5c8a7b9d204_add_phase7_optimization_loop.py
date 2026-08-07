"""add phase 7 bounded optimization checkpoints

Revision ID: e5c8a7b9d204
Revises: d4b7c9e2f601
Create Date: 2026-07-28 15:00:00
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "e5c8a7b9d204"
down_revision: Union[str, Sequence[str], None] = "d4b7c9e2f601"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table("optimization_jobs") as batch:
        batch.drop_constraint(op.f("ck_optimization_jobs_status"), type_="check")
        batch.create_check_constraint(
            op.f("ck_optimization_jobs_status"),
            "status IN ('PENDING', 'RUNNING', 'WAITING_APPROVAL', "
            "'WAITING_HUMAN', 'OPTIMIZATION_FAILED', 'CANCELLED')",
        )
        batch.add_column(
            sa.Column(
                "candidates_per_round", sa.Integer(), server_default="1", nullable=False
            )
        )
        batch.add_column(
            sa.Column(
                "required_stability_runs", sa.Integer(), server_default="2", nullable=False
            )
        )
        batch.add_column(sa.Column("hashes_json", sa.JSON(), nullable=True))
        batch.add_column(sa.Column("provenance_json", sa.JSON(), nullable=True))
        batch.add_column(sa.Column("samples_json", sa.JSON(), nullable=True))
        batch.add_column(sa.Column("failure_trajectory_json", sa.JSON(), nullable=True))
        batch.add_column(sa.Column("last_checkpoint_sha256", sa.String(64), nullable=True))
        batch.add_column(sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True))
        batch.create_check_constraint(
            op.f("ck_optimization_jobs_candidate_limit"),
            "candidates_per_round > 0 AND candidates_per_round <= 10",
        )
        batch.create_check_constraint(
            op.f("ck_optimization_jobs_stability_runs"),
            "required_stability_runs >= 2 AND required_stability_runs <= 20",
        )

    with op.batch_alter_table("optimization_attempts") as batch:
        batch.add_column(
            sa.Column("status", sa.String(32), server_default="STARTED", nullable=False)
        )
        batch.add_column(sa.Column("root_cause_json", sa.JSON(), nullable=True))
        batch.add_column(sa.Column("candidates_json", sa.JSON(), nullable=True))
        batch.add_column(sa.Column("evaluations_json", sa.JSON(), nullable=True))
        batch.add_column(sa.Column("failure_json", sa.JSON(), nullable=True))
        batch.add_column(sa.Column("checkpoint_sha256", sa.String(64), nullable=True))
        batch.add_column(sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True))
        batch.create_check_constraint(
            op.f("ck_optimization_attempts_status"),
            "status IN ('STARTED', 'EVALUATING', 'COMPLETED', 'FAILED', 'WAITING_HUMAN')",
        )


def downgrade() -> None:
    with op.batch_alter_table("optimization_attempts") as batch:
        batch.drop_constraint(op.f("ck_optimization_attempts_status"), type_="check")
        for name in (
            "completed_at",
            "checkpoint_sha256",
            "failure_json",
            "evaluations_json",
            "candidates_json",
            "root_cause_json",
            "status",
        ):
            batch.drop_column(name)

    with op.batch_alter_table("optimization_jobs") as batch:
        batch.drop_constraint(op.f("ck_optimization_jobs_stability_runs"), type_="check")
        batch.drop_constraint(op.f("ck_optimization_jobs_candidate_limit"), type_="check")
        for name in (
            "completed_at",
            "last_checkpoint_sha256",
            "failure_trajectory_json",
            "samples_json",
            "provenance_json",
            "hashes_json",
            "required_stability_runs",
            "candidates_per_round",
        ):
            batch.drop_column(name)
        batch.drop_constraint(op.f("ck_optimization_jobs_status"), type_="check")
        batch.create_check_constraint(
            op.f("ck_optimization_jobs_status"),
            "status IN ('PENDING', 'RUNNING', 'WAITING_APPROVAL', "
            "'COMPLETED', 'FAILED', 'CANCELLED')",
        )
