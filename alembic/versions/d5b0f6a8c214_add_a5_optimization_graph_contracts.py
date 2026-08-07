"""add A5 bounded optimization graph contracts

Revision ID: d5b0f6a8c214
Revises: c4a9e2d7f103
Create Date: 2026-08-06 23:00:00
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "d5b0f6a8c214"
down_revision: Union[str, Sequence[str], None] = "c4a9e2d7f103"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table("optimization_jobs") as batch:
        batch.drop_constraint(op.f("ck_optimization_jobs_status"), type_="check")
        batch.create_check_constraint(
            op.f("ck_optimization_jobs_status"),
            "status IN ('NOT_READY', 'BLOCKED', 'PENDING', 'RUNNING', "
            "'WAITING_APPROVAL', 'WAITING_HUMAN', 'OPTIMIZATION_FAILED', "
            "'CANCELLED')",
        )
        batch.add_column(sa.Column("readiness_json", sa.JSON(), nullable=True))
        batch.add_column(sa.Column("graph_trace_json", sa.JSON(), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("optimization_jobs") as batch:
        batch.drop_column("graph_trace_json")
        batch.drop_column("readiness_json")
        batch.drop_constraint(op.f("ck_optimization_jobs_status"), type_="check")
        batch.create_check_constraint(
            op.f("ck_optimization_jobs_status"),
            "status IN ('PENDING', 'RUNNING', 'WAITING_APPROVAL', "
            "'WAITING_HUMAN', 'OPTIMIZATION_FAILED', 'CANCELLED')",
        )
