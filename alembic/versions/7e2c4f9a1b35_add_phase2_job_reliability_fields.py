"""add phase 2 job reliability fields

Revision ID: 7e2c4f9a1b35
Revises: 6163370fb844
Create Date: 2026-07-27 18:30:00
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "7e2c4f9a1b35"
down_revision: Union[str, Sequence[str], None] = "6163370fb844"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "review_jobs",
        sa.Column(
            "job_type",
            sa.String(length=128),
            server_default="review",
            nullable=False,
        ),
    )
    op.add_column(
        "review_jobs",
        sa.Column(
            "input_reference",
            sa.String(length=1024),
            server_default="",
            nullable=False,
        ),
    )
    op.add_column(
        "review_jobs",
        sa.Column(
            "checkpoint_sequence",
            sa.Integer(),
            server_default="0",
            nullable=False,
        ),
    )
    op.add_column(
        "review_jobs", sa.Column("error_code", sa.String(length=128), nullable=True)
    )
    op.add_column(
        "review_jobs",
        sa.Column("output_reference", sa.String(length=1024), nullable=True),
    )
    op.add_column(
        "review_jobs", sa.Column("output_summary", sa.Text(), nullable=True)
    )
    op.add_column(
        "job_checkpoints",
        sa.Column(
            "sequence", sa.Integer(), server_default="0", nullable=False
        ),
    )
    op.create_index(
        "ix_job_checkpoints_job_sequence",
        "job_checkpoints",
        ["review_job_id", "sequence"],
        unique=False,
    )
    # The queue claim and idempotency replay paths are hot paths in Phase 2.
    # Keep these indexes in the Phase 2 revision so existing Stage 1 databases
    # receive the same access paths as freshly created metadata.
    op.create_index(
        "ix_review_jobs_queue_created",
        "review_jobs",
        ["status", "available_at", "created_at", "id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_review_jobs_queue_created", table_name="review_jobs")
    op.drop_index(
        "ix_job_checkpoints_job_sequence", table_name="job_checkpoints"
    )
    op.drop_column("job_checkpoints", "sequence")
    op.drop_column("review_jobs", "output_summary")
    op.drop_column("review_jobs", "output_reference")
    op.drop_column("review_jobs", "error_code")
    op.drop_column("review_jobs", "checkpoint_sequence")
    op.drop_column("review_jobs", "input_reference")
    op.drop_column("review_jobs", "job_type")
