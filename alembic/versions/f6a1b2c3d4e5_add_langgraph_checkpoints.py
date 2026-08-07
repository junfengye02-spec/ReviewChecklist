"""add LangGraph review checkpoint storage

Revision ID: f6a1b2c3d4e5
Revises: e5c8a7b9d204
Create Date: 2026-08-06 15:00:00
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import mysql


revision: str = "f6a1b2c3d4e5"
down_revision: Union[str, Sequence[str], None] = "e5c8a7b9d204"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


_BLOB = sa.LargeBinary().with_variant(mysql.LONGBLOB(), "mysql")


def upgrade() -> None:
    op.create_table(
        "langgraph_checkpoints",
        sa.Column("checkpoint_key_sha256", sa.String(length=64), nullable=False),
        sa.Column("thread_id", sa.String(length=128), nullable=False),
        sa.Column(
            "checkpoint_ns",
            sa.String(length=255),
            server_default="",
            nullable=False,
        ),
        sa.Column("checkpoint_id", sa.String(length=64), nullable=False),
        sa.Column("parent_checkpoint_id", sa.String(length=64), nullable=True),
        sa.Column("checkpoint_type", sa.String(length=255), nullable=False),
        sa.Column("checkpoint_data", _BLOB, nullable=False),
        sa.Column("metadata_type", sa.String(length=255), nullable=False),
        sa.Column("metadata_data", _BLOB, nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint(
            "checkpoint_key_sha256",
            name=op.f("pk_langgraph_checkpoints"),
        ),
    )
    op.create_index(
        "ix_langgraph_checkpoints_thread_namespace_id",
        "langgraph_checkpoints",
        ["thread_id", "checkpoint_ns", "checkpoint_id"],
        unique=False,
    )

    op.create_table(
        "langgraph_checkpoint_writes",
        sa.Column("write_key_sha256", sa.String(length=64), nullable=False),
        sa.Column("thread_id", sa.String(length=128), nullable=False),
        sa.Column(
            "checkpoint_ns",
            sa.String(length=255),
            server_default="",
            nullable=False,
        ),
        sa.Column("checkpoint_id", sa.String(length=64), nullable=False),
        sa.Column("task_id", sa.String(length=64), nullable=False),
        sa.Column(
            "task_path",
            sa.String(length=1024),
            server_default="",
            nullable=False,
        ),
        sa.Column("write_index", sa.Integer(), nullable=False),
        sa.Column("channel", sa.String(length=255), nullable=False),
        sa.Column("value_type", sa.String(length=255), nullable=False),
        sa.Column("value_data", _BLOB, nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint(
            "write_key_sha256",
            name=op.f("pk_langgraph_checkpoint_writes"),
        ),
    )
    op.create_index(
        "ix_langgraph_checkpoint_writes_checkpoint",
        "langgraph_checkpoint_writes",
        ["thread_id", "checkpoint_ns", "checkpoint_id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(
        "ix_langgraph_checkpoint_writes_checkpoint",
        table_name="langgraph_checkpoint_writes",
    )
    op.drop_table("langgraph_checkpoint_writes")
    op.drop_index(
        "ix_langgraph_checkpoints_thread_namespace_id",
        table_name="langgraph_checkpoints",
    )
    op.drop_table("langgraph_checkpoints")
