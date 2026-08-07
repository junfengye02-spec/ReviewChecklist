"""add immutable review execution specs

Revision ID: a2e7c4d9b801
Revises: f6a1b2c3d4e5
Create Date: 2026-08-06 16:30:00
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "a2e7c4d9b801"
down_revision: Union[str, Sequence[str], None] = "f6a1b2c3d4e5"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table("review_jobs") as batch:
        batch.add_column(
            sa.Column("execution_spec_sha256", sa.String(length=64), nullable=True)
        )

    op.create_table(
        "review_execution_specs",
        sa.Column("job_id", sa.String(length=36), nullable=False),
        sa.Column("input_sha256", sa.String(length=64), nullable=False),
        sa.Column("spec_json", sa.JSON(), nullable=False),
        sa.Column("document_snapshot_id", sa.String(length=36), nullable=False),
        sa.Column("rule_version_id", sa.String(length=36), nullable=False),
        sa.Column("dataset_version_id", sa.String(length=36), nullable=False),
        sa.Column("model_config_id", sa.String(length=36), nullable=False),
        sa.Column("retriever_artifact_id", sa.String(length=36), nullable=False),
        sa.Column("index_artifact_id", sa.String(length=36), nullable=False),
        sa.Column("chunk_artifact_id", sa.String(length=36), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "schema_version",
            sa.String(length=16),
            server_default="1",
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["job_id"],
            ["review_jobs.id"],
            name=op.f("fk_review_execution_specs_job_id_review_jobs"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["document_snapshot_id"],
            ["document_snapshots.id"],
            name=op.f(
                "fk_review_execution_specs_document_snapshot_id_document_snapshots"
            ),
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["rule_version_id"],
            ["rule_versions.id"],
            name=op.f("fk_review_execution_specs_rule_version_id_rule_versions"),
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["dataset_version_id"],
            ["dataset_versions.id"],
            name=op.f("fk_review_execution_specs_dataset_version_id_dataset_versions"),
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["model_config_id"],
            ["model_configs.id"],
            name=op.f("fk_review_execution_specs_model_config_id_model_configs"),
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["retriever_artifact_id"],
            ["document_artifacts.id"],
            name=op.f(
                "fk_review_execution_specs_retriever_artifact_id_document_artifacts"
            ),
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["index_artifact_id"],
            ["document_artifacts.id"],
            name=op.f(
                "fk_review_execution_specs_index_artifact_id_document_artifacts"
            ),
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["chunk_artifact_id"],
            ["document_artifacts.id"],
            name=op.f(
                "fk_review_execution_specs_chunk_artifact_id_document_artifacts"
            ),
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint(
            "job_id", name=op.f("pk_review_execution_specs")
        ),
    )
    op.create_index(
        "ix_review_execution_specs_input_sha256",
        "review_execution_specs",
        ["input_sha256"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(
        "ix_review_execution_specs_input_sha256",
        table_name="review_execution_specs",
    )
    op.drop_table("review_execution_specs")
    with op.batch_alter_table("review_jobs") as batch:
        batch.drop_column("execution_spec_sha256")
