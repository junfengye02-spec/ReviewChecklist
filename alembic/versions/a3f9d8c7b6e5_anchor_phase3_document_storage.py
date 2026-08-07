"""anchor phase 3 document storage lifecycle

Revision ID: a3f9d8c7b6e5
Revises: 7e2c4f9a1b35
Create Date: 2026-07-27 20:00:00
"""

from typing import Sequence, Union


revision: str = "a3f9d8c7b6e5"
down_revision: Union[str, Sequence[str], None] = "7e2c4f9a1b35"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Anchor the Phase 3 lifecycle release; references already exist in Stage 1."""


def downgrade() -> None:
    """The Phase 3 anchor has no schema mutation to reverse."""
