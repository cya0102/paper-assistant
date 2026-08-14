"""Preserve notes when their derived Section is rebuilt.

Revision ID: 0007_phase3
Revises: 0006_phase3
"""

from collections.abc import Sequence

from alembic import op

revision: str = "0007_phase3"
down_revision: str | None = "0006_phase3"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

SECTION_FK = "fk_notes_section_id_sections"

def upgrade() -> None:
    # Migration 0005 created this FK without a name, so PostgreSQL assigned the
    # default name ``notes_section_id_fkey`` (NOT the model-convention name
    # ``fk_notes_section_id_sections``). Drop the real name before recreating.
    op.drop_constraint(SECTION_FK, "notes", type_="foreignkey")
    op.create_foreign_key(
        SECTION_FK,
        "notes",
        "sections",
        ["section_id"],
        ["section_id"],
        ondelete="SET NULL",
    )


def downgrade() -> None:
    op.drop_constraint("fk_notes_section_id_sections", "notes", type_="foreignkey")
    # Restore the original unnamed-FK state created by migration 0005, so a
    # downgrade/upgrade cycle keeps working.
    op.create_foreign_key(
        SECTION_FK,
        "notes",
        "sections",
        ["section_id"],
        ["section_id"],
        ondelete="CASCADE",
    )
