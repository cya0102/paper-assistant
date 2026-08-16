"""Harden Artifact provenance and ResearchTask idempotency.

Revision ID: 0011_rod_hardening
Revises: 0010_research_tasks
"""

from collections.abc import Sequence

from alembic import op


revision: str = "0011_rod_hardening"
down_revision: str | None = "0010_research_tasks"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Blobs remain content-addressed, but Artifact rows represent provenance
    # instances. The same bytes produced by two WorkUnits must therefore be able
    # to reference the shared blob through two distinct catalog rows.
    op.drop_constraint(
        "uq_research_artifacts_project_type_schema_hash",
        "research_artifacts",
        type_="unique",
    )
    op.create_index(
        "ix_research_artifacts_project_type_schema_hash",
        "research_artifacts",
        ["project_id", "artifact_type", "schema_version", "content_hash"],
    )
    # Earlier releases performed a read-before-insert check but had no database
    # constraint, so concurrent delegates could have produced duplicate keys.
    # Keep the most useful completed row canonical and preserve every other
    # historical task by assigning it a deterministic 64-character legacy key
    # before adding the uniqueness constraint. WorkUnits and Artifact
    # provenance remain intact.
    op.execute(
        """
        WITH ranked AS (
            SELECT
                task_id,
                row_number() OVER (
                    PARTITION BY project_id, generation_key
                    ORDER BY
                        CASE status
                            WHEN 'completed' THEN 0
                            WHEN 'partially_completed' THEN 1
                            WHEN 'running' THEN 2
                            WHEN 'planned' THEN 3
                            WHEN 'created' THEN 4
                            ELSE 5
                        END,
                        updated_at DESC,
                        created_at,
                        task_id
                ) AS ordinal
            FROM research_tasks
        )
        UPDATE research_tasks AS task
        SET generation_key =
            md5(task.generation_key || ':' || task.task_id::text)
            || md5('legacy:' || task.generation_key || ':' || task.task_id::text)
        FROM ranked
        WHERE task.task_id = ranked.task_id
          AND ranked.ordinal > 1
        """
    )
    op.create_unique_constraint(
        "uq_research_tasks_project_generation_key",
        "research_tasks",
        ["project_id", "generation_key"],
    )


def downgrade() -> None:
    op.drop_constraint(
        "uq_research_tasks_project_generation_key",
        "research_tasks",
        type_="unique",
    )
    op.drop_index(
        "ix_research_artifacts_project_type_schema_hash",
        table_name="research_artifacts",
    )
    op.create_unique_constraint(
        "uq_research_artifacts_project_type_schema_hash",
        "research_artifacts",
        ["project_id", "artifact_type", "schema_version", "content_hash"],
    )
