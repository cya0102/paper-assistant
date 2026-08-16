"""Add ResearchTask and WorkUnit state for bounded delegation.

Revision ID: 0010_research_tasks
Revises: 0009_context_artifacts
"""

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = "0010_research_tasks"
down_revision: str | None = "0009_context_artifacts"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "research_tasks",
        sa.Column("task_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("project_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("session_id", postgresql.UUID(as_uuid=True)),
        sa.Column("research_question", sa.Text(), nullable=False),
        sa.Column("task_type", sa.String(length=64), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("plan_json", postgresql.JSONB(), server_default=sa.text("'[]'::jsonb"), nullable=False),
        sa.Column("max_workers", sa.Integer(), server_default=sa.text("3"), nullable=False),
        sa.Column("generation_key", sa.String(length=64), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "max_workers >= 1 AND max_workers <= 5",
            name=op.f("ck_research_tasks_max_workers_range"),
        ),
        sa.ForeignKeyConstraint(
            ["project_id"], ["projects.project_id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("task_id", name="pk_research_tasks"),
    )
    op.create_index(
        "ix_research_tasks_project_status", "research_tasks", ["project_id", "status"]
    )
    op.create_index("ix_research_tasks_session", "research_tasks", ["session_id"])

    op.create_table(
        "work_units",
        sa.Column("work_unit_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("task_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("project_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("work_type", sa.String(length=64), nullable=False),
        sa.Column("objective", sa.Text(), nullable=False),
        sa.Column("paper_ids_json", postgresql.JSONB(), server_default=sa.text("'[]'::jsonb"), nullable=False),
        sa.Column("input_artifact_ids_json", postgresql.JSONB(), server_default=sa.text("'[]'::jsonb"), nullable=False),
        sa.Column("dependency_ids_json", postgresql.JSONB(), server_default=sa.text("'[]'::jsonb"), nullable=False),
        sa.Column("requested_worker", sa.String(length=128), nullable=False),
        sa.Column("allowed_tools_json", postgresql.JSONB(), server_default=sa.text("'[]'::jsonb"), nullable=False),
        sa.Column("output_schema_json", postgresql.JSONB()),
        sa.Column("token_budget", sa.Integer(), nullable=False),
        sa.Column("tool_call_budget", sa.Integer(), nullable=False),
        sa.Column("timeout_seconds", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("attempt_count", sa.Integer(), server_default=sa.text("0"), nullable=False),
        sa.Column("output_artifact_id", postgresql.UUID(as_uuid=True)),
        sa.Column("error", sa.Text()),
        sa.Column("generation_key", sa.String(length=64), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "token_budget > 0", name=op.f("ck_work_units_token_budget_positive")
        ),
        sa.CheckConstraint(
            "tool_call_budget > 0", name=op.f("ck_work_units_tool_call_budget_positive")
        ),
        sa.CheckConstraint(
            "timeout_seconds > 0", name=op.f("ck_work_units_timeout_positive")
        ),
        sa.CheckConstraint(
            "attempt_count >= 0", name=op.f("ck_work_units_attempt_nonnegative")
        ),
        sa.ForeignKeyConstraint(
            ["project_id"], ["projects.project_id"], ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(
            ["task_id"], ["research_tasks.task_id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("work_unit_id", name="pk_work_units"),
        sa.UniqueConstraint(
            "task_id", "generation_key", name="uq_work_units_task_generation_key"
        ),
    )
    op.create_index("ix_work_units_task_status", "work_units", ["task_id", "status"])
    op.create_index("ix_work_units_requested_worker", "work_units", ["requested_worker"])


def downgrade() -> None:
    op.drop_index("ix_work_units_requested_worker", table_name="work_units")
    op.drop_index("ix_work_units_task_status", table_name="work_units")
    op.drop_table("work_units")
    op.drop_index("ix_research_tasks_session", table_name="research_tasks")
    op.drop_index("ix_research_tasks_project_status", table_name="research_tasks")
    op.drop_table("research_tasks")
