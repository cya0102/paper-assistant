"""Durable PostgreSQL Interaction/Note/UserPreference repository."""

from uuid import UUID

from sqlalchemy import or_, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session, sessionmaker

from paper_agent.domain.memory import Interaction, Note, UserPreference
from paper_agent.storage.postgres.models import InteractionRow, NoteRow, UserPreferenceRow


class SqlAlchemyMemoryRepository:
    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self._session_factory = session_factory

    def save_interaction(self, interaction: Interaction) -> None:
        with self._session_factory.begin() as session:
            session.add(
                InteractionRow(
                    interaction_id=interaction.interaction_id,
                    user_id=interaction.user_id,
                    session_id=interaction.session_id,
                    query=interaction.query,
                    paper_ids_json=[str(value) for value in interaction.paper_ids],
                    topics_json=list(interaction.topics),
                    interaction_type=interaction.interaction_type,
                    retrieved_chunk_ids_json=[str(value) for value in interaction.retrieved_chunk_ids],
                    answer_summary=interaction.answer_summary,
                    created_at=interaction.created_at,
                )
            )

    def search_interactions(self, user_id: UUID, query: str, limit: int = 10) -> tuple[Interaction, ...]:
        pattern = f"%{query.strip()}%"
        with self._session_factory() as session:
            rows = tuple(
                session.scalars(
                    select(InteractionRow)
                    .where(
                        InteractionRow.user_id == user_id,
                        or_(
                            InteractionRow.query.ilike(pattern),
                            InteractionRow.answer_summary.ilike(pattern),
                        ),
                    )
                    .order_by(InteractionRow.created_at.desc())
                    .limit(limit)
                )
            )
        return tuple(self._interaction(row) for row in rows)

    def save_note(self, note: Note) -> None:
        with self._session_factory.begin() as session:
            session.add(
                NoteRow(
                    note_id=note.note_id,
                    user_id=note.user_id,
                    project_id=note.project_id,
                    paper_id=note.paper_id,
                    section_id=note.section_id,
                    content=note.content,
                    tags_json=list(note.tags),
                )
            )

    def list_notes(self, user_id: UUID, project_id: UUID, limit: int = 50) -> tuple[Note, ...]:
        with self._session_factory() as session:
            rows = tuple(
                session.scalars(
                    select(NoteRow)
                    .where(NoteRow.user_id == user_id, NoteRow.project_id == project_id)
                    .order_by(NoteRow.updated_at.desc())
                    .limit(limit)
                )
            )
        return tuple(
            Note(
                note_id=row.note_id,
                user_id=row.user_id,
                project_id=row.project_id,
                paper_id=row.paper_id,
                section_id=row.section_id,
                content=row.content,
                tags=tuple(row.tags_json),
            )
            for row in rows
        )

    def set_preference(self, preference: UserPreference) -> None:
        statement = insert(UserPreferenceRow).values(
            user_id=preference.user_id,
            preference_key=preference.key,
            preference_value=preference.value,
        )
        statement = statement.on_conflict_do_update(
            index_elements=[UserPreferenceRow.user_id, UserPreferenceRow.preference_key],
            set_={"preference_value": preference.value},
        )
        with self._session_factory.begin() as session:
            session.execute(statement)

    def get_preferences(self, user_id: UUID) -> tuple[UserPreference, ...]:
        with self._session_factory() as session:
            rows = tuple(session.scalars(select(UserPreferenceRow).where(UserPreferenceRow.user_id == user_id)))
        return tuple(UserPreference(user_id=row.user_id, key=row.preference_key, value=row.preference_value) for row in rows)

    @staticmethod
    def _interaction(row: InteractionRow) -> Interaction:
        return Interaction(
            interaction_id=row.interaction_id,
            user_id=row.user_id,
            session_id=row.session_id,
            query=row.query,
            paper_ids=tuple(UUID(value) for value in row.paper_ids_json),
            topics=tuple(row.topics_json),
            interaction_type=row.interaction_type,
            retrieved_chunk_ids=tuple(UUID(value) for value in row.retrieved_chunk_ids_json),
            answer_summary=row.answer_summary,
            created_at=row.created_at,
        )
