"""Per-operation SQLAlchemy transaction boundary."""

from types import TracebackType
from typing import Self

from sqlalchemy.orm import Session, sessionmaker

from paper_agent.ingestion.ports import IngestionUnitOfWork
from paper_agent.storage.postgres.repositories import (
    SqlAlchemyDerivedDataRepository,
    SqlAlchemyIngestionRunRepository,
    SqlAlchemyPaperFileRepository,
    SqlAlchemyParsedDocumentRepository,
    SqlAlchemyProjectRepository,
)


class SqlAlchemyUnitOfWork:
    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self._session_factory = session_factory
        self._session: Session | None = None

    def __enter__(self) -> Self:
        self._session = self._session_factory()
        self.projects = SqlAlchemyProjectRepository(self._session)
        self.files = SqlAlchemyPaperFileRepository(self._session)
        self.documents = SqlAlchemyParsedDocumentRepository(self._session)
        self.derived = SqlAlchemyDerivedDataRepository(self._session)
        self.runs = SqlAlchemyIngestionRunRepository(self._session)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if self._session is None:
            return
        try:
            if exc_type is not None:
                self._session.rollback()
        finally:
            self._session.close()
            self._session = None

    def commit(self) -> None:
        self._require_session().commit()

    def rollback(self) -> None:
        self._require_session().rollback()

    def _require_session(self) -> Session:
        if self._session is None:
            raise RuntimeError("Unit of Work is not active")
        return self._session


class SqlAlchemyUnitOfWorkFactory:
    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self._session_factory = session_factory

    def __call__(self) -> IngestionUnitOfWork:
        return SqlAlchemyUnitOfWork(self._session_factory)
