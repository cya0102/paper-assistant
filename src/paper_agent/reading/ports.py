"""Paper read repository boundary."""

from typing import Protocol

from paper_agent.domain.reading import ReadPaperRequest, ReadPaperResult


class PaperReadRepository(Protocol):
    def read(self, request: ReadPaperRequest) -> ReadPaperResult: ...

