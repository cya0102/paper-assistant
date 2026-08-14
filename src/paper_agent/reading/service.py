"""Application service for traceable section/page/element reading."""

from paper_agent.domain.reading import ReadPaperRequest, ReadPaperResult
from paper_agent.reading.ports import PaperReadRepository


class ReadPaperService:
    def __init__(self, repository: PaperReadRepository) -> None:
        self._repository = repository

    def read_paper(self, request: ReadPaperRequest) -> ReadPaperResult:
        return self._repository.read(request)

