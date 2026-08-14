"""Application assembly for ingestion, retrieval, reading, and Agent Runtime."""

import os
from pathlib import Path
from uuid import UUID

from redis import Redis
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from paper_agent.ingestion.identity import DeterministicPaperIdentityResolver
from paper_agent.ingestion.chunker import SemanticChunker
from paper_agent.ingestion.parsers import PopplerPdfParser, PyMuPdfParser
from paper_agent.ingestion.pipeline import IngestionPipeline
from paper_agent.ingestion.structure_pipeline import DocumentStructureProcessor
from paper_agent.ingestion.ports import IngestionPdfParser
from paper_agent.indexing import HashingEmbeddingProvider, HierarchicalIndexingService
from paper_agent.retrieval import LexicalHybridReranker, SearchKnowledgeService
from paper_agent.retrieval.advanced import (
    AdvancedSearchKnowledgeService,
    ConservativeQueryRewriter,
    LexicalEvidenceJudge,
)
from paper_agent.reading import ReadPaperService
from paper_agent.agent import AgentRuntime, ToolRegistry
from paper_agent.agent.context_builder import ToolEvidenceCitationFormatter
from paper_agent.agent.tool_adapters import (
    ComparePapersToolAdapter,
    ReadPaperToolAdapter,
    SearchKnowledgeToolAdapter,
)
from paper_agent.memory import RedisCheckpointStore, RedisSessionStore
from paper_agent.agent.ports import LanguageModel
from paper_agent.providers import (
    MimoResponsesModel,
    OpenAIEmbeddingProvider,
    OpenAIResponsesModel,
)
from paper_agent.providers.neural_reranker import CrossEncoderReranker
from paper_agent.storage.local import LocalParsedDocumentStore
from paper_agent.storage.postgres import SqlAlchemyUnitOfWorkFactory
from paper_agent.storage.postgres.index_repository import SqlAlchemyIndexRepository
from paper_agent.storage.postgres.search_repository import SqlAlchemySearchRepository
from paper_agent.storage.postgres.read_repository import SqlAlchemyPaperReadRepository
from paper_agent.storage.postgres.neighbor_repository import SqlAlchemyNeighborRepository
from paper_agent.storage.postgres.memory_repository import SqlAlchemyMemoryRepository
from paper_agent.storage.postgres.research_graph_repository import (
    SqlAlchemyResearchGraphRepository,
)
from paper_agent.research_graph import (
    EvidenceBackedComparisonService,
    LexicalEntailmentJudge,
    ResearchGraphService,
    RuleBasedPaperProfileExtractor,
)


def build_ingestion_pipeline(
    *,
    project_root: Path,
    database_url: str,
    parser_name: str = "auto",
) -> IngestionPipeline:
    engine = create_engine(database_url, pool_pre_ping=True)
    session_factory = sessionmaker(engine, class_=Session, expire_on_commit=False)
    parser = select_parser(parser_name)
    embedding_provider = _embedding_provider()
    indexer = HierarchicalIndexingService(
        SqlAlchemyIndexRepository(session_factory), embedding_provider
    )
    return IngestionPipeline(
        unit_of_work_factory=SqlAlchemyUnitOfWorkFactory(session_factory),
        metadata_extractor=parser,
        identity_resolver=DeterministicPaperIdentityResolver(parser_version=parser.version),
        parser=parser,
        parsed_document_store=LocalParsedDocumentStore(project_root),
        structure_processor=DocumentStructureProcessor(),
        chunker=SemanticChunker(),
        indexer=indexer,
    )


def build_search_knowledge_service(*, database_url: str) -> AdvancedSearchKnowledgeService:
    engine = create_engine(database_url, pool_pre_ping=True)
    session_factory = sessionmaker(engine, class_=Session, expire_on_commit=False)
    base = SearchKnowledgeService(
        repository=SqlAlchemySearchRepository(session_factory),
        provider=_embedding_provider(),
        reranker=_reranker(),
    )
    return AdvancedSearchKnowledgeService(
        base,
        ConservativeQueryRewriter(),
        LexicalEvidenceJudge(),
        SqlAlchemyNeighborRepository(session_factory),
    )


def build_read_paper_service(*, database_url: str) -> ReadPaperService:
    engine = create_engine(database_url, pool_pre_ping=True)
    factory = sessionmaker(engine, class_=Session, expire_on_commit=False)
    return ReadPaperService(SqlAlchemyPaperReadRepository(factory))


def build_research_graph_service(*, database_url: str) -> ResearchGraphService:
    engine = create_engine(database_url, pool_pre_ping=True)
    factory = sessionmaker(engine, class_=Session, expire_on_commit=False)
    return ResearchGraphService(
        SqlAlchemyResearchGraphRepository(factory),
        RuleBasedPaperProfileExtractor(),
        LexicalEntailmentJudge(),
    )


def build_comparison_service(*, database_url: str) -> EvidenceBackedComparisonService:
    engine = create_engine(database_url, pool_pre_ping=True)
    factory = sessionmaker(engine, class_=Session, expire_on_commit=False)
    return EvidenceBackedComparisonService(SqlAlchemyResearchGraphRepository(factory))


def build_agent_runtime(
    *,
    project_id: UUID,
    database_url: str,
    redis_url: str,
    model: str,
    provider: str = "openai",
) -> AgentRuntime:
    tools = ToolRegistry()
    tools.register(SearchKnowledgeToolAdapter(build_search_knowledge_service(database_url=database_url), project_id).contract())
    tools.register(ReadPaperToolAdapter(build_read_paper_service(database_url=database_url), project_id).contract())
    tools.register(
        ComparePapersToolAdapter(
            build_comparison_service(database_url=database_url), project_id
        ).contract()
    )
    redis_client: Redis = Redis.from_url(redis_url, decode_responses=True)
    engine = create_engine(database_url, pool_pre_ping=True)
    memory_factory = sessionmaker(engine, class_=Session, expire_on_commit=False)
    return AgentRuntime(
        _language_model(provider=provider, model=model),
        tools,
        RedisCheckpointStore(redis_client),
        answer_finalizer=ToolEvidenceCitationFormatter(),
        sessions=RedisSessionStore(redis_client),
        memory=SqlAlchemyMemoryRepository(memory_factory),
    )


def _language_model(*, provider: str, model: str) -> LanguageModel:
    normalized = provider.strip().lower()
    if normalized == "openai":
        return OpenAIResponsesModel(model=model)
    if normalized == "mimo":
        api_key = (
            os.environ.get("PAPER_AGENT_LLM_API_KEY")
            or os.environ.get("MIMO_API_KEY")
            or os.environ.get("XIAOMI_MIMO_API_KEY")
        )
        if not api_key:
            raise ValueError(
                "Set MIMO_API_KEY (or PAPER_AGENT_LLM_API_KEY) for the MiMo provider"
            )
        base_url = (
            os.environ.get("PAPER_AGENT_LLM_BASE_URL")
            or os.environ.get("MIMO_BASE_URL")
            or os.environ.get("XIAOMI_MIMO_BASE_URL")
            or "https://api.xiaomimimo.com/v1"
        )
        if not base_url.startswith(("https://", "http://")):
            raise ValueError("MiMo base URL must be a plain http(s) URL")
        return MimoResponsesModel(model=model, api_key=api_key, base_url=base_url)
    raise ValueError(f"Unsupported LLM provider: {provider}")


def _embedding_provider() -> HashingEmbeddingProvider | OpenAIEmbeddingProvider:
    if os.environ.get("PAPER_AGENT_EMBEDDING_PROVIDER", "hashing") == "openai":
        return OpenAIEmbeddingProvider(
            model=os.environ.get("PAPER_AGENT_EMBEDDING_MODEL", "text-embedding-3-small"),
            dimension=256,
        )
    return HashingEmbeddingProvider()


def _reranker() -> LexicalHybridReranker | CrossEncoderReranker:
    if os.environ.get("PAPER_AGENT_RERANKER_PROVIDER", "lexical") == "cross-encoder":
        return CrossEncoderReranker(os.environ.get("PAPER_AGENT_RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"))
    return LexicalHybridReranker()


def select_parser(parser_name: str) -> IngestionPdfParser:
    if parser_name == "pymupdf":
        return PyMuPdfParser()
    if parser_name == "poppler":
        return PopplerPdfParser()
    if parser_name != "auto":
        raise ValueError(f"Unsupported parser: {parser_name}")
    if PyMuPdfParser.is_available():
        return PyMuPdfParser()
    if PopplerPdfParser.is_available():
        return PopplerPdfParser()
    raise RuntimeError("No PDF parser is available; install PyMuPDF or Poppler")
