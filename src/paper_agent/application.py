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
from paper_agent.artifacts.materializer import ToolResultMaterializer
from paper_agent.artifacts.policies import OffloadPolicy
from paper_agent.artifacts.service import ArtifactService
from paper_agent.agent.artifact_tool_adapters import (
    ReadArtifactToolAdapter,
    SearchArtifactToolAdapter,
)
from paper_agent.agent.delegation_tool_adapters import (
    CollectResearchTaskToolAdapter,
    DelegateResearchToolAdapter,
)
from paper_agent.delegation.collector import ResultCollector
from paper_agent.delegation.policy import DelegationPolicy
from paper_agent.delegation.scheduler import Scheduler
from paper_agent.delegation.runner import WorkerRunner
from paper_agent.research_tasks.planner import ResearchPlanner
from paper_agent.research_tasks.service import ResearchTaskService
from paper_agent.storage.local.artifact_blob_store import LocalArtifactBlobStore
from paper_agent.storage.postgres.artifact_repository import SqlAlchemyArtifactRepository
from paper_agent.storage.postgres.research_task_repository import SqlAlchemyResearchTaskRepository
from paper_agent.workers import build_worker_registry
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


def build_artifact_service(
    *, database_url: str, project_root: Path
) -> ArtifactService:
    """Assemble the local content-addressed Artifact stack for one project."""
    engine = create_engine(database_url, pool_pre_ping=True)
    session_factory = sessionmaker(engine, class_=Session, expire_on_commit=False)
    return ArtifactService(
        LocalArtifactBlobStore(project_root),
        SqlAlchemyArtifactRepository(session_factory),
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
    project_root: Path | None = None,
    user_id: UUID | None = None,
    session_id: UUID | None = None,
) -> AgentRuntime:
    project_root = (project_root or Path.cwd()).resolve()
    engine = create_engine(database_url, pool_pre_ping=True)
    session_factory = sessionmaker(engine, class_=Session, expire_on_commit=False)
    search_service = build_search_knowledge_service(database_url=database_url)
    read_service = build_read_paper_service(database_url=database_url)
    artifacts = ArtifactService(
        LocalArtifactBlobStore(project_root),
        SqlAlchemyArtifactRepository(session_factory),
    )
    policy = OffloadPolicy()
    materializer = ToolResultMaterializer(artifacts, policy)
    tools = ToolRegistry()
    tools.register(SearchKnowledgeToolAdapter(search_service, project_id).contract())
    tools.register(ReadPaperToolAdapter(read_service, project_id).contract())
    tools.register(
        ComparePapersToolAdapter(
            build_comparison_service(database_url=database_url), project_id
        ).contract()
    )
    tools.register(ReadArtifactToolAdapter(artifacts, project_id).contract())
    tools.register(SearchArtifactToolAdapter(artifacts, project_id).contract())
    llm = _language_model(provider=provider, model=model)
    redis_client: Redis = Redis.from_url(redis_url, decode_responses=True)
    if user_id is not None:
        task_service = build_research_task_service(
            database_url=database_url,
            project_root=project_root,
            redis_url=redis_url,
            model=llm,
            search_service=search_service,
            read_service=read_service,
            artifacts=artifacts,
            materializer=materializer,
        )
        tools.register(
            DelegateResearchToolAdapter(
                task_service, project_id, user_id, session_id
            ).contract()
        )
        tools.register(
            CollectResearchTaskToolAdapter(task_service, project_id).contract()
        )
    return AgentRuntime(
        llm,
        tools,
        RedisCheckpointStore(redis_client),
        answer_finalizer=ToolEvidenceCitationFormatter(),
        sessions=RedisSessionStore(redis_client),
        memory=SqlAlchemyMemoryRepository(session_factory),
        materializer=materializer,
    )


def build_research_task_service(
    *,
    database_url: str,
    project_root: Path,
    redis_url: str,
    model: LanguageModel,
    search_service: object,
    read_service: object,
    artifacts: ArtifactService,
    materializer: ToolResultMaterializer,
) -> ResearchTaskService:
    engine = create_engine(database_url, pool_pre_ping=True)
    session_factory = sessionmaker(engine, class_=Session, expire_on_commit=False)
    worker_checkpoints = RedisCheckpointStore(
        Redis.from_url(redis_url, decode_responses=True)
    )
    runner = WorkerRunner(
        registry=build_worker_registry(),
        model=model,
        checkpoints=worker_checkpoints,
        artifacts=artifacts,
        materializer=materializer,
        search_service=search_service,
        read_service=read_service,
    )
    return ResearchTaskService(
        repository=SqlAlchemyResearchTaskRepository(session_factory),
        planner=ResearchPlanner(),
        policy=DelegationPolicy(),
        scheduler=Scheduler(runner),
        collector=ResultCollector(artifacts),
        artifacts=artifacts,
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
