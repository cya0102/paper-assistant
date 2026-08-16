"""Application assembly for ingestion, retrieval, reading, and Agent Runtime."""

import os
from collections.abc import Callable
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
from paper_agent.agent.rod_tool_adapter import (
    RetrieveAndAnalyzeKnowledgeToolAdapter,
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
from paper_agent.rag import (
    EvidenceArtifactMaterializer,
    NullRagTracer,
    RagConfig,
    RagTraceEvent,
    RagTracer,
    RagWorkUnitPlanner,
    RetrieveOffloadDelegateAnswerFinalizer,
    RetrieveOffloadDelegateService,
    RodResultCollector,
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
    rag_mode: str | None = None,
    rag_tracer: RagTracer | None = None,
) -> AgentRuntime:
    project_root = (project_root or Path.cwd()).resolve()
    engine = create_engine(database_url, pool_pre_ping=True)
    session_factory = sessionmaker(engine, class_=Session, expire_on_commit=False)
    search_service = build_search_knowledge_service(database_url=database_url)
    read_service = build_read_paper_service(database_url=database_url)
    policy = OffloadPolicy()
    artifacts = ArtifactService(
        LocalArtifactBlobStore(project_root),
        SqlAlchemyArtifactRepository(session_factory),
        retention_days=policy.config.artifact_retention_days,
    )
    materializer = ToolResultMaterializer(artifacts, policy)
    llm = _language_model(provider=provider, model=model)
    redis_client: Redis = Redis.from_url(redis_url, decode_responses=True)
    tools = ToolRegistry()
    selected_rag_mode = (
        rag_mode
        or os.environ.get(
            "PAPER_AGENT_RAG_MODE", "retrieve-offload-delegate"
        )
    ).strip().lower()
    tracer = rag_tracer or NullRagTracer()
    required_tool_name: str | None = None
    answer_finalizer: object
    if selected_rag_mode == "retrieve-offload-delegate":
        if user_id is None or session_id is None:
            raise ValueError(
                "retrieve-offload-delegate mode requires user_id and session_id"
            )
        rod_service = build_retrieve_offload_delegate_service(
            database_url=database_url,
            project_root=project_root,
            redis_url=redis_url,
            model=llm,
            search_service=search_service,
            artifacts=artifacts,
            materializer=materializer,
            tracer=tracer,
            worker_model_factory=lambda: _language_model(
                provider=provider, model=model
            ),
        )
        tools.register(
            RetrieveAndAnalyzeKnowledgeToolAdapter(
                rod_service,
                project_id=project_id,
                user_id=user_id,
                session_id=session_id,
            ).contract()
        )
        required_tool_name = "retrieve_and_analyze_knowledge"
        answer_finalizer = RetrieveOffloadDelegateAnswerFinalizer()
    elif selected_rag_mode == "direct":
        tools.register(
            SearchKnowledgeToolAdapter(search_service, project_id).contract()
        )
        tools.register(ReadPaperToolAdapter(read_service, project_id).contract())
        tools.register(
            ComparePapersToolAdapter(
                build_comparison_service(database_url=database_url), project_id
            ).contract()
        )
        tools.register(
            ReadArtifactToolAdapter(
                artifacts,
                project_id,
                max_tokens_cap=policy.config.read_artifact_max_tokens,
            ).contract()
        )
        tools.register(SearchArtifactToolAdapter(artifacts, project_id).contract())
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
        answer_finalizer = ToolEvidenceCitationFormatter()
    else:
        raise ValueError(
            "rag_mode must be retrieve-offload-delegate or direct"
        )
    return AgentRuntime(
        llm,
        tools,
        RedisCheckpointStore(redis_client),
        answer_finalizer=answer_finalizer,
        sessions=RedisSessionStore(redis_client),
        memory=SqlAlchemyMemoryRepository(session_factory),
        materializer=materializer,
        required_tool_name=required_tool_name,
        answer_observer=lambda _answer, _results: tracer.emit(
            RagTraceEvent(
                event="rag.answer.validated",
                details={"rag_mode": selected_rag_mode},
            )
        ),
    )


def build_retrieve_offload_delegate_service(
    *,
    database_url: str,
    project_root: Path,
    redis_url: str,
    model: LanguageModel,
    search_service: object | None = None,
    artifacts: ArtifactService | None = None,
    materializer: ToolResultMaterializer | None = None,
    config: RagConfig | None = None,
    tracer: RagTracer | None = None,
    worker_model_factory: Callable[[], LanguageModel] | None = None,
) -> RetrieveOffloadDelegateService:
    """Assemble the standard RAG service without binding Domain to providers."""
    config = config or _rag_config_from_environment()
    engine = create_engine(database_url, pool_pre_ping=True)
    session_factory = sessionmaker(engine, class_=Session, expire_on_commit=False)
    policy = OffloadPolicy()
    artifacts = artifacts or ArtifactService(
        LocalArtifactBlobStore(project_root),
        SqlAlchemyArtifactRepository(session_factory),
        retention_days=policy.config.artifact_retention_days,
    )
    materializer = materializer or ToolResultMaterializer(artifacts, policy)
    search_service = search_service or build_search_knowledge_service(
        database_url=database_url
    )
    runner = WorkerRunner(
        registry=build_worker_registry(),
        model=model,
        checkpoints=RedisCheckpointStore(
            Redis.from_url(redis_url, decode_responses=True)
        ),
        artifacts=artifacts,
        materializer=materializer,
        search_service=search_service,
        read_service=build_read_paper_service(database_url=database_url),
        model_factory=worker_model_factory,
    )
    return RetrieveOffloadDelegateService(
        search=search_service,  # type: ignore[arg-type]
        repository=SqlAlchemyResearchTaskRepository(session_factory),
        scheduler=Scheduler(
            runner, max_attempts=2, max_workers=config.max_workers
        ),
        evidence_materializer=EvidenceArtifactMaterializer(artifacts),
        planner=RagWorkUnitPlanner(config),
        collector=RodResultCollector(artifacts),
        config=config,
        tracer=tracer,
    )


def _rag_config_from_environment() -> RagConfig:
    def integer(name: str, default: int) -> int:
        return int(os.environ.get(name, str(default)))

    return RagConfig(
        max_evidence=integer("PAPER_AGENT_RAG_MAX_EVIDENCE", 6),
        max_per_paper=integer("PAPER_AGENT_RAG_MAX_PER_PAPER", 2),
        max_workers=integer("PAPER_AGENT_RAG_MAX_WORKERS", 3),
        max_rounds=integer("PAPER_AGENT_RAG_MAX_ROUNDS", 2),
        worker_token_budget=integer(
            "PAPER_AGENT_RAG_WORKER_TOKEN_BUDGET", 1200
        ),
        worker_tool_call_budget=integer(
            "PAPER_AGENT_RAG_WORKER_TOOL_CALL_BUDGET", 2
        ),
        worker_timeout_seconds=integer(
            "PAPER_AGENT_RAG_WORKER_TIMEOUT_SECONDS", 90
        ),
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
