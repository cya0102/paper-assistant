# Paper Agent

Paper Agent provides deterministic PDF ingestion, traceable semantic chunks,
hierarchical hybrid retrieval, precise paper reading, a recoverable model tool
loop, and an evidence-first Research Graph for structured paper comparison.

The technical source of truth is `paper-agent-technical-spec.md`.

## Implemented scope through Research Graph Foundation

- Separate `File`, `Paper`, and `PaperVersion` identities.
- Multiple project-relative file locations for one SHA-256 binary identity.
- Recursive, project-scoped PDF discovery.
- Streaming SHA-256 with changed-during-read detection.
- Pre-parse binary deduplication.
- Versioned Canonical Parsed Document JSON and Markdown persistence.
- PostgreSQL repositories and per-operation Unit of Work.
- Per-file failure isolation and ingestion run/item state.
- PyMuPDF canonical parser with a Poppler CLI fallback.
- Deterministic PDF metadata extraction with evidence and confidence.
- DOI, arXiv, title, author, Unicode, and whitespace normalization.
- Normalized-body content fingerprints that ignore repeated page edges and page numbers.
- Conservative Paper identity and Version resolution.
- `force_reindex`, parser/schema-version invalidation, retry, and missing-path detection.
- Concurrent SHA-256 insert protection through PostgreSQL `ON CONFLICT`.
- Project identity manifest and `init`, `ingest`, `status`, and `db-upgrade` commands.
- Versioned Section Tree with heading hierarchy recovery and Block ownership.
- Figure, Table, Equation, and Algorithm base Elements.
- Section-local semantic dependency groups and Section-aware chunks.
- Chunk provenance back to Paper, Version, Section, Page, Block, and Element.
- Structure/chunk version invalidation with stage-local derived-data rebuilds.
- Batch-oriented, replaceable Embedding Provider with a deterministic offline baseline.
- Paper, Section, and Chunk pgvector indexes with HNSW cosine search.
- PostgreSQL `tsvector`/GIN candidates with BM25 re-scoring.
- Project/Paper/Version/Section scope plus metadata filters.
- Hierarchical Dense + Sparse retrieval, reranking, threshold, and deduplication.
- Fully traceable Evidence and explicit `no_evidence` results.
- Embedding/index version invalidation without unnecessary parsing or chunking.
- Query rewrite, multi-query fusion, neighbor expansion, and Evidence Judge.
- Minimal JSON-Schema Tool Registry for `search_knowledge` and `read_paper`.
- Recoverable Agent Runtime with per-tool checkpoints.
- Token-budgeted, per-paper balanced context and citation validation.
- Redis Session/Agent state with TTL.
- PostgreSQL Interactions, Notes, and long-term Preferences.
- OpenAI stateful Responses, Xiaomi MiMo stateless Responses, OpenAI Embedding,
  and optional Cross-Encoder providers.
- `search`, `read`, and `ask` CLI commands.
- Versioned, queryable `PaperProfile` fields instead of an opaque Profile JSON blob.
- Claim, ResearchEntity, typed Paper/Entity Relation, and polymorphic EvidenceLink models.
- Claim–Evidence entailment boundary with an offline conservative baseline.
- Project-scoped Research Graph Repository with active/superseded history and relation deduplication.
- Deterministic offline PaperProfile extraction from existing Section-aware Chunks.
- Evidence-backed structured comparison with explicit `insufficient_evidence` cells.
- Agent `compare_papers` Tool and offline `profile-extract` / `compare` CLI commands.
- Derived structure state bound to the exact canonical Parsed Document hash.
- Section-bound Notes survive structure replacement through `ON DELETE SET NULL`.
- Content-addressed Artifact Store (SHA-256 blobs + gzip + atomic rename) under
  `.paper-agent/artifacts/` with a project-scoped PostgreSQL catalog.
- `ToolResult` refactor: checkpoints/Redis/Providers only ever see a compact
  `model_payload`, an `artifact_ref`, and a `citation_manifest`; full raw
  payloads are offloaded to the Artifact Store by `OffloadPolicy`.
- `read_artifact` / `search_artifact` Tools with bounded views, cursors, and
  token budgets; cross-project/expired/corrupt reads fail with stable errors.
- `compare_papers` / `read_paper` / `search_knowledge` compact model views with
  full results recoverable through Artifact hydration.
- ResearchTask / WorkUnit persistence (`research_tasks`, `work_units`) with
  stable generation keys for idempotent retries.
- Bounded single-layer delegation: `delegate_research` / `collect_research_task`
  Tools, `paper_analyzer` and `evidence_verifier` workers, and a synchronous
  dependency-aware Scheduler with one automatic retry.

The default ingestion/search path remains offline. Model API credentials are only
required for `ask` or when explicitly selecting an online Embedding provider.
Research Graph extraction and comparison also use offline deterministic baselines
by default.

Use Xiaomi MiMo for the Agent loop without changing the embedding provider:

```bash
export PAPER_AGENT_LLM_PROVIDER=mimo
export PAPER_AGENT_LLM_MODEL=mimo-v2.5-pro
export MIMO_API_KEY='your-real-key'
export MIMO_BASE_URL='https://api.xiaomimimo.com/v1'
uv run paper-agent ask --root /path/to/project 'What is the main method? Cite evidence.'
```

MiMo continuation is stateless: the Agent checkpoint persists and replays model
output items and tool results instead of sending `previous_response_id`.

## Development

Python 3.12 or newer is required.

```bash
uv sync --extra dev
uv run pytest
uv run mypy src
```

Set the PostgreSQL connection used by Alembic:

```bash
brew install pgvector
psql -U <database-admin> -d paper_agent -c 'CREATE EXTENSION IF NOT EXISTS vector;'
export PAPER_AGENT_DATABASE_URL='postgresql+psycopg://paper_agent:paper_agent@localhost:5432/paper_agent'
uv run alembic upgrade head
```

Initialize and ingest a project:

```bash
uv run paper-agent init --root /path/to/project
uv run paper-agent ingest --root /path/to/project papers/
uv run paper-agent status --root /path/to/project
uv run paper-agent search --root /path/to/project 'codebook construction'
uv run paper-agent profile-extract --root /path/to/project PAPER_ID
uv run paper-agent compare --root /path/to/project PAPER_ID_A PAPER_ID_B
```

Use `--force-reindex` to regenerate parsing and all current derived data. Parser,
Canonical Document schema, structure, chunking, embedding, and index versions
invalidate only the affected stage and its downstream data. A full-project scan
also marks previously known paths that no longer exist as `missing`.

Canonical parser artifacts are stored below:

```text
.paper-agent/parsed/{paper_id}/{version_id}/
├── document.json
├── document.md
└── assets/
```

The internal retrieval service is assembled with:

```python
from paper_agent.application import build_search_knowledge_service
from paper_agent.domain.retrieval import SearchRequest, SearchScope

service = build_search_knowledge_service(database_url=database_url)
result = service.search_knowledge(
    SearchRequest(query="How is the codebook built?", scope=SearchScope(project_id=project_id))
)
```

Build and compare Research Graph profiles without an LLM:

```python
from paper_agent.application import (
    build_comparison_service,
    build_research_graph_service,
)

graph = build_research_graph_service(database_url=database_url)
graph.extract_profile(project_id, first_paper_id)
graph.extract_profile(project_id, second_paper_id)

comparison = build_comparison_service(database_url=database_url).compare(
    project_id,
    (first_paper_id, second_paper_id),
)
print(comparison.status.value)
```

Every non-empty comparison cell carries EvidenceLink records back to the exact
PaperVersion, Section, Chunk, page range, source blocks, and evidence text. Missing
Profile/Claim evidence produces an explicit refusal cell rather than generated text.

## Retrieve + Offload + Delegate

`paper-agent ask` now defaults to a deterministic Retrieve-Offload-Delegate
path. `retrieve_and_analyze_knowledge` retrieves bounded Evidence, writes every
selected Chunk to its own content-addressed Artifact, and runs one isolated
`chunk_analyst` per Artifact with bounded parallelism. The main Agent receives
only short reports, Claims, Artifact references, and a Citation Manifest; it
never receives retrieved Chunk text. Evidence insufficiency permits one query
rewrite, then returns `no_evidence`. Use `--trace summary|jsonl` for stage
events and `--rag-mode direct` only for rollback/comparison. The explicit
`delegate` command remains available for advanced research workflows.

See [`docs/retrieve-offload-delegate.md`](docs/retrieve-offload-delegate.md) for the
full design and CLI usage.

The full staged plan is in
[`docs/research-graph-roadmap.md`](docs/research-graph-roadmap.md). The delivered
vertical slices are summarized in
[`docs/Phase4阶段总结.md`](docs/Phase4阶段总结.md) and
[`docs/retrieve-offload-delegate.md`](docs/retrieve-offload-delegate.md).
