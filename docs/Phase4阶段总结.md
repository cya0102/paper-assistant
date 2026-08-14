# Phase 4 阶段总结：研究图谱基础与证据化比较

## 阶段名称

**Phase 4：研究图谱基础与证据化比较（Research Graph Foundation & Evidence-backed Comparison）**

这个名称对应本阶段的实际交付边界：系统不再只在查询时临时检索 Chunk 并生成回答，而是开始把论文中的研究画像、可验证主张、研究实体、论文关系和证据链接沉淀为可查询、可追溯、可重建的研究知识层；同时交付一个严格以证据为输入的多论文比较纵切。

本阶段完成了 `Research Graph + Research Workflow` 路线图中的 Phase A 基础，并提前实现了 Phase B 的无持久化比较内核。持久化 Comparison Task 和完整 Literature Review Workflow 不属于本阶段已完成范围。

## 阶段目标

在不破坏既有 PDF 摄取、解析、索引、检索、阅读和问答链路的前提下，将 Paper Agent 从“论文 RAG 后端”推进为具备结构化研究知识层的科研助手，建立以下能力：

- 将论文内容提取为结构化 `PaperProfile`，而不是只保存一段摘要或不可查询的 JSON Blob。
- 将可独立验证的论文结论建模为 `Claim`，并区分证据支持、反驳、信息不足和未审阅状态。
- 将方法、数据集、指标、任务、机制等统一为可规范化的 `ResearchEntity`。
- 用 `PaperRelation` 表达论文或实体之间的有类型关系，并通过数据库约束和规范化 Key 避免重复边。
- 让所有 Profile 字段、Claim、Entity、Relation 和比较项都能通过 `EvidenceLink` 回链到原始论文证据。
- 保存提取算法、模型、Prompt、Schema、解析文档和 Chunking 版本，支持后续失效、重建和历史追溯。
- 所有 Graph 查询强制携带 `project_id`，避免跨项目读取。
- 默认提供离线、确定性的提取和蕴含判断实现，使测试和最小使用路径不依赖网络或在线模型。
- 提供可由 Agent 和 CLI 调用的证据化多论文比较能力；证据不足时明确拒绝生成内容。

## 核心设计原则

### Evidence First

自动提取或综合出的内容不能只有文本。每条有效证据保存：

- `project_id`
- `paper_id`
- `version_id`
- `section_id`
- `chunk_id`
- 可选 `element_id`
- `page_start` / `page_end`
- `source_block_ids`
- `evidence_text` 快照
- 证据与目标的关系、证据类型和置信度

Section 或 Chunk 在后续重新解析、重建时，历史证据文本和定位快照仍然保留，不因源结构变化而丢失生成依据。

### Claim-level Grounding

本阶段新增 `EntailmentJudge` 边界。Claim 的蕴含状态包括：

- `supported`
- `contradicted`
- `insufficient`
- `unreviewed`

只有经过判断且状态为 `supported` 的 Claim 才能进入 `verified`；引用编号存在本身不再被视为结论得到支持。

### Derivation Versioning

每个派生结果保存生成谱系，包括提取方式、Extractor Version、Schema Version、Prompt Version、Model Name、Canonical Parsed Document Hash 和 Chunking Version。

相同输入和相同生成版本通过稳定 Key 保持幂等。输入或生成版本变化时创建新结果，旧结果进入 `superseded`，不通过原地覆盖抹去历史。

### Project Isolation

Repository 和 Application Service 的 Graph 入口都要求显式传入 `project_id`。Paper 和 PaperVersion 的项目归属通过 `paper_files` 验证；Graph 查询、关系端点校验和数据库状态统计都按项目过滤。

### Offline by Default

Domain 层不依赖 SQLAlchemy 或具体模型 SDK。`PaperProfileExtractor` 和 `EntailmentJudge` 通过 Protocol 抽象，默认实现为可离线执行的规则提取器和词法蕴含判断器，在线 LLM Provider 可以后续替换接入。

## 已完成功能

### Research Graph Domain Model

新增以下核心领域对象：

- `GenerationProvenance`：保存所有生成和输入版本信息，并生成稳定 `generation_key`。
- `EvidenceLink`：保存目标、证据来源、定位、文本快照、证据类型和置信度。
- `PaperProfile` / `PaperProfileFieldValue`：按论文版本保存结构化研究画像及多值字段。
- `Claim`：保存主张类型、原始与归一化陈述、极性、置信度、提取谱系、审阅状态和蕴含状态。
- `ResearchEntity`：统一表达 task、research problem、method、method component、mechanism、dataset、metric、baseline、concept 和 domain。
- `RelationEndpoint` / `PaperRelation`：支持 Paper 与 Entity 两类端点及有向、无向关系。
- `ComparisonCell` / `ComparisonDimension` / `PaperComparisonResult`：表达结构化多论文比较和信息不足状态。

`PaperProfile` 的稳定字段采用行式记录，覆盖研究问题、动机、假设、贡献、方法、方法组件、假设条件、数据集、指标、基线、实验设置、关键结果、限制、失败案例和未来工作。可变扩展属性仍可使用映射结构，但不会替代稳定字段建模。

### Relation 语义与去重

关系类型覆盖：

- `cites` / `cited_by`
- `extends` / `improves` / `simplifies`
- `uses_method` / `uses_dataset` / `evaluates_on`
- `same_problem` / `different_assumption`
- `supports` / `contradicts`
- `reproduces` / `fails_to_reproduce`
- `analogous_to` / `inspired_by`

其中 `same_problem`、`different_assumption` 和 `analogous_to` 使用规范化的无向端点顺序生成 Relation Key；其余关系保留方向。数据库 active 唯一索引与 Repository 幂等逻辑共同避免重复边。

### PostgreSQL Schema 与 Migration

新增 Alembic Migration：

```text
0008_research_graph_foundation.py
revision: 0008_research_graph
down_revision: 0007_phase3
```

新增 7 张表：

1. `paper_profiles`
2. `paper_profile_fields`
3. `claims`
4. `research_entities`
5. `research_entity_aliases`
6. `paper_relations`
7. `evidence_links`

Schema 包含项目、Paper、Version 外键，置信度与来源 Hash 检查约束，Profile/Claim/Relation 的 active 部分唯一索引，以及 Entity Alias 独立索引结构。

### Repository Port 与 PostgreSQL Repository

新增 `ResearchGraphRepository` Protocol 和 `PostgresResearchGraphRepository`，实现：

- 保存和读取指定项目、论文和版本的 PaperProfile。
- 保存并按 Paper、Version、Claim Type 查询 Claim。
- 保存、规范化并按名称、别名和 Entity Type 查询 ResearchEntity。
- 保存 PaperRelation，并按 Paper、Entity、Relation Type 批量查询。
- 批量加载目标对象的 EvidenceLink，避免逐对象 N+1 查询。
- 在写入前验证 Paper、Version 和 Relation Endpoint 的项目归属。
- 在新结果激活时保留旧记录并写入 `superseded_by`。
- 对相同实体和关系执行幂等合并与去重。

### 离线结构化提取

新增 `PaperProfileExtractor` Protocol 和 `RuleBasedPaperProfileExtractor`。

规则提取器从已解析 Chunk 中识别明确的研究问题、方法、数据集、指标、结果、限制等线索，生成稳定 ID 的 Profile、Claim、Entity 和 Relation。它采用保守策略：没有明确文本线索时不补写事实，所有输出附带原始 Chunk 的 EvidenceLink。

新增 `ResearchGraphService` 负责：

```text
项目范围内加载 PaperVersion 和 Chunk
→ 调用 Extractor
→ 执行 Claim–Evidence 判断
→ 规范化 Entity / Relation Endpoint
→ 持久化 Profile、Claim、Entity、Relation 和 Evidence
```

### Claim–Evidence 验证

新增 `EntailmentJudge` Protocol、离线 `LexicalEntailmentJudge` 和 `ClaimVerificationService`。

词法基线仅在 Claim 与证据具备足够重合且未命中明显反向信号时给出支持判断；否则返回 `contradicted` 或 `insufficient`。该边界可替换为在线或本地神经模型，但具体 Provider 不进入 Domain 层。

### Evidence-backed Comparison Service

新增 `EvidenceBackedComparisonService`，输入至少两个同项目 `paper_id`，批量读取已持久化的 Profile 和 Claim，按以下维度生成结构化结果：

- research problem
- assumptions
- method
- datasets
- metrics
- experimental setting
- results
- advantages
- limitations

每个有内容的 Comparison Cell 必须附带至少一个 EvidenceLink。没有结构化内容或证据时，Cell 返回 `insufficient_evidence` 和原因，不调用模型猜测或补全。

当前比较结果是运行时结构化对象，尚未引入 `ComparisonTask`、`ComparisonDimension`、`ComparisonCell` 和 `SynthesisArtifact` 持久化表。

### Agent Tool 与 CLI

新增 Agent Tool：

```text
compare_papers
```

Tool Contract 要求传入 2～20 个 Paper ID，输出比较维度、Cell、可比性、缺失原因、置信度、证据和生成版本信息。证据同时转换为 Agent Runtime 可识别的稳定 Evidence Citation，继续复用现有最终答案引用校验。

新增 CLI：

```bash
paper-agent profile-extract PAPER_ID [--version-id VERSION_ID]
paper-agent compare PAPER_ID PAPER_ID [PAPER_ID ...]
```

CLI 输出结构化 JSON，可在不启动 LLM Agent 的情况下验证提取和比较链路。

### 应用装配与兼容性

应用层新增 Research Graph Service 和 Comparison Service Builder，并把 `compare_papers` 注册到现有 Agent Runtime。现有 `ingest`、`search`、`read` 和 `ask` 行为保持兼容。

`database_status` 改为要求 `project_id`，现有 Paper、Version、Chunk、Vector 统计和新增 Graph 统计都按项目隔离。

项目版本由 `0.5.1` 升级为 `0.6.0`。

## 最小使用流程

在数据库完成迁移并已有解析、Chunk 化的论文后：

```bash
uv run alembic upgrade head

uv run paper-agent profile-extract <paper-id-1>
uv run paper-agent profile-extract <paper-id-2>

uv run paper-agent compare <paper-id-1> <paper-id-2>
```

比较命令只使用当前项目内已经提取并持久化的结构化 Profile、Claim 和 Evidence。若论文尚未提取或某一维度没有证据，JSON 中会给出明确的信息不足状态。

配置在线模型后，Agent 也可以在工具循环中调用 `compare_papers`；在线模型只负责组织表达，结构化比较内容仍由 Evidence-backed Comparison Service 提供。

## 测试与验证

本阶段新增测试覆盖：

- Domain validation。
- EvidenceLink 定位和证据文本完整性。
- Claim 只有在 `supported` 时才能标记为 verified。
- 离线提取器的确定性和稳定 ID。
- 无向 Relation Key 规范化和关系去重。
- Repository Profile、Claim、Entity、Relation round-trip。
- 按 Paper、Entity、Relation Type 查询关系。
- Project 隔离和跨项目写入拒绝。
- Comparison 无证据拒绝。
- `compare_papers` Tool Contract 和 JSON 序列化。
- 现有 CLI、Agent Tool Registry 和 Schema 元数据回归。

最终验证结果：

```text
uv run pytest
73 passed, 6 skipped

uv run mypy src
Success: no issues found in 85 source files
```

6 个集成测试因本地未配置 `PAPER_AGENT_TEST_DATABASE_URL` 或 `PAPER_AGENT_TEST_REDIS_URL` 而跳过，其中包括 Research Graph PostgreSQL 集成测试。因此不能声称真实 PostgreSQL/Redis 集成测试已经通过；Domain、Service、Tool、Schema 元数据和所有不依赖外部服务的回归测试已通过。

Migration 另完成离线 SQL 生成检查：

```text
0007_phase3 → 0008_research_graph：通过
0008_research_graph → 0007_phase3：通过
```

## 主要新增文件

```text
docs/research-graph-roadmap.md
docs/Phase4阶段总结.md
migrations/versions/0008_research_graph_foundation.py
src/paper_agent/domain/research_graph.py
src/paper_agent/domain/comparison.py
src/paper_agent/research_graph/ports.py
src/paper_agent/research_graph/extractor.py
src/paper_agent/research_graph/entailment.py
src/paper_agent/research_graph/service.py
src/paper_agent/storage/postgres/research_graph_repository.py
tests/unit/research_graph/test_research_graph.py
tests/integration/test_postgres_research_graph.py
```

同时修改 Domain 导出和枚举、SQLAlchemy Models、Application Builder、Agent Tool Adapter、Prompt、CLI、数据库状态统计、README、使用说明、版本号及相关回归测试。

## 当前可使用程度

当前系统已形成以下可运行链路：

```text
PDF 增量摄取与身份去重
→ Canonical Parsed Document
→ Section / Element / Semantic Group / Chunk
→ Paper / Section / Chunk 分层索引
→ 混合检索、精确阅读与 Agent 问答
→ 论文结构化 Profile / Claim / Entity / Relation
→ Claim–Evidence 验证
→ Evidence-backed 多论文结构化比较
```

离线环境可执行结构化提取、验证和比较，不需要模型 API Key。当前适合本地研究知识整理、后端集成和后续 Research Workflow 开发。

## 尚未完成的路线图内容

以下能力已在 `research-graph-roadmap.md` 中定义边界，但本阶段没有用空实现冒充完成：

- 可替换的生产级 LLM Profile/Claim/Relation Extractor。
- Profile/Claim 人工审核、Entity 合并拆分和冲突处理界面。
- Citation/Reference 解析及自动 `cites` / `cited_by` 关系。
- 持久化 ComparisonTask、Dimension、Cell、SynthesisArtifact 和人工 Cell 修订。
- Literature Review ResearchTask、筛选决策、阅读状态、Evidence Matrix、共识、矛盾和研究空白。
- 基于机制和假设兼容性的 Cross-domain Inspiration 与 IdeaCard。
- DOI/arXiv/BibTeX/RIS、Crossref/OpenAlex/Semantic Scholar、Zotero 等外部发现与同步。
- OCR、Figure、Table Cell、Equation、Reference 和 PDF 原始区域定位。
- Graph/Comparison 专项评测集和 1,000～10,000 篇论文性能基准。
- HTTP API、后台任务队列、进度/暂停/取消/重试、可观测性、鉴权、ACL 和多租户生产能力。

## 下一阶段建议

下一条最合理的纵切是：

**Phase 5：持久化证据比较与人工修订（Persistent Comparison & Human Review）**

建议按以下顺序实施：

1. 新增 `ComparisonTask`、`ComparisonDimension`、`ComparisonCell` 和 `SynthesisArtifact` Domain 与 Migration。
2. 保存每次比较使用的 PaperVersion、Profile、Claim 和 Evidence 不可变快照。
3. 增加 Comparison Repository 和可恢复状态，支持失败重试和幂等生成。
4. 提供 `paper-agent compare --save`、历史比较读取和 Cell 人工修订入口。
5. 在 PaperVersion 或 Extractor Version 变化后标记结果 stale，并支持增量刷新。
6. 建立 Comparison Correctness、Evidence Coverage 和 `no_evidence` Accuracy 最小评测集。

这一纵切会把当前无持久化的比较内核升级为可审阅、可恢复、可复用的研究产物，并直接成为后续 Literature Review Evidence Matrix 的数据基础。
