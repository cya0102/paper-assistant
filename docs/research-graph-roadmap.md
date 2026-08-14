# Research Graph + Research Workflow 路线图

本文档定义 Paper Agent 从检索问答后端演进为“Research Graph + Research Workflow”科研助手的领域边界、数据生命周期和分阶段交付顺序。它补充现有 `paper-agent-technical-spec.md`，不改变 PDF、Canonical Parsed Document、Section/Element/Chunk 作为证据源的既有语义。

## 1. 目标与不变量

Research Graph 层把论文中的研究问题、方法、实验、主张和关系持久化为可查询、可比较、可审阅的数据；Research Workflow 层在这些数据之上保存持续调研任务、筛选决策、证据矩阵和报告快照。

所有阶段共同遵守以下不变量：

- Evidence First：自动生成的 Profile 字段、Claim、Entity、Relation、Comparison Cell、Research Gap 和 IdeaCard 必须保留论文、版本、Section、Chunk、可选 Element、页面、Source Block 和证据文本快照。
- Claim-level Grounding：引用存在不等于蕴含成立；Claim 的证据状态明确区分 `supported`、`contradicted`、`insufficient` 和 `unreviewed`。只有 `supported` 可以进入 `verified`。
- Derivation Versioning：派生记录保存算法/提取方式、Extractor、Schema、Prompt、Model、Canonical Document Hash 和 Chunking Version。重建生成新记录，旧记录进入 `superseded`，不原地抹去生成历史。
- Project Isolation：所有服务和 Repository 的入口以 `project_id` 为必填边界；Paper/Version 的项目归属通过 `paper_files` 验证，不能依赖全局 `papers.canonical_version_id`。
- Offline by Default：Domain 和应用服务不绑定在线模型；默认 Rule-based Extractor 和 Lexical Entailment Judge 可离线运行。
- Batch and Idempotency：批量加载论文/证据、批量写入派生记录；使用稳定 Key、唯一约束和 active/superseded 语义保证重复执行安全。

## 2. 分层架构

```text
PDF / Canonical Parsed Document
        ↓
Section / Element / Chunk（现有证据源）
        ↓
Research Graph Extraction
        ├── PaperProfile + ProfileField
        ├── Claim + Claim–Evidence status
        ├── ResearchEntity + Alias
        └── PaperRelation + EvidenceLink
        ↓
Evidence-backed Comparison
        ↓
Literature Review Workflow
        ↓
Cross-domain Mechanism Transfer
        ↓
API / Queue / Evaluation / Collaboration
```

Domain 层只包含不可依赖 SQLAlchemy 的模型与规则；Extractor、Entailment Judge、外部元数据、OCR/视觉解析和 Novelty Check 都通过 Protocol/Port 接入；PostgreSQL Repository 负责项目归属检查、持久化、去重和批量查询。

## 3. Phase A：Research Graph Foundation

### 3.1 领域模型

- `PaperProfile` 是一次针对特定 PaperVersion 的版本化提取结果。
- `PaperProfileField` 使用行式结构保存稳定字段；`research_problem`、`method_name`、`datasets`、`metrics`、`key_results` 等不是一个不可查询 JSON Blob。字段允许多值和有序值。
- `Claim` 保存独立可验证主张、归一化文本、极性、置信度、提取谱系、人工审阅与蕴含状态。
- `ResearchEntity` 统一 task/problem/method/component/mechanism/dataset/metric/baseline/concept/domain；Alias 单独建表，支持规范化和反向查询。
- `PaperRelation` 的端点可以是 Paper 或 ResearchEntity。`same_problem`、`different_assumption`、`analogous_to` 采用无向规范化 Key，其余关系保留方向。
- `EvidenceLink` 是多态目标链接，保存原始定位和证据文本快照；Section/Chunk 后续重建时，历史证据快照仍保留。

### 3.2 版本与失效

每次 Profile/Claim/Relation 生成包含 `generation_key`。相同输入和生成版本重复执行为幂等；相同逻辑目标但生成谱系变化时，新记录成为 active，旧记录记录 `superseded_by`。Entity 是项目内规范化概念，使用 canonical key 合并，别名和证据只追加。

### 3.3 本阶段纵切

本轮实现：核心 Domain、7 张 PostgreSQL 表、Alembic `0008`、Graph Repository Port/SQLAlchemy Repository、Rule-based Extractor、Lexical Entailment Judge、Extraction Service、Comparison Service、`compare_papers` Tool，以及 `profile-extract`/`compare` CLI。

后续 Phase A 增强：可替换 LLM Extractor Provider、批量后台提取任务、人工审阅接口、实体合并/拆分、Citation/Reference 解析生成的 `cites` 边，以及增量失效调度。

## 4. Phase B：Evidence-backed Comparison

新增持久化聚合：

- `ComparisonTask`：项目、输入论文/版本快照、状态、创建者和生成谱系。
- `ComparisonDimension`：research problem、assumptions、method、datasets、metrics、experimental setting、results、advantages、limitations。
- `ComparisonCell`：规范化值、原始描述、可比性、不可比原因、置信度、人工修订状态和 EvidenceLink。
- `SynthesisArtifact`：结构化表格、叙述性综合、使用的 Cell/Claim/版本快照和生成谱系。

本轮的 Comparison Service 是 Phase B 的无持久化最小内核：只读取已存 Profile/Claim；有内容的 Cell 必须带 Evidence；缺少证据时返回 `insufficient_evidence`，不调用模型补写。下一纵切将把任务、维度、Cell 和 Artifact 落库，并支持人工修订和增量刷新。

## 5. Phase C：Literature Review Workflow

按顺序实现：

1. `ResearchTask`、`ResearchQuestion`、Inclusion/Exclusion Criterion、Search Strategy。
2. Candidate Paper 与 `ScreeningDecision`，保存 included/excluded/pending、原因、操作者和时间。
3. `PaperReviewState`，保存阅读状态、优先级、目标 PaperVersion 和 Profile 新鲜度。
4. Evidence Matrix，以 Comparison Cell/Claim/EvidenceLink 为输入生成方法分类、时间线、共识和矛盾。
5. `ResearchGap` 与 `SynthesisArtifact`，明确区分论文事实、跨论文综合和系统推断。
6. Corpus 更新后按 PaperVersion、Extractor/Schema Version 和证据 Hash 增量刷新；报告保留不可变论文版本与证据快照。

调研任务采用可恢复状态机：`created → searching → screening → extracting → synthesizing → completed/failed/cancelled`。每个阶段可重试，不重复覆盖已确认的人工决策。

## 6. Phase D：Cross-domain Inspiration

先建立机制级表示，而不是直接跨库向量 Top-K：

```text
Problem → Constraint → Mechanism → Representation
        → Optimization Objective → Assumption → Applicability Condition
```

实现顺序：

1. 从 Claim/Profile 提取带证据的 Mechanism Graph。
2. 候选生成同时使用机制结构匹配、实体规范化和检索召回。
3. 排除相同 Domain 的词面近重复；保留机制相似但对象/表示不同的候选。
4. `AssumptionCompatibilityJudge` 检查目标问题与来源机制的适用条件。
5. 生成 `InnovationHypothesis/IdeaCard`，包含来源域、目标问题、迁移机制、类比、不兼容假设、预期收益、失败模式、最小验证实验、支持/反证、Novelty Check 状态和置信度。

IdeaCard 的字段必须标注 `paper_fact`、`system_inference` 或 `unverified_hypothesis`。Novelty Check 未完成时不得宣称“新颖”。

## 7. Phase E：External Discovery and Multimodal

核心层只定义 Port：

- `BibliographicImportProvider`：DOI、arXiv URL、BibTeX、RIS。
- `ScholarlyMetadataProvider`：Crossref/OpenAlex/Semantic Scholar 等元数据和引用发现。
- `LibrarySyncProvider`：Zotero 导入导出。
- `DocumentOcrProvider`、`FigureUnderstandingProvider`、`TableStructureProvider`、`EquationRecognitionProvider`。
- `ReferenceParser` 与 `CitationLocator`：参考文献解析、正文引用到 PDF 原始区域定位。
- `DiscoverySubscriptionProvider`：新论文、新版本、被引变化提醒。

外部 Provider 输出先转换为内部 DTO，再进入身份解析/证据模型；Provider 响应、请求版本、缓存键和授权范围不泄漏进 Domain。

## 8. Phase F：Evaluation and Production

### 8.1 质量评测

建立版本化 Evaluation Case/Run/Metric Port，最小评测集覆盖：Paper Recall@K、Section/Chunk Recall@K、NDCG、Citation Precision、Evidence Coverage、Claim–Evidence Entailment、Relation Extraction Accuracy、Comparison Correctness、`no_evidence` Accuracy，以及跨领域候选的新颖性/可迁移性人工评分。

每个评测结果固定语料快照、PaperVersion、Parser/Chunk/Index/Extractor/Model/Prompt 版本。性能基准分 1k、5k、10k 论文，记录摄取吞吐、增量刷新耗时、P50/P95/P99 查询延迟、数据库大小和峰值内存。

### 8.2 生产化

- HTTP API 与流式回答；结构化 Tool/Comparison/Workflow 响应。
- 后台任务队列，支持进度、暂停、取消、重试和幂等 Job Key。
- Provider 限流、批处理、缓存、退避和熔断。
- OpenTelemetry Trace、Tool Span、Token/成本和派生数据血缘观测。
- 用户认证、Project ACL、多人批注和多租户隔离；数据库查询层持续以 `project_id` 作为强制条件。

## 9. 建议的后续纵切

第一纵切完成后，最合理的下一条纵切是“持久化 Comparison Task + 人工修订”：新增 ComparisonTask/Dimension/Cell/SynthesisArtifact Migration 和 Repository；让当前确定性 Comparison Service 保存不可变输入版本快照；增加 `paper-agent compare --save`、读取历史比较、Cell 修订和回归评测。它能直接复用本轮 Profile/Claim/EvidenceLink，同时为 Phase C 的 Evidence Matrix 提供稳定数据结构。

## 10. 验收门槛

每个纵切必须同时具备 Domain validation、Migration、Repository、Application Service、至少一个无模型入口、序列化契约、离线测试和文档；涉及数据库唯一性/隔离/事务的行为必须有 PostgreSQL 集成测试。未执行的外部服务测试必须明确报告，不能以空实现或跳过测试冒充完成。
