# Phase 2A 开发总结报告

## 阶段结论

Phase 2A 已完成 Paper/Section/Chunk 分层索引和内部知识检索基础设施。默认 Ingestion Pipeline 现在从 PDF 一直执行到 `indexed`；`SearchKnowledgeService` 可以在 Project Scope 内执行 Metadata Resolution、分层 Dense Retrieval、PostgreSQL FTS/BM25、Hybrid Fusion、Rerank、Threshold 和去重，并返回完整可追溯 Evidence 或明确的 `no_evidence`。

项目版本已升级为 `paper-agent 0.4.0`。

## 本阶段新增功能

### Embedding Provider 与批量生成

- 新增独立于存储和检索实现的 `EmbeddingProvider` Protocol。
- Provider 公开 Provider、Model、Version、Dimension 描述。
- 支持一次输入多个文本并返回等量向量的批量接口。
- 默认实现 `HashingEmbeddingProvider`：256 维、离线、确定性、无需下载模型。
- 默认 Provider 使用 Token 和相邻 Token Feature Hashing、带符号累积和 L2 归一化。
- 相同文本和版本可以稳定生成相同向量，适合测试、离线运行和专有词基线召回。
- Repository 在写入前严格检查向量维度和批次数量。
- 接口允许后续替换为本地神经模型或远程 Embedding API。

### Embedding Version 与增量索引

- 新增 `EmbeddingDescriptor`、`IndexDocument`、`IndexedVector`、`IndexingState` 和 `IndexingReport` Domain Model。
- `embedding_version` 由 Provider、Model、Provider Version 和 Dimension 共同构成。
- `index_version` 独立跟踪 Paper/Section/Chunk 表示构造算法。
- 每个 Index Document 保存内容 SHA-256；同 Embedding Version 下未变化的向量可以复用。
- Embedding 以可配置 Batch Size 生成。
- Index State 保存源数据总摘要、Embedding Version、Index Version 和状态。
- 未变化 PaperVersion 重复导入时不重新 Parse、Chunk 或 Embed。
- 只有 Embedding/Index Version 变化时，只重建索引。
- Structure/Chunk Replace 会在同一事务中失效旧 Index State 和向量，避免陈旧索引被检索。

### pgvector Schema

新增 Alembic Migration `0004_phase2a`：

- 启用 PostgreSQL `vector` Extension。
- 新增 `embedding_configs`。
- 新增 `indexing_states`。
- 新增 `paper_embeddings`。
- 新增 `section_embeddings`。
- 新增 `chunk_embeddings`。
- 三类向量均使用 `vector(256)`。
- 三类向量均建立 HNSW `vector_cosine_ops` 索引。
- Vector Store 只保存派生数据，可从 Paper/Section/Chunk 完整重建。

为避免强制引入 `pgvector-python` Runtime Dependency，本项目实现了一个小型 SQLAlchemy `VectorType` Adapter，负责向 PostgreSQL 发送和读取标准 pgvector 文本格式。

### Paper、Section、Chunk 分层索引

- Paper Index 输入包含 Title、Aliases、Authors、Venue、Abstract、Section Path 和有限正文表示。
- Section Index 输入包含 Section Path 及该 Section 的 Chunk 内容。
- Chunk Index 输入包含 Section Path 和 Chunk Text。
- 所有索引记录保留 Project、Paper、Version、Section/Chunk Target 和源内容 Hash。
- 默认 Ingestion Pipeline 成功状态延伸为 `indexed`。

### PostgreSQL 全文检索与 BM25

- 为 `papers`、`sections` 和 `chunks` 新增持久化 `tsvector` 生成列。
- 三层 Source Table 均建立 GIN Full Text Index。
- Chunk Sparse Retrieval 先使用 PostgreSQL FTS 获取宽召回 Candidate。
- 对 FTS Candidate 使用标准 BM25 公式进行重评分。
- BM25 使用 Query Term Frequency、Document Frequency、Document Length 和 Average Document Length。
- 论文专有名词、模型缩写、数据集名称和公式/表格标签可通过 Sparse Path 提高召回。

### Metadata Filter 与 Scope

新增强类型：

- `SearchScope`
- `MetadataFilter`
- `SearchRequest`

Scope 支持：

- Project
- Paper IDs
- Version IDs
- Section IDs

Metadata Filter 支持：

- Year Range
- Venue
- Author
- Chunk Type

所有 SQL 查询都强制 Project Scope，并与 Paper File/Version 归属交叉验证，避免不同 Project 数据泄漏。

### Vector Search 与 Hybrid Retrieval

`SearchKnowledgeService` 的默认流程：

```text
Query Embedding
→ Exact/Contained Metadata Resolution
→ Paper Dense Retrieval
→ Section Dense Retrieval
→ Scoped Chunk Dense Retrieval
  + Scoped PostgreSQL FTS/BM25 Retrieval
→ Candidate Fusion
```

明确给出 Paper/Version/Section Scope 时，后续每一层都会保留该范围。没有明确 Paper 时，先通过 Paper-level Index 定位候选，再进入 Section 和 Chunk。

### Reranker、Threshold、去重和 no_evidence

- 新增可替换 `Reranker` Protocol。
- 默认 `LexicalHybridReranker` 综合 Dense Score、归一化 BM25 和 Query Token Coverage。
- 最终 Relevance 范围为 0～1。
- `SearchRequest` 可以覆盖默认 Relevance Threshold。
- 所有候选低于 Threshold 时返回 `SearchStatus.NO_EVIDENCE`。
- `no_evidence` 结果保证 `has_sufficient_evidence=false` 且 Evidence 为空。
- 对高 Token Jaccard 相似度的 Chunk 做近重复删除。
- 多 Paper Candidate 使用基础 Per-paper Quota，降低 Evidence 被单篇论文垄断的风险。

### 可追溯 Evidence

新增 `Evidence` Domain Model，包含：

- Evidence ID
- Paper ID / Version ID / Paper Title
- Section ID / Section Path
- Page Start / Page End
- Chunk ID
- Element IDs
- Evidence Text
- Dense Score
- BM25 Score
- Rerank Score
- Final Relevance

Evidence ID 基于规范化 Query Hash 和 Chunk ID 确定性生成。

### search_knowledge 内部服务

新增 `SearchKnowledgeService.search_knowledge(SearchRequest)`。

返回：

- Resolved Papers
- Traceable Evidence
- Search Status
- `has_sufficient_evidence`
- `no_evidence` Reason

该服务目前是内部 Application Service，不是 CLI、LLM Tool 或最终问答接口。

## 验证结果

- `mypy --strict`：54 个源码文件通过。
- 完整测试：44 项全部通过，包括真实 PostgreSQL 用例。
- PostgreSQL：17.10。
- pgvector：0.8.6。
- Migration `0003_phase1c → 0004_phase2a`：真实执行通过。
- pgvector Extension、三层 `vector(256)` 写入与读取：通过。
- Paper/Section/Chunk HNSW Index 存在性：通过。
- Index Version 幂等、Batch Generation 和单文档内容变化复用：通过。
- Ingestion Index Stage 独立恢复：通过，未重跑 Parser/Structure/Chunk。
- PostgreSQL FTS、BM25、Dense+Sparse Hybrid Retrieval：通过。
- Year/Venue Metadata Filter：通过。
- Evidence 完整溯源和 `no_evidence`：通过。

## 当前可使用程度

项目现在可以完成：

```text
PDF 目录
→ 增量摄取与身份解析
→ Canonical Parsed Document
→ Section / Element / Semantic Chunk
→ Paper / Section / Chunk Embedding
→ pgvector HNSW + PostgreSQL FTS/BM25
→ Metadata-scoped Hierarchical Hybrid Retrieval
→ Reranked Traceable Evidence / no_evidence
```

这意味着项目已具备论文知识检索后端，但还没有 Agent 对话入口。调用方可以通过内部 Python Service 检索 Evidence，再自行决定如何展示或传给 LLM。

## 下一阶段待实现内容

下一阶段目标是 Agent Runtime 与精确阅读/上下文层：

1. Agent Runtime 和可恢复 Agent Loop。
2. 最小 Tool Registry 与 Tool Contract。
3. 把内部 `search_knowledge` 封装成安全 Tool Adapter。
4. 实现 `read_paper`，支持 Section、Page、Figure、Table、Equation、Algorithm 和 Neighbor Chunk。
5. Query Understanding、Query Rewrite 与 Multi-query。
6. Neighbor Expansion 和可选 Evidence Judge。
7. Context Builder、Token Budget、Evidence Balance 和引用格式化。
8. Session State 与 Redis Short-term Memory。
9. Interaction Memory、Notes 和长期用户偏好。
10. LLM Provider Port、Tool Calling 与最终答案生成。
11. 生产级神经 Embedding Provider 和 Cross-Encoder Reranker 配置。
12. Agent/Search/Read 的端到端评估与集成测试。

该阶段仍会保持 Tool 数量最少，Parser、Chunker、Embedder 和 Reranker 不直接暴露给 LLM。
