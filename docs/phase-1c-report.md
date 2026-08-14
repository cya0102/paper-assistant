# Phase 1C 开发总结报告

## 阶段结论

Phase 1C 已把 Phase 1B 的 Canonical Parsed Document 转换为可供索引和检索使用的、版本化且可追溯的结构化论文数据。默认 `ingest` 流水线现在会执行到 `chunked`，并把 Section、Element、Semantic Group、Chunk 及派生版本状态写入 PostgreSQL。

## 本阶段新增功能

### Section Tree Domain、Schema 与 Builder

- 新增 `Section`、`StructuredDocument` Domain Model。
- 使用稳定 UUID 生成 Section ID，相同 Version 和相同输入可重复得到相同结果。
- 恢复显式 Heading Level、数字编号、罗马数字、字母编号和常见论文标题。
- 防止错误的 Heading Level 跳级。
- 保存 `parent_section_id`、`level`、`section_order`、`section_path`、页范围和来源 Heading Block。
- 标题前内容归入合成的 `Front Matter`。

### Block 到 Section 的归属

- Canonical Document 中每个 Block 按 Reading Order 唯一归属到一个 Section。
- Section 保存完整的 `source_block_ids`，用于后续重建和证据回溯。
- Chunker 以 Section 为硬边界，不跨 Section 或 Paper 合并内容。

### 基础 Element 模型

- 新增 Figure、Table、Equation、Algorithm 四类 Element。
- 支持从 Canonical Block Type 和 Caption 模式识别 Element。
- 保存 Label、Caption/Content、Page、BBox、来源 Block 和所属 Section。
- 当前是基础模型；图片裁切、表格网格重建、公式 OCR/LaTeX 和 Algorithm 结构恢复不在 Phase 1C 范围。

### Semantic Blocks 与 Dependency Groups

- 新增普通 `text` Group 和 `element_dependency` Group。
- Figure/Table/Algorithm 可绑定 Caption 和邻近引用正文。
- Equation 可绑定前置定义和后置 `where` 等变量解释。
- Heading 被明确视为 Dependency 边界，避免公式组误吞小节标题。
- Dependency Group 作为尽量不可拆的原子语义单元。

### Section-aware Semantic Chunker

- 默认目标长度 600 tokens、硬上限 800 tokens，均可配置。
- 语义完整性和 Dependency 约束优先于长度约束。
- 普通超长文本按句子和 token 安全降级拆分；中文无空格长文本也能受硬上限约束。
- Dependency Group 即使超过硬上限也保持原子性，这是有意的语义优先策略。
- Chunk ID 基于 Version、Chunking Version、Section 和来源 Group 确定性生成。

### 完整溯源

每个 Chunk 保存：

- `paper_id`
- `version_id`
- `section_id` 与 `section_path`
- `page_start` 与 `page_end`
- `source_group_ids`
- `source_block_ids`
- `related_element_ids`
- `chunking_version`

因此 Chunk 可以回到论文、具体版本、Section、页范围、Canonical Block 和 Element。

### Chunking Version、恢复与重建

新增 `derived_data_states`，分别跟踪：

- Structure Version
- Chunking Version

恢复规则：

1. Parser/Schema/Structure/Chunking 全部相同：复用 `chunked` 结果。
2. 只有 Chunking Version 变化：复用 Canonical Document、Section、Element 和 Semantic Group，只重建 Chunk。
3. Structure Version 变化：复用 Canonical Document，重建 Section/Element/Group 和所有下游 Chunk。
4. Parser 或 Canonical Schema Version 变化：重新解析 PDF，并重建全部下游数据。
5. `--force-reindex`：强制从 Metadata/Parsing 开始重新生成当前派生数据。

派生结构和 Chunk 使用事务内 replace 语义，避免同一 Version 混入两个算法版本的数据。

### PostgreSQL Migration 与 Repository

新增 Alembic Migration `0003_phase1c`，创建：

- `sections`
- `elements`
- `semantic_groups`
- `chunks`
- `derived_data_states`

新增派生数据 Repository，支持：

- 读取版本状态。
- 原子替换 Structure/Element/Group。
- 加载已生成结构以恢复后续阶段。
- 原子替换 Chunk。
- Domain Model 与 PostgreSQL Row 的往返转换。

## 验证结果

- `mypy --strict`：通过，41 个源码文件无类型错误。
- 常规测试：36 项通过；未配置测试数据库时 1 项真实 PostgreSQL 测试按预期跳过。
- 真实 PostgreSQL：使用 PostgreSQL 17.10 隔离实例执行通过。
- Migration：真实执行 `0001 → 0002 → 0003` 通过。
- Migration 往返：真实执行 `0003 → 0002 → 0003` 通过。
- Repository 集成：Section、Semantic Group、Chunk、Derived State 的写入、读取与溯源字段断言通过。
- 隔离 PostgreSQL 实例在测试结束后已停止，不影响现有数据库服务。

## 当前可使用程度

项目现在可以用于“论文导入与结构化预处理”：

```text
PDF 目录
→ 增量扫描与 SHA-256/正文去重
→ Paper/File/Version Identity
→ 真实 PDF Parsing
→ Canonical Parsed Document
→ Section Tree 与 Block 归属
→ 基础 Element 与 Dependency Group
→ Section-aware Semantic Chunk
→ PostgreSQL 可追溯持久化
```

可以直接通过 CLI 初始化项目、导入真实 PDF、重复增量运行、强制重建，并用 SQL 查看 Section 和 Chunk。当前还不能直接用自然语言搜索或问答，因为 Chunk 尚未生成 Embedding，也没有 BM25/Hybrid Retrieval 或对外 Search Tool。

## 下一阶段目标

下一阶段进入技术设计文档的第二轮：Paper/Section/Chunk Indexing 与 Retrieval Foundation。

计划目标：

1. Embedding Provider 与批量生成接口。
2. PostgreSQL `pgvector` Schema、索引与 Embedding Version。
3. Paper、Section、Chunk 分层索引。
4. PostgreSQL Full Text Search / BM25 候选召回。
5. Metadata Filter 与 Paper/Version/Section Scope。
6. Vector Search 与 Hybrid Retrieval。
7. Reranker Interface、Threshold、去重和 `no_evidence`。
8. 可追溯 Evidence Domain Model。
9. `search_knowledge` 内部服务接口与集成测试。

Agent Runtime、Session/Long-term Memory 和完整 Tool Calling 仍留到再后一个阶段。
