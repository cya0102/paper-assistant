# Paper Agent 开发日志

> 项目：`paper-agent`  
> 当前版本：`0.4.0`  
> 记录日期：2026-08-12  
> 当前阶段：Phase 2A 完成，论文导入、结构化、分层索引与内部混合检索链路可用

## 1. 项目目标与开发原则

Paper Agent 的目标是构建一个面向本地论文库的、可增量恢复且证据可追溯的知识处理后端。系统需要把本地 PDF 逐步转换为论文身份、结构化章节、语义块、向量索引和检索证据，同时保留从检索结果回到论文、版本、章节、页面、原始 Block 和 Element 的完整链路。

开发过程中逐步形成了以下原则：

1. **身份分层**：区分 Project、Paper、PaperVersion、PaperFile 和 FileLocation，避免把“论文”“论文版本”“本地 PDF 文件”和“文件路径”混为一谈。
2. **增量执行**：解析器、Canonical Schema、Structure、Chunking、Embedding 和 Index 各自有版本，版本未变化时复用已有结果。
3. **派生数据可重建**：Section、Chunk 和 Embedding 等数据均可从上游数据重新生成，不把派生结果当成唯一事实来源。
4. **逐文件失败隔离**：一个 PDF 失败不应中断整个目录的摄取。
5. **事务一致性**：一个阶段的数据使用原子替换，避免数据库中出现一半新数据、一半旧数据。
6. **证据可追溯**：检索结果必须能定位到具体论文、版本、Section、Page、Block、Element 和 Chunk。
7. **接口可替换**：Parser、Embedding Provider、Reranker 和 Repository 通过清晰边界组装，便于后续替换实现。

## 2. 开发历程

### 2.1 Phase 1A：身份与摄取骨架

Phase 1A 建立了最初的数据库和摄取骨架。Alembic Migration `0001_phase1` 创建了：

- `projects`
- `papers`
- `paper_versions`
- `paper_files`
- `paper_file_locations`
- `parsed_documents`
- `ingestion_runs`
- `ingestion_items`

这一阶段确定了核心数据关系：

```text
Project
├── PaperFile
│   └── FileLocation
└── IngestionRun
    └── IngestionItem

Paper
└── PaperVersion
    ├── PaperFile
    └── ParsedDocument
```

同时建立了项目范围扫描、SHA-256 文件身份、导入运行记录和逐文件错误记录等基础能力。

### 2.2 Phase 1B：真实 PDF 增量导入

Phase 1B 将骨架扩展为可以处理真实 PDF 的增量导入 MVP，新增 Migration `0002_phase1b`。

主要完成内容：

- PyMuPDF 默认解析器和 Poppler CLI 后备解析器。
- 页面、文本块、BBox、阅读顺序和基础 Heading/Paragraph 提取。
- PDF Title、Authors、Subject、Keywords、Page Count 元数据提取。
- DOI、arXiv ID 和年份识别。
- Unicode、标题、作者、DOI、arXiv 和空白规范化。
- 二进制 `file_hash` 与正文 `content_hash` 双层去重。
- Paper 与 PaperVersion 的保守身份解析。
- Canonical Parsed Document 的 JSON、Markdown 和 assets 本地持久化。
- Parser Version 和 Canonical Schema Version 失效机制。
- `--force-reindex`、失败重试和 missing-path 检测。
- `.paper-agent/project.json` 稳定项目身份。
- `init`、`ingest`、`status`、`db-upgrade` CLI。

此时系统已经能够完成：

```text
PDF → 扫描 → 文件指纹 → 元数据 → 论文身份 → Canonical Document
```

但尚未构建 Section、Chunk 和检索索引。

### 2.3 Phase 1C：论文结构与语义分块

Phase 1C 新增 Migration `0003_phase1c`，把 Canonical Parsed Document 转换为可检索的结构化论文数据。

新增表：

- `sections`
- `elements`
- `semantic_groups`
- `chunks`
- `derived_data_states`

主要实现：

- Section Tree Domain、Schema 和 Builder。
- 显式 Heading Level、数字编号、罗马数字、字母编号和常见论文标题识别。
- Heading 层级恢复和跳级保护。
- Block 按 Reading Order 唯一归属到 Section。
- Figure、Table、Equation、Algorithm 基础 Element。
- 普通文本组和 Element Dependency Group。
- Section-aware Semantic Chunker。
- 默认 600 token 目标长度和 800 token 硬上限。
- Chunk 到 Paper、Version、Section、Page、Block、Element 的完整溯源。
- Structure Version 和 Chunking Version 独立失效。
- Structure、Group 和 Chunk 的事务化原子替换。

Phase 1C 完成后的流水线为：

```text
Canonical Document
→ Section Tree
→ Element / Semantic Group
→ Section-aware Chunk
→ PostgreSQL 可追溯持久化
```

这一阶段开始使用真实 PostgreSQL 17.10 环境执行 Migration 往返和 Repository 集成测试，补足了 Phase 1B 主要依赖离线 DDL 验证的不足。

### 2.4 Phase 2A：分层索引与混合检索

Phase 2A 新增 Migration `0004_phase2a`，项目版本升级至 `0.4.0`。

新增数据结构：

- `embedding_configs`
- `indexing_states`
- `paper_embeddings`
- `section_embeddings`
- `chunk_embeddings`
- `papers.search_vector`
- `sections.search_vector`
- `chunks.search_vector`

主要实现：

- 可替换的 `EmbeddingProvider` Protocol。
- 默认 256 维、离线、确定性的 `HashingEmbeddingProvider`。
- Embedding 批量生成和内容哈希复用。
- 独立的 `embedding_version` 与 `index_version`。
- Paper、Section、Chunk 三层向量表示。
- pgvector `vector(256)` 和 HNSW cosine 索引。
- PostgreSQL `tsvector` 持久化生成列与 GIN 索引。
- FTS 候选召回和 BM25 重评分。
- Project、Paper、Version、Section 检索范围控制。
- Year、Venue、Author、Chunk Type 元数据过滤。
- Dense + Sparse Candidate Fusion。
- `LexicalHybridReranker`、相关性阈值、近重复去除和 per-paper quota。
- 明确的 `no_evidence` 返回语义。
- 可追溯 `Evidence` Domain。
- 内部 `search_knowledge` Application Service。

Phase 2A 完成后的完整链路为：

```text
PDF 目录
→ 增量摄取与身份解析
→ Canonical Parsed Document
→ Section / Element / Semantic Group / Chunk
→ Paper / Section / Chunk Embedding
→ pgvector HNSW + PostgreSQL FTS/BM25
→ Metadata-scoped Hybrid Retrieval
→ Reranked Traceable Evidence / no_evidence
```

默认 Hashing Embedding 适合离线测试、确定性基线和专有名词召回，但不是最终生产级神经语义模型。

## 3. 数据库迁移记录

| Migration | 阶段 | 主要作用 |
|---|---|---|
| `0001_phase1` | Phase 1A | Project、Paper、Version、File、Location、Parsed Document 和 Ingestion 基础表 |
| `0002_phase1b` | Phase 1B | PDF 元数据、规范化身份、正文哈希、页数和解析证据 |
| `0003_phase1c` | Phase 1C | Section、Element、Semantic Group、Chunk 和派生版本状态 |
| `0004_phase2a` | Phase 2A | pgvector、三层 Embedding、索引状态、FTS/GIN 和 HNSW |

实际项目数据库最终升级到：

```text
0004_phase2a
```

迁移经验：已有项目不需要删除数据库或重新 `init`。正确流程是先执行 `db-upgrade`，再运行普通全项目 `ingest`，让版本状态机制补建新增派生数据。只有上游解析产物损坏或明确需要全部刷新时才使用 `--force-reindex`。

## 4. 开发与联调过程中遇到的问题

### 4.1 把 SQL 直接输入 zsh

最初在 shell 中直接执行：

```text
CREATE USER paper_agent
```

zsh 报告 `command not found: CREATE`。原因是 `CREATE USER` 是 PostgreSQL SQL，不是 shell 命令。

解决方法是先进入 `psql`，或者使用 `psql -c` 执行单条 SQL。这个问题说明安装文档应明确区分 shell 命令和 SQL 命令。

### 4.2 本地不存在 `postgres` 角色

执行 `psql -U postgres` 时出现：

```text
FATAL: role "postgres" does not exist
```

Homebrew PostgreSQL 通常以当前 macOS 用户创建本地管理员，而不一定创建名为 `postgres` 的角色。实际使用 `chenyuan` 连接并创建 `paper_agent` 用户和数据库。

经验：开发文档不应硬编码 `postgres` 管理员名，应写成 `<database-admin>`，并指导使用 `\du` 查看现有角色。

### 4.3 数据库连接变量缺失

执行 `paper-agent init` 时出现：

```text
Set PAPER_AGENT_DATABASE_URL or pass --database-url
```

当前 CLI 直接读取环境变量，但不会自动加载项目 `.env`。解决方法是：

```bash
export PAPER_AGENT_DATABASE_URL='postgresql+psycopg://...'
```

或者显式传递 `--database-url`。

经验：后续可以考虑明确支持 `.env`，或在错误提示中给出完整连接串示例和当前配置来源。

### 4.4 `projects` 外键缺失

第一次 `init` 成功执行 Migration 并生成 `.paper-agent/project.json`，但随后 `ingest` 报告：

```text
ingestion_runs.project_id 不存在于 projects.project_id
```

根因有两层：

1. 当前 `init` 只创建 manifest 和升级 Schema，没有同步写入 `projects`。
2. 首次 `ingest` 在同一个 Unit of Work 中同时添加 Project 和 IngestionRun，SQLAlchemy 的 flush 顺序没有确保 Project 先于 Run 落库，触发外键错误并回滚。

现场通过手动插入对应 `projects` 记录恢复。

建议的根本修复：

- `init` 在 Migration 后显式 upsert Project。
- 或在创建 IngestionRun 前执行 `session.flush()`，确保 Project 已写入。
- 添加“全新数据库 + 全新项目目录”的端到端 CLI 集成测试。

这是当前仍值得在代码中正式修复和回归覆盖的问题。

### 4.5 PyMuPDF 可用性判断混乱

说明文档一度显示 PyMuPDF 不可用，但项目 `.venv` 中实际已经安装 PyMuPDF `1.28.2`。原因通常是检查命令使用了不同 Python 环境，或者之前的隔离测试环境没有安装该包。

最终通过以下方式以项目虚拟环境为准：

```bash
uv run python -c "import pymupdf; print(pymupdf.__version__)"
```

经验：依赖可用性检查必须输出 `sys.executable`，避免 Anaconda、系统 Python、uv 虚拟环境之间混淆。

### 4.6 pgvector 前置条件

Phase 2A 的 `0004_phase2a` 会启用 `vector` Extension 并创建 `vector(256)` 字段与 HNSW 索引。真实环境需要先安装 pgvector，并确保执行 Migration 的数据库角色具备创建 Extension 的权限。

推荐流程：

```bash
brew install pgvector
psql -U <database-admin> -d <database> \
  -c 'CREATE EXTENSION IF NOT EXISTS vector;'
uv run paper-agent db-upgrade
```

集成测试应使用独立测试数据库和 `PAPER_AGENT_TEST_DATABASE_URL`，避免在真实论文库上执行测试清理。

### 4.7 55 个文件因 NUL 字符失败

Phase 2A 实际扫描 282 个文件时，初始结果为：

```text
failed  | 56
indexed | 226
```

错误清单分析显示，其中 55 个文件具有相同错误：

```text
structure_failed
PostgreSQL text fields cannot contain NUL (0x00) bytes
```

根因链路：

```text
PDF 异常文本层
→ PyMuPDF/Poppler 提取出 \x00
→ DocumentBlock.text 接受该字符
→ 本地 JSON 可以保存 \u0000
→ SemanticGroup.text 批量写入 PostgreSQL
→ PostgreSQL text 拒绝 NUL
→ Structure 事务回滚，文件标记 failed
```

SQLAlchemy 错误中的 `Query-invoked autoflush` 只是触发写入的时机，不是根因；通过 `no_autoflush` 不能解决数据不合法的问题。

最终在 `DocumentBlock.text` 的领域边界加入统一清洗：

```python
@field_validator("text")
@classmethod
def remove_nul_characters(cls, value: str | None) -> str | None:
    if value is None:
        return None
    return value.replace("\x00", "")
```

把清洗放在 Canonical Domain 边界的好处：

- PyMuPDF、Poppler 和未来 Parser 统一受保护。
- 新解析结果和旧 `document.json` 加载结果都能清洗。
- 下游 Section、Semantic Group、Chunk、FTS 和 Embedding 不再重复处理同一问题。

修复过程中还出现一次 `NameError: model_validator is not defined`，原因是加入 `field_validator` 时误删了原有 `model_validator` 导入。恢复为同时导入两者后，使用 `hello\x00world → helloworld` 的最小验证通过。

改进建议：

- 增加 NUL 清洗单元测试。
- 增加旧 Canonical JSON 含 `\u0000` 的加载回归测试。
- 考虑统一处理其他数据库不接受的控制字符，同时避免破坏有意义的换行和制表符。

### 4.8 一个文件的 Poppler 布局 XML 无效

另一个失败文件报告：

```text
Poppler 生成的布局 XML 无效：
not well-formed (invalid token): line 4509, column 88
```

Poppler Adapter 使用 `pdftotext -bbox-layout` 生成 XML，再通过 XML Parser 转换为 Canonical Document。PDF 内部异常字符可能产生不符合 XML 1.0 的输出，因此在创建 `DocumentBlock` 之前就失败，Domain 层 NUL 清洗无法介入。

通过显式使用 PyMuPDF 解析该文件后成功，因为 PyMuPDF 直接读取页面字典，不经过 Poppler XML。

改进建议：

- `auto` 模式不仅在启动时选择 Parser，还应支持单文件解析失败后的安全 fallback。
- 记录 primary parser 和 fallback parser 的错误链。
- 对 Poppler XML 中明确非法的控制字符做谨慎预清洗，但不能掩盖真正的 XML 结构损坏。

### 4.9 批量错误输出过长

失败时 CLI 会在 JSON 的 `failed` 数组中打印完整 SQLAlchemy 异常。批量 INSERT 包含数千个参数，导致终端输出巨大、定位困难。

本次采用的恢复策略：

1. 从 Excel 失败清单统计错误类型。
2. 先单文件验证修复。
3. 再验证唯一的 Poppler XML 文件。
4. 小批量验证 5 个文件。
5. 从 PostgreSQL 导出剩余失败路径。
6. 使用 shell 循环逐文件重试，终端只显示成功/失败，完整日志写入 `/tmp`。

改进建议：

- CLI 默认只输出稳定错误码和截断后的错误摘要。
- 完整堆栈与 SQL 参数写入日志文件。
- 增加 `paper-agent retry-failed` 命令，支持 project、run、error-code 和 batch-size。
- 提供 `--quiet`、`--log-file` 和机器可读报告路径。

## 5. 最终联调结果

修复和逐步重试后的数据库状态：

```text
 status  | count
---------+------
 indexed | 282
```

最终结果说明：

- 282 个 PaperFile 全部达到 `indexed`。
- NUL 文本清洗对原失败文件有效。
- Poppler XML 异常文件通过 PyMuPDF 成功处理。
- 逐文件失败隔离机制有效，最初的 56 个失败没有破坏其余 226 个成功文件。
- 普通增量重试能够复用已完成阶段，不需要清空数据库或全量 `--force-reindex`。

## 6. 当前测试策略

### 6.1 静态与单元测试

```bash
uv sync --extra dev
uv run mypy
uv run pytest tests/unit -v
```

重点覆盖：

- Domain 不变量。
- SHA-256、Content Hash 和身份解析。
- Parser/Schema/Structure/Chunk/Embedding/Index 版本失效。
- 增量复用和阶段恢复。
- Chunk 溯源。
- Embedding 批量生成和复用。
- Rerank、Threshold、去重和 `no_evidence`。
- NUL 与非法控制字符边界。

### 6.2 PostgreSQL 集成测试

使用独立数据库：

```bash
export PAPER_AGENT_TEST_DATABASE_URL='postgresql+psycopg://.../paper_assistant_test'
uv run pytest tests/integration -v
```

重点验证：

- `0001 → 0002 → 0003 → 0004` Migration。
- Migration downgrade/upgrade 往返。
- Repository Domain/Row round trip。
- pgvector Extension 和 `vector(256)` 读写。
- HNSW 和 GIN 索引存在。
- FTS、BM25、Dense + Sparse Hybrid Retrieval。
- Metadata Filter 和 Scope 隔离。
- Evidence 溯源与 `no_evidence`。

### 6.3 真实论文库冒烟测试

真实数据测试不应只看命令退出码，还需要检查：

```sql
SELECT status, COUNT(*)
FROM paper_files
GROUP BY status;
```

并检查：

- `indexing_states.status = 'indexed'`
- Paper、Section、Chunk Embedding 数量非零。
- 代表性查询返回正确论文和 Section。
- 不相关查询返回 `no_evidence`。
- Evidence 页码、Chunk 和 Section 溯源正确。

## 7. 当前局限与后续计划

当前系统已经具备论文知识检索后端，但还没有最终 Agent 对话入口。

已知局限：

- 默认 Hashing Embedding 不是生产级语义模型。
- 默认 Reranker 是词法混合基线，不是 Cross-Encoder。
- Figure/Table/Equation/Algorithm 仍是基础 Element 模型。
- 尚未实现图片裁切、表格网格重建、公式 OCR/LaTeX 和复杂版面恢复。
- 双栏 PDF 阅读顺序和 Heading 分类仍依赖启发式规则。
- 没有 `read_paper` 精确阅读工具。
- 没有 Agent Runtime、Tool Calling、Context Builder 和 Memory。
- CLI 的失败重试和日志体验仍需改善。
- `init` 与 `projects` 表同步问题需要正式修复。

建议下一阶段优先级：

1. 修复并测试 `init` 的 Project upsert/flush 顺序。
2. 补充 NUL、旧 JSON 和 Parser fallback 回归测试。
3. 增加 `retry-failed` 与结构化日志。
4. 实现 `read_paper` 和 Neighbor Chunk Expansion。
5. 将 `search_knowledge` 封装为安全 Tool Adapter。
6. 实现 Query Rewrite、Context Builder 和引用格式化。
7. 引入可配置神经 Embedding Provider 和 Cross-Encoder Reranker。
8. 构建 Agent Runtime、Session State 和长期记忆。

## 8. 关键经验总结

1. **真实数据比合成测试更容易暴露边界问题。** 55 个 NUL 文件说明 Parser 输出必须在领域边界统一规范化。
2. **错误阶段不一定等于根因位置。** `structure_failed` 实际是 PostgreSQL 文本约束失败，而不是 Section Builder 算法错误。
3. **不要用事务控制掩盖非法数据。** `no_autoflush` 只能改变 flush 时机，不能让 PostgreSQL 接受 NUL。
4. **Fallback 应以单文件为粒度。** Parser 在大多数 PDF 上可用，不代表对每个 PDF 都可用。
5. **派生数据必须可重建。** 版本化与阶段恢复让故障修复后无需清库重来。
6. **先单文件、再小批量、最后全量。** 这是处理真实数据批量故障时成本最低、反馈最快的策略。
7. **集成测试要使用真实 PostgreSQL，但必须隔离数据库。** 只有真实 Extension、约束和索引才能暴露环境差异。
8. **CLI 错误输出需要面向人设计。** 稳定错误码、摘要、日志文件和失败重试入口比直接打印完整 ORM 异常更实用。

---

截至本日志记录时间，Paper Agent 已完成 Phase 2A，真实项目中的 282 个 PDF 文件全部达到 `indexed` 状态。系统已具备从本地 PDF 到可追溯混合检索 Evidence 的完整后端链路，下一阶段将从“知识检索基础设施”进入“Agent Runtime、精确阅读和上下文组织”。
