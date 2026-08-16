# Retrieve + Offload + Delegate

> 当前状态（2026-08-16）：标准 `paper-agent ask` 已按
> `retrieve-offload-delegate-rag-redesign.md` 升级为复合 ROD 路径。下文第 1～5 节
> 仍说明可复用的 Artifact 层和高级 `delegate` 能力；普通问答的当前执行链、配置和
> 验收以第 6 节为准。

本阶段为 Paper Agent 引入三层分级执行策略，并把它实现为一条可运行的纵切：

- **Retrieve**：定位候选论文、Chunk、Claim、Relation 或 Artifact。
- **Offload**：把完整工具结果、中间产物和审计数据保存到 Artifact Store。
- **Hydrate**：按 Token Budget 把 Artifact 的部分内容重新加载给模型。
- **Delegate**：把可独立执行的研究子问题交给隔离上下文中的 Worker Agent。

```text
User Query
→ Research Router
→ PostgreSQL/pgvector 定位候选
→ Retriever 返回 Candidate Reference
→ ContextBuilder 选择少量 Evidence
→ 完整结果写入 Artifact Store
→ 主 Agent 只接收摘要、Evidence Slice、Artifact ID
→ 主 Agent 需要更多信息时调用 read_artifact/search_artifact
→ 只有复杂、高价值、可并行任务才 Delegate
→ Worker 完整结果仍写入 Artifact Store
→ 主 Agent 只读取 Worker 摘要和必要 Artifact Slice
→ Citation/Entailment Verification
→ 最终答案或研究报告
```

## 1. Artifact Layer

新增模块：

```text
src/paper_agent/
├── domain/artifact.py            # ArtifactDescriptor/Reference/Selector/Slice/CitationReference
├── artifacts/
│   ├── ports.py                  # ArtifactBlobStore / ArtifactRepository / ArtifactServicePort
│   ├── service.py                # materialize / read_slice / search / validate_hash
│   ├── materializer.py           # ToolResultMaterializer + 紧凑视图 + Citation Manifest 提取
│   ├── policies.py               # OffloadPolicy / OffloadPolicyConfig
│   ├── views.py                  # 固定视图提取器（禁止任意 JSONPath / 文件路径）
│   └── tokens.py                 # 确定性 Token 估算
├── storage/local/artifact_blob_store.py
├── storage/postgres/artifact_repository.py
└── agent/artifact_tool_adapters.py   # read_artifact / search_artifact
```

### Artifact Domain

- ArtifactDescriptor：artifact_id、project_id、session_id/research_task_id/work_unit_id/tool_call_id（可空）、artifact_type、schema_version、media_type、content_hash(SHA-256)、storage_backend、storage_key、byte_size、token_estimate、summary、citation_manifest、status、created_by、created_at、expires_at。
- ArtifactReference：主 Agent 收到的紧凑句柄（artifact_id + 视图列表）。
- CitationReference：citation_label、paper/version/section/chunk/element id、paper_title、section_path、页码、evidence_hash。
- ArtifactSelector：artifact_id + project_id + view + cursor + max_tokens（1..4000）。
- ArtifactSlice：content + citations + next_cursor + truncated + token_count。

所有 ID、Hash、大小、Token 和状态在 Domain 层做校验。

### Blob Store（本地内容寻址）

```text
.paper-agent/artifacts/blobs/sha256/<ab>/<full-sha256>.json.gz
```

该路径相对于传给 CLI 的论文项目根目录，而不是代码仓库。按本文测试配置，实际位置为：

```text
/Users/chenyuan/Documents/develop/paperAgentTest/.paper-agent/artifacts/blobs/sha256/<ab>/<full-sha256>.json.gz
```

- 路径完全由服务端生成，不接受用户输入拼接。
- 标准库 gzip 压缩（mtime=0，确定性字节）。
- 临时文件 + os.replace 原子重命名。
- 读取时重新校验 SHA-256，损坏抛 ARTIFACT_CORRUPT。
- Blob 成功写入并校验后才提交数据库 Catalog；Blob 按内容哈希去重，Artifact Catalog 则按 session/task/work-unit/tool-call provenance 生成确定性 ID。相同上下文重试复用同一 Artifact，不同 Worker 即使输出完全相同，也保留不同 Catalog 记录并共享同一 Blob。
- 已存在但损坏的同址 Blob 会通过原子替换自修复；读取发现损坏时 Catalog 状态持久化为 `corrupt`。
- 实现 ArtifactBlobStore 端口，未来可切换 S3/Object Storage。

### PostgreSQL Catalog

迁移 0009_context_artifacts 新增基础表，迁移 0011_rod_hardening 修正 provenance 与幂等约束：

- research_artifacts：`(project_id, artifact_type, schema_version, content_hash)` 为普通检索索引，不再是唯一约束；Catalog 实例保留各自 provenance，底层 Blob 仍内容寻址去重。
- artifact_citations：artifact_id + project_id + citation_label 等；Citation Finalizer 只凭清单校验引用。
- research_tasks：唯一约束 `(project_id, generation_key)`，并发或重放不会创建重复任务。

## 2. ToolResult 重构

ToolResult.payload 被拆分为：

```python
ToolResult(
    call_id,
    name,
    model_payload,       # 紧凑模型视图（≤ 单结果预算）
    artifact_ref,        # 离屏结果 -> ArtifactReference
    citation_manifest,   # Citation Finalizer 只读这个
    is_error,
)
```

完整 Raw Payload 不再进入 Redis、Checkpoint 或 Provider 请求。为兼容保留 payload 属性（= model_payload）。

统一执行链：

```text
Tool.execute
→ RawToolResult
→ ToolResultMaterializer
→ OffloadPolicy
→ ArtifactService
→ Compact ToolResult
→ AgentRuntime
```

### OffloadPolicy 默认阈值

| 配置 | 默认值 |
|---|---|
| max_inline_tokens_per_result | 2000 |
| max_total_tool_tokens | 6000 |
| preview_tokens | 800 |
| artifact_retention_days | 30 |
| read_artifact_max_tokens | 4000 |

规则：小于单结果预算可 Inline；超过预算、二进制结果、compare_papers > 5 篇、read_paper 整章/多页、Worker 结果 → 始终 Offload；同一 Model Payload 不重复相同正文。二进制结果以带 media type、byte size 和 base64 的 JSON envelope 存储，原始二进制不会进入 Provider 或 Checkpoint。`read_artifact` 返回的已经是服务端限额后的 Slice，不会被再次 Offload；`delegate_research`、`collect_research_task`、`search_artifact` 保留控制字段，不会被通用压缩器剥掉 task_id 或 ArtifactReference。

## 3. 现有 Tool 的模型视图

- search_knowledge：模型只收到 status、summary、resolved papers、预算内 selected_evidence、omitted_evidence、artifact_ref。完整 Artifact 保留全部候选、重写查询、Dense/BM25/Rerank 分数与省略原因。
- read_paper：模型只收到论文基本信息 + 预算内少量 passage/element + omitted_passages。passages/elements 不再与统一 evidence 重复正文（删除了旧 evidence 键）。
- compare_papers：模型只收到 status、paper_count、comparable/insufficient dimensions、少量 high_level_findings、citations、artifact_ref、available_views。完整 Artifact 保留 dimensions/cells/evidence/derivation。支持视图：dimension:<name>、paper:<id>、all-cells、evidence、derivation。

## 4. read_artifact / search_artifact

read_artifact：artifact_id、view、cursor、max_tokens。project_id 由 Adapter 绑定；max_tokens 有严格上限；支持 Cursor 分页；超大单项和 full/report/result 视图会返回可无损拼接的 JSON fragment；非法/越界 Cursor 被拒绝。跨项目/过期/损坏返回稳定错误码。不允许任意文件路径或 JSONPath。Worker 还会额外校验 `allowed_artifact_ids`，不能读取未分配的 Artifact。

search_artifact：MVP 结构化过滤（query/artifact_type/created_by/max_results）。

## 5. ResearchTask / WorkUnit / Delegation

迁移 0010_research_tasks 新增 research_tasks 与 work_units。WorkUnit 携带目标、论文/Artifact 输入、依赖、Worker、允许工具、输出 Schema、Token/ToolCall/时间预算、状态、重试次数、输出 Artifact、幂等 Generation Key。

### DelegationPolicy 路由

| 请求 | 决策 |
|---|---|
| 单篇问答 | 不 Delegate |
| 2～5 篇比较 | 单 Agent + Offload |
| 6～20 篇比较 | 确定性批处理，按需按维度 Delegate |
| 系统文献调研 / 跨领域探索 | Delegate |
| PDF 解析、索引、数据库操作 | 不 Delegate |

### WorkerRegistry

已实现：paper_analyzer（search_knowledge/read_paper/read_artifact）、evidence_verifier（read_artifact/read_paper，verdict 只能是 supported/contradicted/insufficient/unreviewed）。已注册未实现：landscape_scout、relation_analyst、contradiction_finder、cross_domain_analogy_scout（Scheduler 明确拒绝运行）。

### 执行流程

```text
delegate_research
→ DelegationPolicy 判定
→ ResearchPlanner 生成确定性 WorkUnit DAG（验证单元依赖所有前置分析单元）
→ Scheduler（同步、单层、拓扑调度、依赖 Artifact 注入、最多重试一次）
→ WorkerRunner：隔离 Checkpoint + 专属简报 + 论文/Artifact 白名单 + 锁定工具 + 递归 Schema/Citation 校验 + ToolCall/Token/时间预算
→ ArtifactService 保存 Worker Artifact（始终 Offload）
→ collect_research_task：紧凑摘要 + artifact_refs + Citation Manifest + 未解决问题 + 失败 WorkUnit
```

### 第一版限制

- 最大并行 Worker 3～5（当前同步顺序执行）。
- 最大委派深度 1：Worker 不允许创建 Worker、Worker 之间不直接聊天，通过 Artifact 共享数据。
- Worker 只读论文数据、只写 Staging Artifact；canonical Research Graph 只能由验证服务写入。
- WorkUnit 最多自动重试一次；使用稳定 Generation Key 保证幂等。
- 主 Agent 不接收 Worker 完整执行轨迹。

## 6. 验证指令与预期结果

本节对应重构后的标准问答路径：

```text
paper-agent ask
→ retrieve_and_analyze_knowledge
→ Retrieve
→ 每个 Evidence Chunk 独立 Offload
→ 同层有界并行 chunk_analyst
→ Collect + Sufficiency
→ 最多一次 Query Rewrite
→ 最终合成或 no_evidence
```

普通问答验收不再要求用户手工执行 `delegate`，也不要求传入 `paper_id`、
`workstream` 或 `max_workers`。`delegate` 只用于高级研究任务和故障调试。

先区分代码仓库和论文项目目录：

| 变量 | 路径 | 用途 |
|---|---|---|
| `PAPER_AGENT_REPO_ROOT` | `/Users/chenyuan/Documents/develop/paper-assistant` | 运行 `uv`、pytest、mypy 和 Alembic |
| `PAPER_AGENT_PROJECT_ROOT` | `/Users/chenyuan/Documents/develop/paperAgentTest` | 已初始化并 ingest 论文的项目；保存 Project Manifest 和 Artifact Blob |

准备环境：

```bash
export PAPER_AGENT_REPO_ROOT="/Users/chenyuan/Documents/develop/paper-assistant"
export PAPER_AGENT_PROJECT_ROOT="/Users/chenyuan/Documents/develop/paperAgentTest"

cd "$PAPER_AGENT_REPO_ROOT"
set -a
source .env
set +a

test -f "$PAPER_AGENT_REPO_ROOT/pyproject.toml" && echo "repo root: OK"
test -f "$PAPER_AGENT_PROJECT_ROOT/.paper-agent/project.json" && echo "project root: OK"
```

除 `paper-agent --root` 明确接收论文项目目录外，下面的命令均在代码仓库根目录执行。

### 6.1 依赖、静态检查和迁移头

```bash
uv sync --extra dev
uv run mypy src
uv run alembic heads
git diff --check
```

预期：

- mypy 输出 `Success: no issues found`；当前检查 127 个源码文件。
- Alembic 只输出 `0011_rod_hardening (head)`。
- 本次重构复用现有字符串类型字段和 0009～0011 表，不新增 Migration。
- `git diff --check` 无输出。

### 6.2 ROD、Worker、并行调度和 Provider 单元测试

先运行新纵切：

```bash
uv run pytest \
  tests/unit/rag/test_rod_rag.py \
  tests/unit/delegation \
  -ra
```

预期全部通过，覆盖：

- 一个选中的 Evidence Chunk 对应一个 `retrieved_evidence` Artifact。
- 主 Agent Payload 中没有原始 Chunk 正文。
- 每个 Chunk 对应一个 `chunk_analysis` WorkUnit。
- `chunk_analyst` 只能使用 `read_artifact`，且只能读取一个获授权 Artifact。
- Worker Claim 的 Citation 必须属于输入 Artifact 的 Citation Manifest。
- 同一 DAG 层的独立 WorkUnit 在 `max_workers` 上限内并行。
- WorkUnit 最多重试一次；依赖层仍按拓扑顺序执行。
- 第一轮 partial/irrelevant 时只改写一次 Query；第二轮仍不足返回
  `insufficient`，不会把模型记忆当作论文事实。
- Retrieve 无候选时直接返回 `no_evidence`，不创建 Worker。
- 相同 Session、Query 重放复用稳定 Task、Artifact 和 WorkUnit ID。

再运行 Artifact、Agent Runtime 和 OpenAI/MiMo 兼容测试：

```bash
uv run pytest \
  tests/unit/artifacts \
  tests/unit/agent \
  tests/unit/providers/test_mimo_provider.py \
  tests/unit/test_project_manifest_and_cli.py \
  -ra
```

预期全部通过。重点检查：

- 默认 `build_agent_runtime` 只向标准 ask 暴露
  `retrieve_and_analyze_knowledge`。
- `--rag-mode direct` 才保留旧 Search/Read/Artifact/Delegate 工具集合。
- OpenAI 和 MiMo 在 ROD Collect 返回 supported/insufficient 后隐藏工具并进入合成。
- MiMo 的文本形式 `<tool_call>` 仍会被规范化；畸形或未执行标记被拒绝。
- supported 答案必须引用 Manifest；insufficient/no_evidence 的模型输出会被确定性替换为
  `no_evidence：<reason>`。
- Checkpoint 只保存紧凑 Worker Report、Claim、ArtifactReference 和 Citation Manifest。

### 6.3 PostgreSQL ROD 纵切

需要 PostgreSQL 16 和 pgvector。下面使用隔离端口，避免连接日常数据库：

```bash
docker run --name paper-agent-test-postgres \
  -e POSTGRES_PASSWORD=paper_agent \
  -e POSTGRES_DB=paper_agent_test \
  -p 55432:5432 \
  -d pgvector/pgvector:pg16

export PAPER_AGENT_TEST_DATABASE_URL='postgresql+psycopg://postgres:paper_agent@localhost:55432/paper_agent_test'

uv run paper-agent db-upgrade \
  --database-url "$PAPER_AGENT_TEST_DATABASE_URL"
```

预期容器为 running，迁移命令输出 `Database upgraded to head.`。

执行：

```bash
uv run pytest \
  tests/integration/test_postgres_metadata.py \
  tests/integration/test_postgres_artifacts.py \
  tests/integration/test_postgres_research_tasks.py \
  tests/integration/test_e2e_offload_delegate.py \
  tests/integration/test_e2e_retrieve_offload_delegate.py \
  -ra
```

预期无 skip、全部通过。新的
`test_e2e_retrieve_offload_delegate.py` 验证：

- 1 个 `rag_evidence_analysis` ResearchTask。
- N 个 `retrieved_evidence` Artifact。
- N 个 `chunk_analysis` WorkUnit。
- N 个成功 Worker 对应的 `worker_result` Artifact。
- Task → Evidence Artifact → WorkUnit → Worker Artifact provenance 完整。
- 相同 Project/User/Session/Query 重放不会重复创建 Task、WorkUnit 或 Artifact。
- 其他 `project_id` 不能读取 Task 或 Artifact。

本次开发环境没有可用的 PostgreSQL 测试服务，因此该组当前未执行并在全量测试中
显示为 skip；不能据此声称 PostgreSQL 集成已经通过。

### 6.4 Redis Checkpoint 集成

如果没有现成测试 Redis，可启动隔离实例：

```bash
docker run --name paper-agent-test-redis -p 56379:6379 -d redis:7
export PAPER_AGENT_TEST_REDIS_URL='redis://localhost:56379/15'
```

如果 Redis 开启密码认证，`PAPER_AGENT_TEST_REDIS_URL` 必须使用带认证信息的完整 URL，
但不要把凭据提交到仓库。

执行：

```bash
uv run pytest \
  tests/integration/test_redis_compact_checkpoint.py \
  tests/integration/test_redis_session_state.py \
  -ra
```

预期 3 个测试全部通过，验证 TTL、Checkpoint/Session round-trip，以及 ROD Checkpoint
不保存 retrieved Chunk 正文。本次开发使用本机带认证 Redis 的隔离 DB 15 实际执行结果为：

```text
3 passed
```

### 6.5 全量回归

同时设置 PostgreSQL 和 Redis 测试 URL 后执行：

```bash
uv run pytest -ra
uv run mypy src
git diff --check
```

预期全部测试通过且无环境型 skip；mypy 成功；`git diff --check` 无输出。

没有设置外部服务时，PostgreSQL/Redis 集成测试必须明确显示为 skip，而不是伪装成
已通过。本次无外部测试变量的实际基线为：

```text
160 passed, 20 skipped
Success: no issues found in 127 source files
0011_rod_hardening (head)
```

其中 skip 全部来自缺少 `PAPER_AGENT_TEST_DATABASE_URL` 或
`PAPER_AGENT_TEST_REDIS_URL`；Redis 组随后按 6.4 使用认证配置单独执行并通过。
启用本机认证 Redis、但仍未配置 PostgreSQL 时，本次全量实际结果为：

```text
163 passed, 17 skipped
```

这 17 个 skip 全部是 PostgreSQL 集成测试，不能算作通过。

### 6.6 标准 ask CLI 人工验收

CLI 人工验收必须使用已经 ingest 论文的日常项目和数据库，不要把空的
`paper_agent_test` 当成论文语料库。

先确认数据：

```bash
uv run paper-agent status \
  --root "$PAPER_AGENT_PROJECT_ROOT" \
  --database-url "$PAPER_AGENT_DATABASE_URL"
```

预期 `files`、`papers` 和 `chunks` 均大于 0。若全部为 0，应先检查数据库 URL 和
Project Manifest，不要对日常项目重新执行 `init`。

设置标准 RAG 预算：

```bash
export PAPER_AGENT_RAG_MODE='retrieve-offload-delegate'
export PAPER_AGENT_RAG_MAX_EVIDENCE='6'
export PAPER_AGENT_RAG_MAX_PER_PAPER='2'
export PAPER_AGENT_RAG_MAX_WORKERS='3'
export PAPER_AGENT_RAG_MAX_ROUNDS='2'
export PAPER_AGENT_RAG_WORKER_TOKEN_BUDGET='1200'
export PAPER_AGENT_RAG_WORKER_TOOL_CALL_BUDGET='2'
export PAPER_AGENT_RAG_WORKER_TIMEOUT_SECONDS='90'
```

用 MiMo 执行普通论文问答：

```bash
export PAPER_AGENT_LLM_PROVIDER='mimo'
export PAPER_AGENT_LLM_MODEL='mimo-v2.5-pro'

uv run paper-agent ask \
  --root "$PAPER_AGENT_PROJECT_ROOT" \
  --database-url "$PAPER_AGENT_DATABASE_URL" \
  --redis-url "$PAPER_AGENT_REDIS_URL" \
  --provider "$PAPER_AGENT_LLM_PROVIDER" \
  --model "$PAPER_AGENT_LLM_MODEL" \
  --rag-mode retrieve-offload-delegate \
  --trace summary \
  '2D-TAN 的主要方法是什么？请给出论文证据。'
```

用户不需要执行 `delegate`，也不需要提供任何 Paper ID。预期：

- stdout 只包含最终 JSON，不混入 Trace。
- stderr 依次出现 `rag.retrieve.started`、`rag.retrieve.completed`、
  `rag.artifact.created`、`rag.delegate.started`、`rag.worker.started`、
  `rag.worker.completed`、`rag.collect.completed`、
  `rag.sufficiency.checked`、`rag.synthesis.started` 和
  `rag.answer.validated`。
- 每个回答中的 `[E编号]` 都存在于 Worker 返回的 Citation Manifest。
- 最终答案不包含 `<tool_call>`、`<function=...>` 或
  `<parameter=...>`。
- 第一轮充分时只执行一轮；第一轮不足时最多执行一次改写后的第二轮。
- 第二轮仍不足时 stdout 中的 answer 以 `no_evidence` 开头。

机器可读 Trace：

```bash
uv run paper-agent ask \
  --root "$PAPER_AGENT_PROJECT_ROOT" \
  --database-url "$PAPER_AGENT_DATABASE_URL" \
  --redis-url "$PAPER_AGENT_REDIS_URL" \
  --provider "$PAPER_AGENT_LLM_PROVIDER" \
  --model "$PAPER_AGENT_LLM_MODEL" \
  --trace jsonl \
  '2D-TAN 的主要方法是什么？请给出论文证据。' \
  2> /tmp/paper-agent-rag-trace.jsonl
```

预期 `/tmp/paper-agent-rag-trace.jsonl` 每行都是一个 JSON Event，且 Event 中不包含
Chunk `text`、Artifact Blob 或 Worker 完整执行历史。

OpenAI 使用相同路径，只替换 Provider 和模型：

```bash
uv run paper-agent ask \
  --root "$PAPER_AGENT_PROJECT_ROOT" \
  --database-url "$PAPER_AGENT_DATABASE_URL" \
  --redis-url "$PAPER_AGENT_REDIS_URL" \
  --provider openai \
  --model "$OPENAI_CHAT_MODEL" \
  --trace summary \
  '2D-TAN 的主要方法是什么？请给出论文证据。'
```

OpenAI 与 MiMo 的 Artifact、WorkUnit、Citation 和 no_evidence 验收标准完全相同。

### 6.7 数据库侧验收

一次有证据的 ask 完成后，在日常 PostgreSQL 会话中执行：

```sql
SELECT task_id, session_id, status, plan_json
FROM research_tasks
WHERE task_type = 'rag_evidence_analysis'
ORDER BY created_at DESC
LIMIT 5;

SELECT artifact_type, count(*)
FROM research_artifacts
WHERE research_task_id = '<上一步 task_id>'
GROUP BY artifact_type
ORDER BY artifact_type;

SELECT work_type, requested_worker, status, count(*)
FROM work_units
WHERE task_id = '<上一步 task_id>'
GROUP BY work_type, requested_worker, status;
```

预期：

- Task 的 `plan_json` 为 retrieve/chunk_analysis/collect。
- `retrieved_evidence` 数量与选中的去重 Chunk 数一致。
- `chunk_analysis + chunk_analyst` WorkUnit 数量与 Evidence Artifact 数一致。
- 每个成功 WorkUnit 有一个 `output_artifact_id` 指向 `worker_result`。
- 同一个 Session/Query 重放后数量不增长。
- Artifact Blob 位于
  `$PAPER_AGENT_PROJECT_ROOT/.paper-agent/artifacts/blobs/sha256/`，而不是代码仓库。

### 6.8 direct 回滚/对照模式

`direct` 只用于故障回滚和评测对照，不是普通用户默认路径：

```bash
uv run paper-agent ask \
  --root "$PAPER_AGENT_PROJECT_ROOT" \
  --database-url "$PAPER_AGENT_DATABASE_URL" \
  --redis-url "$PAPER_AGENT_REDIS_URL" \
  --provider "$PAPER_AGENT_LLM_PROVIDER" \
  --model "$PAPER_AGENT_LLM_MODEL" \
  --rag-mode direct \
  '2D-TAN 的主要方法是什么？'
```

预期 direct 模式保留旧的 Search/Read/Artifact/Delegate 工具行为；默认不传
`--rag-mode` 时必须走 Retrieve–Offload–Delegate。显式 `paper-agent delegate`
仍可用于高级批量研究和调试，但不再是标准 RAG 人工验收步骤。
