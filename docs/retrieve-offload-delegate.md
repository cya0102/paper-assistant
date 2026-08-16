# Retrieve + Offload + Delegate

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

以下命令均在仓库根目录执行。

### 6.1 安装、静态检查与迁移链

```bash
uv sync --extra dev
uv run mypy src
uv run alembic heads
```

预期：依赖安装成功；mypy 输出 `Success: no issues found`；Alembic 只显示 `0011_rod_hardening (head)`。

### 6.2 Artifact / Offload / Hydrate 单元测试

```bash
uv run pytest tests/unit/artifacts tests/unit/agent/test_artifact_tool_adapters.py -ra
uv run pytest tests/unit/agent/test_runtime_offload.py tests/unit/agent/test_tool_adapters.py -ra
uv run pytest tests/unit/providers/test_mimo_provider.py -ra
```

预期全部通过，覆盖：小结果 Inline、大结果和二进制结果 Offload；严格模型预算；Blob gzip/Hash 校验及损坏自修复；相同内容共享 Blob 但保留不同 Worker provenance；`read_artifact` 无递归 Offload、合法 View/Cursor 分页、超大 JSON 无损恢复、跨项目/越界/非法 Cursor/Worker 越权被拒绝；Checkpoint 中仅保存 compact payload；OpenAI/MiMo 读取 passage/element 而非已删除的旧 evidence 键。

### 6.3 Delegate 单元测试

```bash
uv run pytest tests/unit/delegation -ra
```

预期全部通过，覆盖：简单请求拒绝委派、6～20 篇或显式 workstream 允许委派；确定性 Task/WorkUnit generation key；验证 WorkUnit 的 DAG 依赖和 Artifact 注入；失败最多重试一次；实际 ToolCall/Token/时间预算；论文与 Artifact 白名单；递归输出 Schema；未知/歧义/缺失 Citation 拒绝；`verified` verdict 拒绝；未解决问题可由 collect 恢复。

### 6.4 PostgreSQL / Redis 集成测试

先提供空的测试库和 Redis（下面端口仅为示例）：

```bash
docker run --name paper-agent-test-postgres -e POSTGRES_PASSWORD=paper_agent -e POSTGRES_DB=paper_agent_test -p 55432:5432 -d postgres:16
docker run --name paper-agent-test-redis -p 56379:6379 -d redis:7
export PAPER_AGENT_TEST_DATABASE_URL='postgresql+psycopg://postgres:paper_agent@localhost:55432/paper_agent_test'
export PAPER_AGENT_TEST_REDIS_URL='redis://localhost:56379/0'
uv run paper-agent db-upgrade --database-url "$PAPER_AGENT_TEST_DATABASE_URL"
```

预期：两个容器进入 running；迁移命令输出 `Database upgraded to head.`。

执行 retrieve-offload-delegate 的数据库纵切：

```bash
uv run pytest tests/integration/test_postgres_metadata.py tests/integration/test_postgres_artifacts.py tests/integration/test_postgres_research_tasks.py tests/integration/test_e2e_offload_delegate.py -ra
uv run pytest tests/integration/test_redis_compact_checkpoint.py tests/integration/test_redis_session_state.py -ra
```

预期无 skip、全部通过。PostgreSQL 组验证 0011 约束、Artifact provenance、Task/WorkUnit 幂等、真实 Offload→Hydrate 和 Delegate→Collect；Redis 组验证恢复时不重复工具调用，且不保存完整比较矩阵/Worker 正文。

### 6.5 全量回归

```bash
uv run pytest -ra
uv run mypy src
git diff --check
```

预期：配置了 PostgreSQL/Redis 后全部测试通过且无环境型 skip；mypy 成功；`git diff --check` 无输出。未配置两个测试服务时，相关集成测试应明确显示为 skip，而不是失败。

### 6.6 CLI 人工验收

准备至少 6 个已经 ingest 的 paper UUID，并设置模型、数据库和 Redis：

```bash
export PAPER_AGENT_LLM_MODEL='<可用模型名>'
export PAPER_AGENT_LLM_PROVIDER='openai'
export PAPER_AGENT_REDIS_URL="$PAPER_AGENT_TEST_REDIS_URL"
export PAPER_ID_1='<uuid>' PAPER_ID_2='<uuid>' PAPER_ID_3='<uuid>'
export PAPER_ID_4='<uuid>' PAPER_ID_5='<uuid>' PAPER_ID_6='<uuid>'
```

简单请求不得 Delegate：

```bash
uv run paper-agent delegate --root "$PWD" --database-url "$PAPER_AGENT_TEST_DATABASE_URL" '比较两篇论文' --paper-id "$PAPER_ID_1" --paper-id "$PAPER_ID_2"
```

预期退出码 1，JSON 为 `delegated: false`，reason 表明 2～5 篇比较使用主 Agent + Offload。

显式工作流必须完成 Delegate→Collect：

```bash
uv run paper-agent delegate --root "$PWD" --database-url "$PAPER_AGENT_TEST_DATABASE_URL" '比较方法并列出证据不足项' --paper-id "$PAPER_ID_1" --paper-id "$PAPER_ID_2" --workstream method --max-workers 2 --model "$PAPER_AGENT_LLM_MODEL" --provider "$PAPER_AGENT_LLM_PROVIDER"
```

预期退出码 0；输出同时含 `delegation` 和 `collected`；task_id 一致，status 为 `completed` 或存在可解释失败项的 `partially_completed`；每个成功 WorkUnit 只有 ArtifactReference，`unresolved_questions` 不丢失。

6 篇默认批处理必须触发委派并执行验证 DAG：

```bash
uv run paper-agent delegate --root "$PWD" --database-url "$PAPER_AGENT_TEST_DATABASE_URL" '系统比较六篇论文的方法、数据集、结果与局限，并核验证据' --paper-id "$PAPER_ID_1" --paper-id "$PAPER_ID_2" --paper-id "$PAPER_ID_3" --paper-id "$PAPER_ID_4" --paper-id "$PAPER_ID_5" --paper-id "$PAPER_ID_6" --max-workers 3 --model "$PAPER_AGENT_LLM_MODEL" --provider "$PAPER_AGENT_LLM_PROVIDER"
```

预期产生确定性 WorkUnit 列表；verification 最后执行并只能读取前置 Worker Artifact；最终只汇总紧凑摘要、ArtifactReference、Citation Manifest、未解决问题和失败项。

最后通过主 Agent 验收 search_artifact/read_artifact 控制面：

```bash
uv run paper-agent ask --root "$PWD" --database-url "$PAPER_AGENT_TEST_DATABASE_URL" --redis-url "$PAPER_AGENT_REDIS_URL" --model "$PAPER_AGENT_LLM_MODEL" --provider "$PAPER_AGENT_LLM_PROVIDER" '先检索刚才的研究 Artifact；如果摘要不足，选择 available_views 中的视图分页读取，然后基于 Citation Manifest 回答。'
```

预期退出码 0；Agent 能先 search_artifact，再按需 read_artifact；分页结果的 `token_count` 不超过请求上限，`next_cursor` 可继续读取，最终答案只使用返回过的引用标签。
