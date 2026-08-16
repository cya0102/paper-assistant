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
- Blob 成功写入后才提交数据库 Catalog；内容哈希唯一约束保证幂等重试不产生重复 Artifact。
- 实现 ArtifactBlobStore 端口，未来可切换 S3/Object Storage。

### PostgreSQL Catalog

迁移 0009_context_artifacts 新增：

- research_artifacts：唯一约束 (project_id, artifact_type, schema_version, content_hash)。
- artifact_citations：artifact_id + project_id + citation_label 等；Citation Finalizer 只凭清单校验引用。

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

规则：小于单结果预算可 Inline；超过预算、二进制结果、compare_papers > 5 篇、read_paper 整章/多页、Worker 结果 → 始终 Offload；同一 Model Payload 不重复相同正文。

## 3. 现有 Tool 的模型视图

- search_knowledge：模型只收到 status、summary、resolved papers、预算内 selected_evidence、omitted_evidence、artifact_ref。完整 Artifact 保留全部候选、重写查询、Dense/BM25/Rerank 分数与省略原因。
- read_paper：模型只收到论文基本信息 + 预算内少量 passage/element + omitted_passages。passages/elements 不再与统一 evidence 重复正文（删除了旧 evidence 键）。
- compare_papers：模型只收到 status、paper_count、comparable/insufficient dimensions、少量 high_level_findings、citations、artifact_ref、available_views。完整 Artifact 保留 dimensions/cells/evidence/derivation。支持视图：dimension:<name>、paper:<id>、all-cells、evidence、derivation。

## 4. read_artifact / search_artifact

read_artifact：artifact_id、view、cursor、max_tokens。project_id 由 Adapter 绑定；max_tokens 有严格上限；支持 Cursor 分页；跨项目/过期/损坏返回稳定错误码。不允许任意文件路径或 JSONPath。

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
→ ResearchPlanner 生成确定性 WorkUnit DAG
→ Scheduler（同步、单层、依赖感知、最多重试一次）
→ WorkerRunner：隔离 Checkpoint + 专属简报 + 锁定工具 + Schema 校验
→ ArtifactService 保存 Worker Artifact（始终 Offload）
→ collect_research_task：紧凑摘要 + artifact_refs + Citation Manifest + 未解决问题 + 失败 WorkUnit
```

### 第一版限制

- 最大并行 Worker 3～5（当前同步顺序执行）。
- 最大委派深度 1：Worker 不允许创建 Worker、Worker 之间不直接聊天，通过 Artifact 共享数据。
- Worker 只读论文数据、只写 Staging Artifact；canonical Research Graph 只能由验证服务写入。
- WorkUnit 最多自动重试一次；使用稳定 Generation Key 保证幂等。
- 主 Agent 不接收 Worker 完整执行轨迹。

## 6. 验证

```bash
uv run pytest                          # 单元 + 集成（需要 PAPER_AGENT_TEST_DATABASE_URL / PAPER_AGENT_TEST_REDIS_URL）
uv run mypy src                        # strict
```

关键回归点：

- 小结果 Inline、大结果 Offload、Artifact 无损恢复、gzip 生效、Hash 损坏拒绝读取、跨 Project 拒绝、内容 Hash 去重、过期稳定报错、read_artifact 支持 View/Cursor/max_tokens。
- Checkpoint 恢复不重复 Tool Call；Redis 不保存完整比较矩阵；OpenAI/MiMo 只接收 model_payload；Citation Finalizer 使用 Manifest。
- 20 篇比较仍满足上下文预算；简单请求不触发 Delegate；Worker 只收到最小上下文；Worker 输出被 ArtifactService 保存；主 Agent 只收到 ArtifactReference；evidence_verifier 不能把 insufficient 标记为 verified。
