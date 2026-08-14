# Phase 3 阶段总结

## 阶段目标

Phase 3 将 Phase 2A 的索引与检索后端扩展为可恢复的论文 Agent Runtime，并补齐精确阅读、上下文控制、引用校验、短期状态、长期交互记忆和可选神经 Provider。

## 已完成功能

### Agent Runtime 与 Tool Loop

- 新增 `AgentCheckpoint`、`ModelTurn`、`ToolCall`、`ToolResult` 和 `AgentAnswer` 领域模型。
- 实现不依赖 LangChain/LangGraph 的可恢复 Agent Loop。
- 在 Model Response、Pending Tool Calls 和每个 Tool Result 后写入 Checkpoint。
- 支持一个模型响应内的多个 Tool Call；恢复时不会重复执行已完成调用，并会把同一响应的完整 Tool Result 集合返回模型。
- 增加最大循环步数、未知 Tool 拒绝、Tool Error 结构化回传和失败状态。

### Tool Registry 与工具适配器

- 新增最小 `ToolRegistry` 和 JSON Schema `ToolContract`。
- 将 `search_knowledge` 内部服务封装为模型工具。
- 新增 `read_paper` 工具，支持 Paper、Version、Section、Page Range、Element ID、Element Type 和 Neighbor 参数。
- Tool 输出保留 Paper、Version、Section、Page、Chunk 和 Element 溯源。

### 高级低噪检索

- 新增保守 Query Rewrite 与 Multi-query 执行。
- 跨 Query 按 Chunk ID 融合去重，并保留最高相关度。
- 命中后只在同 Version、同 Section、相邻 Chunk Order 内扩展 Neighbor。
- 新增可替换 `EvidenceJudge`，默认使用可离线运行的词法校准实现。
- 保留原有 Threshold、近重复过滤、Paper Quota 和 `no_evidence`。

### Context 与引用

- 新增 Context Builder，使用 Token Budget 控制送入模型的证据总量。
- 对多论文证据执行每篇上限和轮询选择，避免单篇论文占满 Context。
- 给 Evidence 分配 `[E1]` 引用标签。
- 最终答案会校验引用是否真实存在；未知引用或有证据却完全不引用时拒绝完成。
- 输出末尾附 Paper Title、Section Path 和 Page Range 来源。

### Session 与长期记忆

- 新增 Redis Agent Checkpoint Store 和 Session Store，默认 TTL 为 24 小时。
- Session 保存 Recent Messages、Current Paper、Active Chunk 和 Recent Tool Results。
- 同一 Session 的后续问题会回填近期消息，支持“刚才那篇论文”一类指代。
- 新增 PostgreSQL `interactions`、`notes`、`user_preferences`。
- Agent 成功回答后自动记录 Query、Paper IDs、Retrieved Chunk IDs 和 Answer Summary。
- Notes 的 Domain/Schema/Repository 基础支持 Project/Paper/Section 关联和 Tags；尚未提供面向用户的 Notes CLI/Tool/API。Preference 同样处于领域与 Repository 基础层。

### 模型与神经检索 Provider

- 新增 OpenAI Responses API Provider，使用 `previous_response_id` 和 Tool `call_id` 继续模型工具循环。
- 新增 OpenAI 批量 Embedding Provider，输出当前 pgvector Schema 所需的 256 维向量。
- 新增可选 Sentence Transformers Cross-Encoder Reranker。
- 默认仍使用无需 API Key 的 Hashing Embedding 和 Lexical Reranker。

### CLI、Migration 与测试

- 新增 `paper-agent search`、`paper-agent read`、`paper-agent ask`。
- 项目版本升级到 `0.5.1`。
- Migration 链扩展为 `0005_phase3`、`0006_phase3` 和 `0007_phase3`。
- 更新 `.env.example`、`README.md` 和 `使用说明.md`。
- 修复 Parser/Schema 变化后旧结构仍被复用：派生状态保存 Canonical Parsed Document Hash，Hash 为 `NULL` 或不一致时重建 Structure/Chunk。
- 修复 Structure 重建删除 Section 时 Note 被级联删除：Notes→Section 外键改为 `ON DELETE SET NULL`。
- 同时连接真实 PostgreSQL/pgvector 和独立临时 Redis 执行全量测试：64 passed。
- Migration 当前为 `0007_phase3 (head)`，并完成 `0007→0005→0007` 往返验证。
- `mypy --strict` 检查 77 个源码文件：无错误。

## 当前可使用程度

不配置模型 Key 时，项目可以完成：

```text
PDF 导入与去重
→ Canonical Document
→ Section/Element/Chunk
→ 分层索引
→ CLI 混合检索
→ CLI Section/Page/Element 阅读
```

配置 OpenAI Key、模型名和 Redis 后，可以使用 `paper-agent ask` 运行完整 Agent Search/Read/Answer 流程。当前形态适合本机单用户研究、后端集成和继续开发，尚未提供 Web UI、HTTP API、鉴权和生产级任务调度。

## 下一阶段建议

1. HTTP API、流式回答和 Web/客户端接入。
2. 后台 Ingestion/Indexing Queue、并发控制和取消/重试。
3. 用户认证、Project ACL、多租户数据隔离。
4. OpenTelemetry、结构化日志、Tool Trace、Token/成本统计。
5. Interaction/Note Embedding 和长期记忆语义召回。
6. Figure 裁切、Table Cell、Equation LaTeX 和 Algorithm 结构恢复。
7. 建立真实 Agent/Search/Read 评测集，校准 Threshold、Evidence Judge、Reranker 和引用正确率。
8. 增加 Provider Retry、Rate Limit、Batch 调度、模型缓存和生产配置验证。
