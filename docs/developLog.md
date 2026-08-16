# 调试记录：paper-assistant 问题排查与修复

> 用途：秋招面试复盘材料。本文件完整记录一次"存量系统评估 → 发现缺陷 → 设计修复方案 → 实施 → 二次审查 → 修复二次缺陷 → 回归验证"的工程过程，每个问题均附**实际业务场景**，便于面试时把技术问题讲成"用户故事"。
> 项目：paper-agent（论文阅读助手 Agent），版本 0.5.0，Python 3.12 + SQLAlchemy 2.0 + PostgreSQL/pgvector + Redis + Alembic。
> 问题编号说明：问题 1 = 原编号 A，问题 2 = 原编号 C，问题 3 = 原编号 D（原 B 经确认非问题已关闭）。后续新增问题请继续编号 **问题 4、问题 5……**，直接追加章节即可。

---

## 0. 一句话总结

对已宣称"完成"的 Phase3 功能做代码级审查，发现 **3 个问题**：派生数据失效机制不完整（问题 1）、read_paper 隔离与引用缺失（问题 2）、Notes 级联删除数据丢失（问题 3）。三个问题均已修复；其中问题 2 经历两轮——初版修复后**二次审查又发现 15 项衍生问题**，最终重构为"统一 evidence 契约"彻底解决。全程 **66 个测试通过、mypy strict 零错误**，迁移链完整。

---

## 1. 项目背景

paper-agent 是一个本地论文阅读助手，核心链路：

```text
PDF 导入与去重 → Canonical Parsed Document → Section Tree / Element / Semantic Group
→ Section-aware Chunking → 分层 Embedding 索引（pgvector）→ Hybrid 检索（Dense+FTS+BM25）
→ Agent Runtime（Tool Loop + Redis Checkpoint）→ 引用校验的最终答案
```

关键设计：**一套多版本号驱动的派生数据失效机制**（Parser Version / Canonical Schema Version / Structure Version / Chunking Version / Embedding Version / Index Version），目标是"上游版本变化时只重建受影响的派生数据，事务内替换，防止版本混用"。

---

## 2. 评估目标与方法

**任务**：不修改代码，先审查项目是否存在问题；重点核对"派生数据版本失效""恢复一致性""数据安全"三类声明是否真正成立。

**方法**（只读静态审查 + 离线验证，评估阶段全程未改动源码）：

1. 通读核心模块：`ingestion/pipeline.py`、`ingestion/identity.py`、`storage/postgres/repositories.py`、`storage/postgres/models.py`、`retrieval/*`、`agent/*`、`providers/*`、迁移 0001-0005；
2. 追踪关键数据流：解析 → 结构 → 分块 → 索引的版本判断路径；
3. 交叉核对"声明 vs 实现 vs 测试"三方一致性（含测试盲区分析）；
4. 离线执行：`pytest`（评估时 55 passed + 4 skipped，4 个需真实 PG/Redis）、`mypy --strict`（77 文件 0 错误）。

---

## 3. 问题 1：Parser/Schema 版本变化不重建下游数据

### 3.1 业务场景（实际用户影响）

研究者维护一个本地论文库。某天项目升级（安装新版 PyMuPDF，或从 Poppler 后备解析器切回 PyMuPDF），执行 `paper-agent ingest` 重新导入全部论文，期望"用新解析器重解析，章节/索引随之更新"。

**用户实际看到的**：解析器确实重新解析了（`parsed_documents` 出现新记录），但检索"SCANet 的 method"返回的仍是**旧章节划分下的段落**；Agent 阅读 Section 3.2 时给出的内容与论文最新排版不符；而 `paper-agent status` 显示一切正常、不报任何错误。

**为什么严重**：这是**静默的不一致**——证据文本、页码、章节路径来自两套不同的解析结果。对"可追溯证据"这一核心卖点是直接破坏：引用指向的内容与论文实际内容错位，且用户无从察觉。

### 3.2 现象与影响

项目宣称"Parser/Schema 变化时重新解析并重建下游数据"（Phase3 总结第 9 点）。但代码追踪显示：

- Parser 版本变化时，`has_current_document=False` → **确实重新解析**，写入新的 `parsed_documents` 行；
- 但随后结构步骤中 `structure_is_current` 只比较 `structure_version`（处理器版本号），该值未变 → **直接加载数据库里的旧 Section/Element/SemanticGroup**；
- 分块步骤同理（`chunking_version` 未变）→ **不重新分块**；
- 索引 `is_current` 检查的是 DB 中 sections/chunks 的 `source_digest`（未变）→ **不重建索引**。

**后果**：换 parser（或 parser 升级）后，解析产物是新的，但检索/阅读服务的 Section、Chunk、Embedding 全部还是旧解析的结果——**结构与内容不一致**，且是静默的。

### 3.3 根因分析

`derived_data_states` 表只记录 `structure_version` / `chunking_version`，**缺少"这份结构是由哪一版解析产物生成的"的绑定信息**：

```python
# 修复前（pipeline.py 结构步骤）
structure_is_current = bool(
    derived_state
    and derived_state.structure_version == self._structure_processor.version
)
# 只要处理器版本号没变就认为"当前"，与解析产物完全无关
```

而真正的失效路径只有两条：内容变化产生新 version_id（全新派生状态），或手动 `--force-reindex`。**Parser 版本升级恰好两条都不触发**。

### 3.4 为什么测试没发现（测试盲区）

审查既有测试后发现一个组合盲区：

- `test_parser_version_change_reparses_without_force`：验证了"重新解析"，但该 pipeline **未配置 structure_processor/chunker**，根本没走到结构重建步骤；
- `test_phase1c_recovers_each_derived_stage_by_its_version`：验证了结构/分块版本号变化会重建，但 **parser 版本保持不变**。

"parser 版本变化 **且** 结构处理器存在"这一组合恰好从未被测试覆盖——实现与测试盲区互相掩盖。

### 3.5 修复方案与实施

**核心思路**：给派生状态绑定"解析产物哈希"，失效判据从"版本号一致"升级为"版本号一致 **且** 解析哈希一致"。

1. **迁移 0006**：`derived_data_states` 增加 `document_hash VARCHAR(64)` + 64 位 hex 校验约束；**存量行留 NULL 并在注释中明确"NULL 视为 stale，下次 ingest 一次性重建"**——不做有歧义的 backfill（宁可多做一次重建，也不冒用错数据的风险）。
2. **领域层**：`DerivedDataState` 增加 `document_hash` 字段（含格式校验）。
3. **仓储层**：
   - `ParsedDocumentRepository.current_document_hash(version_id, parser_name, parser_version, schema_version)` —— 查当前解析产物的哈希；
   - `DerivedDataRepository.replace_structure(..., document_hash=...)` —— 写结构时同时落哈希。
4. **Pipeline**：

```python
# 修复后（pipeline.py 结构步骤）
current_hash = unit_of_work.documents.current_document_hash(
    version_id, parser.name, parser.version, canonical_schema_version,
)
structure_is_current = bool(
    current_hash is not None
    and derived_state
    and derived_state.structure_version == self._structure_processor.version
    and derived_state.document_hash == current_hash   # ← 关键新增
)
```

   快速路径 `_target_is_current` 同步加入哈希校验；`current_hash` 为 None 时按"非当前"处理（自愈而不丢数据）。

5. **测试**：
   - 单元：`test_parser_version_change_rebuilds_structure_and_chunks`（补齐此前缺失的组合）、`test_unchanged_document_hash_does_not_rebuild_structure_or_chunks`（防过度重建）、`test_schema_version_change_rebuilds_structure_and_chunks`；
   - 集成：新增 `tests/integration/test_postgres_parser_rebuild.py`——真实 PG 上验证 parser 变化重建、`derived_data_states.document_hash` 与 `parsed_documents.document_hash` 一致、force-reindex 场景。

> 细节：`document_hash = sha256(完整 document.json)`，JSON 包含 parser name/version，因此 **parser 版本变化 → 哈希必然变化 → 触发重建**；而 parser 换了但输出完全相同时哈希不变 → 不重建。失效粒度是"解析产物是否真的变了"，比"版本字符串是否变了"语义更正确。

### 3.6 验证

```text
3 个新增单元测试 + 1 个新增 PG 集成测试文件；全量测试通过，mypy strict 零错误
```

---

## 4. 问题 2：read_paper 无项目隔离与无引用校验

### 4.1 业务场景（实际用户影响）

**场景 A（隔离）**：研究者用同一套部署管理多个语料库——"CV 论文库"与"NLP 论文库"是两个 project，未来还要开放给同组同学（多用户）。`read_paper` 不校验 `project_id`：只要拿到 paper_id（从日志、搜索结果、他人分享中泄露）就能**跨项目读取论文全文**；而 `search_knowledge` 全程受 project 约束。同一系统里两条不对称的权限路径——"检索要隔离，阅读不用隔离"没有设计依据，是后续多租户化的现成漏洞。

**场景 B（引用/来源）**：用户问"详细解释 Section 3.2 的公式推导"，Agent 走 `read_paper` 路径。修复前 read 返回的段落**没有任何引用编号**：模型可以编造页码/章节，答案不强制引用、不附来源——幻觉校验在最常见的阅读场景下是关闭的。而 search 路径有完整的 `[E#]` 引用校验，两条路径行为不一致。

**场景 C（记忆）**：用户连续追问"刚才那篇论文的 3.2 节再展开讲讲"。纯 read 回答不写入 Interaction Memory / Redis Session：`current_paper_id`、`active_chunk_ids` 不更新，"刚才那篇论文"的短期指代在 read 路径下失效。

### 4.2 初版问题与第一轮修复

**初版问题**：`ReadPaperRequest` 无 `project_id`；read 返回的 passages 无引用编号；`ToolEvidenceCitationFormatter` 只解析 `payload["evidence"]`，read 路径的引用校验为空。

**第一轮修复**：
1. 领域层 `ReadPaperRequest` 增加 `project_id`；仓储校验项目归属；
2. 工具适配器绑定 `project_id`（与 search 对称，不出现在模型可见 Schema 中）；CLI `read` 增加 `--root`；
3. passages 生成 `[P{数字}]` 引用编号（由 Chunk ID 派生）；Finalizer 正则升级为 `[EP]` 双命名空间；提示词同步。

### 4.3 二次审查：15 项衍生问题（同行评审价值最高的部分）

第一轮修复后，对修复本身再做一轮代码审查，发现 **15 项衍生问题**，按性质分四组：

**隔离层（4 项）**
1. `project_id` 仍是可选字段，直接调用 `ReadPaperService` 传 `None` 可绕过隔离；
2. 仓储只校验 paper 归属，**不校验 version 归属**——同一 Paper 在两个项目各有 Version 时仍可跨项目读取；
3. 默认版本用**全局** `paper.canonical_version_id`，该 Version 可能只属于其他项目；
4. 先查全局 Paper 再查归属，错误信息会**泄露 paper_id 是否真实存在**。

**契约层（5 项）**
5. Read Element（Figure/Table/Equation/Algorithm）仍无引用编号，纯 Element 回答不强制引用；
6. Read 适配器没有产出**统一顶层 `evidence`**，passages 只能被 Finalizer 特殊分支识别，Memory 等模块无法统一消费；
7. CLI `read` 直接输出仍无 Passage Citation；
8. `ReadElement` 领域模型缺 `section_path`，即使补上引用也无法生成完整"论文—章节—页码"来源；
9. 集成测试只验证"随机项目不能读"，**没验证同 Paper 跨 Version 越界**。

**MiMo 层（3 项）**
10. MiMo 最终证据包只在**没有** Search Evidence 时才加入 Read Passage——Search+Read 同时发生时 Read 内容被丢弃；
11. MiMo 只要 Read 返回 `elements` 就允许完成，但 Element 无引用 → 纯 Element 回答可在零引用状态下完成；
12. MiMo 在 `citations` 为空时跳过引用检查，纯 Element Read 恰好进入该分支。

**校验/记忆层（3 项）**
13. Finalizer 只要求整份答案"至少一个合法引用"，Search+Read 混合回答可只引 Search、完全忽略 Read 来源；
14. Runtime `_persist_memory` 只读顶层 `payload["evidence"]` → 纯 Read 回答不写 Interaction/Redis Session；
15. 缺少 Read Element、MiMo Search+Read、Read-only Memory 的回归测试。

**根因归纳**：15 项问题表面上分散，根源只有一个——**read 结果没有统一成与 search 相同的 evidence 契约**，导致每条消费路径（Finalizer / MiMo / Memory / 隔离）都要各自特殊处理、各自漏。

### 4.4 最终修复：统一 evidence 契约

1. **隔离层**：`ReadPaperRequest.project_id` 改为**必填字段**；仓储先查项目归属（统一返回 `"Paper not found in project"`，不泄露存在性），版本在**项目内解析**（is_canonical 优先），显式请求的 version 必须属于当前项目；
2. **契约层**：`_serialize` 汇出统一 `"evidence"` 列表（passages + elements，标准字段：citation/paper_id/version_id/paper_title/section_id/section_path/page/chunk_id/element_id/text）；`ReadElement` 增加 `section_path`；CLI 输出补 citation；
3. **MiMo 层**：`_finalization_input` 去掉 `if not blocks:` 门控**无条件合并** Search+Read；`_should_finalize` 改为基于 `payload.get("evidence")` 判定；`_is_valid_final_response` 空引用视为契约错误；
4. **校验/记忆层**：Finalizer 增加**每命名空间至少引用一次**（`{E,P} - 已引用前缀` 非空即拒绝）；统一 evidence 后 Runtime Memory 自动消费，纯 Read 回答正确写入 Interaction/Session；
5. **回归测试**：补齐 4 个单元测试（序列化统一 evidence、每命名空间校验、read-only 记忆持久化、MiMo 合并）+ 集成测试扩展（双项目双 Version 双向越界拒绝、project_id 必填、fake 模型同时引用 E/P）。

### 4.5 验证

```text
全量测试 66 passed + 5 skipped（新增 4 个单元测试）；mypy strict 77 文件 0 错误
（真实 PG 集成测试建议：uv run pytest tests/integration/test_phase3_agent_search_read.py -vv）
```

---

## 5. 问题 3：Notes 级联删除数据丢失

### 5.1 业务场景（实际用户影响）

研究者在论文 Section 上记笔记（如"3.2 节与 4.1 节的方法对比待补充"）。某天论文解析器升级，或用户执行 `--force-reindex`，触发结构重建：`replace_structure` 会 `DELETE FROM sections WHERE version_id=...`，而 `notes.section_id` 外键是 `ON DELETE CASCADE`——**用户笔记随结构重建一起被静默删除，没有任何提示**。

**为什么严重**：科研笔记是研究者的长期资产（跨会话的思考痕迹），静默丢失不可接受。更隐蔽的是：重建后 Section 的 UUID 全部重新生成，用户甚至不知道笔记曾经存在。

### 5.2 根因

`notes.section_id` 外键为 `ON DELETE CASCADE`，而 `replace_structure` 重建结构时会 `DELETE FROM sections WHERE version_id=...`（`repositories.py`）——**绑定到 Section 的用户笔记会在任何结构重建/force-reindex 时被静默删除**。

### 5.3 修复（实施部分）

- `models.py::NoteRow.section_id` FK 改为 `ON DELETE SET NULL`（该列本身可空，无需改类型）；
- 新增迁移 0007：drop 旧 FK → 重建为 SET NULL；
- 测试：`test_postgres_metadata.py` 新增断言 FK `ondelete == "SET NULL"`；集成测试 `test_postgres_parser_rebuild.py` 覆盖"结构重建后笔记仍在且 `section_id` 为 NULL""force-reindex 后同样存活"。

### 5.4 二次审查发现：修复本身引入了迁移 Bug（"对修复保持怀疑"的实例）

**现象**：迁移 0007 在真实 PostgreSQL 上执行会失败：

```text
psycopg.errors.UndefinedObject: constraint "fk_notes_section_id_sections" of relation "notes" does not exist
```

**根因**（SQLAlchemy 命名约定陷阱）：

- 迁移 **0005** 创建 notes 表时，三个外键是**无名约束**（`sa.ForeignKeyConstraint([...], ondelete=...)` 未指定 `name`）；
- 无名外键在 PostgreSQL 上由数据库自动命名：**`notes_section_id_fkey`**（`{表}_{列}_fkey`）；
- 迁移 **0007** drop 的却是 `fk_notes_section_id_sections`——这是 `models.py` 中 `NAMING_CONVENTION` 生成的约定名，**只在 SQLAlchemy `Base.metadata` 建库路径下才会出现**；
- Alembic 迁移里的约束对象不挂在应用 MetaData 上，**不应用命名约定** → 0005 实际生成的名字与 0007 期望的名字不一致。

**为什么本地没暴露**：需要真实 PG 的集成测试全部被 `skipif` 跳过；`test_postgres_parser_rebuild.py` 中 `upgrade_database()` 一旦在真实 PG 上运行就会立即触发。属于"测试没跑 ≠ 测试通过"的典型盲区。

### 5.5 最终修复（改动仅 1 个文件）

`migrations/versions/0007_phase3_notes_section_fk_set_null.py`：

```python
def upgrade() -> None:
    # 0005 创建的是无名 FK → PostgreSQL 默认名 notes_section_id_fkey
    op.drop_constraint("notes_section_id_fkey", "notes", type_="foreignkey")
    op.create_foreign_key(
        "fk_notes_section_id_sections", "notes", "sections",
        ["section_id"], ["section_id"], ondelete="SET NULL",
    )

def downgrade() -> None:
    op.drop_constraint("fk_notes_section_id_sections", "notes", type_="foreignkey")
    # 还原 0005 的原始状态（无名 → 默认名），保证 downgrade→upgrade 往返可用
    op.create_foreign_key(
        "notes_section_id_fkey", "notes", "sections",
        ["section_id"], ["section_id"], ondelete="CASCADE",
    )
```

要点：**downgrade 也要还原成 upgrade 期望 drop 的名字**，否则降级再升级会再次失败。

### 5.6 验证

```text
$ alembic ScriptDirectory heads → ['0007_phase3']   # 迁移链完整
$ py_compile 0007...py                              # 语法通过
```

（真实 PG 验证：`uv run pytest tests/integration/test_postgres_parser_rebuild.py tests/integration/test_postgres_phase1c_repository.py -vv`）

---

## 6. 问题 4：MiMo Worker 的 Markdown JSON 围栏导致合法分析结果被拒绝

### 6.1 业务场景与现象

在 `retrieve-offload-delegate` 模式下询问 2D-TAN 的主要方法时，主流程已经成功完成检索、Evidence Artifact Offload 和 Worker 分发，但多个 Chunk Analyst 最终被标记为失败：

```text
Worker output is not valid JSON: Expecting value
```

数据库中的 `work_units` 显示这些 Worker 均已消耗两次尝试，任务只能以 `partially_completed` 或 `no_evidence` 结束。进一步读取 Redis Checkpoint 的 `model_history` 后发现，MiMo 实际返回了字段完整、引用正确且符合 Worker Schema 的 JSON，只是在外层增加了 Markdown 代码围栏：

````text
```json
{
  "relevance": "relevant",
  "summary": "...",
  "claims": [...],
  "unresolved_questions": [...]
}
```
````

因此，这不是检索、Artifact、引用或 JSON 内容错误，而是 Provider 输出格式与 Worker 校验器之间的兼容问题。合法的 Evidence Claim 被误判为失败，最终表现为用户收到 `no_evidence`。

### 6.2 根因

`WorkerOutputValidator` 修复前直接把模型原始文本传给 `json.loads`：

```python
parsed = json.loads(answer_text)
```

当 `answer_text` 的第一个字符是 Markdown 围栏的反引号而不是 `{` 时，`json.loads` 会在第一个字符处抛出 `JSONDecodeError: Expecting value`。虽然围栏内部是合法 JSON，后续 JSON Schema、Claim 和 Citation Manifest 校验完全没有机会执行。

### 6.3 修复方案

在解析前增加一个边界严格的标准化步骤：只剥离**包裹整个 Worker 答案的单个 JSON Markdown 围栏**，然后继续使用原有 JSON 解析与校验流程。

```python
normalized = self._unwrap_json_fence(answer_text)
parsed = json.loads(normalized)
```

`_unwrap_json_fence` 使用 `fullmatch`，仅接受以下两种完整答案：

````text
```json
{...}
```
````

或：

````text
```
{...}
```
````

修复刻意不采用“从任意自然语言中截取第一个 `{...}`”的宽松策略。类似 `下面是结果：{...}` 或在围栏前后添加解释文字的输出仍然会被拒绝，避免掩盖真正不符合 Worker Contract 的响应。Token Budget、JSON Schema、字段类型、Claim 和 Citation Manifest 校验均保持不变。

### 6.4 测试与验证

新增两项回归测试：

1. 接受包裹整个答案的单个 `json` Markdown 围栏，并正确还原为标准 JSON；
2. 拒绝围栏外仍包含自然语言前缀或后缀的响应，确保兼容处理没有放宽安全边界。

验证结果：

```text
Delegation/Worker 专项测试：18 passed
全部单元测试：151 passed
git diff --check：通过
```

旧 Work Unit 的重试预算已经耗尽，因此修复后必须使用新的 `session_id` 重新执行 ROD 请求。新的 Worker 可以正常接收 MiMo 的围栏 JSON，继续完成 Schema、引用校验和 Worker Artifact 持久化。

---

## 7. 验证结果汇总（全部修复后）

| 项目          | 结果                                                                                      |
| ------------- | ----------------------------------------------------------------------------------------- |
| 全量测试      | **66 passed + 5 skipped**（4 个需真实 PG/Redis，1 个新增的 PG 集成测试）                  |
| 新增测试      | 问题1：单元 3 + 集成 1 文件；问题2：单元 4 + 集成扩展；问题3：metadata 断言 + 集成 1 文件 |
| mypy --strict | **77 个源码文件，0 错误**                                                                 |
| 迁移链        | `0001 → 0007_phase3` 头一致，`heads == ['0007_phase3']`                                   |

---

## 8. 经验总结（面试可讲的点）

1. **"声明已实现"不等于"实现正确"**：版本失效机制在单一维度（版本号）下看似完整，但缺少"绑定到上游产物"的锚点，导致跨维度（parser 变化）失效失败。审查时要沿数据流逐环节核对，而不是只看接口名。
2. **失效粒度：内容哈希 > 版本字符串**。`document_hash` 直接对解析产物建模，天然覆盖"任何导致输出变化的来源"，同时避免"版本变了但输出没变"的无谓重建。
3. **测试盲区会互相掩盖**：parser 变化测试没配结构处理器、结构重建测试没换 parser——单看都绿，组合场景是灰的。补测试要盯"组合参数空间"而非单个维度。
4. **数据库迁移的命名约定陷阱**：SQLAlchemy `NAMING_CONVENTION` 只作用于 ORM 建库路径；手写 Alembic 迁移里的无名约束由 PostgreSQL 自动命名（`{table}_{col}_fkey`）。迁移里 drop 约束前必须确认它在迁移链中的**真实名字**。
5. **downgrade 与 upgrade 必须对称可往返**：降级重建的约束名要还原成升级时能 drop 的名字，否则 downgrade→upgrade 死循环。
6. **"测试没跑" ≠ "测试通过"**：被 `skipif` 跳过的集成测试意味着核心声明（真实 PG 迁移、pgvector 检索、Redis 恢复）从未被验证。评审时要把"哪些测试没跑"和"哪些测试失败"同等对待。
7. **只读审查的价值**：先定位、出方案、再动手；修复前先补能复现的测试，避免"修了但不知道修对没有"。
8. **统一数据契约消除特殊分支**：问题 2 的 15 项衍生问题根源是 read 结果没有统一成与 search 相同的 evidence 契约——每加一个消费方（Finalizer/MiMo/Memory）就要为 read 特判一次、漏一次。**"让不同来源产出同一契约，再让所有消费方只认一种契约"** 是消除这类系统性漏判的根本方法。
9. **对修复保持怀疑（二次审查）**：问题 3 的迁移 Bug 与问题 2 的 15 项衍生问题，都是"修复通过测试"之后才被发现的。**修复本身也是待审查对象**——先补能复现的测试，再对自己写的修复做同样严格的审查。

---

## 9. 面试可能追问与回答要点

**Q1：为什么用 document_hash 而不是 parser_version 作为失效键？**
答：parser_version 是"输入侧"信息，document_hash 是"输出侧"结果。换 parser 但输出相同（如只是元数据顺序变化）时，版本号变了但内容没变，按版本号会无谓重建；按内容哈希则精确反映"下游派生数据是否真的需要重算"。同时哈希由完整 document.json（含 parser 信息）计算，parser 真变时哈希必然变，不漏判。

**Q2：存量数据 document_hash 为 NULL 怎么处理？**
答：把 NULL 视为"未知/过期"，下次 ingest 强制重建一次并回填哈希。宁可多做一次重建（幂等、成本可控），也不冒险把未经验证的结构当成"当前"。这是典型的 fail-safe 取舍。

**Q3：为什么测试没发现问题 1？**
答：两个既有测试各自只覆盖了问题的一半——parser 版本变化测试的 pipeline 没配置结构处理器，走不到重建逻辑；结构重建测试又固定用同一 parser。组合场景缺失，实现缺陷与测试盲区相互掩盖。这次补的三个单测就是针对这个组合空间。

**Q4：迁移约束名 bug 的教训？**
答：Alembic 迁移里的约束对象不挂在 ORM 的 MetaData 上，不应用命名约定；无名外键由 PG 用 `{表}_{列}_fkey` 自动命名。所以"模型里的名字"和"迁移链里的真实名字"可能不一致，drop 前必须核对迁移链历史。另外这个 bug 本地测不出来（集成测试被环境变量跳过），说明"被跳过的测试"和"失败的测试"都需要关注。

**Q5：你如何保证这次修复不回退？**
答：三层保障——单元测试覆盖失效逻辑的组合场景；集成测试在真实 PG 上走完整迁移链并断言数据存活性；mypy --strict 保证接口契约（协议方法签名）在所有实现方一致。

**Q6：问题 2 为什么修了两轮？**
答：第一轮解决的是"表面症状"（没有 project_id、passages 没有引用编号），但这只是给每条消费路径打了补丁。第二轮审查发现 15 项衍生问题，根因是 read 结果没有统一成与 search 相同的 evidence 契约——Finalizer、MiMo、Memory 各自要特判，各自漏。于是重构为**统一 evidence 契约**：read 的 passages 和 elements 都汇入标准 `evidence` 列表，所有消费方只认一种结构，隔离、引用、记忆、校验四条路径一次性对齐。这轮经历让我理解"修复的深度取决于对数据契约的抽象"，而不是补丁数量。

**Q7：这些问题在你的真实使用场景里会怎样表现出来？**
答：（结合业务场景讲）——解析器升级后检索内容与论文实际不一致但系统不报错（问题 1）；多语料库/多用户时 read 可跨项目越权、Agent 回答"展开讲讲 3.2 节"时可以编造页码（问题 2）；用户在 Section 上的笔记在重新索引后凭空消失（问题 3）。三个问题都直接伤害"可追溯、可信、不丢数据"这三个科研助手最核心的信任点。

---

## 10. 本次改动的文件清单

| 文件                                                           | 改动                                                       | 归属     |
| -------------------------------------------------------------- | ---------------------------------------------------------- | -------- |
| `migrations/versions/0006_phase3_derived_document_hash.py`     | 新增：derived_data_states.document_hash                    | 问题 1   |
| `migrations/versions/0007_phase3_notes_section_fk_set_null.py` | 新增：notes.section_id FK → SET NULL；修正约束名           | 问题 3   |
| `src/paper_agent/domain/chunk.py`                              | DerivedDataState 增加 document_hash + 校验                 | 问题 1   |
| `src/paper_agent/domain/reading.py`                            | ReadPaperRequest.project_id 必填；ReadElement.section_path | 问题 2   |
| `src/paper_agent/storage/postgres/models.py`                   | DerivedDataStateRow.document_hash；NoteRow FK SET NULL     | 问题 1/3 |
| `src/paper_agent/storage/postgres/repositories.py`             | current_document_hash()；replace_structure(document_hash)  | 问题 1   |
| `src/paper_agent/storage/postgres/read_repository.py`          | 项目/版本级隔离；项目内解析默认版本；统一错误              | 问题 2   |
| `src/paper_agent/ingestion/ports.py`                           | 协议方法签名同步                                           | 问题 1   |
| `src/paper_agent/ingestion/pipeline.py`                        | structure_is_current / _target_is_current 增加哈希校验     | 问题 1   |
| `src/paper_agent/agent/tool_adapters.py`                       | 适配器绑定 project_id；统一顶层 evidence；Element citation | 问题 2   |
| `src/paper_agent/agent/context_builder.py`                     | Finalizer 支持 E/P 双命名空间 + 每命名空间至少引用一次     | 问题 2   |
| `src/paper_agent/agent/prompts.py`                             | 提示词补充 [E编号]/[P编号]                                 | 问题 2   |
| `src/paper_agent/providers/openai_provider.py`                 | MiMo 合并 Search+Read；基于 evidence 判定；空引用契约错误  | 问题 2   |
| `src/paper_agent/application.py`                               | ReadPaperToolAdapter 注入 project_id                       | 问题 2   |
| `src/paper_agent/cli.py`                                       | read 增加 --root；输出补 citation                          | 问题 2   |
| `tests/unit/ingestion/test_pipeline.py`                        | 3 个新单测 + 测试 fake 同步                                | 问题 1   |
| `tests/unit/agent/test_tool_adapters.py`                       | 新增：统一 evidence / Element citation 序列化测试          | 问题 2   |
| `tests/unit/agent/test_runtime_and_context.py`                 | 每命名空间校验、read-only 记忆持久化测试                   | 问题 2   |
| `tests/unit/providers/test_mimo_provider.py`                   | Search+Read 合并 + 每命名空间重试测试                      | 问题 2   |
| `tests/integration/test_postgres_parser_rebuild.py`            | 新增：真实 PG 端到端（问题 1/3）                           | 问题 1/3 |
| `tests/integration/test_postgres_metadata.py`                  | 新增 document_hash 列/约束、FK ondelete 断言               | 问题 1/3 |
| `tests/integration/test_postgres_phase1c_repository.py`        | round-trip 补 document_hash 断言                           | 问题 1   |
| `tests/integration/test_phase3_agent_search_read.py`           | 双项目双 Version 越界拒绝、project_id 必填、E/P 双引用     | 问题 2   |
| `docs/使用说明.md`                                             | read 项目隔离与 P 引用说明同步                             | 问题 2   |
