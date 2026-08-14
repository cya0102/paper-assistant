# Phase 1B 开发总结报告

## 阶段目标

Phase 1B 将 Phase 1A 的 Ingestion 骨架扩展为可处理真实 PDF 的增量导入 MVP。系统现在能够解析 PDF、提取轻量元数据、判断论文与版本身份、生成 Canonical Parsed Document，并根据文件、正文、Parser 和 Schema 版本决定是否重跑。

## 本阶段新增功能

### 真实 PDF Parser

- 增加 PyMuPDF 默认适配器。
- 增加 Poppler CLI 后备适配器。
- 提取页面尺寸、文本块、BBox、阅读顺序和基础 Heading/Paragraph 类型。
- 对加密、损坏、空 PDF 和 Parser 不可用返回稳定错误码。

### Metadata 与规范化

- 提取 PDF Title、Authors、Subject、Keywords、Page Count。
- 从正文识别 DOI、arXiv ID 和年份。
- 对 Title、Authors、DOI、arXiv、Unicode、空白和连字符进行规范化。
- 保存 Metadata 来源与置信度。

### Content Fingerprint

- 从规范化正文计算 SHA-256 `content_hash`。
- 忽略重复页眉、页脚和页码。
- 支持不同 PDF 二进制但正文相同时复用已有 PaperVersion 和 Parsed Document。

### Paper 与 Version Identity

- 分层使用 Content Hash、DOI、arXiv ID、Normalized Title + Authors。
- Content Hash 相同时复用现有 Version。
- DOI/arXiv 相同但正文不同时创建同 Paper 下的新 Version。
- 低置信度条件不会自动模糊合并。

### 增量与恢复

- 实现 `force_reindex`。
- Parser Version 或 Canonical Schema Version 改变时重新解析。
- 失败文件可在下一次 Ingestion 中重试。
- 全项目扫描时把已移除路径标记为 `missing`。
- PostgreSQL 通过 `ON CONFLICT` 防止并发创建相同 SHA-256 File。
- 每篇文件继续保持独立失败隔离。

### PostgreSQL 与 CLI

- 增加 0002 migration，保存 normalized metadata、content hash、page count 和 metadata evidence。
- 增加稳定项目 ID manifest：`.paper-agent/project.json`。
- 增加 `paper-agent init`、`ingest`、`status`、`db-upgrade` 命令。
- 生成并提交 `uv.lock`。

## 验证结果

- PyMuPDF 对真实测试 PDF 的 Metadata 和 Canonical Document 输出通过。
- Poppler 对真实测试 PDF 的 Metadata 和 Canonical Document 输出通过。
- SHA-256、Content Hash、Paper/Version、force reindex、Parser Version、missing、失败隔离测试通过。
- Alembic upgrade 和 downgrade PostgreSQL SQL 生成通过。
- mypy strict 检查通过。

当前机器没有运行中的 PostgreSQL Server，因此 Migration 已完成离线 PostgreSQL DDL 往返验证，但尚未在本机执行数据库连接级 upgrade/downgrade。

## 当前可用程度

在提供 PostgreSQL 连接后，项目可以初始化本地论文目录、递归扫描 PDF、完成二进制和正文去重、解析 PDF、建立 Paper/Version 身份，并持久化 `document.json` 和 `document.md`。它已经是可运行的论文导入工具，但还不是检索或问答系统。

## 下一阶段目标：Phase 1C

下一阶段聚焦从 Canonical Parsed Document 生成可检索的结构化论文数据：

1. Section Tree Domain、Schema 和 Builder。
2. Heading hierarchy 与 Block-to-Section 归属。
3. Figure/Table/Equation/Algorithm 基础 Element 模型。
4. Semantic Blocks 与 Dependency Groups。
5. Section-aware Semantic Chunker。
6. Chunk 对 Paper、Version、Section、Page、Block、Element 的完整 provenance。
7. Chunking Version、阶段恢复和重建测试。
8. 真实 PostgreSQL 集成环境与 Migration 往返测试。

Phase 1C 不进入 Embedding、Search/RAG、Agent Runtime 或 Redis。
