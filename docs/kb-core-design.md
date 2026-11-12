# Personal KB Agent 个人知识库问答系统核心设计文档

> 本文为设计稿（目标态）；当前实现见 README「架构」章节。

## 一、项目简介

本系统是一个生产级的个人知识库问答 Agent，旨在通过大语言模型和检索增强生成（RAG）技术，实现基于用户私有文档的精准问答。

## 二、核心强制技术栈

- **后端 Web 框架**：FastAPI（要求异步，自动生成 OpenAPI 文档）。
- **Agent 编排引擎**：LangGraph StateGraph（负责多节点状态流转和工具调用）。
- **RAG 管道**：向量检索配合重排序，向量数据库选用 Chroma（支持替换为 Qdrant 或 Pinecone）。
- **关系型数据库**：PostgreSQL，配合 SQLAlchemy 2.0 异步 ORM 和 Alembic 迁移工具。
- **认证与安全**：使用 JWT 进行鉴权，密码采用 bcrypt 哈希。
- **流式输出**：采用 SSE（Server-Sent Events）技术，基于 LangGraph 的 `stream_mode="messages"` 实现逐 token 返回。
- **部署与测试**：使用 Docker Compose 一键启动，测试框架使用 pytest。

## 三、核心 Agent 架构（LangGraph）

主图包含一个意图路由节点（`intent_router`），负责判断用户问题应该走 RAG 检索还是 SQL 查询。

如果走 SQL 查询，会进入一个独立的 SQL 子图。该子图严格按照六节点设计：

1. `check_relevance`：判断问题是否适合查库。
2. `fetch_schema`：动态读取当前数据库 schema，严禁硬编码表结构。
3. `generate_sql`：基于 schema 生成 PostgreSQL 查询。
4. `validate_sql`：执行 dry-run 验证语法，失败则带上错误信息回到 `generate_sql` 重试。
5. `execute_sql`：执行并返回结果集。
6. `narrate_result`：把结果翻译成自然语言，保留原始行数信息。

## 四、不可违反的工程约束

1. **强制输出校验**：所有 Agent 的输出必须经过 Pydantic 模式验证后才能返回用户，非法结构直接抛出 502 错误。
2. **禁用硬编码限制**：SQL 生成节点禁止在 system prompt 中写死 "LIMIT 10" 等会截断多行答案的指令，行数保护策略应放在执行层。
3. **I/O 韧性**：工具调用必须包裹超时与重试机制，禁止无限等待外部网络 I/O。
4. **成本控制**：每次对话必须记录 prompt/completion token 用量与估算成本，并且对话上下文只携带最近 N 轮，防止上下文窗口爆炸。
5. **代码分层**：禁止把 RAG、Agent、API 逻辑混在一个文件里。必须做到 api 不含 SQL、agents 不碰 HTTP、rag 不认路由。
