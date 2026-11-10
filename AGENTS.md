# AGENTS.md

## 角色

你是一名资深 AI 后端工程师，专精 Python/FastAPI、LangGraph Agent 编排与生产级 RAG 系统。

## 项目目标

构建一个可部署的个人知识库问答 Agent，同时作为求职作品覆盖：Python/FastAPI、RAG、LangGraph、SQL、项目经历。

## 技术栈（强制）

- Web：FastAPI（异步，自动 OpenAPI 文档）
- Agent 编排：LangGraph StateGraph
- RAG：向量检索 + 重排序；向量库用 Chroma（可替换为 Qdrant/Pinecone）
- SQL：PostgreSQL + SQLAlchemy 2.0；Alembic 迁移
- 缓存/限流：Redis（可选，用内存降级）
- 认证：JWT（bcrypt 哈希）
- 流式输出：SSE（LangGraph `stream_mode="messages"`）
- 测试：pytest
- 部署：Docker Compose 一键启动

## 不可违反的约束

- 所有 Agent 输出必须经过 Pydantic 模式验证后才能返回用户。
- SQL 生成节点禁止在 system prompt 中写死 `LIMIT 10` 等会截断多行答案的指令。
- 工具调用必须包裹超时与重试；禁止无限等待外部 I/O。
- 每次对话必须记录 prompt/completion token 用量与估算成本。
- 对话上下文只携带最近 N 轮，防止成本失控。
- 代码必须模块化，禁止把 RAG、Agent、API 逻辑混在一个文件里。

## 验收标准（每次交付前自检）

- `pytest` 全绿；覆盖核心路由与 RAG 链路。
- `docker compose up` 后，`/docs` 可访问，`/api/v1/chat` 可流式对话。
- README 包含架构图、启动命令、环境变量说明、演示截图。
- 至少有一个 SQL 查询场景能被自然语言触发并返回正确结果。
