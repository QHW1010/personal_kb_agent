"""应用入口。

启动顺序：日志 → 缓存（Redis 可用则用，否则内存）→ 建表 → 首次灌入演示知识。
生产环境建议设 AUTO_CREATE_TABLES=false 并执行 `alembic upgrade head`。
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from collections.abc import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1 import auth as auth_router
from app.api.v1 import chat as chat_router
from app.api.v1 import health as health_router
from app.api.v1 import knowledge as knowledge_router
from app.core.cache import close_cache, init_cache
from app.core.config import get_settings
from app.core.errors import register_exception_handlers
from app.core.logging import configure_logging, get_logger
from app.db.session import dispose_engine, get_session_factory, init_models
from app.rag.ingest import seed_demo_knowledge
from app.rag.vectorstore import get_vector_store

logger = get_logger(__name__)
API_PREFIX = get_settings().api_v1_prefix


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    cfg = get_settings()
    configure_logging()
    logger.info("starting app=%s env=%s version=%s", cfg.app_name, cfg.env, cfg.app_version)

    await init_cache()

    if cfg.auto_create_tables:
        await init_models()

    try:
        backend = get_vector_store().name
        factory = get_session_factory()
        async with factory() as session:
            seeded = await seed_demo_knowledge(session)
        logger.info("startup_ready vectorstore=%s seeded_docs=%d", backend, seeded)
    except Exception as exc:  # noqa: BLE001 - 索引初始化失败不阻断启动，/health 会报 degraded
        logger.exception("startup_index_init_failed: %s", exc)

    yield

    logger.info("shutting down")
    await close_cache()
    await dispose_engine()


def create_app() -> FastAPI:
    cfg = get_settings()
    app = FastAPI(
        title=cfg.app_name,
        version=cfg.app_version,
        description=(
            "个人知识库问答 Agent：FastAPI + LangGraph StateGraph + 向量检索/重排 + NL2SQL。\n\n"
            "认证方式：先调用 `/api/v1/auth/register` 或 `/login` 获取 Bearer Token，"
            "再在 `Authorize` 中填入即可调试受保护接口。\n\n"
            "对话接口 `POST /api/v1/chat` 默认返回 SSE 流（stream=true）。"
        ),
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=cfg.cors_origin_list or ["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    register_exception_handlers(app)

    # /health 与 /health/live 暴露在根路径，方便容器探针
    app.include_router(health_router.router)
    # 业务接口统一挂在 /api/v1
    app.include_router(health_router.router, prefix=API_PREFIX)
    app.include_router(auth_router.router, prefix=API_PREFIX)
    app.include_router(chat_router.router, prefix=API_PREFIX)
    app.include_router(knowledge_router.router, prefix=API_PREFIX)

    @app.get("/", tags=["meta"], summary="服务元信息")
    async def root() -> dict:
        return {
            "app": cfg.app_name,
            "version": cfg.app_version,
            "docs": "/docs",
            "api": API_PREFIX,
            "endpoints": [
                f"{API_PREFIX}/auth/register",
                f"{API_PREFIX}/auth/login",
                f"{API_PREFIX}/chat",
                f"{API_PREFIX}/knowledge/ingest",
                "/health",
            ],
        }

    return app


app = create_app()
