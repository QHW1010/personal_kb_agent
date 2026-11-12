"""健康检查：区分 liveness 与依赖可用性。"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.llm import describe_llm
from app.core.cache import get_cache
from app.core.config import get_settings
from app.core.logging import get_logger
from app.db.session import get_session
from app.rag.vectorstore import get_vector_store

logger = get_logger(__name__)
router = APIRouter(tags=["health"])


@router.get("/health", summary="健康检查", response_model=dict[str, Any])
async def health(session: AsyncSession = Depends(get_session)) -> dict[str, Any]:
    cfg = get_settings()
    checks: dict[str, str] = {}

    started_ok = True

    try:
        await session.execute(text("SELECT 1"))
        checks["database"] = "ok"
    except Exception as exc:  # noqa: BLE001
        checks["database"] = f"error: {type(exc).__name__}"
        started_ok = False

    try:
        count = await get_vector_store().count()
        checks["vectorstore"] = f"ok ({count} chunks, {get_vector_store().name})"
    except Exception as exc:  # noqa: BLE001
        checks["vectorstore"] = f"error: {type(exc).__name__}"
        started_ok = False

    try:
        cache = get_cache()
        checks["cache"] = "ok" if await cache.ping() else "degraded"
    except Exception as exc:  # noqa: BLE001
        checks["cache"] = f"degraded: {type(exc).__name__}"

    # 只报"配没配"，不回显 Key；模型名与网关地址对排障很有用，可以放
    llm_info = describe_llm()
    checks["llm"] = (
        f"ok ({llm_info['model']} via {llm_info['base_url'] or 'default'})"
        if llm_info["provider"] == "openai" and llm_info["api_key_configured"]
        else f"{llm_info['provider']} ({llm_info['model']})"
    )

    return {
        "status": "ok" if started_ok else "degraded",
        "app": cfg.app_name,
        "version": cfg.app_version,
        "env": cfg.env,
        "checks": checks,
        "llm": llm_info,
    }


@router.get("/health/live", summary="存活探针", response_model=dict[str, str])
async def live() -> dict[str, str]:
    """容器 livenessProbe 用，不触碰任何外部依赖。"""
    return {"status": "alive"}
