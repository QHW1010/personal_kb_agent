"""异步引擎与会话工厂。

引擎惰性创建（lru_cache），测试可通过 reset_engine() 切换数据库后重建。
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from functools import lru_cache

from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine

from app.core.config import get_settings
from app.core.logging import get_logger
from app.db.models import Base

logger = get_logger(__name__)


@lru_cache
def get_engine() -> AsyncEngine:
    cfg = get_settings()
    if cfg.is_sqlite:
        # SQLite 需要保证父目录存在
        db_path = cfg.database_url.split("///")[-1]
        if db_path and db_path != ":memory:":
            parent = os.path.dirname(os.path.abspath(db_path))
            os.makedirs(parent, exist_ok=True)
        return create_async_engine(
            cfg.database_url,
            echo=cfg.db_echo,
            connect_args={"check_same_thread": False},
        )
    return create_async_engine(
        cfg.database_url,
        echo=cfg.db_echo,
        pool_size=5,
        max_overflow=10,
        pool_pre_ping=True,
        pool_recycle=1800,
    )


@lru_cache
def get_session_factory() -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(
        bind=get_engine(),
        class_=AsyncSession,
        expire_on_commit=False,
        autoflush=False,
    )


async def get_session() -> AsyncIterator[AsyncSession]:
    """FastAPI 依赖：每请求一个会话，异常自动回滚。"""
    factory = get_session_factory()
    async with factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def init_models() -> None:
    """dev/test 下自动建表；生产走 alembic upgrade head。"""
    engine = get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("database_ready url=%s", get_settings().database_url.split("@")[-1])


async def dispose_engine() -> None:
    await get_engine().dispose()
    get_engine.cache_clear()
    get_session_factory.cache_clear()


async def reset_engine() -> None:
    """测试用：释放并按最新 settings 重建。"""
    await dispose_engine()
