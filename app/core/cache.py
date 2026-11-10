"""缓存与限流：优先 Redis，连接失败自动降级为进程内存实现。

降级是明确设计：本地开发 / CI 不应因为没起 Redis 就跑不起来。
"""

from __future__ import annotations

import asyncio
import time
from typing import Protocol

from app.core.config import get_settings
from app.core.errors import RateLimitError
from app.core.logging import get_logger

logger = get_logger(__name__)


class CacheBackend(Protocol):
    name: str

    async def get(self, key: str) -> str | None: ...
    async def set(self, key: str, value: str, ttl: int | None = None) -> None: ...
    async def incr(self, key: str, ttl: int | None = None) -> int: ...
    async def delete(self, key: str) -> None: ...
    async def ping(self) -> bool: ...
    async def close(self) -> None: ...


class MemoryCache:
    """带 TTL 的进程内实现，仅适用于单进程/测试场景。"""

    name = "memory"

    def __init__(self) -> None:
        self._data: dict[str, tuple[str, float | None]] = {}
        self._lock = asyncio.Lock()

    def _expired(self, key: str) -> bool:
        item = self._data.get(key)
        if item is None:
            return True
        _, expires_at = item
        if expires_at is not None and expires_at <= time.monotonic():
            self._data.pop(key, None)
            return True
        return False

    async def get(self, key: str) -> str | None:
        async with self._lock:
            if self._expired(key):
                return None
            return self._data[key][0]

    async def set(self, key: str, value: str, ttl: int | None = None) -> None:
        async with self._lock:
            expires_at = time.monotonic() + ttl if ttl else None
            self._data[key] = (value, expires_at)

    async def incr(self, key: str, ttl: int | None = None) -> int:
        async with self._lock:
            if self._expired(key):
                self._data[key] = ("1", time.monotonic() + ttl if ttl else None)
                return 1
            value, expires_at = self._data[key]
            new_value = int(value) + 1
            self._data[key] = (str(new_value), expires_at)
            return new_value

    async def delete(self, key: str) -> None:
        async with self._lock:
            self._data.pop(key, None)

    async def ping(self) -> bool:
        return True

    async def close(self) -> None:
        self._data.clear()


class RedisCache:
    name = "redis"

    def __init__(self, client) -> None:  # noqa: ANN001 - redis.asyncio.Redis
        self._client = client

    async def get(self, key: str) -> str | None:
        return await self._client.get(key)

    async def set(self, key: str, value: str, ttl: int | None = None) -> None:
        await self._client.set(key, value, ex=ttl)

    async def incr(self, key: str, ttl: int | None = None) -> int:
        value = await self._client.incr(key)
        if value == 1 and ttl:
            await self._client.expire(key, ttl)
        return int(value)

    async def delete(self, key: str) -> None:
        await self._client.delete(key)

    async def ping(self) -> bool:
        return bool(await self._client.ping())

    async def close(self) -> None:
        await self._client.aclose()


_cache: CacheBackend | None = None


async def init_cache() -> CacheBackend:
    """应用启动时调用一次；Redis 不可用则静默降级。"""
    global _cache
    cfg = get_settings()
    if cfg.redis_url:
        try:
            import redis.asyncio as aioredis

            client = aioredis.from_url(
                cfg.redis_url,
                encoding="utf-8",
                decode_responses=True,
                socket_connect_timeout=2,
                socket_timeout=2,
            )
            await asyncio.wait_for(client.ping(), timeout=2)
            _cache = RedisCache(client)
            logger.info("cache_backend=redis url=%s", cfg.redis_url)
            return _cache
        except Exception as exc:  # noqa: BLE001 - 任何连接问题都降级
            if cfg.cache_required:
                raise
            logger.warning("redis_unavailable, falling back to memory cache: %s", exc)
    _cache = MemoryCache()
    logger.info("cache_backend=memory")
    return _cache


def get_cache() -> CacheBackend:
    global _cache
    if _cache is None:
        _cache = MemoryCache()
    return _cache


async def close_cache() -> None:
    global _cache
    if _cache is not None:
        await _cache.close()
        _cache = None


async def enforce_rate_limit(identity: str, *, limit: int | None = None, window: int = 60) -> int:
    """滑动窗口近似限流：固定 60s 窗口计数，超限抛 429。"""
    cfg = get_settings()
    limit = cfg.rate_limit_per_minute if limit is None else limit
    if limit <= 0:
        return 0
    cache = get_cache()
    key = f"rl:{identity}:{int(time.time()) // window}"
    count = await cache.incr(key, ttl=window)
    if count > limit:
        raise RateLimitError(
            f"请求过于频繁，请稍后重试（上限 {limit} 次/分钟）",
            details={"limit": limit, "window_seconds": window},
        )
    return count
