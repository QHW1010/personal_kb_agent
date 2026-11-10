"""外部 I/O 韧性封装：强制超时 + 指数退避重试。

AGENTS.md 约束：「工具调用必须包裹超时与重试；禁止无限等待外部 I/O」。
所有 LLM / 向量库 / DB / Redis 调用都必须经过这里，或在自身内部显式设置超时。
"""

from __future__ import annotations

import asyncio
import functools
import time
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

from app.core.config import get_settings
from app.core.errors import ToolExecutionError
from app.core.logging import get_logger

logger = get_logger(__name__)
T = TypeVar("T")


async def call_with_resilience(
    fn: Callable[[], Awaitable[T]],
    *,
    label: str = "tool",
    timeout: float | None = None,
    retries: int | None = None,
    backoff: float | None = None,
    retry_on: tuple[type[BaseException], ...] = (Exception,),
) -> T:
    """执行一个异步调用，带硬超时与有限次重试。

    - 每次尝试都受 timeout 约束，绝不会无限挂起
    - 重试耗尽后抛 ToolExecutionError（上层转 504），错误信息保留最后一次原因
    """
    cfg = get_settings()
    timeout = cfg.tool_timeout_seconds if timeout is None else timeout
    retries = cfg.tool_max_retries if retries is None else retries
    backoff = cfg.tool_backoff_seconds if backoff is None else backoff

    last_exc: BaseException | None = None
    for attempt in range(1, retries + 2):
        started = time.perf_counter()
        try:
            result = await asyncio.wait_for(fn(), timeout=timeout)
            elapsed = (time.perf_counter() - started) * 1000
            if attempt > 1:
                logger.info("tool_recovered label=%s attempt=%d elapsed_ms=%.0f", label, attempt, elapsed)
            return result
        except asyncio.CancelledError:
            raise
        except asyncio.TimeoutError as exc:
            last_exc = exc
            logger.warning("tool_timeout label=%s attempt=%d timeout=%.1fs", label, attempt, timeout)
        except retry_on as exc:  # noqa: BLE001 - 明确由调用方指定可重试类型
            last_exc = exc
            logger.warning("tool_error label=%s attempt=%d err=%s", label, attempt, exc)

        if attempt <= retries:
            await asyncio.sleep(backoff * (2 ** (attempt - 1)))

    raise ToolExecutionError(
        f"调用 {label} 失败（已重试 {retries} 次）",
        details={"label": label, "timeout_seconds": timeout, "reason": str(last_exc)[:200]},
    )


async def run_blocking(
    fn: Callable[..., T],
    *args: Any,
    label: str = "blocking",
    timeout: float | None = None,
    **kwargs: Any,
) -> T:
    """把同步阻塞调用（Chroma / bcrypt 等）丢进线程池，并施加超时。"""

    async def _inner() -> T:
        return await asyncio.to_thread(functools.partial(fn, *args, **kwargs))

    return await call_with_resilience(_inner, label=label, timeout=timeout)


def resilient(label: str, *, timeout: float | None = None) -> Callable:
    """装饰器版本，便于在 service 层复用同一套策略。"""

    def decorator(func: Callable[..., Awaitable[Any]]) -> Callable[..., Awaitable[Any]]:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            return await call_with_resilience(
                lambda: func(*args, **kwargs), label=label, timeout=timeout
            )

        return wrapper

    return decorator
