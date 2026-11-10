"""Token 用量估算与成本台账。

约束：「每次对话必须记录 prompt/completion token 用量与估算成本」。
这里用字符级估算（中日韩 1 字≈1 token，其余 4 字符≈1 token），
真实模型返回 usage 时用真实值覆盖。
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.db.models import UsageRecord
from app.schemas.agent import Usage

_CJK_RE = re.compile(r"[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]")


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    cjk = len(_CJK_RE.findall(text))
    rest = len(text) - cjk
    return max(1, cjk + (rest + 3) // 4)


def compute_cost(prompt_tokens: int, completion_tokens: int) -> float:
    cfg = get_settings()
    cost = (
        prompt_tokens / 1000 * cfg.price_prompt_per_1k
        + completion_tokens / 1000 * cfg.price_completion_per_1k
    )
    return round(cost, 8)


def build_usage(
    *,
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    latency_ms: int = 0,
) -> Usage:
    return Usage(
        model=model,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
        cost_usd=compute_cost(prompt_tokens, completion_tokens),
        latency_ms=latency_ms,
    )


async def record_usage(
    session: AsyncSession,
    *,
    user_id: str,
    conversation_id: str,
    usage: Usage,
    route: str,
) -> UsageRecord:
    record = UsageRecord(
        user_id=user_id,
        conversation_id=conversation_id,
        model=usage.model,
        route=route,
        prompt_tokens=usage.prompt_tokens,
        completion_tokens=usage.completion_tokens,
        total_tokens=usage.total_tokens,
        cost_usd=usage.cost_usd,
        latency_ms=usage.latency_ms,
    )
    session.add(record)
    await session.flush()
    return record


@dataclass(slots=True)
class UsageSummary:
    calls: int
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost_usd: float


async def summarize_usage(session: AsyncSession, user_id: str) -> UsageSummary:
    row = (
        await session.execute(
            select(
                func.count(UsageRecord.id),
                func.coalesce(func.sum(UsageRecord.prompt_tokens), 0),
                func.coalesce(func.sum(UsageRecord.completion_tokens), 0),
                func.coalesce(func.sum(UsageRecord.total_tokens), 0),
                func.coalesce(func.sum(UsageRecord.cost_usd), 0.0),
            ).where(UsageRecord.user_id == user_id)
        )
    ).one()
    return UsageSummary(
        calls=int(row[0]),
        prompt_tokens=int(row[1]),
        completion_tokens=int(row[2]),
        total_tokens=int(row[3]),
        cost_usd=round(float(row[4]), 8),
    )
