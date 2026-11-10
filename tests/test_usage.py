"""用量台账、上下文窗口与限流。"""

from __future__ import annotations

import pytest
from httpx import AsyncClient

from app.core.cache import enforce_rate_limit
from app.core.errors import RateLimitError
from app.services.chat_service import load_history
from app.services.usage import compute_cost, estimate_tokens


def test_estimate_tokens_counts_cjk_and_ascii() -> None:
    assert estimate_tokens("") == 0
    assert estimate_tokens("你好世界") == 4
    assert estimate_tokens("abcdefgh") == 2


def test_compute_cost_scales_with_tokens() -> None:
    small = compute_cost(1000, 1000)
    large = compute_cost(10_000, 10_000)
    assert large > small > 0
    assert compute_cost(0, 0) == 0.0


async def test_usage_endpoint_records_every_call(client: AsyncClient, auth_headers: dict) -> None:
    before = await client.get("/api/v1/chat/usage", headers=auth_headers)
    assert before.status_code == 200
    base_calls = before.json()["calls"]

    for question in ("你好", "重排序是怎么工作的？"):
        resp = await client.post(
            "/api/v1/chat", headers=auth_headers, json={"message": question, "stream": False}
        )
        assert resp.status_code == 200

    after = await client.get("/api/v1/chat/usage", headers=auth_headers)
    body = after.json()
    assert body["calls"] == base_calls + 2
    assert body["prompt_tokens"] > 0
    assert body["completion_tokens"] > 0
    assert body["total_tokens"] >= body["prompt_tokens"] + body["completion_tokens"]
    assert body["cost_usd"] > 0


async def test_history_window_is_bounded(client: AsyncClient, auth_headers: dict) -> None:
    """只携带最近 N 轮，避免上下文无界增长。"""
    first = await client.post(
        "/api/v1/chat", headers=auth_headers, json={"message": "你好", "stream": False}
    )
    conversation_id = first.json()["conversation_id"]
    me = await client.get("/api/v1/auth/me", headers=auth_headers)
    user_id = me.json()["id"]

    for _ in range(4):
        await client.post(
            "/api/v1/chat",
            headers=auth_headers,
            json={"message": "再来一次", "stream": False, "conversation_id": conversation_id},
        )

    trimmed = await load_history(user_id, conversation_id, max_turns=2)
    assert len(trimmed) <= 4, "max_turns=2 时应最多 4 条消息"
    assert await load_history(user_id, conversation_id, max_turns=0) == []


async def test_rate_limit_blocks_after_threshold() -> None:
    identity = "test-identity-for-limit"
    await enforce_rate_limit(identity, limit=2)
    await enforce_rate_limit(identity, limit=2)
    with pytest.raises(RateLimitError):
        await enforce_rate_limit(identity, limit=2)
