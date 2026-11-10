"""对话链路：路由 / RAG 回答 / NL2SQL 回答 / SSE 流式。"""

from __future__ import annotations

from httpx import AsyncClient

from app.agents.nodes import classify


def test_router_classification() -> None:
    assert classify("你好") == "direct"
    assert classify("你是谁？") == "direct"
    assert classify("知识库里一共有多少篇文档？") == "sql"
    assert classify("累计 token 消耗是多少") == "sql"
    assert classify("重排序是怎么工作的？") == "rag"
    assert classify("RAG 链路分为哪几个阶段") == "rag"


async def test_rag_question_returns_citations(client: AsyncClient, auth_headers: dict) -> None:
    resp = await client.post(
        "/api/v1/chat",
        headers=auth_headers,
        json={"message": "重排序是怎么工作的？", "stream": False},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["route"] == "rag"
    assert body["citations"], "RAG 回答必须附带引用"
    assert len(body["answer"]) > 10
    assert body["usage"]["total_tokens"] > 0
    assert body["usage"]["cost_usd"] > 0
    assert 0.0 <= body["confidence"] <= 1.0


async def test_sql_question_executes_and_returns_rows(client: AsyncClient, auth_headers: dict) -> None:
    """验收标准：至少一个 SQL 场景可被自然语言触发并返回正确结果。"""
    resp = await client.post(
        "/api/v1/chat",
        headers=auth_headers,
        json={"message": "知识库里一共有多少篇文档？", "stream": False},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["route"] == "sql"
    assert body["sql"] and "count" in body["sql"].lower()
    assert body["rows"], "SQL 路线必须带回原始结果行"
    total = body["rows"][0]["total_documents"]
    assert isinstance(total, int) and total >= 4
    assert str(total) in body["answer"]


async def test_greeting_goes_direct(client: AsyncClient, auth_headers: dict) -> None:
    resp = await client.post(
        "/api/v1/chat", headers=auth_headers, json={"message": "你好", "stream": False}
    )
    assert resp.status_code == 200
    assert resp.json()["route"] == "direct"


async def test_conversation_id_is_preserved(client: AsyncClient, auth_headers: dict) -> None:
    first = await client.post(
        "/api/v1/chat", headers=auth_headers, json={"message": "你好", "stream": False}
    )
    conversation_id = first.json()["conversation_id"]

    second = await client.post(
        "/api/v1/chat",
        headers=auth_headers,
        json={"message": "再说一次", "stream": False, "conversation_id": conversation_id},
    )
    assert second.json()["conversation_id"] == conversation_id

    history = await client.get(f"/api/v1/chat/history/{conversation_id}", headers=auth_headers)
    assert history.status_code == 200
    roles = [m["role"] for m in history.json()]
    assert roles.count("user") >= 2
    assert roles.count("assistant") >= 2


async def test_stream_returns_sse_events(client: AsyncClient, auth_headers: dict) -> None:
    async with client.stream(
        "POST",
        "/api/v1/chat",
        headers=auth_headers,
        json={"message": "向量库默认用什么？", "stream": True},
    ) as resp:
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        body = "".join([chunk async for chunk in resp.aiter_text()])

    assert "event: start" in body
    assert "event: token" in body, "必须真正逐块推送 token"
    assert "event: done" in body
    assert '"route": "rag"' in body or '"route":"rag"' in body


async def test_empty_message_is_rejected(client: AsyncClient, auth_headers: dict) -> None:
    resp = await client.post(
        "/api/v1/chat", headers=auth_headers, json={"message": "", "stream": False}
    )
    assert resp.status_code == 422
