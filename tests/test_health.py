"""健康检查与 OpenAPI 可见性。"""

from __future__ import annotations

from httpx import AsyncClient


async def test_health_reports_dependencies(client: AsyncClient) -> None:
    resp = await client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["checks"]["database"] == "ok"
    assert "vectorstore" in body["checks"]
    assert body["checks"]["llm"] == "fake"


async def test_liveness_does_not_touch_dependencies(client: AsyncClient) -> None:
    resp = await client.get("/health/live")
    assert resp.status_code == 200
    assert resp.json() == {"status": "alive"}


async def test_health_also_mounted_under_api_prefix(client: AsyncClient) -> None:
    resp = await client.get("/api/v1/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


async def test_docs_and_openapi_are_served(client: AsyncClient) -> None:
    assert (await client.get("/docs")).status_code == 200
    schema = await client.get("/openapi.json")
    assert schema.status_code == 200
    paths = schema.json()["paths"]
    # 验收标准要求 /docs 可访问且核心路由存在
    assert "/api/v1/chat" in paths
    assert "/api/v1/auth/login" in paths
    assert "/health" in paths
