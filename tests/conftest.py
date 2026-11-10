"""pytest 全局配置。

关键点：
1. 环境变量必须在 import app 之前设置（Settings 是模块级缓存）。
2. 所有异步测试共享一个 session 级事件循环，否则 aiosqlite 引擎会跨 loop 报错。
3. 向量库强制走 memory 且不落盘，保证测试之间互不污染。
"""

from __future__ import annotations

import inspect
import os
import tempfile
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

_TMP = Path(tempfile.mkdtemp(prefix="kbtest-"))

os.environ.update(
    {
        "ENV": "test",
        "DEBUG": "false",
        "DATABASE_URL": f"sqlite+aiosqlite:///{(_TMP / 'test.db').as_posix()}",
        "AUTO_CREATE_TABLES": "true",
        "VECTOR_BACKEND": "memory",
        "CHROMA_DIR": (_TMP / "chroma").as_posix(),
        "LLM_PROVIDER": "fake",
        "EMBEDDING_PROVIDER": "hashing",
        "RERANKER": "lexical",
        "SECRET_KEY": "test-secret-key-should-be-32-bytes-min",
        "REDIS_URL": "",
        "RATE_LIMIT_PER_MINUTE": "10000",
        "LOG_LEVEL": "WARNING",
        "HISTORY_MAX_TURNS": "6",
        "TOOL_TIMEOUT_SECONDS": "10",
    }
)

from httpx import ASGITransport, AsyncClient  # noqa: E402

from app.core.cache import init_cache  # noqa: E402
from app.db.session import get_session_factory, init_models  # noqa: E402
from app.main import app as fastapi_app  # noqa: E402
from app.rag.ingest import seed_demo_knowledge  # noqa: E402
from app.rag.vectorstore import InMemoryVectorStore, set_vector_store  # noqa: E402


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """把所有协程测试钉在 session loop 上。

    append=False 让我们的 marker 排在 pytest-asyncio 自动添加的那个之前，
    get_closest_marker 才会读到 loop_scope="session"。
    """
    for item in items:
        func = getattr(item, "function", None)
        if func is not None and inspect.iscoroutinefunction(func):
            item.add_marker(pytest.mark.asyncio(loop_scope="session"), append=False)


@pytest.fixture(scope="session", autouse=True)
async def _bootstrap() -> AsyncIterator[None]:
    await init_cache()
    set_vector_store(InMemoryVectorStore(path=None))
    await init_models()
    factory = get_session_factory()
    async with factory() as session:
        await seed_demo_knowledge(session)
    yield


@pytest.fixture()
async def client() -> AsyncIterator[AsyncClient]:
    transport = ASGITransport(app=fastapi_app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as ac:
        yield ac


@pytest.fixture()
async def auth_headers(client: AsyncClient) -> dict[str, str]:
    """每个测试用独立邮箱注册，避免相互影响。"""
    import uuid

    email = f"user-{uuid.uuid4().hex[:8]}@example.com"
    resp = await client.post(
        "/api/v1/auth/register", json={"email": email, "password": "sup3rsecret!"}
    )
    assert resp.status_code == 201, resp.text
    token = resp.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}
