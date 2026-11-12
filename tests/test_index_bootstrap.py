"""启动期索引自检：seed_demo 开关与幂等性。"""

from __future__ import annotations

from sqlalchemy import func, select

from app.db.models import Document
from app.db.session import get_session_factory
from app.rag.ingest import ensure_index_ready, seed_demo_knowledge


async def _count_documents() -> int:
    factory = get_session_factory()
    async with factory() as session:
        return int(await session.scalar(select(func.count()).select_from(Document)) or 0)


async def test_seed_demo_false_does_not_add_documents() -> None:
    before = await _count_documents()
    assert before > 0, "session 级 fixture 应已灌入演示语料"

    factory = get_session_factory()
    async with factory() as session:
        seeded = await ensure_index_ready(session, seed_demo=False)

    assert seeded == 0
    assert await _count_documents() == before


async def test_seed_demo_is_idempotent() -> None:
    """已经有索引时再跑一次不能变成 8 篇。"""
    before = await _count_documents()

    factory = get_session_factory()
    async with factory() as session:
        seeded = await ensure_index_ready(session, seed_demo=True)

    assert seeded == 0
    assert await _count_documents() == before


async def test_legacy_seed_demo_knowledge_still_works() -> None:
    """旧调用点（scripts/seed_demo.py、conftest）保持可用。"""
    before = await _count_documents()

    factory = get_session_factory()
    async with factory() as session:
        seeded = await seed_demo_knowledge(session)

    assert seeded == 0
    assert await _count_documents() == before
