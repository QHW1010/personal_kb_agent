"""往知识库里灌一份本地文档。

用法：
    python -m scripts.seed_demo                 # 灌入内置演示语料
    python -m scripts.seed_demo ./docs/a.md     # 灌入指定 markdown/txt
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from app.core.cache import init_cache
from app.db.session import dispose_engine, get_session_factory, init_models
from app.rag.ingest import ingest_document, seed_demo_knowledge
from app.rag.vectorstore import get_vector_store


async def main(path: str | None) -> None:
    await init_cache()
    await init_models()
    factory = get_session_factory()

    async with factory() as session:
        if path is None:
            count = await seed_demo_knowledge(session)
            print(f"已灌入演示语料 {count} 篇")
        else:
            file_path = Path(path)
            if not file_path.exists():
                raise SystemExit(f"文件不存在: {file_path}")
            content = file_path.read_text(encoding="utf-8")
            doc, chunks = await ingest_document(
                session, title=file_path.stem, content=content, source=str(file_path)
            )
            await session.commit()
            print(f"已入库: {doc.title} -> {chunks} 个切片")

    store = get_vector_store()
    print(f"向量库: {store.name}, 当前切片总数: {await store.count()}")
    await dispose_engine()


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1] if len(sys.argv) > 1 else None))
