"""把 docs/ 下的真实文档摄入向量库。

用法：
    python -m scripts.ingest --download-only        # 只预热/下载模型，不写库
    python -m scripts.ingest --dry-run              # 只切分，打印统计，不写 DB/向量库
    python -m scripts.ingest                        # 正式入库（已存在的跳过）
    python -m scripts.ingest --force                # 覆盖已存在的同 source 文档
    python -m scripts.ingest --docs-dir docs --pattern "*.md"

输出里一定会打印「实际 embedder + 实际 collection 名 + 维度」——
这是判断「是不是在用旧索引 / 旧模型」的唯一可靠手段。
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

from app.core.cache import init_cache
from app.core.config import get_settings
from app.db.session import dispose_engine, get_session_factory, init_models
from app.rag.chunking import split_markdown
from app.rag.embeddings import get_embedder, warmup_embedder
from app.rag.ingest import (
    document_by_source,
    ingest_document,
    iter_source_docs,
    reindex_documents,
)
from app.rag.vectorstore import current_collection_name, get_vector_store
from sqlalchemy import func, select

from app.db.models import Document


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="把 docs/ 下的文档摄入 Chroma 向量库")
    parser.add_argument("--docs-dir", default="docs", help="文档目录（默认 docs）")
    parser.add_argument("--pattern", default=None, help="文件通配符，如 '*.md'")
    parser.add_argument("--force", action="store_true", help="覆盖已入库的同 source 文档")
    parser.add_argument("--dry-run", action="store_true", help="只切分并打印统计，不写任何数据")
    parser.add_argument("--download-only", action="store_true", help="只下载/加载模型后退出")
    parser.add_argument("--limit", type=int, default=None, help="最多处理多少个文件")
    return parser.parse_args(argv)


def _print_env() -> None:
    cfg = get_settings()
    embedder = get_embedder()
    print(f"Embedder   : {embedder.name} / {cfg.embedding_model} (dim={embedder.dim})")
    print(f"Collection : {current_collection_name()}")
    print(f"Backend    : {get_vector_store().name}")
    print(f"Chunk      : size={cfg.rag_chunk_size} overlap={cfg.rag_chunk_overlap}")
    print()


async def _run(args: argparse.Namespace) -> int:
    docs_dir = Path(args.docs_dir)
    if not docs_dir.is_dir():
        print(f"目录不存在: {docs_dir.resolve()}", file=sys.stderr)
        return 2

    _print_env()

    if args.download_only:
        print("正在预热 embedding 模型（首次会从 hf-mirror 下载权重）…")
        started = time.perf_counter()
        ok = await warmup_embedder()
        print(f"预热{'成功' if ok else '失败'}，用时 {time.perf_counter() - started:.1f}s")
        return 0 if ok else 1

    docs = list(iter_source_docs(docs_dir, pattern=args.pattern))
    if args.limit:
        docs = docs[: args.limit]
    if not docs:
        print(f"在 {docs_dir} 下没找到 .md/.markdown/.txt 文档")
        return 1

    cfg = get_settings()

    if args.dry_run:
        total = 0
        for i, doc in enumerate(docs, 1):
            pieces = split_markdown(doc.content, size=cfg.rag_chunk_size, overlap=cfg.rag_chunk_overlap)
            total += len(pieces)
            print(f"[{i}/{len(docs)}] {doc.source}  title={doc.title!r}  chars={len(doc.content)}  chunks={len(pieces)}")
            for piece in pieces[:3]:
                head = piece.text.splitlines()[0][:50]
                print(
                    f"      #{piece.paragraph_index} chars=[{piece.char_start},{piece.char_end})"
                    f" kind={piece.kind} section={piece.section!r} :: {head}"
                )
            if len(pieces) > 3:
                print(f"      … 其余 {len(pieces) - 3} 片略")
        print(f"\n汇总: {len(docs)} 篇文档，共 {total} 个切片（未写入任何数据）")
        return 0

    await init_cache()
    await init_models()
    factory = get_session_factory()

    added = skipped = failed = 0
    total_chunks = 0
    failures: list[tuple[str, str]] = []

    for i, doc in enumerate(docs, 1):
        started = time.perf_counter()
        try:
            async with factory() as session:
                existing = await document_by_source(session, doc.source)
                if existing is not None and not args.force:
                    skipped += 1
                    print(f"[{i}/{len(docs)}] {doc.source}  skip (已入库，--force 可覆盖)")
                    continue
                _, chunks = await ingest_document(
                    session,
                    title=doc.title,
                    content=doc.content,
                    source=doc.source,
                    force=args.force,
                )
                await session.commit()
            added += 1
            total_chunks += chunks
            elapsed = time.perf_counter() - started
            print(
                f"[{i}/{len(docs)}] {doc.source}  ok  title={doc.title!r} "
                f"chunks={chunks} elapsed={elapsed:.1f}s"
            )
        except Exception as exc:  # noqa: BLE001 - 单篇失败不能中断整批
            failed += 1
            failures.append((doc.source, f"{type(exc).__name__}: {exc}"))
            print(f"[{i}/{len(docs)}] {doc.source}  FAIL {type(exc).__name__}: {exc}")

    store = get_vector_store()
    async with factory() as session:
        ledger = int(
            await session.scalar(select(func.coalesce(func.sum(Document.chunk_count), 0))) or 0
        )
        actual = await store.count()
        if ledger != actual and ledger > 0:
            print(f"\n台账切片 {ledger} 与向量库 {actual} 不一致，正在全量重建索引…")
            await reindex_documents(session)
            actual = await store.count()

    print()
    print(f"汇总: 新增 {added} 篇 / 跳过 {skipped} / 失败 {failed}")
    print(f"本次新增切片 {total_chunks} | 向量库总数 {actual} | 台账切片 {ledger}")
    print(f"Collection : {current_collection_name()}")
    for source, reason in failures:
        print(f"  失败 {source}: {reason}", file=sys.stderr)

    await dispose_engine()
    return 1 if failed else 0


def main(argv: list[str] | None = None) -> int:
    return asyncio.run(_run(_parse_args(argv)))


if __name__ == "__main__":
    raise SystemExit(main())
