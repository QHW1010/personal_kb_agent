"""独立检索测试脚本：输入一个问题，看它能不能从 docs/ 里捞回正确的段落。

用法：
    python -m scripts.test_retrieval "重排序是怎么工作的"
    python -m scripts.test_retrieval --top-k 10 --top-n 5 "SQL 子图有几个节点"
    python -m scripts.test_retrieval --no-rerank "向量召回长什么样"
    python -m scripts.test_retrieval --json "问题"      # 机器可读
    python -m scripts.test_retrieval                    # 无参数进入交互模式

每条结果都带：排名、重排分、向量原始分、来源文件名、段落号、字符区间、标题路径。
加 --verify 会回查 DB 原文，校验 content[char_start:char_end] 是否等于切片文本。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys

from sqlalchemy import select

from app.core.cache import init_cache
from app.core.config import get_settings
from app.db.models import Document
from app.db.session import dispose_engine, get_session_factory, init_models
from app.rag.embeddings import get_embedder, warmup_embedder
from app.rag.retriever import retrieve, search_vectors
from app.rag.vectorstore import Hit, current_collection_name, get_vector_store


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="检索链路联调：问题 → 最相关段落")
    parser.add_argument("query", nargs="?", default=None, help="要检索的问题（省略则进入交互模式）")
    parser.add_argument("--top-k", type=int, default=None, help="粗召回条数（默认取配置 8）")
    parser.add_argument("--top-n", type=int, default=None, help="重排后保留条数（默认 3）")
    parser.add_argument("--no-rerank", action="store_true", help="只看向量召回，不做重排")
    parser.add_argument("--json", action="store_true", help="输出 JSON，便于管道消费")
    parser.add_argument("--verify", action="store_true", help="回查原文校验字符区间是否对齐")
    return parser.parse_args(argv)


async def _verify_offsets(hits: list[Hit], session) -> dict[str, bool]:
    """校验 content[char_start:char_end] == 切片文本。"""
    result: dict[str, bool] = {}
    for hit in hits:
        meta = hit.metadata or {}
        document_id = str(meta.get("document_id", ""))
        start, end = meta.get("char_start"), meta.get("char_end")
        if not document_id or start is None or end is None:
            result[hit.chunk_id] = False
            continue
        doc = await session.scalar(select(Document).where(Document.id == document_id))
        if doc is None:
            result[hit.chunk_id] = False
            continue
        result[hit.chunk_id] = doc.content[int(start) : int(end)] == hit.text
    return result


async def _run_query(query: str, args: argparse.Namespace, session) -> dict:
    cfg = get_settings()
    top_k = args.top_k or cfg.rag_top_k
    top_n = args.top_n or cfg.rag_rerank_top_n

    candidates = await search_vectors(query, top_k=top_k)
    vector_scores = {h.chunk_id: h.score for h in candidates}

    if args.no_rerank:
        hits = candidates[:top_n]
    else:
        hits = await retrieve(query, top_k=top_k, top_n=top_n)

    verified = await _verify_offsets(hits, session) if args.verify else {}

    return {
        "query": query,
        "top_k": top_k,
        "top_n": top_n,
        "reranked": not args.no_rerank,
        "candidates": len(candidates),
        "hits": [
            {
                "rank": i,
                "chunk_id": hit.chunk_id,
                "rerank_score": round(hit.score, 4) if not args.no_rerank else None,
                "vector_score": round(vector_scores.get(hit.chunk_id, hit.score), 4),
                "source": (hit.metadata or {}).get("source", ""),
                "title": (hit.metadata or {}).get("title", ""),
                "section": (hit.metadata or {}).get("section", ""),
                "paragraph_index": (hit.metadata or {}).get("paragraph_index"),
                "char_start": (hit.metadata or {}).get("char_start"),
                "char_end": (hit.metadata or {}).get("char_end"),
                "kind": (hit.metadata or {}).get("kind", "text"),
                "chars": len(hit.text),
                "text": hit.text,
                "offsets_ok": verified.get(hit.chunk_id) if args.verify else None,
            }
            for i, hit in enumerate(hits, 1)
        ],
    }


def _print_result(payload: dict, *, verify: bool) -> None:
    cfg = get_settings()
    embedder = get_embedder()
    print(f"问题     : {payload['query']}")
    print(f"Embedder : {embedder.name} / {cfg.embedding_model} (dim={embedder.dim})")
    print(
        f"Collection: {current_collection_name()}   "
        f"候选 {payload['candidates']} 条  → 展示 {len(payload['hits'])} 条"
        f"  {'（已重排）' if payload['reranked'] else '（未重排）'}"
    )
    print()

    if not payload["hits"]:
        print("没有命中任何片段：索引可能是空的，先跑 python -m scripts.ingest")
        return

    for item in payload["hits"]:
        rerank = f"{item['rerank_score']:.4f}" if item["rerank_score"] is not None else "  -   "
        print(f"#{item['rank']}  rerank={rerank}  vector={item['vector_score']:.4f}  {item['kind']}")
        print(f"    source : {item['source']}")
        print(f"    title  : {item['title']}")
        if item["section"]:
            print(f"    section: {item['section']}")
        print(
            f"    para   : #{item['paragraph_index']}   "
            f"chars=[{item['char_start']},{item['char_end']})  len={item['chars']}"
        )
        if verify:
            mark = "✅" if item["offsets_ok"] else "❌"
            print(f"    偏移校验: {mark} content[char_start:char_end] == 切片文本")
        preview = " ".join(item["text"].split())
        print(f"    preview: {preview[:160]}{'…' if len(preview) > 160 else ''}")
        print()


async def _main_async(args: argparse.Namespace) -> int:
    await init_cache()
    await init_models()
    factory = get_session_factory()

    embedder = get_embedder()
    if embedder.name != "hashing":
        # 首次运行需要下载模型，给个明确提示，别让人以为卡死了
        print("正在确认 embedding 模型可用（首次会下载权重）…", file=sys.stderr)
        if not await warmup_embedder():
            print("模型预热失败，检查 HF_ENDPOINT / 网络后重试", file=sys.stderr)
            await dispose_engine()
            return 1

    if args.query:
        queries = [args.query]
    else:
        print("交互模式：输入问题回车检索，空行退出。\n")
        queries = []

    async with factory() as session:
        count = await get_vector_store().count()
        if count == 0:
            print("向量库是空的，请先执行： python -m scripts.ingest", file=sys.stderr)
            await dispose_engine()
            return 1

        if queries:
            payload = await _run_query(queries[0], args, session)
            if args.json:
                print(json.dumps(payload, ensure_ascii=False, indent=2))
            else:
                _print_result(payload, verify=args.verify)
        else:
            while True:
                try:
                    query = input("问题> ").strip()
                except (EOFError, KeyboardInterrupt):
                    break
                if not query:
                    break
                payload = await _run_query(query, args, session)
                if args.json:
                    print(json.dumps(payload, ensure_ascii=False, indent=2))
                else:
                    _print_result(payload, verify=args.verify)

    await dispose_engine()
    return 0


def main(argv: list[str] | None = None) -> int:
    return asyncio.run(_main_async(_parse_args(argv)))


if __name__ == "__main__":
    raise SystemExit(main())
