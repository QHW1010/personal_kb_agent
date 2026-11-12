"""检索链路：embedding → 向量召回 → 重排序。

整条链路包在 call_with_resilience 里，任何一环卡住都会超时退出，
不会把请求挂死。真实 embedding 是 CPU 密集的同步调用，必须走 aembed_one
丢线程池，否则会阻塞整个事件循环（SSE 流式期间尤其明显）。
"""

from __future__ import annotations

from typing import Any

from app.core.config import get_settings
from app.core.logging import get_logger
from app.core.resilience import call_with_resilience
from app.rag.embeddings import aembed_one, get_embedder
from app.rag.reranker import get_reranker
from app.rag.vectorstore import Hit, get_vector_store
from app.schemas.agent import Citation

logger = get_logger(__name__)


def _as_int(value: Any) -> int | None:
    """旧索引里的元数据可能没有这些 key，不能直接 int()。"""
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


async def search_vectors(query: str, *, top_k: int | None = None) -> list[Hit]:
    """只做向量粗召回（不重排），供测试脚本单独展示向量分数。"""
    cfg = get_settings()
    top_k = top_k or cfg.rag_top_k

    async def _pipeline() -> list[Hit]:
        vector = await aembed_one(get_embedder(), query)
        return await get_vector_store().search(vector, top_k)

    hits = await call_with_resilience(
        _pipeline,
        label="rag.search_vectors",
        timeout=cfg.rag_retrieve_timeout_seconds,
        retries=1,
    )
    logger.info("search_vectors query_len=%d top_k=%d hits=%d", len(query), top_k, len(hits))
    return hits


async def retrieve(
    query: str,
    *,
    top_k: int | None = None,
    top_n: int | None = None,
    rerank: bool = True,
) -> list[Hit]:
    """两阶段检索。返回已重排的命中片段（按相关性降序）。"""
    cfg = get_settings()
    top_n = top_n or cfg.rag_rerank_top_n

    candidates = await search_vectors(query, top_k=top_k)
    if not candidates:
        return []
    if not rerank:
        return candidates[:top_n]

    # 重排是纯 CPU 的字符级打分（毫秒级），不需要线程池
    return get_reranker().rerank(query, candidates, top_n)


def hits_to_citations(hits: list[Hit]) -> list[Citation]:
    citations: list[Citation] = []
    for hit in hits:
        meta = hit.metadata or {}
        snippet = hit.text.strip().replace("\n", " ")
        citations.append(
            Citation(
                chunk_id=hit.chunk_id,
                document_id=str(meta.get("document_id", "")),
                title=str(meta.get("title", "未命名片段")),
                source=str(meta.get("source", "kb")),
                snippet=snippet[:400],
                score=max(0.0, min(1.0, hit.score)),
                paragraph_index=_as_int(meta.get("paragraph_index")),
                char_start=_as_int(meta.get("char_start")),
                char_end=_as_int(meta.get("char_end")),
                section=str(meta.get("section") or ""),
            )
        )
    return citations
