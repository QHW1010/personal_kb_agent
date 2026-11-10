"""检索链路：embedding → 向量召回 → 重排序。

整条链路包在 call_with_resilience 里，任何一环卡住都会超时退出，
不会把请求挂死。
"""

from __future__ import annotations

from app.core.config import get_settings
from app.core.logging import get_logger
from app.core.resilience import call_with_resilience
from app.rag.embeddings import get_embedder
from app.rag.reranker import get_reranker
from app.rag.vectorstore import Hit, get_vector_store
from app.schemas.agent import Citation

logger = get_logger(__name__)


async def retrieve(
    query: str,
    *,
    top_k: int | None = None,
    top_n: int | None = None,
) -> list[Hit]:
    """两阶段检索。返回已重排的命中片段（按相关性降序）。"""
    cfg = get_settings()
    top_k = top_k or cfg.rag_top_k
    top_n = top_n or cfg.rag_rerank_top_n

    async def _pipeline() -> list[Hit]:
        embedder = get_embedder()
        vector = embedder.embed_one(query)
        store = get_vector_store()
        candidates = await store.search(vector, top_k)
        reranker = get_reranker()
        return reranker.rerank(query, candidates, top_n)

    hits = await call_with_resilience(_pipeline, label="rag.retrieve")
    logger.info("retrieve query_len=%d top_k=%d hits=%d", len(query), top_k, len(hits))
    return hits


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
            )
        )
    return citations
