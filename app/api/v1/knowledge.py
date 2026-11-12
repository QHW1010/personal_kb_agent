"""知识库路由：入库 / 检索 / 统计。"""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_user
from app.core.config import get_settings
from app.core.logging import get_logger
from app.db.models import Document, User
from app.db.session import get_session
from app.rag.embeddings import get_embedder
from app.rag.ingest import ingest_document
from app.rag.reranker import get_reranker
from app.rag.retriever import hits_to_citations, retrieve
from app.rag.vectorstore import current_collection_name, get_vector_store
from app.schemas.chat import (
    KnowledgeIngestRequest,
    KnowledgeIngestResponse,
    KnowledgeSearchRequest,
    KnowledgeSearchResponse,
    KnowledgeStats,
)

logger = get_logger(__name__)
router = APIRouter(prefix="/knowledge", tags=["knowledge"])


@router.post("/ingest", response_model=KnowledgeIngestResponse, summary="上传一段文档并建索引")
async def ingest(
    payload: KnowledgeIngestRequest,
    current: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
) -> KnowledgeIngestResponse:
    doc, chunks = await ingest_document(
        session, title=payload.title, content=payload.content, source=payload.source
    )
    return KnowledgeIngestResponse(
        document_id=doc.id,
        title=doc.title,
        chunks=chunks,
        backend=get_vector_store().name,
    )


@router.post("/search", response_model=KnowledgeSearchResponse, summary="只跑检索链路（含重排）")
async def search(
    payload: KnowledgeSearchRequest,
    current: User = Depends(get_current_user),
) -> KnowledgeSearchResponse:
    cfg = get_settings()
    # top_k 是粗召回条数；重排后条数不能超过它，也不能超过全局配置
    hits = await retrieve(
        payload.query,
        top_k=payload.top_k,
        top_n=min(payload.top_k, cfg.rag_rerank_top_n),
    )
    return KnowledgeSearchResponse(query=payload.query, results=hits_to_citations(hits))


@router.get("/stats", response_model=KnowledgeStats, summary="索引与检索配置概览")
async def stats(session: AsyncSession = Depends(get_session)) -> KnowledgeStats:
    cfg = get_settings()
    documents = int(await session.scalar(select(func.count()).select_from(Document)) or 0)
    embedder = get_embedder()
    return KnowledgeStats(
        documents=documents,
        chunks=await get_vector_store().count(),
        backend=get_vector_store().name,
        embedding_provider=embedder.name,
        reranker=get_reranker().name,
        embedding_model=cfg.embedding_model if cfg.embedding_provider != "hashing" else "hashing",
        embedding_dim=embedder.dim,
        collection=current_collection_name(),
    )
