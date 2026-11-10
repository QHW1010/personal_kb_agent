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
from app.rag.vectorstore import get_vector_store
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
    hits = await retrieve(payload.query, top_k=payload.top_k, top_n=payload.top_k)
    return KnowledgeSearchResponse(query=payload.query, results=hits_to_citations(hits))


@router.get("/stats", response_model=KnowledgeStats, summary="索引与检索配置概览")
async def stats(session: AsyncSession = Depends(get_session)) -> KnowledgeStats:
    cfg = get_settings()
    documents = int(await session.scalar(select(func.count()).select_from(Document)) or 0)
    chunks = await get_vector_store().count()
    return KnowledgeStats(
        documents=documents,
        chunks=chunks,
        backend=get_vector_store().name,
        embedding_provider=get_embedder().name,
        reranker=get_reranker().name,
    )
