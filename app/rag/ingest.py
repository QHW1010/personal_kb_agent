"""文档入库：切块 → 向量化 → 写向量库 + 写 DB 台账。"""

from __future__ import annotations

from app.core.config import get_settings
from app.core.logging import get_logger
from app.core.resilience import call_with_resilience
from app.db.models import Document
from app.rag.embeddings import get_embedder
from app.rag.vectorstore import get_vector_store, make_chunk
from sqlalchemy.ext.asyncio import AsyncSession

logger = get_logger(__name__)

DEMO_DOCS: list[tuple[str, str, str]] = [
    (
        "项目概览：个人知识库问答 Agent",
        "本项目是一个可部署的个人知识库问答 Agent，采用 FastAPI + LangGraph + Chroma 技术栈。"
        "请求经 JWT 鉴权后进入 /api/v1/chat，由 LangGraph StateGraph 依次执行路由、检索、"
        "SQL 查询与答案生成四个节点，最终经 Pydantic 模式校验后以 SSE 流式返回。"
        "每次对话都会写入 usage_records 表，记录 prompt/completion token 与估算成本。",
        "demo",
    ),
    (
        "RAG 链路说明",
        "检索增强生成分为两阶段：第一阶段用向量库做粗召回，默认取前 8 个片段；"
        "第二阶段用重排序模型精选前 3 个片段作为上下文。重排序采用向量相似度、"
        "查询词项覆盖率与短语命中三项融合打分，能显著压掉语义相近但不含答案的长段落。"
        "向量库默认使用 Chroma，可通过实现 VectorStore 协议替换为 Qdrant 或 Pinecone。",
        "demo",
    ),
    (
        "SQL 查询能力",
        "当用户问题涉及统计、数量、排名、占比这类结构化问题时，路由器会把请求交给 NL2SQL 节点。"
        "该节点生成的 SQL 仅允许 SELECT/WITH 开头的只读语句，任何 INSERT、UPDATE、DELETE、DROP "
        "等写操作都会被 Pydantic 校验拒绝。系统不会在提示词里硬编码 LIMIT 限制，"
        "以免多行答案被截断；仅在执行层设置行数上限防止恶意查询拖垮数据库。",
        "demo",
    ),
    (
        "运维与部署",
        "服务通过 Docker Compose 一键启动，包含 api、PostgreSQL 与 Redis 三个容器。"
        "启动后访问 /docs 可查看自动生成的 OpenAPI 文档。Redis 不可用时会自动降级为"
        "进程内存缓存，保证本地开发环境不依赖外部中间件也能跑通。"
        "数据库迁移使用 Alembic 管理，生产环境建议关闭自动建表并执行 alembic upgrade head。",
        "demo",
    ),
]


def chunk_text(text: str, *, size: int | None = None, overlap: int | None = None) -> list[str]:
    """按段落切块，超长段落再硬切，块间保留 overlap 字符保证跨块语义连续。"""
    cfg = get_settings()
    size = size or cfg.rag_chunk_size
    overlap = overlap if overlap is not None else cfg.rag_chunk_overlap
    text = text.strip()
    if not text:
        return []
    if len(text) <= size:
        return [text]

    chunks: list[str] = []
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    buffer = ""
    for para in paragraphs:
        if len(buffer) + len(para) + 1 <= size:
            buffer = f"{buffer}\n{para}".strip()
            continue
        if buffer:
            chunks.append(buffer)
        # 单段就超长 → 硬切成 size 长度、带 overlap 的窗口
        while len(para) > size:
            chunks.append(para[:size])
            para = para[size - overlap :]
        buffer = para
    if buffer:
        chunks.append(buffer)
    return chunks


async def ingest_document(
    session: AsyncSession,
    *,
    title: str,
    content: str,
    source: str = "manual",
) -> tuple[Document, int]:
    """写入一条文档并建立索引。返回 (文档, 切片数)。"""
    cfg = get_settings()
    pieces = chunk_text(content, size=cfg.rag_chunk_size, overlap=cfg.rag_chunk_overlap)

    doc = Document(title=title, source=source, content=content, chunk_count=len(pieces))
    session.add(doc)
    await session.flush()  # 先拿到 doc.id

    if pieces:
        chunks = [
            make_chunk(piece, document_id=doc.id, title=title, source=source, index=i)
            for i, piece in enumerate(pieces)
        ]

        async def _index() -> None:
            embedder = get_embedder()
            vectors = embedder.embed([c.text for c in chunks])
            await get_vector_store().add(chunks, vectors)

        await call_with_resilience(_index, label="rag.ingest")
    logger.info("ingested doc_id=%s title=%s chunks=%d", doc.id, title, len(pieces))
    return doc, len(pieces)


async def reindex_documents(session: AsyncSession) -> int:
    """从 DB 全量重建向量索引。

    用于"关系库里有文档、向量库里没切片"的场景（换了向量后端、索引目录被清空等），
    避免出现"文档明明在库里但一条都检索不到"的静默故障。
    """
    from sqlalchemy import select

    cfg = get_settings()
    docs = list((await session.execute(select(Document))).scalars())
    if not docs:
        return 0

    store = get_vector_store()
    await store.reset()
    embedder = get_embedder()

    for doc in docs:
        pieces = chunk_text(doc.content, size=cfg.rag_chunk_size, overlap=cfg.rag_chunk_overlap)
        if not pieces:
            continue
        chunks = [
            make_chunk(piece, document_id=doc.id, title=doc.title, source=doc.source, index=i)
            for i, piece in enumerate(pieces)
        ]

        async def _index(items=chunks) -> None:
            await store.add(items, embedder.embed([c.text for c in items]))

        await call_with_resilience(_index, label="rag.reindex")
        doc.chunk_count = len(pieces)

    await session.commit()
    logger.info("reindexed documents=%d chunks=%d", len(docs), await store.count())
    return len(docs)


async def seed_demo_knowledge(session: AsyncSession) -> int:
    """首次启动灌入演示文档；若库里有文档但索引为空则自动重建索引。"""
    from sqlalchemy import func, select

    existing = await session.scalar(select(func.count()).select_from(Document))
    if existing:
        if await get_vector_store().count() == 0:
            logger.warning("documents_exist_but_index_empty, rebuilding index")
            await reindex_documents(session)
        return 0
    for title, content, source in DEMO_DOCS:
        await ingest_document(session, title=title, content=content, source=source)
    await session.commit()
    logger.info("seeded_demo_docs count=%d", len(DEMO_DOCS))
    return len(DEMO_DOCS)
