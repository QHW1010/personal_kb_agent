"""文档入库：读文件 → 切块 → 向量化 → 写向量库 + 写 DB 台账。

切分用 chunking.split_markdown（结构感知 + 段落级溯源），embedding 走
aembed 丢线程池，整段索引过程在 call_with_resilience 里，超时用
rag_model_load_timeout_seconds（首次可能要下载模型）。
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.logging import get_logger
from app.core.resilience import call_with_resilience
from app.db.models import Document
from app.rag.chunking import chunk_text, normalize_content, split_markdown  # noqa: F401
from app.rag.embeddings import aembed, get_embedder
from app.rag.vectorstore import get_vector_store, make_chunk

logger = get_logger(__name__)

SUPPORTED_SUFFIXES = {".md", ".markdown", ".txt"}
SKIP_DIR_PARTS = {"images", ".git", "__pycache__", ".venv"}

DEMO_DOCS: list[tuple[str, str, str]] = [
    (
        "项目概览：个人知识库问答 Agent",
        "本项目是一个可部署的个人知识库问答 Agent，采用 FastAPI + LangGraph + Chroma 技术栈。"
        "请求经 JWT 鉴权后进入 /api/v1/chat，由 LangGraph StateGraph 依次执行路由、检索、"
        "SQL 查询与答案生成四个节点，最终经 Pydantic 模式校验后以 SSE 流式返回。"
        "每次对话都会写入 usage_records 表，记录 prompt/completion token 与估算成本。",
        "demo:overview",
    ),
    (
        "RAG 链路说明",
        "检索增强生成分为两阶段：第一阶段用向量库做粗召回，默认取前 8 个片段；"
        "第二阶段用重排序模型精选前 3 个片段作为上下文。重排序采用向量相似度、"
        "查询词项覆盖率与短语命中三项融合打分，能显著压掉语义相近但不含答案的长段落。"
        "向量库默认使用 Chroma，可通过实现 VectorStore 协议替换为 Qdrant 或 Pinecone。",
        "demo:rag",
    ),
    (
        "SQL 查询能力",
        "当用户问题涉及统计、数量、排名、占比这类结构化问题时，路由器会把请求交给 NL2SQL 节点。"
        "该节点生成的 SQL 仅允许 SELECT/WITH 开头的只读语句，任何 INSERT、UPDATE、DELETE、DROP "
        "等写操作都会被 Pydantic 校验拒绝。系统不会在提示词里硬编码 LIMIT 限制，"
        "以免多行答案被截断；仅在执行层设置行数上限防止恶意查询拖垮数据库。",
        "demo:sql",
    ),
    (
        "运维与部署",
        "服务通过 Docker Compose 一键启动，包含 api、PostgreSQL 与 Redis 三个容器。"
        "启动后访问 /docs 可查看自动生成的 OpenAPI 文档。Redis 不可用时会自动降级为"
        "进程内存缓存，保证本地开发环境不依赖外部中间件也能跑通。"
        "数据库迁移使用 Alembic 管理，生产环境建议关闭自动建表并执行 alembic upgrade head。",
        "demo:ops",
    ),
]


@dataclass(frozen=True, slots=True)
class SourceDoc:
    """一份待入库的源文件。"""

    source: str  # 相对 docs 目录的 posix 路径，稳定且跨平台
    title: str
    content: str


def read_text_file(path: Path) -> tuple[str, str]:
    """读取文本文件，返回 (文本, 实际编码)。

    中文文档常见 GBK，硬用 utf-8 会直接抛 UnicodeDecodeError 中断整批入库。
    """
    for encoding in ("utf-8-sig", "gbk"):
        try:
            return path.read_text(encoding=encoding), encoding
        except UnicodeDecodeError:
            continue
    logger.warning("decode_fallback_replace file=%s", path)
    return path.read_text(encoding="utf-8", errors="replace"), "utf-8(replace)"


def _first_heading(text: str) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            title = stripped.lstrip("#").strip()
            if title:
                return title
        if stripped:
            # 第一个非空行不是标题就当没有标题
            break
    return ""


def iter_source_docs(docs_dir: Path, *, pattern: str | None = None) -> Iterator[SourceDoc]:
    """递归读取目录下的 Markdown / TXT，跳过图片等非语料目录。"""
    docs_dir = docs_dir.resolve()
    glob_pattern = pattern or "*"
    for path in sorted(docs_dir.rglob(glob_pattern)):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        rel = path.relative_to(docs_dir).as_posix()
        if any(part in SKIP_DIR_PARTS for part in Path(rel).parts[:-1]):
            continue
        raw, encoding = read_text_file(path)
        content = normalize_content(raw)
        if not content:
            logger.warning("skip_empty_document source=%s", rel)
            continue
        title = _first_heading(content) or path.stem
        logger.debug("read_document source=%s encoding=%s chars=%d", rel, encoding, len(content))
        yield SourceDoc(source=rel, title=title, content=content)


async def document_by_source(session: AsyncSession, source: str) -> Document | None:
    return await session.scalar(select(Document).where(Document.source == source))


async def ingest_document(
    session: AsyncSession,
    *,
    title: str,
    content: str,
    source: str = "manual",
    force: bool = False,
) -> tuple[Document, int]:
    """写入一条文档并建立索引。返回 (文档, 切片数)。

    幂等：同一 source 已存在且 force=False 时直接返回旧记录，不重复入库。
    force=True 会先删掉该文档的旧向量再重建（用于文档内容变更）。
    """
    cfg = get_settings()
    # 落库与切分必须用同一份规范化文本，否则 char_start/char_end 会反查错位
    content = normalize_content(content)

    existing = await document_by_source(session, source)
    if existing is not None and not force:
        return existing, existing.chunk_count
    if existing is not None and force:
        await get_vector_store().delete_document(existing.id)
        await session.delete(existing)
        await session.flush()

    pieces = split_markdown(
        content, size=cfg.rag_chunk_size, overlap=cfg.rag_chunk_overlap
    )

    doc = Document(title=title, source=source, content=content, chunk_count=len(pieces))
    session.add(doc)
    await session.flush()  # 先拿到 doc.id

    if pieces:
        chunks = [
            make_chunk(
                piece.text,
                document_id=doc.id,
                title=title,
                source=source,
                index=i,
                paragraph_index=piece.paragraph_index,
                char_start=piece.char_start,
                char_end=piece.char_end,
                section=piece.section,
                kind=piece.kind,
            )
            for i, piece in enumerate(pieces)
        ]

        async def _index() -> None:
            embedder = get_embedder()
            vectors = await aembed(embedder, [c.text for c in chunks])
            if vectors and len(vectors[0]) != embedder.dim:
                raise ValueError(
                    f"向量维度 {len(vectors[0])} 与 embedder.dim {embedder.dim} 不一致"
                )
            await get_vector_store().add(chunks, vectors)

        # 首次调用可能要下载并加载模型，用模型加载超时而不是默认 10s
        await call_with_resilience(
            _index,
            label="rag.ingest",
            timeout=cfg.rag_model_load_timeout_seconds,
            retries=0,
        )

    logger.info("ingested doc_id=%s source=%s chunks=%d", doc.id, source, len(pieces))
    return doc, len(pieces)


async def reindex_documents(session: AsyncSession) -> int:
    """从 DB 全量重建向量索引。

    用于「关系库里有文档、向量库里没切片」的场景（换了 embedding 模型、
    索引目录被清空等），避免出现「文档明明在库里但一条都检索不到」的静默故障。
    """
    cfg = get_settings()
    docs = list((await session.execute(select(Document))).scalars())
    if not docs:
        return 0

    store = get_vector_store()
    await store.reset()
    embedder = get_embedder()

    for doc in docs:
        pieces = split_markdown(
            doc.content, size=cfg.rag_chunk_size, overlap=cfg.rag_chunk_overlap
        )
        if not pieces:
            continue
        chunks = [
            make_chunk(
                piece.text,
                document_id=doc.id,
                title=doc.title,
                source=doc.source,
                index=i,
                paragraph_index=piece.paragraph_index,
                char_start=piece.char_start,
                char_end=piece.char_end,
                section=piece.section,
                kind=piece.kind,
            )
            for i, piece in enumerate(pieces)
        ]

        async def _index(items=chunks) -> None:
            vectors = await aembed(embedder, [c.text for c in items])
            await store.add(items, vectors)

        await call_with_resilience(
            _index,
            label="rag.reindex",
            timeout=cfg.rag_model_load_timeout_seconds,
            retries=0,
        )
        doc.chunk_count = len(pieces)

    await session.commit()
    logger.info("reindexed documents=%d chunks=%d", len(docs), await store.count())
    return len(docs)


async def ensure_index_ready(session: AsyncSession, *, seed_demo: bool = True) -> int:
    """启动期索引自检：先比对台账与向量数，必要时重建；再按需灌演示语料。

    这里用 sum(chunk_count) 与向量库条数比对，而不是「库里有没有文档」——
    换了 embedding 模型后 collection 会变空，旧逻辑会静默跳过，导致明明有
    文档却检索不到任何东西。
    """
    expected = await session.scalar(select(func.coalesce(func.sum(Document.chunk_count), 0)))
    expected = int(expected or 0)
    actual = await get_vector_store().count()

    if expected and actual != expected:
        logger.warning("index_inconsistent expected=%d actual=%d, rebuilding", expected, actual)
        await reindex_documents(session)

    if not expected and seed_demo:
        for title, content, source in DEMO_DOCS:
            await ingest_document(session, title=title, content=content, source=source)
        await session.commit()
        logger.info("seeded_demo_docs count=%d", len(DEMO_DOCS))
        return len(DEMO_DOCS)

    return 0


async def seed_demo_knowledge(session: AsyncSession) -> int:
    """兼容旧调用点：等价于 ensure_index_ready(seed_demo=True)。"""
    return await ensure_index_ready(session, seed_demo=True)
