"""向量库抽象：Chroma 为主，内存实现兜底。

切换 Qdrant / Pinecone 只需新增一个 VectorStore 实现并改 get_vector_store()，
上层 retriever / ingest 不受影响。

阶段二增加了两件事：
1. 集合名按 embedding provider+model+dim 版本化，换模型不会污染旧索引；
2. chunk_id 改为确定性哈希，配合 upsert 让重复入库幂等。
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import threading
from dataclasses import dataclass, field
from typing import Any, Protocol

# 必须在 import chromadb 之前设置：chromadb 在导入时就会读这个开关，
# 晚了就关不掉匿名遥测（离线环境下它会持续报错刷屏）。
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
os.environ.setdefault("CHROMA_ANONYMIZED_TELEMETRY", "False")

from app.core.config import get_settings  # noqa: E402
from app.core.logging import get_logger  # noqa: E402
from app.core.resilience import run_blocking  # noqa: E402

logger = get_logger(__name__)


@dataclass(slots=True)
class Chunk:
    chunk_id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Hit:
    chunk_id: str
    text: str
    metadata: dict[str, Any]
    score: float  # 余弦相似度，0-1


class VectorStore(Protocol):
    name: str

    async def add(self, chunks: list[Chunk], vectors: list[list[float]]) -> None: ...
    async def search(self, vector: list[float], top_k: int) -> list[Hit]: ...
    async def count(self) -> int: ...
    async def reset(self) -> None: ...
    async def delete_document(self, document_id: str) -> None: ...


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


class InMemoryVectorStore:
    """进程内实现，带 JSON 持久化，重启不丢演示数据。"""

    name = "memory"

    def __init__(self, path: str | None = None) -> None:
        self._items: dict[str, tuple[list[float], Chunk]] = {}
        self._lock = threading.Lock()
        self._path = path
        if path and os.path.exists(path):
            self._load()

    def _load(self) -> None:
        try:
            with open(self._path, encoding="utf-8") as f:  # type: ignore[arg-type]
                payload = json.load(f)
            for row in payload:
                chunk = Chunk(row["chunk_id"], row["text"], row.get("metadata", {}))
                self._items[chunk.chunk_id] = (row["vector"], chunk)
            logger.info("memory_vectorstore_loaded items=%d", len(self._items))
        except Exception as exc:  # noqa: BLE001 - 缓存文件损坏就当空库
            logger.warning("memory_vectorstore_load_failed: %s", exc)

    def _persist(self) -> None:
        if not self._path:
            return
        os.makedirs(os.path.dirname(os.path.abspath(self._path)), exist_ok=True)
        payload = [
            {"chunk_id": c.chunk_id, "text": c.text, "metadata": c.metadata, "vector": v}
            for v, c in self._items.values()
        ]
        with open(self._path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)

    async def add(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        with self._lock:
            for chunk, vector in zip(chunks, vectors):
                self._items[chunk.chunk_id] = (vector, chunk)
            self._persist()

    async def search(self, vector: list[float], top_k: int) -> list[Hit]:
        with self._lock:
            scored = [
                Hit(chunk.chunk_id, chunk.text, dict(chunk.metadata), _cosine(vector, vec))
                for vec, chunk in self._items.values()
            ]
        scored.sort(key=lambda h: h.score, reverse=True)
        return scored[:top_k]

    async def count(self) -> int:
        return len(self._items)

    async def reset(self) -> None:
        with self._lock:
            self._items.clear()
            self._persist()

    async def delete_document(self, document_id: str) -> None:
        with self._lock:
            doomed = [
                key
                for key, (_vec, chunk) in self._items.items()
                if str(chunk.metadata.get("document_id", "")) == document_id
            ]
            for key in doomed:
                self._items.pop(key, None)
            if doomed:
                self._persist()
        logger.info("memory_delete_document document_id=%s chunks=%d", document_id, len(doomed))


class ChromaVectorStore:
    """Chroma 持久化实现。Chroma SDK 是同步的，统一丢线程池 + 超时。"""

    name = "chroma"

    def __init__(self, persist_dir: str, collection: str, *, expected_dim: int | None = None) -> None:
        import logging

        # 客户端构造时就会发遥测事件，所以静音必须在构造之前完成
        logging.getLogger("chromadb.telemetry").setLevel(logging.CRITICAL)

        import chromadb
        from chromadb.config import Settings as ChromaSettings

        os.makedirs(persist_dir, exist_ok=True)
        self._dim = expected_dim
        self._client = chromadb.PersistentClient(
            path=persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False, allow_reset=True),
        )
        metadata: dict[str, Any] = {"hnsw:space": "cosine"}
        if expected_dim:
            metadata["embedding_dim"] = int(expected_dim)
        self._collection = self._client.get_or_create_collection(
            name=collection,
            metadata=metadata,
        )

        # 维度守卫：集合已存在且记录了维度，就必须和当前 embedder 一致，
        # 否则写入会抛出难懂的 InvalidDimensionError。
        existing_dim = (self._collection.metadata or {}).get("embedding_dim")
        if expected_dim and existing_dim and int(existing_dim) != int(expected_dim):
            raise ValueError(
                f"集合 {collection} 维度 {existing_dim} 与当前 embedder 维度 {expected_dim} 不一致；"
                "请更换 CHROMA_COLLECTION 或清空该集合的索引"
            )
        if expected_dim and not existing_dim:
            logger.warning(
                "collection_dim_unknown collection=%s (旧集合，未记录维度)", collection
            )
        logger.info(
            "chroma_ready dir=%s collection=%s dim=%s", persist_dir, collection, expected_dim
        )

    async def add(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        if not chunks:
            return
        if len(chunks) != len(vectors):
            raise ValueError(f"chunks({len(chunks)}) 与 vectors({len(vectors)}) 数量不一致")
        if self._dim and vectors and len(vectors[0]) != self._dim:
            raise ValueError(
                f"向量维度 {len(vectors[0])} 与集合维度 {self._dim} 不一致，拒绝写入"
            )

        def _upsert() -> None:
            # upsert 而非 add：同一 chunk_id 重复写入是幂等的，
            # 中断后重跑不会因为 duplicate id 报错
            self._collection.upsert(
                ids=[c.chunk_id for c in chunks],
                embeddings=vectors,
                documents=[c.text for c in chunks],
                metadatas=[c.metadata for c in chunks],
            )

        await run_blocking(_upsert, label="chroma.upsert")

    async def search(self, vector: list[float], top_k: int) -> list[Hit]:
        def _query() -> dict[str, Any]:
            return self._collection.query(
                query_embeddings=[vector],
                n_results=max(1, top_k),
                include=["documents", "metadatas", "distances"],
            )

        raw = await run_blocking(_query, label="chroma.query")
        ids = (raw.get("ids") or [[]])[0]
        docs = (raw.get("documents") or [[]])[0]
        metas = (raw.get("metadatas") or [[]])[0]
        dists = (raw.get("distances") or [[]])[0]
        hits: list[Hit] = []
        for chunk_id, text, meta, dist in zip(ids, docs, metas, dists):
            # cosine 空间下 distance = 1 - similarity
            hits.append(
                Hit(
                    chunk_id=chunk_id,
                    text=text or "",
                    metadata=dict(meta or {}),
                    score=max(0.0, min(1.0, 1.0 - float(dist))),
                )
            )
        return hits

    async def count(self) -> int:
        return int(await run_blocking(self._collection.count, label="chroma.count"))

    async def reset(self) -> None:
        def _reset() -> None:
            existing = self._collection.get(include=[])
            if existing.get("ids"):
                self._collection.delete(ids=existing["ids"])

        await run_blocking(_reset, label="chroma.reset")

    async def delete_document(self, document_id: str) -> None:
        def _delete() -> None:
            self._collection.delete(where={"document_id": document_id})

        await run_blocking(_delete, label="chroma.delete")
        logger.info("chroma_delete_document document_id=%s", document_id)


_store: VectorStore | None = None

_COLLECTION_SAFE_RE = re.compile(r"[^a-z0-9]+")


def _slug(text: str) -> str:
    return _COLLECTION_SAFE_RE.sub("_", text.lower()).strip("_")


def collection_name_for(base: str, provider: str, model: str | None, dim: int) -> str:
    """按模型版本化集合名，避免换 embedding 后污染旧索引。

    hashing 保持用基础名（旧数据完全不动）；
    fastembed 派生成 kb_documents_fastembed_bge_small_zh_v1_5_512 这类名字。
    Chroma 要求 3-63 个字符、以字母数字开头结尾。
    """
    if provider == "hashing":
        return base
    suffix = _slug(model or provider)[:40]
    name = f"{base}_{_slug(provider)}_{suffix}_{dim}"
    return name[:63].rstrip("_-")


def get_vector_store() -> VectorStore:
    global _store
    if _store is not None:
        return _store

    cfg = get_settings()
    if cfg.vector_backend == "memory":
        _store = InMemoryVectorStore(path=os.path.join("data", "memory_store.json"))
        logger.info("vectorstore=memory (forced)")
        return _store

    # 延迟 import：embeddings 只依赖 config，不会形成循环
    from app.rag.embeddings import get_embedder

    embedder = get_embedder()
    collection = collection_name_for(
        cfg.chroma_collection,
        cfg.embedding_provider,
        cfg.embedding_model if cfg.embedding_provider != "hashing" else None,
        embedder.dim,
    )
    try:
        _store = ChromaVectorStore(cfg.chroma_dir, collection, expected_dim=embedder.dim)
    except Exception as exc:  # noqa: BLE001 - 缺依赖/目录不可写都降级
        if cfg.vector_backend == "chroma":
            raise
        logger.warning("chroma_unavailable, falling back to memory: %s", exc)
        _store = InMemoryVectorStore(path=os.path.join("data", "memory_store.json"))
    return _store


def set_vector_store(store: VectorStore | None) -> None:
    global _store
    _store = store


def current_collection_name() -> str:
    """当前生效的集合名（脚本与 /knowledge/stats 用来暴露真实状态）。"""
    cfg = get_settings()
    from app.rag.embeddings import get_embedder

    embedder = get_embedder()
    return collection_name_for(
        cfg.chroma_collection,
        cfg.embedding_provider,
        cfg.embedding_model if cfg.embedding_provider != "hashing" else None,
        embedder.dim,
    )


def make_chunk(
    text: str,
    *,
    document_id: str,
    title: str,
    source: str,
    index: int,
    paragraph_index: int = 0,
    char_start: int = 0,
    char_end: int = 0,
    section: str = "",
    kind: str = "text",
) -> Chunk:
    """构造切片。chunk_id 由内容决定，重复入库得到同一个 id（配合 upsert 幂等）。

    元数据全部为 str/int（Chroma 不接受 None）。
    """
    raw = f"{document_id}|{index}|{text}".encode()
    return Chunk(
        chunk_id=hashlib.sha1(raw).hexdigest(),
        text=text,
        metadata={
            "document_id": document_id,
            "title": title,
            "source": source,
            "chunk_index": int(index),
            "paragraph_index": int(paragraph_index),
            "char_start": int(char_start),
            "char_end": int(char_end),
            "section": section or "",
            "kind": kind or "text",
        },
    )
