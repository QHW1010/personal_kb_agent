"""向量库抽象：Chroma 为主，内存实现兜底。

切换 Qdrant / Pinecone 只需新增一个 VectorStore 实现并改 get_vector_store()，
上层 retriever / ingest 不受影响。
"""

from __future__ import annotations

import json
import math
import os
import threading
import uuid
from dataclasses import dataclass, field
from typing import Any, Protocol

# 必须在 import chromadb 之前设置：chromadb 在导入时就会读这个开关，
# 晚了就关不掉匿名遥测（离线环境下它会持续刷 ERROR 日志）。
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


class ChromaVectorStore:
    """Chroma 持久化实现。Chroma SDK 是同步的，统一丢线程池 + 超时。"""

    name = "chroma"

    def __init__(self, persist_dir: str, collection: str) -> None:
        import logging

        # 客户端构造时就会发遥测事件，所以静音必须在构造之前完成
        logging.getLogger("chromadb.telemetry").setLevel(logging.CRITICAL)

        import chromadb
        from chromadb.config import Settings as ChromaSettings

        os.makedirs(persist_dir, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False, allow_reset=True),
        )
        self._collection = self._client.get_or_create_collection(
            name=collection,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("chroma_ready dir=%s collection=%s", persist_dir, collection)

    async def add(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        if not chunks:
            return

        def _add() -> None:
            self._collection.add(
                ids=[c.chunk_id for c in chunks],
                embeddings=vectors,
                documents=[c.text for c in chunks],
                metadatas=[c.metadata for c in chunks],
            )

        await run_blocking(_add, label="chroma.add")

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


_store: VectorStore | None = None


def get_vector_store() -> VectorStore:
    global _store
    if _store is not None:
        return _store

    cfg = get_settings()
    if cfg.vector_backend == "memory":
        _store = InMemoryVectorStore(path=os.path.join("data", "memory_store.json"))
        logger.info("vectorstore=memory (forced)")
        return _store

    try:
        _store = ChromaVectorStore(cfg.chroma_dir, cfg.chroma_collection)
    except Exception as exc:  # noqa: BLE001 - 缺依赖/目录不可写都降级
        if cfg.vector_backend == "chroma":
            raise
        logger.warning("chroma_unavailable, falling back to memory: %s", exc)
        _store = InMemoryVectorStore(path=os.path.join("data", "memory_store.json"))
    return _store


def set_vector_store(store: VectorStore | None) -> None:
    global _store
    _store = store


def make_chunk(text: str, *, document_id: str, title: str, source: str, index: int) -> Chunk:
    return Chunk(
        chunk_id=uuid.uuid4().hex,
        text=text,
        metadata={
            "document_id": document_id,
            "title": title,
            "source": source,
            "chunk_index": index,
        },
    )
