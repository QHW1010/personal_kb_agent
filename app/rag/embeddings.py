"""Embedding 抽象与实现。

三种实现，通过 EMBEDDING_PROVIDER 切换，上层接口（Embedder 协议）不变：

- hashing：确定性、零下载、纯 CPU，CI 与离线兜底（默认）
- fastembed：ONNX 真实语义模型，复用 onnxruntime，不需要 torch（阶段二主用）
- sentence-transformers：历史实现，需要 torch，保留但不再推荐

同步 encode 一律通过 aembed()/aembed_one() 丢进线程池，避免阻塞事件循环。
"""

from __future__ import annotations

import asyncio
import hashlib
import importlib.util
import math
import os
import re
from typing import Protocol

from app.core.config import get_settings
from app.core.errors import ToolExecutionError
from app.core.logging import get_logger
from app.core.resilience import call_with_resilience

logger = get_logger(__name__)

_WORD_RE = re.compile(r"[a-zA-Z0-9_]{2,}")
_CJK_RE = re.compile(r"[\u4e00-\u9fff]")

# fastembed 模型名 → 向量维度。
# 构造期就要知道维度（用来派生 collection 名），但绝不能为此去下载权重，
# 所以这里用静态表兜底，表里没有的再问 fastembed 的注册表（同样不下载）。
_KNOWN_DIMS: dict[str, int] = {
    "BAAI/bge-small-zh-v1.5": 512,
    "BAAI/bge-small-en-v1.5": 384,
    "sentence-transformers/all-MiniLM-L6-v2": 384,
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2": 384,
    "jinaai/jina-embeddings-v2-base-zh": 768,
    "intfloat/multilingual-e5-large": 1024,
}


class Embedder(Protocol):
    name: str
    dim: int

    def embed(self, texts: list[str]) -> list[list[float]]: ...
    def embed_one(self, text: str) -> list[float]: ...


def tokenize(text: str) -> list[str]:
    """中英混排分词：英文词 + 中文单字 + 中文二元组。

    二元组让"知识库"这类词能作为一个信号命中，单字保证未登录词也有覆盖。
    """
    lowered = text.lower()
    tokens: list[str] = _WORD_RE.findall(lowered)
    cjk = _CJK_RE.findall(lowered)
    tokens.extend(cjk)
    tokens.extend(a + b for a, b in zip(cjk, cjk[1:]))
    return tokens


def _l2_normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vector))
    if norm == 0:
        return vector
    return [v / norm for v in vector]


class HashingEmbedder:
    """Hashing trick + 符号哈希，L2 归一化后可直接用点积算余弦。"""

    name = "hashing"

    def __init__(self, dim: int = 384) -> None:
        self.dim = dim

    def _bucket(self, token: str) -> tuple[int, float]:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        value = int.from_bytes(digest, "big")
        index = value % self.dim
        sign = 1.0 if (value >> 63) & 1 else -1.0
        return index, sign

    def embed_one(self, text: str) -> list[float]:
        vector = [0.0] * self.dim
        tokens = tokenize(text)
        if not tokens:
            return vector
        for token in tokens:
            index, sign = self._bucket(token)
            vector[index] += sign
        return _l2_normalize(vector)

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_one(t) for t in texts]


def _resolve_dim(model_name: str) -> int:
    if model_name in _KNOWN_DIMS:
        return _KNOWN_DIMS[model_name]
    try:  # 兜底：fastembed 的静态注册表，只读元数据，不下载权重
        from fastembed import TextEmbedding

        return int(TextEmbedding.get_embedding_size(model_name))
    except Exception as exc:  # noqa: BLE001
        raise ValueError(
            f"未知 embedding 模型 {model_name}，请把维度加入 _KNOWN_DIMS"
        ) from exc


class FastEmbedEmbedder:
    """ONNX 推理的真实语义 embedding（复用 onnxruntime，不依赖 torch）。

    模型懒加载：构造只确定维度，第一次 embed 才建 onnx session / 触发下载。
    """

    name = "fastembed"

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-zh-v1.5",
        *,
        cache_dir: str = "./data/models",
        batch_size: int = 32,
        threads: int | None = None,
        query_prefix: str = "",
        passage_prefix: str = "",
        client: object | None = None,
    ) -> None:
        if client is None and importlib.util.find_spec("fastembed") is None:
            raise RuntimeError(
                "未安装 fastembed，请执行 pip install fastembed 或把 EMBEDDING_PROVIDER 切回 hashing"
            )
        self._model_name = model_name
        self._cache_dir = cache_dir
        self._batch_size = batch_size
        self._threads = threads
        self._query_prefix = query_prefix
        self._passage_prefix = passage_prefix
        self._model = client  # 测试注入点：传 fake 就完全不 import fastembed
        self.dim = _resolve_dim(model_name)

    # ---- 内部 ----
    def _ensure(self):  # noqa: ANN202
        if self._model is not None:
            return self._model

        cfg = get_settings()
        # 必须在 import fastembed / huggingface_hub 之前生效：
        # 这两个库在 import 时就把 HF_ENDPOINT 读成常量
        if cfg.hf_endpoint:
            os.environ.setdefault("HF_ENDPOINT", cfg.hf_endpoint)
        if cfg.hf_hub_disable_xet:
            os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

        from fastembed import TextEmbedding

        model = TextEmbedding(
            model_name=self._model_name,
            cache_dir=self._cache_dir,
            threads=self._threads,
            lazy_load=False,
        )
        real_dim = int(model.embedding_size)
        if real_dim != self.dim:
            logger.warning(
                "embedding_dim_corrected declared=%d real=%d model=%s",
                self.dim,
                real_dim,
                self._model_name,
            )
            self.dim = real_dim
        self._model = model
        logger.info(
            "fastembed_ready model=%s dim=%d cache=%s",
            self._model_name,
            self.dim,
            self._cache_dir,
        )
        return model

    # ---- Embedder 协议 ----
    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        model = self._ensure()
        prepared = [self._passage_prefix + t for t in texts] if self._passage_prefix else list(texts)
        return [
            _l2_normalize([float(x) for x in vec])
            for vec in model.embed(prepared, batch_size=self._batch_size)
        ]

    def embed_one(self, text: str) -> list[float]:
        model = self._ensure()
        payload = self._query_prefix + text
        vector = next(iter(model.query_embed(payload)))
        return _l2_normalize([float(x) for x in vector])


class SentenceTransformerEmbedder:
    """历史实现，需要 torch。保留是为了老的 .env 不至于启动即失败。

    deprecated：请改用 EMBEDDING_PROVIDER=fastembed。
    """

    name = "sentence-transformers"

    def __init__(self, model_name: str = "BAAI/bge-small-zh-v1.5") -> None:
        self._model_name = model_name
        self._model = None
        self.dim = _resolve_dim(model_name)

    def _ensure(self):  # noqa: ANN202
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self._model_name)
            self.dim = int(self._model.get_sentence_embedding_dimension())
        return self._model

    def embed(self, texts: list[str]) -> list[list[float]]:
        model = self._ensure()
        vectors = model.encode(texts, normalize_embeddings=True)
        return [list(map(float, v)) for v in vectors]

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]


# ---------------------------------------------------------------- 异步门面
async def aembed(embedder: Embedder, texts: list[str]) -> list[list[float]]:
    """把同步批量编码丢进线程池。

    刻意不用 run_blocking：它内部又套一层 call_with_resilience，
    而调用方（retrieve / ingest）外层已经有超时重试，会放大成 N×M 次重试。
    """
    if not texts:
        return []
    return await asyncio.to_thread(embedder.embed, texts)


async def aembed_one(embedder: Embedder, text: str) -> list[float]:
    return await asyncio.to_thread(embedder.embed_one, text)


_embedder: Embedder | None = None


def get_embedder() -> Embedder:
    global _embedder
    if _embedder is not None:
        return _embedder

    cfg = get_settings()
    if cfg.embedding_provider == "fastembed":
        try:
            _embedder = FastEmbedEmbedder(
                cfg.embedding_model,
                cache_dir=cfg.fastembed_cache_dir,
                batch_size=cfg.embedding_batch_size,
                threads=cfg.embedding_threads,
                query_prefix=cfg.embedding_query_prefix,
                passage_prefix=cfg.embedding_passage_prefix,
            )
        except Exception as exc:  # noqa: BLE001 - 缺包/未知模型都可降级
            if not cfg.embedding_fallback_to_hashing:
                raise
            logger.error("fastembed_unavailable, fallback to hashing: %s", exc)
            _embedder = HashingEmbedder(dim=cfg.embedding_dim)
    elif cfg.embedding_provider == "sentence-transformers":
        _embedder = SentenceTransformerEmbedder(cfg.embedding_model)
    else:
        _embedder = HashingEmbedder(dim=cfg.embedding_dim)
    return _embedder


def set_embedder(embedder: Embedder | None) -> None:
    """测试注入点。"""
    global _embedder
    _embedder = embedder


async def warmup_embedder() -> bool:
    """预热：把「下载权重 + 建 onnx session」从请求链路挪到启动阶段。

    失败不抛异常——录音里已经明确说 RAG 是可选能力，服务应该起得来，
    由 /health 体现 degraded，而不是让整个进程起不来。
    """
    cfg = get_settings()
    embedder = get_embedder()
    if isinstance(embedder, HashingEmbedder):
        return True
    try:
        await call_with_resilience(
            lambda: aembed_one(embedder, "预热"),
            label="rag.embed_warmup",
            timeout=cfg.rag_model_load_timeout_seconds,
            retries=0,  # 下载类操作不重试，避免 3 倍等待
        )
        logger.info("embedder_warmup_ok provider=%s dim=%d", embedder.name, embedder.dim)
        return True
    except ToolExecutionError as exc:
        logger.error("embedder_warmup_failed provider=%s err=%s", embedder.name, exc)
        return False
