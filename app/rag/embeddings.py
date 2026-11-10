"""Embedding 抽象。

默认 hashing embedder：确定性、零下载、纯 CPU，保证 CI 与离线演示可跑。
需要更高召回时把 EMBEDDING_PROVIDER 切成 sentence-transformers 即可，
上层接口不变。
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Protocol

from app.core.config import get_settings

_WORD_RE = re.compile(r"[a-zA-Z0-9_]{2,}")
_CJK_RE = re.compile(r"[\u4e00-\u9fff]")


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
        norm = math.sqrt(sum(v * v for v in vector))
        if norm == 0:
            return vector
        return [v / norm for v in vector]

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_one(t) for t in texts]


class SentenceTransformerEmbedder:
    """可选高质量 embedding；模型懒加载，首次调用会有下载耗时。"""

    name = "sentence-transformers"

    def __init__(self, model_name: str = "BAAI/bge-small-zh-v1.5") -> None:
        self._model_name = model_name
        self._model = None
        self.dim = 512

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


_embedder: Embedder | None = None


def get_embedder() -> Embedder:
    global _embedder
    if _embedder is None:
        cfg = get_settings()
        if cfg.embedding_provider == "sentence-transformers":
            _embedder = SentenceTransformerEmbedder()
        else:
            _embedder = HashingEmbedder(dim=cfg.embedding_dim)
    return _embedder


def set_embedder(embedder: Embedder | None) -> None:
    """测试注入点。"""
    global _embedder
    _embedder = embedder
