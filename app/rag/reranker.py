"""重排序层：向量召回容易"语义近似但不精确"，重排把真正含答案的片段顶上来。

默认 lexical 重排（词项覆盖率 + 短语命中 + 向量分融合），零依赖、可解释；
可选 cross-encoder（需 sentence-transformers），精度更高但更慢。
"""

from __future__ import annotations

from typing import Protocol

from app.core.config import get_settings
from app.core.logging import get_logger
from app.rag.embeddings import tokenize
from app.rag.vectorstore import Hit

logger = get_logger(__name__)


class Reranker(Protocol):
    name: str

    def rerank(self, query: str, hits: list[Hit], top_n: int) -> list[Hit]: ...


class LexicalReranker:
    """lightweight hybrid reranker。

    最终分 = 0.55 * 向量余弦 + 0.35 * 查询词项覆盖率 + 0.10 * 整句短语命中
    覆盖率取"查询 token 有多少比例出现在片段里"，能有效压掉泛泛而谈的长段落。
    """

    name = "lexical"
    VECTOR_WEIGHT = 0.55
    COVERAGE_WEIGHT = 0.35
    PHRASE_WEIGHT = 0.10

    def rerank(self, query: str, hits: list[Hit], top_n: int) -> list[Hit]:
        if not hits:
            return []
        query_tokens = {t for t in tokenize(query) if len(t) > 0}
        if not query_tokens:
            return hits[:top_n]

        rescored: list[Hit] = []
        for hit in hits:
            chunk_tokens = set(tokenize(hit.text))
            coverage = len(query_tokens & chunk_tokens) / len(query_tokens)
            phrase_bonus = 1.0 if query.strip() and query.strip() in hit.text else 0.0
            score = (
                self.VECTOR_WEIGHT * hit.score
                + self.COVERAGE_WEIGHT * coverage
                + self.PHRASE_WEIGHT * phrase_bonus
            )
            rescored.append(
                Hit(hit.chunk_id, hit.text, hit.metadata, max(0.0, min(1.0, round(score, 4))))
            )

        rescored.sort(key=lambda h: h.score, reverse=True)
        return rescored[:top_n]


class CrossEncoderReranker:
    """可选：bge-reranker 之类，精度优先。首次调用会下载模型。"""

    name = "cross-encoder"

    def __init__(self, model_name: str = "BAAI/bge-reranker-base") -> None:
        self._model_name = model_name
        self._model = None

    def _ensure(self):  # noqa: ANN202
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self._model_name)
        return self._model

    def rerank(self, query: str, hits: list[Hit], top_n: int) -> list[Hit]:
        if not hits:
            return []
        model = self._ensure()
        scores = model.predict([(query, h.text) for h in hits])
        rescored = [
            Hit(h.chunk_id, h.text, h.metadata, float(1 / (1 + 2.718281828 ** -float(s))))
            for h, s in zip(hits, scores)
        ]
        rescored.sort(key=lambda h: h.score, reverse=True)
        return rescored[:top_n]


_reranker: Reranker | None = None


def get_reranker() -> Reranker:
    global _reranker
    if _reranker is None:
        cfg = get_settings()
        if cfg.reranker == "cross-encoder":
            try:
                _reranker = CrossEncoderReranker()
            except Exception as exc:  # noqa: BLE001
                logger.warning("cross_encoder_unavailable, fallback lexical: %s", exc)
                _reranker = LexicalReranker()
        else:
            _reranker = LexicalReranker()
    return _reranker


def set_reranker(reranker: Reranker | None) -> None:
    global _reranker
    _reranker = reranker
