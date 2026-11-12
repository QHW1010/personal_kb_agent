"""fastembed 分支测试。

不触发任何网络请求：用注入的 fake client 代替 TextEmbedding，
真实模型只在显式设置 RUN_FASTEMBED_IT=1 时才跑（integration marker）。
"""

from __future__ import annotations

import importlib.util
import math
import os

import pytest

from app.rag.embeddings import FastEmbedEmbedder, HashingEmbedder, aembed, aembed_one

DIM = 512


class _FakeModel:
    """最小可用的 TextEmbedding 替身：记录收到的文本，返回归一化向量。"""

    embedding_size = DIM

    def __init__(self) -> None:
        self.seen_docs: list[str] = []
        self.seen_queries: list[str] = []

    @staticmethod
    def _vector(seed: float = 1.0) -> list[float]:
        vec = [0.0] * DIM
        vec[0] = seed
        return vec

    def embed(self, documents, batch_size: int = 32, **_: object):
        docs = [documents] if isinstance(documents, str) else list(documents)
        self.seen_docs.extend(docs)
        return [self._vector() for _ in docs]

    def query_embed(self, query, **_: object):
        queries = [query] if isinstance(query, str) else list(query)
        self.seen_queries.extend(queries)
        return [self._vector() for _ in queries]


def _embedder(**kwargs) -> tuple[FastEmbedEmbedder, _FakeModel]:
    fake = _FakeModel()
    return FastEmbedEmbedder("BAAI/bge-small-zh-v1.5", client=fake, **kwargs), fake


def test_dim_resolved_without_loading_model() -> None:
    embedder, _ = _embedder()
    assert embedder.dim == DIM
    assert embedder.name == "fastembed"


def test_embed_and_embed_one_shapes_are_normalized() -> None:
    embedder, _ = _embedder()
    vectors = embedder.embed(["第一段", "第二段"])
    assert len(vectors) == 2
    assert all(len(v) == DIM for v in vectors)
    single = embedder.embed_one("问题")
    assert len(single) == DIM
    assert math.isclose(math.sqrt(sum(x * x for x in single)), 1.0, rel_tol=1e-9)


def test_empty_input_short_circuits() -> None:
    embedder, fake = _embedder()
    assert embedder.embed([]) == []
    assert fake.seen_docs == []


def test_query_and_passage_prefixes_are_applied() -> None:
    embedder, fake = _embedder(query_prefix="query: ", passage_prefix="passage: ")
    embedder.embed(["文档正文"])
    embedder.embed_one("用户问题")
    assert fake.seen_docs == ["passage: 文档正文"]
    assert fake.seen_queries == ["query: 用户问题"]


def test_missing_dependency_raises_so_factory_can_fall_back(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    with pytest.raises(RuntimeError, match="未安装 fastembed"):
        FastEmbedEmbedder("BAAI/bge-small-zh-v1.5")


def test_unknown_model_name_raises() -> None:
    with pytest.raises(ValueError, match="未知 embedding 模型"):
        FastEmbedEmbedder("not-a-real/model-xyz")


async def test_async_wrappers_delegate_to_thread() -> None:
    embedder, fake = _embedder()
    vectors = await aembed(embedder, ["a", "b"])
    assert len(vectors) == 2
    assert len(await aembed_one(embedder, "q")) == DIM
    assert fake.seen_docs == ["a", "b"]


async def test_async_wrappers_on_hashing_embedder() -> None:
    embedder = HashingEmbedder(dim=32)
    vectors = await aembed(embedder, ["知识库"])
    assert len(vectors) == 1
    assert len(await aembed_one(embedder, "知识库")) == 32


@pytest.mark.integration
@pytest.mark.skipif(
    not os.getenv("RUN_FASTEMBED_IT"),
    reason="需要真实模型与网络（会从 hf-mirror 下载权重），设置 RUN_FASTEMBED_IT=1 才跑",
)
async def test_real_model_recalls_chinese_semantics() -> None:
    embedder = FastEmbedEmbedder("BAAI/bge-small-zh-v1.5", cache_dir="./data/models")
    docs = embedder.embed(["重排序阶段把粗召回结果重新打分", "数据库连接池配置"])
    query = await aembed_one(embedder, "重排是怎么做的")
    dot = sum(x * y for x, y in zip(query, docs[0]))
    assert dot > 0.5, f"语义召回异常，相似度仅 {dot:.3f}"
