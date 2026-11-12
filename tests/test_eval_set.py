"""评估集的「可判中性」：锚点必须真实存在于语料，并且在单个切片内。

这道测试守的是一个很容易悄悄退化的东西：文档改了措辞、切分参数变了、
或者出题时手滑写错一个字，那道题的排名就永远是「未命中」，指标无声无息掉一截。
把它放进 CI，改文档的人会立刻知道哪道题失效了。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.core.config import get_settings
from app.rag.chunking import split_markdown
from app.rag.ingest import iter_source_docs
from scripts.eval_set import EVAL_SET, by_topic

DOCS_DIR = Path(__file__).resolve().parents[1] / "docs"


@pytest.fixture(scope="module")
def corpus() -> dict[str, list[str]]:
    cfg = get_settings()
    out: dict[str, list[str]] = {}
    for doc in iter_source_docs(DOCS_DIR):
        out[doc.source] = [
            piece.text
            for piece in split_markdown(
                doc.content, size=cfg.rag_chunk_size, overlap=cfg.rag_chunk_overlap
            )
        ]
    return out


def test_corpus_is_discoverable(corpus: dict[str, list[str]]) -> None:
    assert corpus, "docs/ 下没有可摄入的文档，评估集无从判分"
    assert sum(len(v) for v in corpus.values()) >= 30


def test_every_source_exists(corpus: dict[str, list[str]]) -> None:
    for case in EVAL_SET:
        assert case.source in corpus, f"{case.source} 不在 docs/ 里"


def test_anchor_falls_inside_one_chunk(corpus: dict[str, list[str]]) -> None:
    """锚点跨切片就等于永远判不中，这种题是纯噪声。"""
    for case in EVAL_SET:
        hits = [text for text in corpus[case.source] if case.anchor in text]
        assert hits, f"锚点不在任何切片内: {case.question} :: {case.anchor!r}"


def test_anchor_is_selective(corpus: dict[str, list[str]]) -> None:
    """命中四五个切片的锚点太泛，等于「来源对了就算命中」，指标会虚高。"""
    for case in EVAL_SET:
        hits = sum(1 for text in corpus[case.source] if case.anchor in text)
        assert hits <= 3, f"锚点过泛（命中 {hits} 个切片）: {case.anchor!r}"


def test_eval_set_covers_multiple_documents() -> None:
    sources = {case.source for case in EVAL_SET}
    assert len(sources) >= 5, "评估集至少要覆盖 5 篇文档，否则测得的是单篇检索"
    assert len(by_topic()) >= 15, "分组过少，分档统计看不出退化点"
