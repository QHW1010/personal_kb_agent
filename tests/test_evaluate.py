"""评估指标的判分逻辑。

指标本身不难，难在"什么算命中"。这里把判分规则钉死：
来源文件名 + 锚点原文必须同时命中，只对文件名会让指标虚高。
"""

from __future__ import annotations

from types import SimpleNamespace

from scripts.evaluate import (
    EVAL_SET,
    EvalCase,
    _display_width,
    _fit,
    first_relevant_rank,
    is_relevant,
    mrr,
    recall_at,
)


def _hit(source: str, text: str, score: float = 0.5) -> SimpleNamespace:
    return SimpleNamespace(text=text, metadata={"source": source}, score=score)


CASE = EvalCase(question="q", source="doc.md", anchor="关键句")


def test_relevance_needs_both_source_and_anchor() -> None:
    assert is_relevant(_hit("doc.md", "这里有 关键句 存在"), CASE)
    # 锚点对但来源不对 → 不算命中
    assert not is_relevant(_hit("other.md", "这里有 关键句 存在"), CASE)
    # 来源对但锚点不在正文 → 不算命中（防止同篇文档的无关片段混进来）
    assert not is_relevant(_hit("doc.md", "完全无关的内容"), CASE)


def test_first_relevant_rank_is_one_based() -> None:
    hits = [
        _hit("doc.md", "无关"),
        _hit("doc.md", "含 关键句 的片段"),
        _hit("doc.md", "也含 关键句"),
    ]
    assert first_relevant_rank(hits, CASE) == 2


def test_first_relevant_rank_returns_zero_on_miss() -> None:
    assert first_relevant_rank([_hit("doc.md", "无关")], CASE) == 0
    assert first_relevant_rank([], CASE) == 0


def test_recall_at_counts_hits_within_k() -> None:
    ranks = [1, 3, 0, 5]
    assert recall_at(ranks, 3) == 0.5  # 1 和 3 命中
    assert recall_at(ranks, 5) == 0.75
    assert recall_at(ranks, 1) == 0.25
    assert recall_at(ranks, 0) == 0.0


def test_recall_at_handles_empty_set() -> None:
    assert recall_at([], 3) == 0.0


def test_mrr_uses_reciprocal_rank() -> None:
    # (1/1 + 1/2 + 0 + 1/5) / 4
    assert abs(mrr([1, 2, 0, 5]) - (1 + 0.5 + 0 + 0.2) / 4) < 1e-9
    assert mrr([]) == 0.0


def test_eval_set_is_well_formed() -> None:
    """评估集的每道题都必须有来源与锚点，否则判分永远是 0。"""
    assert 30 <= len(EVAL_SET) <= 50, "题量太少撑不起统计，太多则维护成本过高"
    for case in EVAL_SET:
        assert case.question.strip()
        assert case.source.endswith(".md")
        assert case.anchor.strip()
    # 题面不能重复：重复题会让均值被同一道题拉偏
    questions = [case.question for case in EVAL_SET]
    assert len(set(questions)) == len(questions)


def test_display_width_counts_cjk_as_two() -> None:
    assert _display_width("abcd") == 4
    assert _display_width("中文") == 4
    assert _display_width("a中") == 3


def test_fit_truncates_and_pads() -> None:
    assert _fit("ab", 4) == "ab  "
    truncated = _fit("这是一段很长的中文句子", 10)
    assert _display_width(truncated) == 10
    assert "…" in truncated, "超宽内容必须被截断并加省略号"
