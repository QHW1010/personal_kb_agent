"""忠实度判定：解析健壮性与打分口径。

这里的测试重点不是「模型判得准不准」（那要人工标注才能测），而是
「模型输出的各种脏形态能不能被正确接住」，以及「弃答 / 判定失败不能被算成满分」。
后者是评估脚本最容易骗自己的地方：一个解析 bug 能让忠实度凭空变成 1.00。
"""

from __future__ import annotations

from scripts.faithfulness import (
    ClaimVerdict,
    FaithfulnessVerdict,
    aggregate,
    build_judge_messages,
    parse_verdict,
)


def test_parse_plain_json() -> None:
    verdict = parse_verdict(
        '{"claims":[{"text":"A","supported":true,"support":1,"reason":"原文"},'
        '{"text":"B","supported":false,"support":0}]}'
    )
    assert not verdict.error
    assert verdict.total == 2
    assert verdict.supported == 1
    assert verdict.score == 0.5


def test_parse_tolerates_code_fence_and_chatter() -> None:
    raw = '好的，核查结果如下：\n```json\n{"claims":[{"text":"A","supported":true}]}\n```\n以上。'
    verdict = parse_verdict(raw)
    assert not verdict.error
    assert verdict.total == 1


def test_parse_accepts_bare_list() -> None:
    assert parse_verdict('[{"text":"A","supported":true}]').total == 1


def test_parse_accepts_chinese_boolean_and_numeric_support() -> None:
    verdict = parse_verdict('{"claims":[{"text":"A","supported":"是","support":"2"}]}')
    assert verdict.claims[0].supported is True
    assert verdict.claims[0].support == 2


def test_parse_skips_entries_without_text() -> None:
    verdict = parse_verdict('{"claims":[{"supported":true},{"text":"B","supported":true}]}')
    assert verdict.total == 1
    assert verdict.claims[0].text == "B"


def test_parse_failure_is_reported_not_swallowed() -> None:
    verdict = parse_verdict("我认为这个答案基本正确。")
    assert verdict.error
    assert verdict.score is None


def test_missing_claims_key_is_an_error() -> None:
    assert parse_verdict('{"verdict":"ok"}').error


def test_abstention_scores_none_not_one() -> None:
    """弃答既不是满分也不是零分：单独记一档，避免奖励「什么都不说」。"""
    assert FaithfulnessVerdict(claims=[]).score is None
    assert FaithfulnessVerdict(claims=[ClaimVerdict("A", True)]).score == 1.0
    assert FaithfulnessVerdict(claims=[ClaimVerdict("A", False)]).score == 0.0


def test_aggregate_separates_abstain_and_failure() -> None:
    verdicts = [
        FaithfulnessVerdict(claims=[ClaimVerdict("A", True), ClaimVerdict("B", False)]),
        FaithfulnessVerdict(claims=[ClaimVerdict("C", True)]),
        FaithfulnessVerdict(claims=[]),  # 弃答
        FaithfulnessVerdict(error="bad json"),  # 判定失败
    ]
    summary = aggregate(verdicts)
    assert summary["questions"] == 4
    assert summary["scored_questions"] == 2
    assert summary["abstained"] == 1
    assert summary["judge_failed"] == 1
    assert summary["claims"] == 3
    assert summary["unsupported_claims"] == 1
    # 题级平均：(0.5 + 1.0) / 2；微平均：2/3
    assert summary["score_avg"] == 0.75
    assert abs(summary["score_micro"] - 2 / 3) < 1e-3
    assert summary["perfect_questions"] == 1


def test_aggregate_on_empty_is_zero_not_crash() -> None:
    summary = aggregate([])
    assert summary["score_avg"] == 0.0
    assert summary["score_micro"] == 0.0


def test_judge_prompt_only_sees_citations() -> None:
    """证据只给引用片段：喂整篇文档会把「文档里确实这么写」当成「被引用支持」。"""
    messages = build_judge_messages(
        "为什么禁 LIMIT？",
        "因为会截断多行答案。[1]",
        [{"source": "a.md", "section": "约束", "snippet": "行数保护放在执行层"}],
    )
    assert messages[0]["role"] == "system"
    body = messages[1]["content"]
    assert "行数保护放在执行层" in body
    assert "a.md" in body
    assert "[1]" in body
    assert "为什么禁 LIMIT？" in body


def test_judge_prompt_handles_no_citations() -> None:
    body = build_judge_messages("q", "a", [])[1]["content"]
    assert "没有返回任何引用片段" in body


def test_judge_prompt_truncates_long_snippet() -> None:
    body = build_judge_messages("q", "a", [{"source": "a.md", "snippet": "长" * 900}])[1]["content"]
    assert body.count("长") <= 400
