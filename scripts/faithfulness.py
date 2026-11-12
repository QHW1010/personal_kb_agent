"""忠实度判定的纯逻辑：判定提示词、鲁棒解析、claim 级打分。

刻意不做任何 IO：真实调用模型的部分在 scripts/evaluate_faithfulness.py，
这里只负责「给模型看什么」和「模型吐出来的东西怎么算分」，因此可以单测。

判定的核心是 claim 级，而不是让模型给一个整体可信度分。整体分没有定义、
不可复现，而且几乎总是偏高——模型对「大致正确」的容忍度远高于事实核查该有的标准。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "ClaimVerdict",
    "FaithfulnessVerdict",
    "JUDGE_SYSTEM_PROMPT",
    "build_judge_messages",
    "parse_verdict",
    "aggregate",
]

JUDGE_SYSTEM_PROMPT = """你是严格的事实核查员。判断【答案】里的每一条事实性断言，能否由【引用片段】的原文直接推出。

规则：
1. 只依据引用片段的原文，严禁用你自己的常识或领域知识补全。
2. 把答案拆成尽可能小的原子断言。一句话里有两个独立事实，就拆成两条。
3. 寒暄、过渡句、以及「根据资料」这类元话语不算断言，不要列入。
4. 每条断言给出 supported（布尔值）与 support（支撑它的引用编号，从 1 开始；没有就填 0）。
5. 引用片段没有提到，不等于断言错误：只要这条断言无法由片段推出，就判 supported=false。
6. 数字、单位、否定词、限定条件必须与原文一致，差一个字都算不支持。
7. 如果整段答案只是在说明资料不足、没有给出任何实质性结论（例如「知识库中没有找到相关信息」），
   则 claims 返回空数组，不要把这些说明拆成断言。
8. 只输出一个 JSON 对象，不要解释、不要代码块：
{"claims":[{"text":"断言原文","supported":true,"support":1,"reason":"十个字以内"}]}
"""


@dataclass(frozen=True)
class ClaimVerdict:
    """单条断言的判定结果。"""

    text: str
    supported: bool
    support: int = 0  # 支撑它的引用编号（1 起）；0 = 未给出
    reason: str = ""


@dataclass
class FaithfulnessVerdict:
    """一道题的判定结果。`error` 非空表示裁判输出解析失败。"""

    claims: list[ClaimVerdict] = field(default_factory=list)
    error: str = ""

    @property
    def total(self) -> int:
        return len(self.claims)

    @property
    def supported(self) -> int:
        return sum(1 for c in self.claims if c.supported)

    @property
    def score(self) -> float | None:
        """没有断言时返回 None（弃答），而不是 1.0。

        把「什么都没说」算成满分，等于奖励模型不回答；算成 0 分又会冤枉
        正常的「知识库中没有找到相关信息」。所以单独记一档。
        """
        if self.error or not self.claims:
            return None
        return self.supported / self.total


_JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y", "是", "支持"}
    return False


def _as_int(value: Any) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def _extract_payload(raw: str) -> Any:
    """从模型输出里挖出 JSON。围栏、前后废话、裸列表都要能接住。"""
    text_out = (raw or "").strip()
    fenced = _FENCE_RE.search(text_out)
    if fenced:
        text_out = fenced.group(1).strip()

    for candidate in (text_out, *(m.group(0) for m in _JSON_OBJ_RE.finditer(text_out))):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    raise ValueError(f"裁判输出不是合法 JSON: {raw[:80]!r}")


def parse_verdict(raw: str) -> FaithfulnessVerdict:
    """解析裁判输出。解析失败不抛异常，如实记进 error 字段。

    静默当成「通过」是最糟的处理方式：一个格式 bug 会让忠实度凭空变成 1.00。
    """
    try:
        payload = _extract_payload(raw)
    except ValueError as exc:
        return FaithfulnessVerdict(error=str(exc))

    if isinstance(payload, dict):
        items = payload.get("claims")
    elif isinstance(payload, list):
        items = payload
    else:
        items = None

    if not isinstance(items, list):
        return FaithfulnessVerdict(error="裁判输出缺少 claims 数组")

    claims: list[ClaimVerdict] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text") or "").strip()
        if not text:
            continue
        claims.append(
            ClaimVerdict(
                text=text,
                supported=_as_bool(item.get("supported")),
                support=_as_int(item.get("support")),
                reason=str(item.get("reason") or "")[:40],
            )
        )
    return FaithfulnessVerdict(claims=claims)


def build_judge_messages(
    question: str,
    answer: str,
    citations: list[dict[str, Any]],
    *,
    snippet_chars: int = 400,
) -> list[dict[str, str]]:
    """拼裁判的消息。证据只给 citations，不给整篇文档。

    给了全文，模型会把「文档里确实这么写」当成「答案被这段引用支持」，
    而这恰好是我们要测的东西 —— 答案必须被它自己声称的依据支持。
    """
    blocks: list[str] = []
    for index, cite in enumerate(citations, start=1):
        snippet = str(cite.get("snippet") or "").strip().replace("\n", " ")
        source = cite.get("source") or "unknown"
        section = cite.get("section") or ""
        head = f"[{index}] 来源：{source}" + (f" / {section}" if section else "")
        blocks.append(f"{head}\n{snippet[:snippet_chars]}")

    evidence = "\n\n".join(blocks) if blocks else "（本次没有返回任何引用片段）"
    user = (
        f"【问题】\n{question}\n\n"
        f"【引用片段】\n{evidence}\n\n"
        f"【答案】\n{answer}\n\n"
        "请按规则逐条核查，只输出 JSON。"
    )
    return [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def aggregate(verdicts: list[FaithfulnessVerdict]) -> dict[str, Any]:
    """汇总判定结果。

    同时给题级平均与微平均（按断言汇总）：
    - 题级平均：每道题权重相同，用户视角的「一次问答有多可信」。
    - 微平均：每条断言权重相同，问题视角的「模型多久编一句」。
    两者差距大，说明某几道题断言特别多，均值被它们拉动。
    """
    scored = [v.score for v in verdicts if v.score is not None]
    total_claims = sum(v.total for v in verdicts)
    supported_claims = sum(v.supported for v in verdicts)
    unsupported = total_claims - supported_claims

    return {
        "questions": len(verdicts),
        "scored_questions": len(scored),
        "abstained": sum(1 for v in verdicts if not v.error and not v.claims),
        "judge_failed": sum(1 for v in verdicts if v.error),
        "claims": total_claims,
        "supported_claims": supported_claims,
        "unsupported_claims": unsupported,
        "score_avg": round(sum(scored) / len(scored), 4) if scored else 0.0,
        "score_micro": round(supported_claims / total_claims, 4) if total_claims else 0.0,
        "perfect_questions": sum(1 for s in scored if s == 1.0),
    }
