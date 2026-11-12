"""引用隔离：保证 citations 只描述当前轮次的检索结果。

多轮对话有三个真实的污染路径，各有对策：

1. **历史答案里的旧编号被照抄**。上一轮的答案正文里带着 `[1]`、`[2]`，
   它们随 messages 一起喂给模型；模型很自然地继续写 `[1]`，而这一轮的
   citations 可能是完全不同的三段。编号看上去合法，指向的却是上一轮的
   片段 —— 这是最隐蔽的一种错。`sanitize_history()` 在拼接提示词前抹掉
   历史答案里的标记，让模型没有旧编号可抄。

2. **模型写出越界编号**。只检索到 3 条却标了 `[7]`，前端会渲染成指向
   不存在来源的链接。`align_citation_marks()` 按本轮 citations 的长度
   校验，越界的删掉，合法的保留。

3. **上游把历史 citations 混进 state**。`validate_answer()` 在唯一出口
   用本轮 hits 的 chunk_id 做白名单，兜底过滤。

这里全部是纯函数，不碰 IO、不认 route，可以单独测。

已知边界：`arr[1]` 这种下标写法与引用标记形态相同，无法区分，会被当成
引用处理。历史答案是自然语言，出现代码下标的概率极低；`[0]` 则因为编号
从 1 开始而被排除在外。
"""

from __future__ import annotations

import re
from typing import Any

from langchain_core.messages import BaseMessage

__all__ = [
    "strip_citation_marks",
    "sanitize_history",
    "align_citation_marks",
    "filter_citations_by_hits",
]

# 匹配 [1]、[1,2]、[1，2]，允许方括号内有空格。
# 编号从 1 开始（[0] 不是合法引用），这样 arr[0] 这类代码片段不会被误伤。
_CITE_GROUP_RE = re.compile(r"\[\s*([1-9]\d*(?:\s*[,，]\s*\d+)*)\s*\]")
# 抹掉标记后可能留下的多余空格，以及"结论 ，"这种标点前的空格
_MULTI_SPACE_RE = re.compile(r"[ \t]{2,}")
_SPACE_BEFORE_PUNCT_RE = re.compile(r"[ \t]+([，。；：！？、）】」])")


def strip_citation_marks(text: str) -> str:
    """去掉全部 [n] 标记，供净化历史消息使用。"""
    if not text:
        return text
    cleaned = _CITE_GROUP_RE.sub("", text)
    cleaned = _MULTI_SPACE_RE.sub(" ", cleaned)
    cleaned = _SPACE_BEFORE_PUNCT_RE.sub(r"\1", cleaned)
    return cleaned.strip()


def sanitize_history(messages: list[BaseMessage]) -> list[BaseMessage]:
    """把历史 AI 消息里的引用标记抹掉，用户的提问原样保留。

    只处理 type == "ai" 且 content 是字符串的消息；多模态 content（list）
    直接放行，不冒险改结构。

    返回的是 copy，不会改动调用方持有的消息对象 —— 历史消息还要落库/复用。
    """
    sanitized: list[BaseMessage] = []
    for message in messages:
        content = getattr(message, "content", "")
        if (
            getattr(message, "type", "") == "ai"
            and isinstance(content, str)
            and _CITE_GROUP_RE.search(content)
        ):
            sanitized.append(message.model_copy(update={"content": strip_citation_marks(content)}))
        else:
            sanitized.append(message)
    return sanitized


def align_citation_marks(
    answer: str, citations: list[dict[str, Any]] | list[Any] | None
) -> tuple[str, list[int]]:
    """按当前轮 citations 的条数校正答案里的 [n]。

    返回 `(校正后的答案, 被丢弃的编号)`。丢弃的编号交给调用方记日志 ——
    出现越界编号说明模型在编，这本身是值得观测的信号。
    """
    if not answer:
        return answer, []
    limit = len(citations or [])
    dropped: list[int] = []

    def _replace(match: re.Match[str]) -> str:
        nums = [int(part) for part in re.split(r"[,，]", match.group(1)) if part.strip()]
        keep = [n for n in nums if 1 <= n <= limit]
        dropped.extend(n for n in nums if not 1 <= n <= limit)
        if not keep:
            return ""
        return "[" + ",".join(str(n) for n in keep) + "]"

    cleaned = _CITE_GROUP_RE.sub(_replace, answer)
    cleaned = _MULTI_SPACE_RE.sub(" ", cleaned)
    cleaned = _SPACE_BEFORE_PUNCT_RE.sub(r"\1", cleaned)
    return cleaned.strip(), dropped


def filter_citations_by_hits(
    citations: list[dict[str, Any]],
    hits: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    """按本轮检索命中的 chunk_id 白名单过滤引用。

    第二个返回值是被丢弃的条数，用于告警：正常情况下它恒为 0，
    一旦大于 0 就说明有别的轮次（或别的代码路径）把引用混了进来。
    """
    if not citations:
        return [], 0
    allowed = {str(hit.get("chunk_id")) for hit in hits or [] if hit.get("chunk_id")}
    if not allowed:
        # 本轮没有检索命中却带着引用，一律视为跨轮污染
        return [], len(citations)
    kept = [c for c in citations if str(c.get("chunk_id")) in allowed]
    return kept, len(citations) - len(kept)
