"""多轮引用隔离：历史轮次的 citations 不得污染当前轮次的答案。

三层防线各有独立用例：
1. sanitize_history —— 历史答案里的 [1] 在进提示词前被抹掉；
2. align_citation_marks —— 越界编号按本轮引用条数删掉；
3. filter_citations_by_hits —— 出口按本轮 chunk_id 白名单兜底。

最后两个用例走完整 HTTP 链路，验证连续两轮真实对话的行为。
"""

from __future__ import annotations

import re
from typing import Any

from httpx import AsyncClient
from langchain_core.language_models.chat_models import SimpleChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from app.agents import nodes
from app.agents.llm import reset_chat_model, set_chat_model
from app.agents.state import new_state
from app.rag.citation_guard import (
    align_citation_marks,
    filter_citations_by_hits,
    sanitize_history,
    strip_citation_marks,
)
from app.services.chat_service import validate_answer


def _cite(chunk_id: str, title: str = "片段") -> dict[str, Any]:
    return {"chunk_id": chunk_id, "title": title, "snippet": "正文", "score": 0.6}


# ---------------------------------------------------------------- 纯函数


def test_strip_citation_marks_removes_all_forms() -> None:
    assert strip_citation_marks("结论。[1]") == "结论。"
    assert strip_citation_marks("A[1]B[2,3]C") == "ABC"
    assert strip_citation_marks("多组[1][2]，补充[1，4]。") == "多组，补充。"
    assert strip_citation_marks("没有标记") == "没有标记"
    assert strip_citation_marks("") == ""


def test_strip_citation_marks_does_not_touch_non_numeric_brackets() -> None:
    """Markdown 链接、代码里的方括号不能被误删。"""
    text = "见 [文档](https://example.com) 与 arr[0]"
    assert strip_citation_marks(text) == text


def test_sanitize_history_strips_only_ai_messages() -> None:
    history = [
        HumanMessage(content="上一轮我问题里的 [1] 要保留"),
        AIMessage(content="上一轮答案，引用了片段[1][2]。"),
    ]
    cleaned = sanitize_history(history)

    assert cleaned[0].content == "上一轮我问题里的 [1] 要保留"
    assert cleaned[1].content == "上一轮答案，引用了片段。"
    # 不能就地修改调用方持有的消息对象
    assert history[1].content == "上一轮答案，引用了片段[1][2]。"


def test_sanitize_history_leaves_clean_messages_untouched() -> None:
    history = [AIMessage(content="上一轮没有引用"), HumanMessage(content="继续")]
    cleaned = sanitize_history(history)
    assert [m.content for m in cleaned] == ["上一轮没有引用", "继续"]


def test_align_drops_out_of_range_marks() -> None:
    citations = [_cite("a"), _cite("b")]
    answer, dropped = align_citation_marks("结论 A[1]，结论 B[2]，结论 C[7]。", citations)

    assert "[7]" not in answer
    assert "[1]" in answer and "[2]" in answer
    assert dropped == [7]


def test_align_strips_everything_when_no_citations() -> None:
    answer, dropped = align_citation_marks("共有 5 篇文档。[1][2]", [])
    assert answer == "共有 5 篇文档。"
    assert dropped == [1, 2]


def test_align_keeps_valid_part_of_mixed_group() -> None:
    citations = [_cite("a"), _cite("b")]
    answer, dropped = align_citation_marks("结论[1,5]。", citations)
    assert answer == "结论[1]。"
    assert dropped == [5]


def test_filter_by_hits_removes_foreign_chunks() -> None:
    citations = [_cite("cur-1"), _cite("prev-9")]
    hits = [{"chunk_id": "cur-1"}, {"chunk_id": "cur-2"}]

    kept, dropped = filter_citations_by_hits(citations, hits)

    assert [c["chunk_id"] for c in kept] == ["cur-1"]
    assert dropped == 1


def test_filter_by_hits_drops_all_when_nothing_retrieved() -> None:
    """本轮没有检索（如 SQL / 寒暄路线）却带着引用，一律视为跨轮污染。"""
    kept, dropped = filter_citations_by_hits([_cite("prev-1"), _cite("prev-2")], [])
    assert kept == []
    assert dropped == 2


def test_validate_answer_drops_citations_not_in_current_hits() -> None:
    """出口兜底：即使 state 里被塞了历史引用，也不能流到响应里。"""
    state = new_state(question="q", messages=[], conversation_id="c", user_id="u")
    state.update(
        {
            "route": "rag",
            "answer": "答案[1]",
            "hits": [{"chunk_id": "cur-1", "text": "本轮片段"}],
            "citations": [_cite("cur-1"), _cite("prev-9")],
        }
    )

    answer = validate_answer(state, latency_ms=3)

    assert [c.chunk_id for c in answer.citations] == ["cur-1"]


# ---------------------------------------------------------------- 提示词侧


class _RecordingChat(SimpleChatModel):
    """记录每次收到的消息，便于断言历史已被消毒。"""

    reply: str = "回答如下。[1]"
    seen: list[list[BaseMessage]] = []

    @property
    def _llm_type(self) -> str:
        return "recording-fake"

    def _call(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> str:
        self.seen.append(list(messages))
        return self.reply


async def test_synthesize_does_not_feed_stale_marks_to_model() -> None:
    """历史答案里的 [1][2] 必须在进模型前被抹掉，否则会被照抄。"""
    fake = _RecordingChat(seen=[])
    set_chat_model(fake)
    try:
        state = new_state(
            question="那它的执行层怎么做？",
            messages=[
                HumanMessage(content="SQL 为什么不写 LIMIT？"),
                AIMessage(content="因为会截断多行答案[1][2]。"),
            ],
            conversation_id="conv",
            user_id="u",
        )
        state.update({"route": "rag", "hits": [], "citations": []})

        result = await nodes.synthesize_node(state)
    finally:
        reset_chat_model()

    sent = fake.seen[0]
    ai_messages = [m for m in sent if getattr(m, "type", "") == "ai"]
    assert ai_messages, "历史应当被传给模型"
    assert all("[1]" not in str(m.content) for m in ai_messages), "历史里的旧编号未被消毒"
    # 本轮没有检索命中 → 答案里的 [1] 也要被抹掉
    assert "[1]" not in result["answer"]


async def test_synthesize_keeps_marks_within_current_citation_count() -> None:
    fake = _RecordingChat(reply="结论一[1]，结论二[2]，结论三[9]。", seen=[])
    set_chat_model(fake)
    try:
        state = new_state(question="q", messages=[], conversation_id="conv", user_id="u")
        state.update(
            {
                "route": "rag",
                "hits": [{"chunk_id": "a", "text": "x"}, {"chunk_id": "b", "text": "y"}],
                "citations": [_cite("a"), _cite("b")],
            }
        )
        result = await nodes.synthesize_node(state)
    finally:
        reset_chat_model()

    assert result["answer"] == "结论一[1]，结论二[2]，结论三。"


# ---------------------------------------------------------------- 完整链路


async def test_second_turn_greeting_has_no_citations(client: AsyncClient, auth_headers: dict) -> None:
    """第一轮 RAG 有引用，第二轮寒暄必须一条都不带。"""
    first = await client.post(
        "/api/v1/chat",
        headers=auth_headers,
        json={"message": "SQL 提示词为什么禁止硬编码 LIMIT？", "stream": False},
    )
    assert first.status_code == 200, first.text
    body1 = first.json()
    assert body1["route"] == "rag"
    assert body1["citations"], "第一轮应当带引用"

    second = await client.post(
        "/api/v1/chat",
        headers=auth_headers,
        json={
            "message": "你好",
            "stream": False,
            "conversation_id": body1["conversation_id"],
        },
    )
    assert second.status_code == 200, second.text
    body2 = second.json()

    assert body2["route"] == "direct"
    assert body2["citations"] == [], "寒暄轮不得继承上一轮引用"
    assert "[1]" not in body2["answer"]


async def test_second_turn_citations_stay_in_current_turn(
    client: AsyncClient, auth_headers: dict
) -> None:
    """连续两轮 RAG：第二轮的引用必须都能在本轮检索结果里找到出处。"""
    first = await client.post(
        "/api/v1/chat",
        headers=auth_headers,
        json={"message": "向量数据库用的是什么？", "stream": False},
    )
    assert first.status_code == 200, first.text
    conv = first.json()["conversation_id"]

    second = await client.post(
        "/api/v1/chat",
        headers=auth_headers,
        json={
            "message": "那 PostgreSQL 用什么 ORM？",
            "stream": False,
            "conversation_id": conv,
        },
    )
    assert second.status_code == 200, second.text
    body = second.json()
    assert body["citations"], "第二个问题同样命中文档，应当带引用"

    # 答案里出现的最大编号不得超出本轮引用条数
    marks = [int(n) for n in re.findall(r"\[(\d+)\]", body["answer"])]
    assert max(marks or [0]) <= len(body["citations"]), "答案出现了本轮不存在的引用编号"
