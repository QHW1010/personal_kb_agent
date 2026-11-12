"""对话服务：编排 Graph、校验输出、落库、计费。

API 层只做协议转换（HTTP/SSE），业务全在这里，满足"分层不混写"约束。
"""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from pydantic import ValidationError
from sqlalchemy import select

from app.agents.graph import get_graph
from app.agents.llm import model_display_name
from app.agents.state import AgentState, new_state
from app.core.config import get_settings
from app.core.errors import AgentOutputError
from app.core.logging import get_logger
from app.db.models import ChatMessage
from app.db.session import get_session_factory
from app.rag.citation_guard import filter_citations_by_hits
from app.schemas.agent import AgentAnswer
from app.schemas.chat import ChatResponse
from app.services.usage import build_usage, estimate_tokens, record_usage

logger = get_logger(__name__)


def sse_event(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


async def load_history(user_id: str, conversation_id: str, max_turns: int) -> list[BaseMessage]:
    """只取最近 N 轮，防止上下文无界增长导致成本失控。"""
    if max_turns <= 0:
        return []
    factory = get_session_factory()
    async with factory() as session:
        stmt = (
            select(ChatMessage)
            .where(
                ChatMessage.user_id == user_id,
                ChatMessage.conversation_id == conversation_id,
            )
            .order_by(ChatMessage.created_at.desc())
            .limit(max_turns * 2)
        )
        rows = list((await session.execute(stmt)).scalars())
    messages: list[BaseMessage] = []
    for row in reversed(rows):
        if row.role == "user":
            messages.append(HumanMessage(content=row.content))
        else:
            messages.append(AIMessage(content=row.content))
    return messages


async def _persist_user_message(user_id: str, conversation_id: str, content: str) -> None:
    factory = get_session_factory()
    async with factory() as session:
        session.add(
            ChatMessage(
                conversation_id=conversation_id,
                user_id=user_id,
                role="user",
                content=content,
            )
        )
        await session.commit()


async def _persist_answer(
    *,
    user_id: str,
    conversation_id: str,
    answer: AgentAnswer,
    route: str,
) -> str:
    factory = get_session_factory()
    async with factory() as session:
        message = ChatMessage(
            conversation_id=conversation_id,
            user_id=user_id,
            role="assistant",
            content=answer.answer,
            citations=[c.model_dump() for c in answer.citations],
        )
        session.add(message)
        await record_usage(
            session,
            user_id=user_id,
            conversation_id=conversation_id,
            usage=answer.usage,
            route=route,
        )
        await session.commit()
        return message.id


def validate_answer(state: AgentState, *, latency_ms: int) -> AgentAnswer:
    """唯一出口：所有 Agent 产物必须过这里。"""
    route = state.get("route") or "direct"
    sql = state.get("sql")
    rows = state.get("rows")
    answer_text = (state.get("answer") or "").strip()

    error = state.get("error")
    # SQL 路线失败（生成不出/执行失败）→ 降级为直答，且不冒充查询结果
    if route == "sql" and not sql:
        route = "direct"
        answer_text = answer_text or (
            "这个问题我没能转换成受支持的查询语句。目前支持的统计维度包括：文档数量、"
            "用户数量、对话数量、token 用量与调用成本。你可以换个更具体的问法。"
        )
        rows = None

    if not answer_text:
        answer_text = "抱歉，本次没有生成有效答案，请稍后重试或换个问法。"

    prompt_tokens = int(state.get("prompt_tokens") or 0)
    completion_tokens = int(state.get("completion_tokens") or estimate_tokens(answer_text))

    # 出口兜底：引用只能属于本轮检索命中的 chunk。
    # 多轮对话里历史 citations 若混进 state，用户会看到"这一轮的答案
    # 引用了这一轮根本没查过的段落"，比没有引用更难察觉。
    citations, leaked = filter_citations_by_hits(
        state.get("citations") or [], state.get("hits") or []
    )
    if leaked:
        logger.warning(
            "citations_out_of_scope dropped=%d conversation=%s",
            leaked,
            state.get("conversation_id"),
        )

    payload = {
        "answer": answer_text,
        "route": route,
        "citations": citations,
        "sql": sql,
        "rows": rows,
        "confidence": float(state.get("confidence") or 0.5),
        "needs_clarification": bool(state.get("needs_clarification")),
        "truncated": bool(state.get("sql_truncated")),
        "usage": build_usage(
            model=model_display_name(),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=latency_ms,
        ).model_dump(),
    }
    try:
        answer = AgentAnswer.model_validate(payload)
    except ValidationError as exc:
        logger.error("agent_output_invalid errors=%s payload_keys=%s", exc.errors()[:3], list(payload))
        raise AgentOutputError("Agent 输出未通过模式校验", details={"errors": exc.error_count()}) from exc

    if error:
        logger.warning("agent_soft_error route=%s error=%s", route, error)
    return answer


def _initial_state(
    user_id: str,
    conversation_id: str,
    message: str,
    history: list[BaseMessage],
    top_k: int | None = None,
) -> AgentState:
    return new_state(
        question=message,
        messages=history,
        conversation_id=conversation_id,
        user_id=user_id,
        top_k=top_k,
    )


def _to_response(answer: AgentAnswer, conversation_id: str, message_id: str) -> ChatResponse:
    return ChatResponse(
        conversation_id=conversation_id,
        message_id=message_id,
        answer=answer.answer,
        route=answer.route,
        citations=answer.citations,
        sql=answer.sql,
        rows=answer.rows,
        confidence=answer.confidence,
        needs_clarification=answer.needs_clarification,
        usage=answer.usage,
    )


async def ask(
    *,
    user_id: str,
    message: str,
    conversation_id: str | None = None,
    top_k: int | None = None,
) -> ChatResponse:
    """非流式：一次拿完整答案。"""
    cfg = get_settings()
    conversation_id = conversation_id or uuid.uuid4().hex
    started = time.perf_counter()

    history = await load_history(user_id, conversation_id, cfg.history_max_turns)
    await _persist_user_message(user_id, conversation_id, message)

    state = _initial_state(user_id, conversation_id, message, history, top_k)
    graph = get_graph()
    result: AgentState = await graph.ainvoke(state)  # type: ignore[assignment]

    answer = validate_answer(result, latency_ms=int((time.perf_counter() - started) * 1000))
    message_id = await _persist_answer(
        user_id=user_id, conversation_id=conversation_id, answer=answer, route=answer.route
    )
    logger.info(
        "chat_done conversation=%s route=%s tokens=%d cost=%.8f",
        conversation_id,
        answer.route,
        answer.usage.total_tokens,
        answer.usage.cost_usd,
    )
    return _to_response(answer, conversation_id, message_id)


async def stream(
    *,
    user_id: str,
    message: str,
    conversation_id: str | None = None,
    top_k: int | None = None,
) -> AsyncIterator[str]:
    """SSE 流式：token 增量推送，最后一帧 done 携带完整结构化结果。"""
    cfg = get_settings()
    conversation_id = conversation_id or uuid.uuid4().hex
    started = time.perf_counter()

    history = await load_history(user_id, conversation_id, cfg.history_max_turns)
    await _persist_user_message(user_id, conversation_id, message)

    yield sse_event("start", {"conversation_id": conversation_id, "model": model_display_name()})

    state = _initial_state(user_id, conversation_id, message, history, top_k)
    graph = get_graph()

    emitted: list[str] = []
    final_state: dict[str, Any] = {}
    route_sent = False
    citations_sent = False
    try:
        async for mode, payload in graph.astream(state, stream_mode=["messages", "values"]):
            if mode == "messages":
                chunk, meta = payload  # type: ignore[misc]
                # 只有 synthesize 的 token 才是给用户看的正文；
                # intent_router / sql_builder 也会调模型，它们的输出必须挡掉
                if meta.get("langgraph_node") != "synthesize":
                    continue
                content = getattr(chunk, "content", "")
                if isinstance(content, str) and content:
                    emitted.append(content)
                    yield sse_event("token", {"t": content})
            elif mode == "values":
                final_state = payload  # type: ignore[assignment]
                # 意图一旦确定就先告诉前端，用户不必等正文才知道走的是哪条链路
                if not route_sent and final_state.get("intent_source"):
                    route_sent = True
                    yield sse_event(
                        "route",
                        {
                            "route": final_state.get("route"),
                            "source": final_state.get("intent_source"),
                            "reason": final_state.get("intent_reason", ""),
                        },
                    )
                # 检索完就把引用推给前端，边生成边能显示来源
                citations = final_state.get("citations") or []
                if not citations_sent and citations:
                    citations_sent = True
                    yield sse_event("citations", {"citations": citations})
    except Exception as exc:  # noqa: BLE001 - 流中断要保证客户端收到明确错误
        logger.exception("stream_failed conversation=%s err=%s", conversation_id, exc)
        if not emitted:
            # 一个 token 都没发出 → 退回一次性执行，仍然给用户完整答案
            try:
                final_state = await graph.ainvoke(state)  # type: ignore[assignment]
                answer_text = str(final_state.get("answer") or "")
                for start in range(0, len(answer_text), 24):
                    piece = answer_text[start : start + 24]
                    emitted.append(piece)
                    yield sse_event("token", {"t": piece})
            except Exception:  # noqa: BLE001
                yield sse_event("error", {"code": "stream_failed", "message": "生成过程中断，请重试"})
                return
        else:
            yield sse_event("error", {"code": "stream_failed", "message": "生成过程中断，请重试"})
            return

    if not final_state:
        final_state = await graph.ainvoke(state)  # type: ignore[assignment]

    latency_ms = int((time.perf_counter() - started) * 1000)
    try:
        answer = validate_answer(final_state, latency_ms=latency_ms)
    except AgentOutputError as exc:
        yield sse_event("error", {"code": exc.code, "message": exc.message})
        return

    message_id = await _persist_answer(
        user_id=user_id, conversation_id=conversation_id, answer=answer, route=answer.route
    )
    response = _to_response(answer, conversation_id, message_id)
    yield sse_event("done", response.model_dump())
    logger.info(
        "chat_stream_done conversation=%s route=%s tokens=%d",
        conversation_id,
        answer.route,
        answer.usage.total_tokens,
    )
