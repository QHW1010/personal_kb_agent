"""LangGraph 状态定义。

messages 使用 add_messages reducer，节点只返回增量即可。
其余字段为单值覆盖语义。
"""

from __future__ import annotations

from typing import Annotated, Any, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict, total=False):
    # 对话消息（含历史窗口 + 本轮问题）
    messages: Annotated[list[BaseMessage], add_messages]
    # 本轮原始问题
    question: str
    # 会话标识
    conversation_id: str
    user_id: str
    # 路由结果：rag | sql | direct
    route: str
    # 路由来源：fast_rule（寒暄直判）| llm（模型分类）| fallback（模型失败回落规则）
    intent_source: str
    # 路由理由（LLM 给的短理由，或规则命中的说明），仅用于可观测性
    intent_reason: str
    # 本轮检索条数（来自 ChatRequest.top_k，None 表示用全局配置）
    top_k: int | None
    # 检索命中的片段（已序列化，便于跨节点传递/落库）
    hits: list[dict[str, Any]]
    citations: list[dict[str, Any]]
    # NL2SQL 产物
    sql: str | None
    rows: list[dict[str, Any]] | None
    sql_truncated: bool
    # 最终答案
    answer: str
    confidence: float
    needs_clarification: bool
    # 本次调用的 token 估算（供 usage 台账使用）
    prompt_tokens: int
    completion_tokens: int
    # 失败降级信息
    error: str | None


def new_state(
    *,
    question: str,
    messages: list[BaseMessage],
    conversation_id: str,
    user_id: str,
    top_k: int | None = None,
) -> AgentState:
    return AgentState(
        messages=messages,
        question=question,
        conversation_id=conversation_id,
        user_id=user_id,
        route="rag",
        intent_source="",
        intent_reason="",
        top_k=top_k,
        hits=[],
        citations=[],
        sql=None,
        rows=None,
        sql_truncated=False,
        answer="",
        confidence=0.5,
        needs_clarification=False,
        prompt_tokens=0,
        completion_tokens=0,
        error=None,
    )
