"""LangGraph StateGraph 编译入口。

拓扑：
    START → intent_router ─┬→ retrieve ────┐
                           ├→ sql_builder ─┼→ synthesize → END
                           └→ direct ──────────────────────→ END

intent_router 决定"要不要查、查文档还是查库"，synthesize 是唯一产出自然语言的节点。
两个节点名都不与 AgentState 的键重名 —— 重名会被 LangGraph 直接拒绝。
"""

from __future__ import annotations

from functools import lru_cache

from langgraph.graph import END, START, StateGraph

from app.agents import nodes
from app.agents.state import AgentState


def _route_selector(state: AgentState) -> str:
    route = state.get("route") or "rag"
    return route if route in {"rag", "sql", "direct"} else "rag"


def build_graph():
    graph = StateGraph(AgentState)
    # 注意：节点 id 不能与 state 的键重名（langgraph 会直接抛错），
    # 所以这里用 intent_router / sql_builder 而不是 route / sql。
    graph.add_node("intent_router", nodes.intent_router_node)
    graph.add_node("retrieve", nodes.retrieve_node)
    graph.add_node("sql_builder", nodes.sql_node)
    graph.add_node("synthesize", nodes.synthesize_node)
    graph.add_node("direct", nodes.direct_node)

    graph.add_edge(START, "intent_router")
    graph.add_conditional_edges(
        "intent_router",
        _route_selector,
        {"rag": "retrieve", "sql": "sql_builder", "direct": "direct"},
    )
    graph.add_edge("retrieve", "synthesize")
    graph.add_edge("sql_builder", "synthesize")
    graph.add_edge("synthesize", END)
    graph.add_edge("direct", END)
    return graph.compile()


def graph_topology() -> list[tuple[str, str]]:
    """给 README / 排障用的边列表（不含条件边展开）。"""
    return [
        ("START", "intent_router"),
        ("intent_router", "retrieve | sql_builder | direct"),
        ("retrieve", "synthesize"),
        ("sql_builder", "synthesize"),
        ("synthesize", "END"),
        ("direct", "END"),
    ]


@lru_cache
def get_graph():
    return build_graph()
