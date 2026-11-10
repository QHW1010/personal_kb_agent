"""LangGraph StateGraph 编译入口。

拓扑：
    START → route ─┬→ retrieve ─┐
                   ├→ sql ──────┼→ compose → END
                   └→ direct ───────────────→ END
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
    # 所以这里用 router / sql_builder 而不是 route / sql。
    graph.add_node("router", nodes.route_node)
    graph.add_node("retrieve", nodes.retrieve_node)
    graph.add_node("sql_builder", nodes.sql_node)
    graph.add_node("compose", nodes.compose_node)
    graph.add_node("direct", nodes.direct_node)

    graph.add_edge(START, "router")
    graph.add_conditional_edges(
        "router",
        _route_selector,
        {"rag": "retrieve", "sql": "sql_builder", "direct": "direct"},
    )
    graph.add_edge("retrieve", "compose")
    graph.add_edge("sql_builder", "compose")
    graph.add_edge("compose", END)
    graph.add_edge("direct", END)
    return graph.compile()


@lru_cache
def get_graph():
    return build_graph()
