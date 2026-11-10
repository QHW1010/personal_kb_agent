"""LangGraph 节点实现。

节点职责单一，互不越界：
route → retrieve / generate_sql → compose
所有外部 I/O 都经 call_with_resilience，答案组装后由 service 层过 Pydantic。
"""

from __future__ import annotations

import re
from datetime import date, datetime
from decimal import Decimal
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from sqlalchemy import text

from app.agents import prompts
from app.agents.llm import get_chat_model
from app.agents.state import AgentState
from app.core.config import get_settings
from app.core.errors import ToolExecutionError
from app.core.logging import get_logger
from app.core.resilience import call_with_resilience
from app.db.session import get_session_factory
from app.rag.retriever import hits_to_citations, retrieve
from app.schemas.agent import SQLPlan
from app.services.usage import estimate_tokens

logger = get_logger(__name__)

# ---------------------------------------------------------------- 路由规则
_GREETING_RE = re.compile(
    r"^\s*(你好|您好|hi|hello|hey|在吗|谢谢|感谢|你是谁|你叫什么|what can you do|帮助|help)"
    r"\s*[!！。.~？?、,，]*\s*$",
    re.IGNORECASE,
)
_AGG_RE = re.compile(r"(多少|几个|几条|几篇|几位|数量|统计|总数|总计|排名|占比|平均|最高|最低|汇总|sum|count|avg|max|min|top)")
_ENTITY_RE = re.compile(
    r"(文档|资料|知识库|用户|账号|对话|消息|会话|token|成本|花费|费用|记录|延迟|用量)",
    re.IGNORECASE,
)


def classify(question: str) -> str:
    """确定性路由：greeting → direct，聚合类问题 → sql，其余 → rag。

    规则优先而非 LLM 优先，是为了让路由可解释、可单测、零成本。
    """
    q = question.strip()
    if _GREETING_RE.match(q):
        return "direct"
    if _AGG_RE.search(q) and _ENTITY_RE.search(q):
        return "sql"
    return "rag"


# ---------------------------------------------------------------- 离线 NL2SQL
# 小型意图库：无 Key 时也能真实命中 SQL 场景（验收标准要求）
# 每条规则 = (实体模式, 聚合模式或 None, SQL)，两个模式同时出现在问句里才算命中，
# 这样"一共有几个用户"和"用户一共有几个"都能匹配。
_OFFLINE_SQL_RULES: list[tuple[re.Pattern[str], re.Pattern[str] | None, str]] = [
    (
        re.compile(r"(文档|资料|知识库)"),
        re.compile(r"(多少|几篇|几个|数量|总数|统计)"),
        "SELECT count(*) AS total_documents FROM documents",
    ),
    (
        re.compile(r"(用户|账号)"),
        re.compile(r"(多少|几个|几位|数量|总数|统计|注册)"),
        "SELECT count(*) AS total_users FROM users",
    ),
    (
        re.compile(r"(对话|消息|会话)"),
        re.compile(r"(多少|几条|数量|总数|统计)"),
        "SELECT count(*) AS total_messages FROM chat_messages",
    ),
    (
        re.compile(r"(token|用量)", re.IGNORECASE),
        re.compile(r"(多少|总数|统计|合计|消耗|用了)"),
        "SELECT sum(total_tokens) AS total_tokens, sum(prompt_tokens) AS prompt_tokens, "
        "sum(completion_tokens) AS completion_tokens FROM usage_records",
    ),
    (
        re.compile(r"(成本|花费|费用|开销)"),
        None,
        "SELECT sum(cost_usd) AS total_cost_usd, count(*) AS calls FROM usage_records",
    ),
    (
        re.compile(r"(延迟|耗时)"),
        re.compile(r"(平均|均值)"),
        "SELECT avg(latency_ms) AS avg_latency_ms FROM usage_records",
    ),
    (
        re.compile(r"(最新|最近|最后)"),
        re.compile(r"(文档|资料)"),
        "SELECT title, source, created_at FROM documents ORDER BY created_at DESC",
    ),
    (
        re.compile(r"(最多|排名|排行)"),
        re.compile(r"(文档|切片)"),
        "SELECT title, chunk_count FROM documents ORDER BY chunk_count DESC",
    ),
]


def offline_sql(question: str) -> SQLPlan | None:
    q = question.strip()
    for entity_re, agg_re, sql in _OFFLINE_SQL_RULES:
        if not entity_re.search(q):
            continue
        if agg_re is not None and not agg_re.search(q):
            continue
        return SQLPlan(sql=sql, rationale="离线意图库匹配", tables=[_guess_table(sql)])
    return None


def _guess_table(sql: str) -> str:
    match = re.search(r"\bfrom\s+(\w+)", sql, re.IGNORECASE)
    return match.group(1) if match else "unknown"


def _extract_sql(raw: str) -> str:
    text_out = raw.strip()
    fenced = re.search(r"```(?:sql)?\s*(.+?)```", text_out, re.DOTALL | re.IGNORECASE)
    if fenced:
        return fenced.group(1).strip()
    return text_out


# ---------------------------------------------------------------- 工具函数
def _jsonify(value: Any) -> Any:
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8", errors="replace")
    return value


def _jsonify_row(row: dict[str, Any]) -> dict[str, Any]:
    return {k: _jsonify(v) for k, v in row.items()}


async def _execute_sql(sql: str) -> tuple[list[dict[str, Any]], bool]:
    """执行只读 SQL。行数上限只作为 DoS 防护，并显式回报是否截断。"""
    cap = get_settings().sql_max_rows

    async def _run() -> tuple[list[dict[str, Any]], bool]:
        factory = get_session_factory()
        async with factory() as session:
            result = await session.execute(text(sql))
            fetched = result.mappings().fetchmany(cap + 1)
            rows = [_jsonify_row(dict(r)) for r in fetched]
        truncated = len(rows) > cap
        return rows[:cap], truncated

    return await call_with_resilience(_run, label="sql.execute", timeout=8.0)


# ---------------------------------------------------------------- 节点
async def route_node(state: AgentState) -> dict[str, Any]:
    question = state["question"]
    route = classify(question)
    logger.info("route conversation=%s route=%s q=%s", state.get("conversation_id"), route, question[:40])
    return {"route": route}


async def retrieve_node(state: AgentState) -> dict[str, Any]:
    question = state["question"]
    hits = await retrieve(question)
    citations = [c.model_dump() for c in hits_to_citations(hits)]
    return {
        "hits": [
            {"chunk_id": h.chunk_id, "text": h.text, "score": h.score, "metadata": h.metadata}
            for h in hits
        ],
        "citations": citations,
    }


async def sql_node(state: AgentState) -> dict[str, Any]:
    """NL → SQL → 校验 → 执行。

    生成结果必须先过 SQLPlan（只读校验），再执行；任何一步失败都降级为
    可用文案而不是 500。
    """
    question = state["question"]
    cfg = get_settings()

    plan: SQLPlan | None = None
    if cfg.llm_provider != "fake":
        try:
            plan = await _llm_sql_plan(question)
        except (ToolExecutionError, ValueError) as exc:
            logger.warning("llm_sql_failed, fallback to offline rules: %s", exc)
    if plan is None:
        plan = offline_sql(question)
    if plan is None:
        return {
            "sql": None,
            "rows": None,
            "error": "无法把该问题转换为受支持的 SQL 查询",
            "confidence": 0.25,
        }

    try:
        rows, truncated = await _execute_sql(plan.sql)
    except ToolExecutionError as exc:
        return {
            "sql": plan.sql,
            "rows": None,
            "error": str(exc.message),
            "confidence": 0.2,
        }
    return {"sql": plan.sql, "rows": rows, "sql_truncated": truncated, "confidence": 0.85}


async def _llm_sql_plan(question: str) -> SQLPlan:
    model = get_chat_model()
    messages = [
        SystemMessage(content=prompts.SQL_SYSTEM_PROMPT + "\n" + prompts.SQL_FEWSHOT),
        HumanMessage(content=question),
    ]
    response = await call_with_resilience(
        lambda: model.ainvoke(messages), label="llm.nl2sql", timeout=20.0
    )
    raw = response.content if isinstance(response.content, str) else str(response.content)
    return SQLPlan(sql=_extract_sql(raw), rationale="LLM 生成", tables=[_guess_table(raw)])


def _build_context_block(state: AgentState) -> str:
    parts: list[str] = []
    hits = state.get("hits") or []
    for idx, hit in enumerate(hits, start=1):
        title = hit.get("metadata", {}).get("title", "未命名片段")
        parts.append(f"[片段{idx}] 标题：{title}\n正文：{hit.get('text', '')}")

    if state.get("route") == "sql":
        sql = state.get("sql") or "(未生成)"
        rows = state.get("rows") or []
        parts.append(f"[SQL] {sql}")
        lines = [f"{i}. {row}" for i, row in enumerate(rows, start=1)]
        parts.append("[SQL结果] 共 {} 行\n{}".format(len(rows), "\n".join(lines) or "(空)"))

    if not parts:
        return prompts.NO_CONTEXT_HINT
    return "\n\n".join(parts)


async def compose_node(state: AgentState) -> dict[str, Any]:
    """调用 LLM 组装答案。流式 token 由 graph 的 stream_mode="messages" 捕获。"""
    question = state["question"]
    context = _build_context_block(state)
    system = SystemMessage(content=prompts.RAG_SYSTEM_PROMPT.format(context=context))

    history = state.get("messages") or []
    messages = [system, *[m for m in history if m.type != "system"], HumanMessage(content=question)]

    model = get_chat_model()

    async def _run() -> str:
        buffer: list[str] = []
        async for chunk in model.astream(messages):
            content = chunk.content
            if isinstance(content, str):
                buffer.append(content)
        return "".join(buffer)

    answer = await call_with_resilience(_run, label="llm.compose", timeout=45.0)
    if not answer.strip():
        answer = "抱歉，本次生成没有返回有效内容，请换个问法再试一次。"

    prompt_text = system.content + "\n" + question + "\n" + "".join(m.content for m in history if isinstance(m.content, str))
    return {
        "answer": answer,
        "prompt_tokens": estimate_tokens(prompt_text),
        "completion_tokens": estimate_tokens(answer),
    }


async def direct_node(state: AgentState) -> dict[str, Any]:
    answer = (
        "你好，我是个人知识库问答助手。你可以：\n"
        "1. 直接提问，我会检索知识库并给出带引用的答案；\n"
        "2. 问统计类问题（如「知识库里有多少篇文档」），我会翻译成只读 SQL 并执行；\n"
        "3. 通过 POST /api/v1/knowledge/ingest 上传新文档后再问我。"
    )
    return {
        "answer": answer,
        "confidence": 0.6,
        "prompt_tokens": 0,
        "completion_tokens": estimate_tokens(answer),
    }
