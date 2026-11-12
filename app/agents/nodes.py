"""LangGraph 节点实现。

节点职责单一，互不越界：
route → retrieve / generate_sql → compose
所有外部 I/O 都经 call_with_resilience，答案组装后由 service 层过 Pydantic。
"""

from __future__ import annotations

import json
import re
from datetime import date, datetime
from decimal import Decimal
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from sqlalchemy import text

from app.agents import prompts
from app.agents.llm import get_chat_model, get_intent_model, is_real_llm
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

    它是 LLM 路由的**兜底**，不是唯一路径：真实模型可用时由模型判断意图，
    模型超时/返回脏格式时回落到这里，保证路由永远不会"没有答案"。
    """
    q = question.strip()
    if _GREETING_RE.match(q):
        return "direct"
    if _AGG_RE.search(q) and _ENTITY_RE.search(q):
        return "sql"
    return "rag"


# ---------------------------------------------------------------- LLM 意图路由
_VALID_INTENTS = ("rag", "sql", "direct")
_INTENT_JSON_RE = re.compile(r"\{.*?\}", re.DOTALL)


def parse_intent(raw: str) -> tuple[str, str]:
    """解析模型的意图输出，返回 (intent, reason)。

    真实模型的输出不可控：可能带 ```json 围栏、可能多一句解释、
    也可能直接回一个单词。三种形态都兼容，实在解析不出才抛错交给上层回落。
    """
    text_out = (raw or "").strip()
    match = _INTENT_JSON_RE.search(text_out)
    if match:
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, dict):
            intent = str(payload.get("intent", "")).strip().lower()
            if intent in _VALID_INTENTS:
                return intent, str(payload.get("reason", "")).strip()[:40]

    lowered = text_out.lower()
    for intent in _VALID_INTENTS:
        if re.search(rf"\b{intent}\b", lowered):
            return intent, "关键词解析"
    raise ValueError(f"无法解析意图输出: {text_out[:60]!r}")


async def _llm_intent(question: str) -> tuple[str, str]:
    cfg = get_settings()
    model = get_intent_model()
    messages = [
        SystemMessage(content=prompts.INTENT_SYSTEM_PROMPT),
        HumanMessage(content=question),
    ]
    response = await call_with_resilience(
        lambda: model.ainvoke(messages),
        label="llm.intent",
        timeout=cfg.llm_intent_timeout_seconds,
        retries=1,
    )
    content = response.content
    raw = content if isinstance(content, str) else str(content)
    return parse_intent(raw)


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
async def intent_router_node(state: AgentState) -> dict[str, Any]:
    """意图路由：寒暄直判 → LLM 分类 → 规则兜底。"""
    question = state["question"]
    cfg = get_settings()

    # 寒暄用正则整句匹配即可，不值得为"你好"花一次 API 调用与 800ms
    if _GREETING_RE.match(question.strip()):
        return {
            "route": "direct",
            "intent_source": "fast_rule",
            "intent_reason": "寒暄直判",
            "confidence": 0.6,
        }

    route = classify(question)
    source = "fast_rule"
    reason = "规则命中"

    if cfg.llm_provider != "fake":
        try:
            route, reason = await _llm_intent(question)
            source = "llm"
        except Exception as exc:  # noqa: BLE001 - 路由失败绝不能拖垮整轮问答
            logger.warning("intent_llm_failed, fallback to rules: %s", exc)
            route = classify(question)
            source = "fallback"
            reason = "模型路由失败，回落规则"

    logger.info(
        "intent conversation=%s route=%s source=%s q=%s",
        state.get("conversation_id"),
        route,
        source,
        question[:40],
    )
    return {
        "route": route,
        "intent_source": source,
        "intent_reason": reason,
        # 模型给出的路由比正则更可信，但也不给满分
        "confidence": 0.85 if source == "llm" else 0.7,
    }


# 节点 id 与函数名的历史别名：外部（测试/文档）仍可能按旧名引用
route_node = intent_router_node


async def retrieve_node(state: AgentState) -> dict[str, Any]:
    question = state["question"]
    # top_k 由 ChatRequest 透传（None 时走全局配置），不再是死参数
    hits = await retrieve(question, top_k=state.get("top_k"))
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

    plan: SQLPlan | None = None
    if is_real_llm():
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
    cfg = get_settings()
    model = get_chat_model()
    messages = [
        SystemMessage(content=prompts.SQL_SYSTEM_PROMPT + "\n" + prompts.SQL_FEWSHOT),
        HumanMessage(content=question),
    ]
    response = await call_with_resilience(
        lambda: model.ainvoke(messages),
        label="llm.nl2sql",
        timeout=cfg.llm_timeout_seconds,
        retries=1,
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


def _usage_to_tokens(usage: Any) -> tuple[int, int]:
    """把 LangChain 的 usage_metadata 归一成 (prompt, completion)。

    不同 provider 的字段名不完全一致（input_tokens / prompt_tokens），
    取不到就返回 0，由上层用 estimate_tokens 兜底。
    """
    if not isinstance(usage, dict):
        return 0, 0
    prompt = usage.get("input_tokens") or usage.get("prompt_tokens") or 0
    completion = usage.get("output_tokens") or usage.get("completion_tokens") or 0
    return int(prompt), int(completion)


_CITE_MARK_RE = re.compile(r"\s*\[\d+(?:\s*[,，]\s*\d+)*\]")


def strip_unsupported_citation_marks(answer: str, citations: list[dict[str, Any]]) -> str:
    """没有引用时，把模型自作主张写的 [1] 抹掉。

    引用列表是唯一事实来源。留着裸的 [1] 会让前端渲染成指向不存在的来源，
    比不给标记更糟。有引用时原样保留 —— 那是模型按提示词要求标注的。
    """
    if citations or not answer:
        return answer
    cleaned = _CITE_MARK_RE.sub("", answer)
    return re.sub(r"[ \t]{2,}", " ", cleaned).strip()


def _prompt_for_route(state: AgentState, context: str) -> str:
    """按路由选提示词：SQL 路线不该出现文档引用规则，否则模型会编造 [1]。"""
    if state.get("route") == "sql" and not state.get("hits"):
        return prompts.SQL_ANSWER_PROMPT.format(context=context)
    return prompts.RAG_SYSTEM_PROMPT.format(context=context)


async def synthesize_node(state: AgentState) -> dict[str, Any]:
    """调用 LLM 合成答案（ synthesize 阶段）。

    关键点：这里必须**真的流式消费** model.astream。token 经 LangChain 的
    回调冒泡到 LangGraph，再由 stream_mode="messages" 推给 SSE；
    如果把 astream 换成 ainvoke，客户端会等整段答案生成完才收到第一个字节。
    """
    cfg = get_settings()
    question = state["question"]
    context = _build_context_block(state)
    system = SystemMessage(content=_prompt_for_route(state, context))

    history = state.get("messages") or []
    messages = [system, *[m for m in history if m.type != "system"], HumanMessage(content=question)]

    model = get_chat_model()

    async def _run() -> tuple[str, tuple[int, int]]:
        buffer: list[str] = []
        usage: Any = None
        async for chunk in model.astream(messages):
            content = getattr(chunk, "content", "")
            if isinstance(content, str) and content:
                buffer.append(content)
            meta = getattr(chunk, "usage_metadata", None)
            if meta:
                usage = meta
        return "".join(buffer), _usage_to_tokens(usage)

    try:
        # retries=0：流式调用一旦重试，客户端会收到重复 token，宁可让上层报错
        answer, (prompt_tokens, completion_tokens) = await call_with_resilience(
            _run, label="llm.synthesize", timeout=cfg.llm_timeout_seconds, retries=0
        )
    except ToolExecutionError as exc:
        logger.error("synthesize_failed: %s", exc)
        return {
            "answer": "",
            "error": str(exc.message),
            "confidence": 0.2,
            "prompt_tokens": 0,
            "completion_tokens": 0,
        }

    if not answer.strip():
        answer = "抱歉，本次生成没有返回有效内容，请换个问法再试一次。"
    answer = strip_unsupported_citation_marks(answer, state.get("citations") or [])

    hits = state.get("hits") or []
    prompt_text = system.content + "\n" + question + "\n" + "".join(
        m.content for m in history if isinstance(m.content, str)
    )
    return {
        "answer": answer,
        # 真实模型给了 usage 就用真实值；离线模型没有，走估算
        "prompt_tokens": prompt_tokens or estimate_tokens(prompt_text),
        "completion_tokens": completion_tokens or estimate_tokens(answer),
        "confidence": 0.85 if hits else 0.4,
        "needs_clarification": not hits,
    }


# 旧名兼容（README / 早期文档里叫 compose）
compose_node = synthesize_node


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
