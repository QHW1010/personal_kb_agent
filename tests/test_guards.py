"""约束守卫测试：SQL 只读校验、Agent 输出模式闸门、提示词不含截断指令。"""

from __future__ import annotations

import re

import pytest
from pydantic import ValidationError

from app.agents import prompts
from app.agents.state import new_state
from app.schemas.agent import AgentAnswer, SQLPlan
from app.services.chat_service import validate_answer


@pytest.mark.parametrize(
    "bad_sql",
    [
        "DELETE FROM users",
        "UPDATE documents SET title='x'",
        "DROP TABLE users",
        "select 1; drop table users",
        "INSERT INTO users(id) VALUES ('1')",
        "TRUNCATE chat_messages",
        "WITH x AS (DELETE FROM users RETURNING id) SELECT * FROM x",
    ],
)
def test_sqlplan_rejects_non_readonly(bad_sql: str) -> None:
    with pytest.raises(ValidationError):
        SQLPlan(sql=bad_sql)


def test_sqlplan_accepts_select_and_cte() -> None:
    assert SQLPlan(sql="SELECT count(*) FROM documents").sql.startswith("SELECT")
    assert SQLPlan(sql="WITH t AS (SELECT 1 AS a) SELECT a FROM t").sql.startswith("WITH")


def test_sqlplan_strips_trailing_semicolon() -> None:
    assert SQLPlan(sql="SELECT 1;").sql == "SELECT 1"


def test_agent_answer_rejects_blank() -> None:
    with pytest.raises(ValidationError):
        AgentAnswer(answer="   ")


def test_agent_answer_requires_sql_when_route_is_sql() -> None:
    with pytest.raises(ValidationError):
        AgentAnswer(answer="结果是 3", route="sql")


def test_agent_answer_rejects_unsafe_sql_payload() -> None:
    with pytest.raises(ValidationError):
        AgentAnswer(answer="x", route="sql", sql="DELETE FROM users")


def test_citations_are_deduplicated() -> None:
    citation = {
        "chunk_id": "same",
        "title": "t",
        "snippet": "s",
        "score": 0.5,
    }
    answer = AgentAnswer(answer="ok", citations=[citation, citation])
    assert len(answer.citations) == 1


def test_validate_answer_downgrades_failed_sql_route() -> None:
    """SQL 生成失败时不得冒充查询结果，路由降级为直答。"""
    state = new_state(question="随便问", messages=[], conversation_id="c1", user_id="u1")
    state.update({"route": "sql", "sql": None, "rows": None, "answer": ""})
    answer = validate_answer(state, latency_ms=5)
    assert answer.route == "direct"
    assert answer.sql is None
    assert "查询语句" in answer.answer or "没" in answer.answer


def test_validate_answer_raises_on_invalid_payload() -> None:
    from app.core.errors import AgentOutputError

    state = new_state(question="q", messages=[], conversation_id="c1", user_id="u1")
    state.update({"route": "sql", "sql": "SELECT 1", "rows": [], "answer": ""})
    # answer 为空会被兜底话术填上，因此这里不会抛错；
    # 真正会抛错的是结构非法，例如 route 取值越界
    assert validate_answer(state, latency_ms=1).answer
    state.update({"route": "unknown-route"})
    with pytest.raises(AgentOutputError):
        validate_answer(state, latency_ms=1)


def test_sql_prompt_does_not_hardcode_row_limit() -> None:
    """AGENTS.md 约束：禁止在 system prompt 里写死 LIMIT 10 之类截断指令。"""
    assert re.search(r"\blimit\s+\d+", prompts.SQL_SYSTEM_PROMPT, re.IGNORECASE) is None
    assert re.search(r"\blimit\s+\d+", prompts.SQL_FEWSHOT, re.IGNORECASE) is None
    # 并且要明确告诉模型不要自行加行数限制
    assert "不要" in prompts.SQL_SYSTEM_PROMPT


def test_offline_sql_rules_cover_required_scenarios() -> None:
    from app.agents.nodes import offline_sql

    assert offline_sql("知识库有多少篇文档") is not None
    assert offline_sql("一共有几个用户") is not None
    assert offline_sql("累计消耗多少 token") is not None
    assert offline_sql("总花费多少钱") is not None
    assert offline_sql("今天天气怎么样") is None
