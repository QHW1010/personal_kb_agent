"""编排层行为：LLM 意图路由、解析容错、真实 token 用量、SSE 事件顺序。

全部离线：用注入的假模型替代 DeepSeek，真实网络只出现在
pytest -m integration 的用例里（见 tests/test_fastembed_branch.py）。
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import pytest
from httpx import AsyncClient
from langchain_core.language_models.chat_models import SimpleChatModel
from langchain_core.messages import AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGenerationChunk

from app.agents import nodes
from app.agents.llm import reset_chat_model, set_chat_model
from app.core.config import get_settings


class _StaticChat(SimpleChatModel):
    """固定回复的 BaseChatModel：不联网、可断言。"""

    reply: str = ""
    explode: bool = False

    @property
    def _llm_type(self) -> str:
        return "static-fake"

    def _call(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> str:
        if self.explode:
            raise RuntimeError("模型炸了")
        return self.reply


class _UsageChat(SimpleChatModel):
    """流式返回带 usage_metadata 的 chunk，模拟真实网关的计量方式。"""

    reply: str = "答案正文"
    prompt_tokens: int = 100
    completion_tokens: int = 20

    @property
    def _llm_type(self) -> str:
        return "usage-fake"

    def _call(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> str:
        return self.reply

    async def _astream(  # type: ignore[override]
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        midpoint = len(self.reply) // 2
        for piece in (self.reply[:midpoint], self.reply[midpoint:]):
            yield ChatGenerationChunk(message=AIMessageChunk(content=piece))
        # 真实网关的 usage 在最后一个 chunk 上
        yield ChatGenerationChunk(
            message=AIMessageChunk(
                content="",
                usage_metadata={
                    "input_tokens": self.prompt_tokens,
                    "output_tokens": self.completion_tokens,
                    "total_tokens": self.prompt_tokens + self.completion_tokens,
                },
            )
        )


@pytest.fixture()
def real_llm_settings(monkeypatch: pytest.MonkeyPatch):
    """把 provider 切成 openai，但模型仍是注入的假模型 —— 走 LLM 分支，不联网。"""
    cfg = get_settings()
    monkeypatch.setattr(cfg, "llm_provider", "openai")
    yield cfg
    reset_chat_model()


# ---------------------------------------------------------------- 解析容错
@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ('{"intent": "rag", "reason": "要查文档"}', "rag"),
        ('```json\n{"intent": "sql", "reason": "统计"}\n```', "sql"),
        ('{"intent":"DIRECT","reason":"寒暄"}', "direct"),
        ("我认为这是 rag 类问题", "rag"),
        ("sql", "sql"),
    ],
)
def test_parse_intent_tolerates_model_noise(raw: str, expected: str) -> None:
    intent, _reason = nodes.parse_intent(raw)
    assert intent == expected


def test_parse_intent_rejects_unparseable_output() -> None:
    with pytest.raises(ValueError):
        nodes.parse_intent("我无法判断")


# ---------------------------------------------------------------- 路由
async def test_greeting_never_calls_llm(real_llm_settings) -> None:
    """寒暄走正则直判：模型即使会炸也不该被调用。"""
    set_chat_model(_StaticChat(reply="不该被调用", explode=True))

    result = await nodes.intent_router_node({"question": "你好"})

    assert result["route"] == "direct"
    assert result["intent_source"] == "fast_rule"


async def test_intent_router_uses_llm(real_llm_settings) -> None:
    set_chat_model(_StaticChat(reply='{"intent": "sql", "reason": "问数量"}'))

    result = await nodes.intent_router_node({"question": "知识库里有多少文档"})

    assert result["route"] == "sql"
    assert result["intent_source"] == "llm"
    assert result["intent_reason"] == "问数量"
    assert result["confidence"] > 0.7


async def test_intent_router_vetoes_rules_when_llm_disagrees(real_llm_settings) -> None:
    """规则会判 sql（有聚合词+实体词），模型判 rag 时应以模型为准。"""
    assert nodes.classify("文档一共有多少种检索策略的区别") in {"sql", "rag"}
    set_chat_model(_StaticChat(reply='{"intent": "rag", "reason": "问区别"}'))

    result = await nodes.intent_router_node({"question": "文档一共有多少种检索策略的区别"})

    assert result["route"] == "rag"
    assert result["intent_source"] == "llm"


async def test_intent_router_falls_back_to_rules_when_llm_fails(real_llm_settings) -> None:
    set_chat_model(_StaticChat(reply="", explode=True))

    result = await nodes.intent_router_node({"question": "知识库里一共有多少篇文档？"})

    assert result["route"] == "sql"  # 规则兜底命中
    assert result["intent_source"] == "fallback"
    assert "回落" in result["intent_reason"]


async def test_intent_router_stays_rule_based_when_provider_is_fake(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = get_settings()
    monkeypatch.setattr(cfg, "llm_provider", "fake")
    set_chat_model(_StaticChat(reply='{"intent": "sql", "reason": "x"}'))

    result = await nodes.intent_router_node({"question": "重排序是怎么工作的？"})
    reset_chat_model()

    assert result["route"] == "rag"
    assert result["intent_source"] == "fast_rule"


# ---------------------------------------------------------------- 生成
async def test_synthesize_uses_real_usage_from_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """真实网关会回 usage，台账必须用它，而不是自己估算。"""
    set_chat_model(_UsageChat(reply="硬编码 LIMIT 会截断多行答案。", prompt_tokens=321, completion_tokens=45))

    result = (
        await nodes.synthesize_node(
            {
                "question": "为什么禁止硬编码 LIMIT？",
                "hits": [{"text": "片段", "metadata": {"title": "设计文档"}}],
                "messages": [],
                "route": "rag",
            }
        )
    )
    reset_chat_model()

    assert result["answer"].startswith("硬编码")
    assert result["prompt_tokens"] == 321
    assert result["completion_tokens"] == 45
    assert result["needs_clarification"] is False


async def test_synthesize_marks_clarification_when_no_hits() -> None:
    set_chat_model(_StaticChat(reply="知识库中没有找到相关信息。"))

    result = (
        await nodes.synthesize_node({"question": "无关问题", "hits": [], "messages": [], "route": "rag"})
    )
    reset_chat_model()

    assert result["needs_clarification"] is True
    assert result["confidence"] < 0.5


async def test_synthesize_degrades_without_raising() -> None:
    set_chat_model(_StaticChat(reply="", explode=True))

    result = (
        await nodes.synthesize_node({"question": "随便问", "hits": [], "messages": [], "route": "rag"})
    )
    reset_chat_model()

    assert result["answer"] == ""
    assert result["error"]


def test_strip_citation_marks_when_no_citations() -> None:
    """SQL 路线没有引用，模型若自作主张写 [1] 必须抹掉。"""
    assert nodes.strip_unsupported_citation_marks("共有 5 篇文档。[1]", []) == "共有 5 篇文档。"
    assert nodes.strip_unsupported_citation_marks("结论[1][2]，补充说明。", []) == "结论，补充说明。"


def test_keep_citation_marks_when_citations_exist() -> None:
    cites = [{"source": "kb-core-design.md"}]
    assert nodes.strip_unsupported_citation_marks("结论。[1]", cites) == "结论。[1]"


# ---------------------------------------------------------------- SSE 事件
async def test_stream_emits_route_and_citations_events(
    client: AsyncClient, auth_headers: dict
) -> None:
    """前端需要"先知道走哪条链路、再看到正文"，两个事件必须在 token 之前到达。"""
    async with client.stream(
        "POST",
        "/api/v1/chat",
        headers=auth_headers,
        json={"message": "重排序是怎么工作的？", "stream": True},
    ) as resp:
        assert resp.status_code == 200
        body = "".join([chunk async for chunk in resp.aiter_text()])

    assert "event: route" in body
    assert '"route": "rag"' in body or '"route":"rag"' in body
    assert "event: citations" in body
    # 顺序：route → citations → token → done
    assert body.index("event: route") < body.index("event: citations") < body.index("event: token")
    assert body.rindex("event: done") > body.index("event: token")
