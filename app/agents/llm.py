"""LLM 工厂。

两个实现：
- offline（fake）：LangChain 自定义 ChatModel，确定性、零网络、零成本，
  用于本地演示与 CI。它仍然走 LangChain 的流式回调，所以 SSE 的
  stream_mode="messages" 在两种 provider 下行为一致。
- openai：真实模型，需要装 requirements-llm.txt 并配置 OPENAI_API_KEY。
"""

from __future__ import annotations

import re
from collections.abc import AsyncIterator
from typing import Any

from langchain_core.language_models.chat_models import SimpleChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGenerationChunk

from app.core.config import get_settings
from app.core.errors import AppError
from app.core.logging import get_logger

logger = get_logger(__name__)

_CHUNK_RE = re.compile(r"\[片段\d+\] 标题：(?P<title>.*?)\n正文：(?P<body>.*?)(?=\n\[片段|\Z)", re.S)
_SQL_RESULT_RE = re.compile(r"\[SQL结果\] 共 (?P<rows>\d+) 行\n(?P<body>.*)\Z", re.S)
_STREAM_PIECE = 12  # 每个 chunk 的字符数，模拟真实 token 粒度


class OfflineChatModel(SimpleChatModel):
    """离线确定性模型：把提示词里的检索片段抽句式地重新组织成答案。

    它不是"智能"的，但它诚实地反映检索到的内容，且完全可复现 ——
    这正是测试与演示需要的性质。
    """

    model_name: str = "offline-deterministic"

    @property
    def _llm_type(self) -> str:
        return "offline-deterministic"

    # ---------- 核心生成逻辑 ----------
    def _compose(self, messages: list[BaseMessage]) -> str:
        system_text = "\n".join(m.content for m in messages if m.type == "system" and isinstance(m.content, str))
        human_text = ""
        for m in reversed(messages):
            if m.type == "human" and isinstance(m.content, str):
                human_text = m.content
                break

        sql_match = _SQL_RESULT_RE.search(system_text)
        if sql_match:
            return self._compose_sql(sql_match, human_text)

        fragments = _CHUNK_RE.findall(system_text)
        if fragments:
            return self._compose_rag(fragments, human_text)

        return (
            "我是个人知识库问答助手，可以基于你已上传的文档回答问题，也能把统计类问题翻译成 SQL 查询。"
            "目前知识库中没有检索到与本次提问相关的资料；你可以先通过 "
            "`POST /api/v1/knowledge/ingest` 上传文档，再问我一次。"
        )

    @staticmethod
    def _compose_rag(fragments: list[tuple[str, str]], question: str) -> str:
        lines: list[str] = []
        titles = [t.strip() for t, _ in fragments if t.strip()]
        head = "、".join(f"《{t}》" for t in dict.fromkeys(titles[:3]))
        lines.append(f"围绕「{question.strip()[:40]}」，知识库中{head or '相关资料'}给出了以下要点：")
        lines.append("")
        for idx, (title, body) in enumerate(fragments[:3], start=1):
            sentence = re.split(r"[。！？\n]", body.strip())[0].strip()
            if not sentence:
                continue
            lines.append(f"{idx}. {sentence}。")
        lines.append("")
        lines.append("以上内容严格来自检索片段，未做外部扩写；具体出处见下方引用列表。")
        return "\n".join(lines)

    @staticmethod
    def _compose_sql(match: re.Match[str], question: str) -> str:
        row_count = int(match.group("rows"))
        body = match.group("body").strip()
        # 上下文里已经带了序号，这里去掉避免出现 "- 1. {...}" 这种双重编号
        preview = [
            re.sub(r"^\d+\.\s*", "", ln.strip())
            for ln in body.splitlines()
            if ln.strip()
        ][:5]
        if row_count == 0:
            return f"针对「{question.strip()[:40]}」，查询执行成功但没有命中任何数据（0 行）。"
        detail = "\n".join(f"- {ln}" for ln in preview)
        suffix = "" if row_count <= len(preview) else f"\n（共 {row_count} 行，此处仅展示前 {len(preview)} 行）"
        return (
            f"针对「{question.strip()[:40]}」，我把问题转换成了只读 SQL 并执行，"
            f"共返回 {row_count} 行结果：\n\n{detail}{suffix}"
        )

    def _call(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> str:
        return self._compose(messages)

    async def _astream(  # type: ignore[override]
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        text = self._compose(messages)
        for start in range(0, len(text), _STREAM_PIECE):
            piece = text[start : start + _STREAM_PIECE]
            yield ChatGenerationChunk(message=AIMessageChunk(content=piece))


class OpenAICompatibleChatModel:
    """薄封装，让上层只依赖 astream/ainvoke 两个方法。"""

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self.model_name = getattr(inner, "model_name", "unknown")

    async def astream(self, messages: list[BaseMessage]) -> AsyncIterator[BaseMessage]:
        async for chunk in self._inner.astream(messages):
            yield chunk

    async def ainvoke(self, messages: list[BaseMessage]) -> AIMessage:
        return await self._inner.ainvoke(messages)


_model: Any = None


def get_chat_model() -> Any:
    global _model
    if _model is not None:
        return _model

    cfg = get_settings()
    if cfg.llm_provider == "openai":
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:  # pragma: no cover - 仅真实模型路径触发
            raise AppError(
                "缺少 langchain-openai，请先 pip install -r requirements-llm.txt",
                details={"provider": cfg.llm_provider},
            ) from exc

        inner = ChatOpenAI(
            model=cfg.llm_model,
            api_key=cfg.llm_api_key or "not-set",
            base_url=cfg.llm_base_url,
            temperature=cfg.llm_temperature,
            max_tokens=cfg.llm_max_tokens,
            streaming=True,
            timeout=cfg.tool_timeout_seconds,
            max_retries=cfg.tool_max_retries,
        )
        _model = OpenAICompatibleChatModel(inner)
        logger.info("llm_provider=openai model=%s", cfg.llm_model)
    else:
        _model = OfflineChatModel()
        logger.info("llm_provider=offline model=%s", _model.model_name)
    return _model


def set_chat_model(model: Any) -> None:
    global _model
    _model = model


def model_display_name() -> str:
    cfg = get_settings()
    return cfg.llm_model if cfg.llm_provider == "openai" else "offline-deterministic"
