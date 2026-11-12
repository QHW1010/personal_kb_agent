"""LLM 工厂。

两个实现：
- openai：真实模型。DeepSeek / 通义 / vLLM / Azure 都兼容 OpenAI 协议，
  所以统一走 ChatOpenAI，只换 base_url 与 model。
- offline（fake）：LangChain 自定义 ChatModel，确定性、零网络、零成本，
  用于本地演示与 CI。

**两者都是 LangChain BaseChatModel**，这一点是硬要求：
LangGraph 的 stream_mode="messages" 靠 BaseChatModel 的流式回调工作，
任何"薄封装"过的普通 Python 对象都不会发出 token（历史 bug 就是这么来的）。
"""

from __future__ import annotations

import asyncio
import re
import time
from collections.abc import AsyncIterator
from typing import Any

from langchain_core.language_models.chat_models import SimpleChatModel
from langchain_core.messages import AIMessageChunk, BaseMessage, HumanMessage
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


def _build_openai_chat_model(cfg: Any) -> Any:
    """构造真实的 OpenAI 兼容模型。DeepSeek 只是 base_url 不同。"""
    try:
        from langchain_openai import ChatOpenAI
    except ImportError as exc:  # pragma: no cover - 仅在缺包时触发
        raise AppError(
            "缺少 langchain-openai，请先 pip install -r requirements-llm.txt",
            details={"provider": cfg.llm_provider},
        ) from exc

    if not cfg.llm_api_key:
        raise AppError(
            "LLM_PROVIDER=openai 但没有配置 API Key（OPENAI_API_KEY / LLM_API_KEY）",
            details={"provider": cfg.llm_provider, "model": cfg.llm_model},
        )

    return ChatOpenAI(
        model=cfg.llm_model,
        api_key=cfg.llm_api_key,
        base_url=cfg.llm_base_url,
        temperature=cfg.llm_temperature,
        max_tokens=cfg.llm_max_tokens,
        streaming=True,
        # usage 只在最后一个 chunk 回来的话，流式台账就记不到真实用量。
        # 走 extra_body 而不是 stream_options 参数，避免 langchain 把它塞进
        # model_kwargs 并打 UserWarning。
        extra_body={"stream_options": {"include_usage": True}},
        timeout=cfg.llm_timeout_seconds,
        # 重试统一由 call_with_resilience 管：SDK 自己再重试会变成 3×3 放大
        max_retries=0,
    )


_model: Any = None


def get_chat_model() -> Any:
    """返回统一的 BaseChatModel。流式与非流式都用它。"""
    global _model
    if _model is not None:
        return _model

    cfg = get_settings()
    if cfg.llm_provider == "openai":
        _model = _build_openai_chat_model(cfg)
        logger.info(
            "llm_provider=openai model=%s base_url=%s",
            cfg.llm_model,
            cfg.llm_base_url or "(默认)",
        )
    else:
        _model = OfflineChatModel()
        logger.info("llm_provider=offline model=%s", _model.model_name)
    return _model


def get_intent_model() -> Any:
    """意图分类专用：同模型、温度为 0、只允许极短输出。

    分类任务不该带创造性，也不该消耗正文的 max_tokens。
    """
    base = get_chat_model()
    bind = getattr(base, "bind", None)
    if bind is None:  # OfflineChatModel 也支持 bind（BaseChatModel 提供）
        return base
    return base.bind(temperature=0.0, max_tokens=64)


def set_chat_model(model: Any) -> None:
    """测试注入点：绕开真实网络，同时保证是 BaseChatModel。"""
    global _model
    _model = model


def reset_chat_model() -> None:
    global _model
    _model = None


def is_real_llm() -> bool:
    return get_settings().llm_provider == "openai"


def model_display_name() -> str:
    cfg = get_settings()
    return cfg.llm_model if cfg.llm_provider == "openai" else "offline-deterministic"


def describe_llm() -> dict[str, Any]:
    """给 /health 用的自述信息：不泄露 Key，只报有没有配。"""
    cfg = get_settings()
    return {
        "provider": cfg.llm_provider,
        "model": model_display_name(),
        "base_url": cfg.llm_base_url or "",
        "api_key_configured": bool(cfg.llm_api_key),
        "temperature": cfg.llm_temperature,
        "max_tokens": cfg.llm_max_tokens,
    }


async def warmup_llm() -> bool:
    """启动期预热：建连接 + 打一次 1-token 请求。

    首次调用真实网关要付 TLS 握手、DNS、连接池初始化的成本，实测能把
    第一个用户问题的首字节推迟 5-7 秒。把它挪到启动期，用户感知的 TTFT
    就只剩推理本身。失败不阻断启动 —— 网络抖动不该让服务起不来。
    """
    if not is_real_llm():
        return True

    started = time.perf_counter()
    try:
        model = get_chat_model()
        await asyncio.wait_for(
            model.bind(max_tokens=1).ainvoke([HumanMessage(content="ping")]),
            timeout=get_settings().llm_intent_timeout_seconds,
        )
        elapsed = (time.perf_counter() - started) * 1000
        logger.info("llm_warmup_ok model=%s elapsed_ms=%.0f", model_display_name(), elapsed)
        return True
    except Exception as exc:  # noqa: BLE001 - 预热失败只影响首个请求的快慢
        logger.warning("llm_warmup_failed: %s", exc)
        return False
