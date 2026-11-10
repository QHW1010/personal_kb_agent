"""对话路由：SSE 流式 + 一次性 JSON。"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_user
from app.core.cache import enforce_rate_limit
from app.core.logging import get_logger
from app.db.models import User
from app.db.session import get_session
from app.schemas.chat import ChatRequest
from app.services import chat_service
from app.services.usage import summarize_usage

logger = get_logger(__name__)
router = APIRouter(prefix="/chat", tags=["chat"])

_SSE_HEADERS = {
    "Cache-Control": "no-cache, no-transform",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",  # 关掉 nginx 缓冲，否则流式会被攒包
}


@router.post(
    "",
    response_model=None,
    summary="提问（stream=true 走 SSE，false 返回 JSON）",
    responses={
        200: {
            "description": "stream=true 时为 text/event-stream；false 时为 ChatResponse JSON",
            "content": {
                "application/json": {"schema": {"$ref": "#/components/schemas/ChatResponse"}},
                "text/event-stream": {
                    "schema": {"type": "string"},
                    "example": (
                        "event: start\ndata: {\"conversation_id\": \"...\"}\n\n"
                        "event: token\ndata: {\"t\": \"根据\"}\n\n"
                        "event: done\ndata: {\"answer\": \"...\", \"citations\": []}\n\n"
                    ),
                },
            },
        },
        429: {"description": "触发限流"},
        502: {"description": "Agent 输出未通过模式校验"},
    },
)
async def chat(
    payload: ChatRequest,
    current: User = Depends(get_current_user),
) -> Any:
    await enforce_rate_limit(f"chat:{current.id}")

    if payload.stream:
        generator = chat_service.stream(
            user_id=current.id,
            message=payload.message,
            conversation_id=payload.conversation_id,
            top_k=payload.top_k,
        )
        return StreamingResponse(
            generator,
            media_type="text/event-stream",
            headers=_SSE_HEADERS,
        )

    return await chat_service.ask(
        user_id=current.id,
        message=payload.message,
        conversation_id=payload.conversation_id,
        top_k=payload.top_k,
    )


@router.get("/usage", summary="当前用户的 token 用量与成本汇总", response_model=dict)
async def usage(
    current: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
) -> dict:
    summary = await summarize_usage(session, current.id)
    return {
        "calls": summary.calls,
        "prompt_tokens": summary.prompt_tokens,
        "completion_tokens": summary.completion_tokens,
        "total_tokens": summary.total_tokens,
        "cost_usd": summary.cost_usd,
    }


@router.get("/history/{conversation_id}", summary="回看某个会话的消息", response_model=list[dict])
async def history(
    conversation_id: str,
    current: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
) -> list[dict]:
    from sqlalchemy import select

    from app.db.models import ChatMessage

    rows = (
        await session.execute(
            select(ChatMessage)
            .where(
                ChatMessage.user_id == current.id,
                ChatMessage.conversation_id == conversation_id,
            )
            .order_by(ChatMessage.created_at)
        )
    ).scalars()
    return [
        {
            "id": m.id,
            "role": m.role,
            "content": m.content,
            "citations": m.citations or [],
            "created_at": m.created_at.isoformat(),
        }
        for m in rows
    ]
