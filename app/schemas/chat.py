"""对话与知识库 API 模式。"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from app.schemas.agent import Citation, Usage


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000, description="用户问题")
    conversation_id: str | None = Field(default=None, description="为空则新建会话")
    stream: bool = Field(default=True, description="true 返回 SSE 流，false 返回一次性 JSON")
    top_k: int | None = Field(default=None, ge=1, le=20, description="覆盖默认检索条数")


class ChatResponse(BaseModel):
    conversation_id: str
    message_id: str
    answer: str
    route: str
    citations: list[Citation] = Field(default_factory=list)
    sql: str | None = None
    rows: list[dict[str, Any]] | None = None
    confidence: float = 0.5
    needs_clarification: bool = False
    usage: Usage


class KnowledgeIngestRequest(BaseModel):
    title: str = Field(min_length=1, max_length=200)
    content: str = Field(min_length=1, max_length=200_000)
    source: str = Field(default="manual", max_length=200)


class KnowledgeIngestResponse(BaseModel):
    document_id: str
    title: str
    chunks: int
    backend: str


class KnowledgeSearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=1000)
    top_k: int = Field(default=5, ge=1, le=20)


class KnowledgeSearchResponse(BaseModel):
    query: str
    results: list[Citation]


class KnowledgeStats(BaseModel):
    documents: int
    chunks: int
    backend: str
    embedding_provider: str
    reranker: str
