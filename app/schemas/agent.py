"""Agent 输出的模式定义 —— 这是返回给用户前的最后一道闸门。

约束：所有 Agent 输出必须经过 Pydantic 模式验证后才能返回用户。
因此这里不只是"类型声明"，还承担语义校验（空答案、非法 SQL、悬空引用等）。
"""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

Route = Literal["rag", "sql", "direct"]

# 只允许只读查询：任何写操作 / DDL / 权限语句一律拒绝
_FORBIDDEN_SQL = re.compile(
    r"\b(insert|update|delete|drop|alter|truncate|create|grant|revoke|copy|merge|call|vacuum)\b",
    re.IGNORECASE,
)


class Citation(BaseModel):
    """一条可追溯的证据。chunk_id 用于回查向量库。"""

    chunk_id: str
    document_id: str = ""
    title: str
    source: str = "kb"
    snippet: str = Field(max_length=800)
    score: float = Field(default=0.0, ge=0.0, le=1.0)


class Usage(BaseModel):
    model: str = "offline-deterministic"
    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)
    total_tokens: int = Field(default=0, ge=0)
    cost_usd: float = Field(default=0.0, ge=0.0)
    latency_ms: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def _fill_total(self) -> "Usage":
        if not self.total_tokens:
            self.total_tokens = self.prompt_tokens + self.completion_tokens
        return self


class SQLPlan(BaseModel):
    """NL → SQL 节点的结构化产物。"""

    sql: str = Field(min_length=1, max_length=4000)
    rationale: str = ""
    tables: list[str] = Field(default_factory=list)

    @field_validator("sql")
    @classmethod
    def _read_only(cls, v: str) -> str:
        cleaned = v.strip().rstrip(";").strip()
        if not cleaned:
            raise ValueError("SQL 为空")
        if _FORBIDDEN_SQL.search(cleaned):
            raise ValueError("仅允许只读 SELECT 查询")
        head = cleaned.split(None, 1)[0].lower()
        if head not in {"select", "with"}:
            raise ValueError("语句必须以 SELECT 或 WITH 开头")
        return cleaned


class AgentAnswer(BaseModel):
    """最终交付给用户的答案。任何节点产出的 dict 都必须先过这里。"""

    answer: str = Field(min_length=1, max_length=8000)
    route: Route = "direct"
    citations: list[Citation] = Field(default_factory=list)
    sql: str | None = None
    rows: list[dict[str, Any]] | None = None
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    needs_clarification: bool = False
    truncated: bool = False
    usage: Usage = Field(default_factory=Usage)

    @field_validator("answer")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("答案不能为空")
        return v.strip()

    @field_validator("sql")
    @classmethod
    def _sql_read_only(cls, v: str | None) -> str | None:
        if v is None:
            return None
        if _FORBIDDEN_SQL.search(v):
            raise ValueError("答案携带的 SQL 非只读语句")
        return v

    @model_validator(mode="after")
    def _consistency(self) -> "AgentAnswer":
        # 走 SQL 路线就必须带回可复核的 SQL 文本
        if self.route == "sql" and not self.sql:
            raise ValueError("route=sql 但缺少 sql 字段")
        # 引用去重（按 chunk_id），避免同一片段重复计分
        seen: set[str] = set()
        deduped: list[Citation] = []
        for c in self.citations:
            if c.chunk_id in seen:
                continue
            seen.add(c.chunk_id)
            deduped.append(c)
        self.citations = deduped
        return self
