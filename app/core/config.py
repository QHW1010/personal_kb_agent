"""应用配置：全部通过环境变量注入，便于 Docker / 本地一键切换。"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """集中配置。命名与 .env.example 严格一致。"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ---------- 基础 ----------
    app_name: str = "Personal KB Agent"
    app_version: str = "0.1.0"
    env: Literal["dev", "test", "prod"] = "dev"
    debug: bool = False
    api_v1_prefix: str = "/api/v1"
    log_level: str = "INFO"

    # ---------- 安全 / JWT ----------
    secret_key: str = "change-me-in-production-please-use-32-bytes-min"
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 60 * 24

    # ---------- 数据库 ----------
    # 本地默认走 SQLite，Docker Compose 下由环境变量覆盖为 PostgreSQL
    database_url: str = "sqlite+aiosqlite:///./data/app.db"
    db_echo: bool = False
    auto_create_tables: bool = True  # 生产环境请关闭，改用 alembic upgrade head

    # ---------- Redis（可选，不可用时自动降级为进程内存） ----------
    redis_url: str | None = None
    cache_required: bool = False

    # ---------- 向量库 ----------
    vector_backend: Literal["auto", "chroma", "memory"] = "auto"
    chroma_dir: str = "./data/chroma"
    chroma_collection: str = "kb_documents"
    embedding_provider: Literal["hashing", "sentence-transformers"] = "hashing"
    embedding_dim: int = 384

    # ---------- 检索 / 重排 ----------
    rag_top_k: int = 8
    rag_rerank_top_n: int = 3
    rag_chunk_size: int = 600
    rag_chunk_overlap: int = 80
    reranker: Literal["lexical", "cross-encoder"] = "lexical"

    # ---------- LLM ----------
    # fake = 离线确定性模型（无 Key 也能跑通全链路、可跑测试）
    llm_provider: Literal["fake", "openai"] = "fake"
    llm_model: str = "gpt-4o-mini"
    llm_base_url: str | None = None
    llm_api_key: str | None = None
    llm_temperature: float = 0.2
    llm_max_tokens: int = 1024

    # ---------- 成本核算（USD / 1K tokens） ----------
    price_prompt_per_1k: float = 0.00015
    price_completion_per_1k: float = 0.0006

    # ---------- 上下文窗口控制 ----------
    history_max_turns: int = Field(default=6, ge=0, le=50)

    # ---------- 工具调用韧性（约束：禁止无限等待外部 I/O） ----------
    tool_timeout_seconds: float = 10.0
    tool_max_retries: int = 2
    tool_backoff_seconds: float = 0.3

    # ---------- 限流 ----------
    rate_limit_per_minute: int = 30

    # ---------- 安全执行 SQL 的行数上限（防 DoS，非答案截断） ----------
    sql_max_rows: int = 200

    # 逗号分隔的字符串：pydantic-settings 对 list[str] 会先做 JSON 解码，
    # 用字符串字段 + property 拆分，兼容 .env 里写 http://a,http://b 的习惯
    cors_origins: str = "http://localhost:3000,http://localhost:8000"

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def is_sqlite(self) -> bool:
        return self.database_url.startswith("sqlite")


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
