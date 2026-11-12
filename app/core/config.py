"""应用配置：全部通过环境变量注入，便于 Docker / 本地一键切换。"""

from __future__ import annotations

import os

# huggingface_hub 在 import 时就把 HF_ENDPOINT 读成常量，晚于 import 设置无效。
# 本模块是所有入口（app.main / scripts/* / tests/*）最早被 import 的模块之一，
# 因此在这里兜底设置镜像源；用 setdefault 保证用户显式设置的值优先。
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from functools import lru_cache  # noqa: E402
from typing import Literal  # noqa: E402

from pydantic import Field  # noqa: E402
from pydantic_settings import BaseSettings, SettingsConfigDict  # noqa: E402


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
    # 基础集合名；fastembed 会按 provider+model+dim 派生带后缀的独立集合，
    # 保证旧 hashing 索引（384 维）不被破坏
    chroma_collection: str = "kb_documents"

    # ---------- Embedding ----------
    # hashing：零依赖确定性向量（CI / 离线兜底）
    # fastembed：ONNX 真实语义模型（复用 onnxruntime，不需要 torch）
    # sentence-transformers：保留兼容，需要 torch，默认不装
    embedding_provider: Literal["hashing", "fastembed", "sentence-transformers"] = "hashing"
    embedding_model: str = "BAAI/bge-small-zh-v1.5"
    embedding_dim: int = 384  # 仅 hashing 生效；fastembed 以模型实际维度为准
    embedding_batch_size: int = 32
    embedding_threads: int | None = None  # None = 交给 onnxruntime 自决
    # e5 系列必须加前缀；bge-zh-v1.5 官方说明"无需前缀"，默认留空
    embedding_query_prefix: str = ""
    embedding_passage_prefix: str = ""
    # 缺包 / 模型不可用时降级为 hashing，而不是让整个服务起不来
    embedding_fallback_to_hashing: bool = True

    # ---------- 模型下载（国内镜像） ----------
    hf_endpoint: str = "https://hf-mirror.com"
    hf_hub_disable_xet: bool = True  # 镜像站不支持 XET CAS，不关会先失败再回退
    fastembed_cache_dir: str = "./data/models"

    # ---------- 检索 / 重排 ----------
    rag_top_k: int = 8
    rag_rerank_top_n: int = 3
    # bge-small-zh 上下文 512 token，中文约 1 字 1 token，块长必须留余量，
    # 否则超长块会被 tokenizer 静默截断尾部
    rag_chunk_size: int = 400
    rag_chunk_overlap: int = 60
    reranker: Literal["lexical", "cross-encoder"] = "lexical"

    # RAG 链路独立超时：全局 tool_timeout 约束的是 Chroma/DB，
    # 首次加载模型要下载 90MB，不能套用 10s
    rag_retrieve_timeout_seconds: float = 15.0
    rag_embed_timeout_seconds: float = 20.0
    rag_model_load_timeout_seconds: float = 600.0

    # 首次启动灌入 4 篇演示语料（让新 clone 的仓库零配置就能问出东西）。
    # 只想看到自己 docs/ 里的内容时设为 false，索引一致性检查仍然会跑。
    seed_demo_docs: bool = True

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
