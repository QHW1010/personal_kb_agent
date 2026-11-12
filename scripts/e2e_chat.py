"""端到端真实问答测试：FastAPI → 权限 → LangGraph → DeepSeek → SSE。

跑的是完整链路，不是单元测试：
  HTTP/ASGI 路由 → JWT 鉴权 → chat_service → LangGraph StateGraph
  → intent_router（真实模型分类）→ retrieve（Chroma 检索）
  → synthesize（真实模型流式生成）→ SSE → 客户端逐帧解析

用法：
    python -m scripts.e2e_chat
    python -m scripts.e2e_chat "SQL 提示词为什么不能写 LIMIT"
    python -m scripts.e2e_chat --base-url http://127.0.0.1:8000   # 打真实服务
    python -m scripts.e2e_chat --base-url http://127.0.0.1:8000 --trace 12
    python -m scripts.e2e_chat --json                             # 只输出结构化结果

关于两种入口：
- 默认走 httpx 的 ASGITransport，不需要起端口，链路与真实服务完全相同（路由、依赖、
  SSE 生成器都跑），但**它会把整个响应缓冲下来**再交给你，所以逐帧时间戳没有意义。
- 想验证"token 真的一点点到达"，用 --base-url 打一个真实 uvicorn 进程，配 --trace 看时刻。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
import uuid
from typing import Any

import httpx

DEFAULT_QUESTION = "根据文档，这个项目为什么禁止在 SQL 提示词里硬编码 LIMIT？"

_GREEN = "\033[32m"
_CYAN = "\033[36m"
_YELLOW = "\033[33m"
_DIM = "\033[2m"
_BOLD = "\033[1m"
_RESET = "\033[0m"


def _supports_color() -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    return sys.stdout.isatty()


class Painter:
    def __init__(self) -> None:
        self.on = _supports_color()

    def __call__(self, text: str, color: str) -> str:
        return f"{color}{text}{_RESET}" if self.on else text


paint = Painter()


def parse_sse_block(block: str) -> tuple[str, dict[str, Any]] | None:
    """把一段 SSE 文本（event + data）解析成 (事件名, payload)。"""
    event = ""
    data_lines: list[str] = []
    for line in block.splitlines():
        if line.startswith("event:"):
            event = line[len("event:") :].strip()
        elif line.startswith("data:"):
            data_lines.append(line[len("data:") :].lstrip())
    if not event:
        return None
    raw = "\n".join(data_lines)
    try:
        payload = json.loads(raw) if raw else {}
    except json.JSONDecodeError:
        payload = {"_raw": raw}
    return event, payload


async def bootstrap_in_process() -> None:
    """ASGITransport 不跑 lifespan，启动期初始化要自己补上。"""
    from app.agents.llm import warmup_llm
    from app.core.cache import init_cache
    from app.db.session import get_session_factory, init_models
    from app.rag.embeddings import warmup_embedder
    from app.rag.ingest import ensure_index_ready

    await init_cache()
    await init_models()
    await warmup_embedder()
    # 与真实服务一致：先建好连接，否则首个问题的 TTFT 会把建连时间也算进去
    await warmup_llm()
    factory = get_session_factory()
    async with factory() as session:
        await ensure_index_ready(session, seed_demo=True)


async def ensure_account(client: httpx.AsyncClient, email: str, password: str) -> str:
    resp = await client.post("/auth/register", json={"email": email, "password": password})
    if resp.status_code not in (200, 201):
        resp = await client.post("/auth/login", json={"email": email, "password": password})
    resp.raise_for_status()
    return resp.json()["access_token"]


async def run(
    question: str,
    *,
    base_url: str | None,
    json_only: bool,
    email: str | None,
    password: str,
    trace: int = 0,
) -> int:
    from app.agents.llm import describe_llm

    llm_info = describe_llm()

    if base_url:
        transport: httpx.AsyncBaseTransport | None = None
        api_root = base_url.rstrip("/") + "/api/v1"
        endpoint = base_url
    else:
        from httpx import ASGITransport

        from app.main import app as fastapi_app

        await bootstrap_in_process()
        transport = ASGITransport(app=fastapi_app)
        api_root = "http://testserver/api/v1"
        endpoint = "in-process ASGI"

    email = email or f"e2e-{uuid.uuid4().hex[:8]}@example.com"

    if not json_only:
        print(paint("═" * 72, _DIM))
        print(paint("  端到端真实问答（LangGraph + DeepSeek + SSE）", _BOLD))
        print(paint("═" * 72, _DIM))
        print(f"  入口     : {endpoint}")
        print(f"  模型     : {llm_info['model']}  @ {llm_info['base_url'] or 'default'}")
        print(f"  Provider : {llm_info['provider']}  (api_key={'已配置' if llm_info['api_key_configured'] else '缺失'})")
        print(f"  问题     : {question}")
        if not base_url:
            print(
                paint(
                    "  注意     : in-process ASGI 会缓冲整个响应，逐帧时间戳无意义；"
                    "要看真实流式请加 --base-url",
                    _DIM,
                )
            )
        print(paint("─" * 72, _DIM))

    async with httpx.AsyncClient(
        transport=transport, base_url=api_root, timeout=180.0
    ) as client:
        token = await ensure_account(client, email, password)
        headers = {"Authorization": f"Bearer {token}"}

        route = ""
        citations: list[dict[str, Any]] = []
        usage: dict[str, Any] = {}
        conversation_id = ""
        answer_parts: list[str] = []
        chunks = 0
        started = time.perf_counter()
        first_token_ms: float | None = None
        error: dict[str, Any] | None = None

        async with client.stream(
            "POST",
            "/chat",
            headers=headers,
            json={"message": question, "stream": True},
        ) as resp:
            if resp.status_code != 200:
                body = (await resp.aread()).decode("utf-8", errors="replace")
                print(paint(f"HTTP {resp.status_code}: {body[:400]}", _YELLOW))
                return 1

            buffer = ""
            async for text in resp.aiter_text():
                buffer += text
                while "\n\n" in buffer:
                    block, buffer = buffer.split("\n\n", 1)
                    parsed = parse_sse_block(block)
                    if parsed is None:
                        continue
                    event, payload = parsed

                    if event == "start":
                        conversation_id = payload.get("conversation_id", "")
                        if not json_only:
                            print(
                                paint("[start] ", _CYAN)
                                + f"conversation={conversation_id[:8]}… model={payload.get('model')}"
                            )
                    elif event == "route":
                        route = payload.get("route", "")
                        if not json_only:
                            print(
                                paint("[route] ", _CYAN)
                                + f"{paint(route, _BOLD)}  "
                                + paint(
                                    f"来源={payload.get('source')} 理由={payload.get('reason') or '-'}",
                                    _DIM,
                                )
                            )
                    elif event == "citations":
                        citations = payload.get("citations") or []
                        if not json_only:
                            print(paint(f"[citations] {len(citations)} 条", _CYAN))
                            for idx, cit in enumerate(citations, start=1):
                                loc = f"para#{cit.get('paragraph_index')}"
                                if cit.get("char_start") is not None:
                                    loc += f" chars[{cit['char_start']},{cit['char_end']})"
                                print(
                                    paint(
                                        f"   {idx}. {cit.get('source')} · {loc}"
                                        f" · score={cit.get('score')}",
                                        _DIM,
                                    )
                                )
                            print(paint("─" * 72, _DIM))
                            print(paint("[生成中] ", _CYAN), end="", flush=True)
                    elif event == "token":
                        piece = payload.get("t", "")
                        if piece:
                            if first_token_ms is None:
                                first_token_ms = (time.perf_counter() - started) * 1000
                            chunks += 1
                            answer_parts.append(piece)
                            if not json_only and trace and chunks <= trace:
                                # 逐帧时间戳：证明 token 是陆续到达的，不是最后一次性吐出
                                offset = (time.perf_counter() - started) * 1000
                                print(
                                    paint(f"[+{offset:>7.1f}ms] ", _DIM) + repr(piece),
                                    flush=True,
                                )
                            elif not json_only:
                                print(piece, end="", flush=True)
                    elif event == "done":
                        citations = payload.get("citations") or citations
                        usage = payload.get("usage") or {}
                        route = payload.get("route") or route
                        if not json_only:
                            print()
                    elif event == "error":
                        error = payload

        elapsed_ms = (time.perf_counter() - started) * 1000
        answer = "".join(answer_parts)

        if json_only:
            print(
                json.dumps(
                    {
                        "question": question,
                        "route": route,
                        "answer": answer,
                        "citations": citations,
                        "usage": usage,
                        "chunks": chunks,
                        "ttft_ms": round(first_token_ms, 1) if first_token_ms else None,
                        "elapsed_ms": round(elapsed_ms, 1),
                        "error": error,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
        else:
            print(paint("─" * 72, _DIM))
            if error:
                print(paint(f"[error] {error}", _YELLOW))
            print(
                paint("[done] ", _CYAN)
                + f"route={route}  citations={len(citations)}  "
                + f"tokens={usage.get('total_tokens')} "
                + f"(prompt {usage.get('prompt_tokens')} / completion {usage.get('completion_tokens')})  "
                + f"cost=${usage.get('cost_usd')}"
            )
            if first_token_ms is None:
                timing = f"总耗时 {elapsed_ms:.0f} ms"
            else:
                timing = (
                    f"首字节 {first_token_ms:.0f} ms · 总耗时 {elapsed_ms:.0f} ms · "
                    f"{chunks} 个 token 帧"
                )
            print(paint("[timing] ", _CYAN) + timing)

    # ---- 验收断言：链路真的通了，不是"看起来像" ----
    problems: list[str] = []
    if error:
        problems.append(f"流中收到 error 事件: {error}")
    if not answer.strip():
        problems.append("没有收到任何正文 token")
    if route == "rag" and not citations:
        problems.append("route=rag 但 citations 为空")
    if chunks < 5:
        problems.append(f"token 帧只有 {chunks} 个，怀疑没有真正流式")
    if llm_info["provider"] == "openai" and not (usage.get("total_tokens") or 0) > 0:
        problems.append("真实模型未返回 usage，token 台账会是估算值")

    if not json_only:
        if problems:
            print(paint("\n验收失败：", _YELLOW))
            for item in problems:
                print(f"  - {item}")
        else:
            print(paint("\n验收通过：意图路由、检索、真实生成、SSE 流式、引用元数据全部到位", _GREEN))
    return 1 if problems else 0


def main() -> None:
    parser = argparse.ArgumentParser(description="端到端真实问答测试（SSE）")
    parser.add_argument("question", nargs="?", default=DEFAULT_QUESTION, help="要问的问题")
    parser.add_argument("--base-url", default=None, help="打已启动的服务，如 http://127.0.0.1:8000")
    parser.add_argument("--json", action="store_true", help="只输出结构化结果")
    parser.add_argument("--email", default=None, help="复用已有账号")
    parser.add_argument("--password", default="sup3rsecret!", help="账号密码")
    parser.add_argument(
        "--trace", type=int, default=0, metavar="N", help="打印前 N 个 token 的到达时刻"
    )
    args = parser.parse_args()

    code = asyncio.run(
        run(
            args.question,
            base_url=args.base_url,
            json_only=args.json,
            email=args.email,
            password=args.password,
            trace=args.trace,
        )
    )
    raise SystemExit(code)


if __name__ == "__main__":
    main()
