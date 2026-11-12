"""答案忠实度评估：用 LLM 逐条核查答案是否被 citations 支持。

跑法：
    python -m scripts.evaluate_faithfulness                 # 全量 41 题
    python -m scripts.evaluate_faithfulness --limit 5       # 先跑 5 题试水
    python -m scripts.evaluate_faithfulness --json          # 给 CI / 汇总脚本消费
    python -m scripts.evaluate_faithfulness --rows          # 逐题打印
    python -m scripts.evaluate_faithfulness --min-score 0.9 # 低于阈值非零退出

和检索评估的分工：
- 检索评估（scripts.evaluate）只看「找到的段落对不对」，不调生成；
- 这里跑完整链路（LangGraph → 真实模型 → 出口校验），再用裁判模型核查答案。

生成与判定走的是同一条出口：`validate_answer()`。也就是说被评估的答案，
就是用户实际会拿到的那一份（引用已按 chunk_id 白名单过滤过）。

前提：需要真实模型（LLM_PROVIDER=openai）。离线确定性模型只是把片段首句拼起来，
没有「推理」可评估，用它算忠实度只会得到一个恒等于 1 的假数字。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
import unicodedata
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.eval_set import EVAL_SET  # noqa: E402
from scripts.faithfulness import (  # noqa: E402
    FaithfulnessVerdict,
    aggregate,
    build_judge_messages,
    parse_verdict,
)

DEFAULT_CONCURRENCY = 4
JUDGE_MAX_TOKENS = 1200

# 对照实验用：一句与引用无关、但听起来很专业的话。
# 裁判要是连它都判"支持"，那满分只说明裁判睡着了。
INJECTED_CLAIM = "此外，该模块默认开启多租户隔离，并在启动时自动向 Prometheus 推送指标。"


async def _bootstrap() -> None:
    from app.core.cache import init_cache
    from app.db.session import get_session_factory, init_models
    from app.rag.embeddings import warmup_embedder
    from app.rag.ingest import ensure_index_ready

    await init_cache()
    await init_models()
    await warmup_embedder()
    factory = get_session_factory()
    async with factory() as session:
        await ensure_index_ready(session, seed_demo=True)


def _build_judge() -> Any:
    """裁判模型：温度 0、不流式、不给重试（重试由 call_with_resilience 统一管）。"""
    from langchain_openai import ChatOpenAI

    from app.core.config import get_settings

    cfg = get_settings()
    return ChatOpenAI(
        model=cfg.llm_model,
        api_key=cfg.llm_api_key,
        base_url=cfg.llm_base_url,
        temperature=0.0,
        max_tokens=JUDGE_MAX_TOKENS,
        max_retries=0,
        timeout=cfg.llm_timeout_seconds,
    )


async def _generate(question: str) -> tuple[str, list[dict[str, Any]], int]:
    """跑完整图 + 出口校验，返回 (答案, 引用列表, 引用条数)。不落库。"""
    import uuid

    from app.agents.graph import get_graph
    from app.agents.state import new_state
    from app.services.chat_service import validate_answer

    started = time.perf_counter()
    state = new_state(
        question=question,
        messages=[],
        conversation_id=f"eval-{uuid.uuid4().hex}",
        user_id="eval",
    )
    result = await get_graph().ainvoke(state)
    answer = validate_answer(result, latency_ms=int((time.perf_counter() - started) * 1000))
    return answer.answer, [c.model_dump() for c in answer.citations], len(answer.citations)


async def _judge(
    judge: Any, question: str, answer: str, citations: list[dict[str, Any]]
) -> FaithfulnessVerdict:
    from langchain_core.messages import HumanMessage, SystemMessage

    from app.core.resilience import call_with_resilience

    messages = build_judge_messages(question, answer, citations)
    typed = [
        SystemMessage(content=messages[0]["content"]),
        HumanMessage(content=messages[1]["content"]),
    ]

    async def _run() -> str:
        response = await judge.ainvoke(typed)
        content = response.content
        return content if isinstance(content, str) else str(content)

    try:
        raw = await call_with_resilience(_run, label="eval.judge", timeout=90.0, retries=1)
    except Exception as exc:  # noqa: BLE001 - 判定失败要如实记录，不能当成通过
        return FaithfulnessVerdict(error=f"{type(exc).__name__}: {exc}")
    return parse_verdict(raw)


async def calibration_check(judge: Any | None = None) -> dict[str, Any]:
    """对照实验：往一份真实答案里塞一句编造，看裁判会不会扣分。

    忠实度接近 1.00 时，第一件该怀疑的事是「裁判根本没在判」。
    这里用同一批引用、同一道题做 A/B：注入一句与引用无关、但语气专业的断言，
    如果分数不降，说明这套判定只能当装饰，不能写进报告。
    """
    judge = judge or _build_judge()
    case = EVAL_SET[0]
    answer, citations, _ = await _generate(case.question)

    base = await _judge(judge, case.question, answer, citations)
    poisoned_answer = answer.rstrip() + "\n" + INJECTED_CLAIM
    poisoned = await _judge(judge, case.question, poisoned_answer, citations)

    base_score = base.score
    poisoned_score = poisoned.score
    return {
        "question": case.question,
        "base_score": base_score,
        "poisoned_score": poisoned_score,
        "detected": (
            base_score is not None
            and poisoned_score is not None
            and poisoned_score < base_score
        ),
        "unsupported": [c.text for c in poisoned.claims if not c.supported],
    }


async def _evaluate_one(
    judge: Any,
    case: Any,
    semaphore: asyncio.Semaphore,
    show_claims: bool,
) -> tuple[dict[str, Any], FaithfulnessVerdict]:
    async with semaphore:
        started = time.perf_counter()
        try:
            answer, citations, cited = await _generate(case.question)
        except Exception as exc:  # noqa: BLE001
            message = f"generate_failed: {type(exc).__name__}: {exc}"
            return (
                {
                    "question": case.question,
                    "topic": case.topic,
                    "source": case.source,
                    "citations": 0,
                    "answer_chars": 0,
                    "claims": 0,
                    "supported": 0,
                    "score": None,
                    "error": message,
                    "latency_ms": int((time.perf_counter() - started) * 1000),
                    "unsupported": [],
                },
                FaithfulnessVerdict(error=message),
            )

        verdict = await _judge(judge, case.question, answer, citations)
        if show_claims:
            print(f"\n── {case.question}")
            print(f"   答案: {answer.strip().replace(chr(10), ' ')[:160]}")
            for claim in verdict.claims:
                flag = "✅" if claim.supported else "❌"
                print(f"     {flag} [{claim.support}] {claim.text[:70]}  {claim.reason}")
            if verdict.error:
                print(f"     ⚠ 判定失败: {verdict.error}")

        return (
            {
                "question": case.question,
                "topic": case.topic,
                "source": case.source,
                "citations": cited,
                "answer_chars": len(answer),
                "claims": verdict.total,
                "supported": verdict.supported,
                "score": None if verdict.score is None else round(verdict.score, 4),
                "error": verdict.error,
                "latency_ms": int((time.perf_counter() - started) * 1000),
                "unsupported": [c.text[:80] for c in verdict.claims if not c.supported][:5],
            },
            verdict,
        )


async def evaluate_faithfulness(
    *,
    limit: int | None = None,
    concurrency: int = DEFAULT_CONCURRENCY,
    show_claims: bool = False,
    calibration: bool = False,
) -> dict[str, Any]:
    from app.core.config import get_settings
    from app.rag.vectorstore import get_vector_store

    cfg = get_settings()
    cases = EVAL_SET[:limit] if limit else list(EVAL_SET)
    judge = _build_judge()
    semaphore = asyncio.Semaphore(max(1, concurrency))

    started = time.perf_counter()
    pairs = await asyncio.gather(
        *(_evaluate_one(judge, case, semaphore, show_claims) for case in cases)
    )
    elapsed_ms = int((time.perf_counter() - started) * 1000)

    rows = [row for row, _ in pairs]
    summary = aggregate([verdict for _, verdict in pairs])
    summary["avg_citations"] = (
        round(sum(int(r["citations"]) for r in rows) / len(rows), 2) if rows else 0.0
    )

    try:
        chunks = int(await get_vector_store().count())
    except Exception:  # noqa: BLE001 - 汇总用，取不到不影响主流程
        chunks = 0

    report: dict[str, Any] = {
        "corpus": {
            "chunks": chunks,
            "model": cfg.llm_model,
            "judge_model": cfg.llm_model,
        },
        "questions": len(rows),
        "concurrency": concurrency,
        "elapsed_ms": elapsed_ms,
        "summary": summary,
        "rows": rows,
    }
    if calibration:
        report["calibration"] = await calibration_check(judge)
    return report



def _display_width(text: str) -> int:
    return sum(2 if unicodedata.east_asian_width(ch) in {"W", "F"} else 1 for ch in text)


def _fit(text: str, width: int) -> str:
    if _display_width(text) <= width:
        return text + " " * (width - _display_width(text))
    out, used = "", 0
    for ch in text:
        step = 2 if unicodedata.east_asian_width(ch) in {"W", "F"} else 1
        if used + step > width - 1:
            break
        out += ch
        used += step
    return out + "…" + " " * max(0, width - used - 1)


def print_report(report: dict[str, Any], *, rows: bool = False) -> None:
    summary = report["summary"]
    print("═" * 78)
    print("  答案忠实度评估（LLM 裁判 · claim 级）")
    print("═" * 78)
    print(f"  语料      : {report['corpus']['chunks']} 个切片")
    print(f"  生成模型  : {report['corpus']['model']}")
    print(f"  裁判模型  : {report['corpus']['judge_model']} (temperature=0)")
    print(f"  题量      : {report['questions']}   并发: {report['concurrency']}")
    print("─" * 78)
    print(f"  参与判定  : {summary['scored_questions']} 题   弃答: {summary['abstained']}   "
          f"判定失败: {summary['judge_failed']}   平均引用: {summary['avg_citations']}")
    print(f"  原子断言  : {summary['claims']} 条  被支持 {summary['supported_claims']}  "
          f"不被支持 {summary['unsupported_claims']}")
    print("─" * 78)
    print(f"  题级平均忠实度 : {summary['score_avg']:.4f}   "
          f"满分题 {summary['perfect_questions']}/{summary['scored_questions']}")
    print(f"  微平均(按断言) : {summary['score_micro']:.4f}")

    calib = report.get("calibration")
    if calib:
        base = calib["base_score"]
        poison = calib["poisoned_score"]
        mark = "✅ 检出" if calib["detected"] else "✗ 未检出"
        print("─" * 78)
        print(
            "  裁判自检 : 原答案 "
            + ("弃答" if base is None else f"{base:.2f}")
            + " → 注入一句编造后 "
            + ("弃答" if poison is None else f"{poison:.2f}")
            + f"   {mark}"
        )
        for claim in calib["unsupported"][:2]:
            print(f"     ❌ {claim}")
    print("═" * 78)

    imperfect = [r for r in report["rows"] if r["score"] is not None and r["score"] < 1.0]
    if rows:
        for index, row in enumerate(report["rows"], start=1):
            score = "—" if row["score"] is None else f"{row['score']:.2f}"
            print(
                "  " + _fit(str(index), 4) + _fit(row["question"], 34) + " "
                + _fit(score, 6) + f"{row['supported']}/{row['claims']} 条  "
                + f"{row['citations']} 引用"
            )
        print("═" * 78)
    elif imperfect:
        print(f"  未完全被引用支持的题（{len(imperfect)} 道）：")
        for row in imperfect:
            print(f"   · {row['question']}")
            for claim in row["unsupported"]:
                print(f"     ❌ {claim}")
        print("═" * 78)
    print()


async def main(argv: list[str] | None = None) -> int:
    from app.agents.llm import is_real_llm

    parser = argparse.ArgumentParser(description="答案忠实度评估（LLM 裁判）")
    parser.add_argument("--limit", type=int, default=None, help="只跑前 N 题")
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY, help="并发题数")
    parser.add_argument("--json", action="store_true", help="只输出 JSON")
    parser.add_argument("--rows", action="store_true", help="逐题打印得分")
    parser.add_argument("--show-claims", action="store_true", help="打印每题的逐条断言判定")
    parser.add_argument(
        "--calibration", action="store_true", help="跑一次注入式对照实验，验证裁判真能扣分"
    )
    parser.add_argument(
        "--min-score", type=float, default=None, metavar="S",
        help="题级平均忠实度低于阈值时以非零码退出，便于接 CI",
    )
    args = parser.parse_args(argv)

    if not is_real_llm():
        print(
            "忠实度评估需要真实模型：请把 LLM_PROVIDER 设为 openai 并配置 OPENAI_API_KEY。\n"
            "离线确定性模型只是拼接片段首句，评估它得到的数字没有意义。",
            file=sys.stderr,
        )
        return 2

    await _bootstrap()
    report = await evaluate_faithfulness(
        limit=args.limit,
        concurrency=args.concurrency,
        show_claims=args.show_claims,
        calibration=args.calibration,
    )

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print_report(report, rows=args.rows)
        print(f"  耗时 {report['elapsed_ms'] / 1000:.1f}s")

    if args.min_score is not None and report["summary"]["score_avg"] < args.min_score:
        print(
            f"  ✗ 忠实度 {report['summary']['score_avg']:.4f} 低于阈值 {args.min_score:.2f}",
            file=sys.stderr,
        )
        return 1
    calib = report.get("calibration")
    if calib and not calib["detected"]:
        print("  ✗ 裁判自检未通过：注入编造后分数没有下降，判定结果不可信", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
