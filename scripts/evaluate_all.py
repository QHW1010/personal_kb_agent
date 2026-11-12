"""双维度评估：检索（Recall@K / MRR）+ 忠实度（LLM 裁判），一张报告出结论。

跑法：
    python -m scripts.evaluate_all                    # 两个维度全跑（需要真实模型）
    python -m scripts.evaluate_all --limit 10         # 先跑 10 题
    python -m scripts.evaluate_all --retrieval-only   # 没 Key 也能跑，只看检索
    python -m scripts.evaluate_all --json > eval.json # 机器可读
    python -m scripts.evaluate_all --min-recall 0.8 --min-score 0.9   # CI 门禁

为什么必须放一起看：
检索命中但答案编造，是比「没检索到」更危险的状态 —— 用户看到引用齐全，
不会怀疑结论。单看任一维度都会把这类题算成「还行」，只有交叉表能把它挑出来。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.evaluate import _bootstrap, _fit, evaluate, topic_stats  # noqa: E402
from scripts.evaluate_faithfulness import evaluate_faithfulness  # noqa: E402

DEFAULT_DEPTH = 5


def _cross(rows: list[dict[str, Any]], prod_ranks: list[int]) -> dict[str, int]:
    """检索与忠实度的联合分布。

    `rows` 与 `prod_ranks` 按 EVAL_SET 顺序一一对应（两个评估都按同一顺序跑）。
    """
    buckets = {
        "both_ok": 0,          # 检索命中 + 答案完全被引用支持
        "cited_but_unsupported": 0,  # 检索命中，但答案里有不被引用支持的断言（最危险）
        "miss_but_ok": 0,      # 检索没命中，答案却全被支持（可能是别的片段兜住了）
        "both_bad": 0,         # 检索没命中 + 答案不被支持
        "abstained": 0,        # 拒答：既不算幻觉也不算答对，单独记一档
    }
    for row, rank in zip(rows, prod_ranks):
        hit = 0 < rank <= 3
        score = row.get("score")
        if score is None:
            buckets["abstained"] += 1
            continue
        if hit and score == 1.0:
            buckets["both_ok"] += 1
        elif hit:
            buckets["cited_but_unsupported"] += 1
        elif score == 1.0:
            buckets["miss_but_ok"] += 1
        else:
            buckets["both_bad"] += 1
    return buckets


def print_report(report: dict[str, Any]) -> None:
    corpus = report["retrieval"]["corpus"]
    questions = report["retrieval"]["questions"]
    print("═" * 78)
    print("  个人知识库 Agent · 双维度评估报告")
    print("═" * 78)
    print(f"  语料      : {corpus['documents']} 篇文档 / {corpus['chunks']} 个切片")
    print(f"  Embedder  : {corpus['embedder']} (dim={corpus['dim']})")
    print(f"  题量      : {questions}   语料来源: docs/")
    print("─" * 78)
    print("  维度一 · 检索（Recall@K / MRR）")
    print("─" * 78)
    for arm in report["retrieval"]["arms"]:
        parts = [f"Recall@{k}={arm[f'recall@{k}']:.2f}" for k in arm["depths"]]
        print("  " + _fit(arm["arm"], 18) + "  ".join(parts) + f"  MRR={arm['mrr']:.3f}")
    print("─" * 78)
    print("  分档（线上 top-N 口径）")
    for stat in topic_stats(report["retrieval"]["rows"], int(report["retrieval"]["depth"])):
        print(
            "  " + _fit(stat["topic"], 12) + _fit(f"n={stat['n']}", 7)
            + f"R@3={stat['recall@3']:.2f}  MRR={stat['mrr']:.3f}"
        )

    faith = report.get("faithfulness")
    if faith:
        summary = faith["summary"]
        print("─" * 78)
        print("  维度二 · 忠实度（LLM 裁判，claim 级）")
        print("─" * 78)
        print(f"  参与判定  : {summary['scored_questions']} 题   弃答 {summary['abstained']}   "
              f"判定失败 {summary['judge_failed']}   平均引用 {summary['avg_citations']}")
        print(f"  原子断言  : {summary['claims']} 条  →  被支持 {summary['supported_claims']}  "
              f"不被支持 {summary['unsupported_claims']}")
        print(f"  题级平均  : {summary['score_avg']:.4f}   微平均 {summary['score_micro']:.4f}   "
              f"满分题 {summary['perfect_questions']}/{summary['scored_questions']}")
        calib = faith.get("calibration")
        if calib:
            base, poison = calib["base_score"], calib["poisoned_score"]
            print(
                "  裁判自检  : 原答案 " + ("弃答" if base is None else f"{base:.2f}")
                + " → 注入编造后 " + ("弃答" if poison is None else f"{poison:.2f}")
                + ("   ✅ 检出" if calib["detected"] else "   ✗ 未检出")
            )
        print("═" * 78)
        cross = report["cross"]
        print("  交叉：检索 × 忠实度")
        print(f"   · 双通过（命中且无编造）      : {cross['both_ok']}")
        print(f"   · 命中但有编造（最危险）      : {cross['cited_but_unsupported']}")
        print(f"   · 未命中但答案全被支持        : {cross['miss_but_ok']}")
        print(f"   · 双失败                      : {cross['both_bad']}")
        print(f"   · 拒答（不判幻觉，也不算答对）: {cross['abstained']}")
        risky = [r for r in report["rows_joined"] if 0 < r["rank_prod"] <= 3
                 and r["score"] is not None and r["score"] < 1.0]
        if risky:
            print("─" * 78)
            print(f"  有引用却存在不被支持断言的题（{len(risky)} 道）：")
            for row in risky:
                print(f"   · {row['question']}  (忠实度 {row['score']:.2f})")
                for claim in row["unsupported"]:
                    print(f"     ❌ {claim}")
    else:
        print("─" * 78)
        print("  维度二 · 忠实度：未运行（--retrieval-only 或缺少真实模型）")
    print("═" * 78)
    print()


async def run(*, limit: int | None, depth: int, retrieval_only: bool, concurrency: int) -> dict[str, Any]:
    started = time.perf_counter()
    retrieval = await evaluate(depth=depth, show_hits=False, limit=limit)

    faith: dict[str, Any] | None = None
    if not retrieval_only:
        faith = await evaluate_faithfulness(
            limit=limit, concurrency=concurrency, calibration=True
        )

    merged_rows: list[dict[str, Any]] = []
    cross: dict[str, int] = {}
    if faith:
        for index, row in enumerate(retrieval["rows"]):
            frow = faith["rows"][index]
            merged_rows.append(
                {
                    "question": row["question"],
                    "topic": row["topic"],
                    "rank_prod": row["rank_prod"],
                    "score": frow["score"],
                    "citations": frow["citations"],
                    "claims": frow["claims"],
                    "unsupported": frow["unsupported"],
                }
            )
        cross = _cross(merged_rows, [r["rank_prod"] for r in merged_rows])

    return {
        "retrieval": retrieval,
        "faithfulness": faith,
        "cross": cross,
        "rows_joined": merged_rows,
        "elapsed_ms": int((time.perf_counter() - started) * 1000),
    }


async def main(argv: list[str] | None = None) -> int:
    from app.agents.llm import is_real_llm

    parser = argparse.ArgumentParser(description="双维度评估：检索 + 忠实度")
    parser.add_argument("--limit", type=int, default=None, help="只跑前 N 题")
    parser.add_argument("--top-k", type=int, default=DEFAULT_DEPTH, help="检索召回深度（默认 5）")
    parser.add_argument("--concurrency", type=int, default=4, help="忠实度评估并发")
    parser.add_argument("--retrieval-only", action="store_true", help="只跑检索维度")
    parser.add_argument("--json", action="store_true", help="输出 JSON")
    parser.add_argument("--min-recall", type=float, default=None, help="线上 Recall@3 阈值")
    parser.add_argument("--min-score", type=float, default=None, help="题级平均忠实度阈值")
    args = parser.parse_args(argv)

    retrieval_only = args.retrieval_only or not is_real_llm()
    if args.retrieval_only is False and not is_real_llm():
        print("（未配置真实模型，自动跳过忠实度维度）", file=sys.stderr)

    await _bootstrap()
    report = await run(
        limit=args.limit,
        depth=args.top_k,
        retrieval_only=retrieval_only,
        concurrency=args.concurrency,
    )

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print_report(report)
        print(f"  耗时 {report['elapsed_ms'] / 1000:.1f}s")

    failed = False
    if args.min_recall is not None:
        prod = next(a for a in report["retrieval"]["arms"] if a["arm"].startswith("线上"))
        if prod["recall@3"] < args.min_recall:
            print(f"  ✗ 线上 Recall@3={prod['recall@3']:.2f} < {args.min_recall:.2f}", file=sys.stderr)
            failed = True
    if args.min_score is not None:
        if not report["faithfulness"]:
            print("  ! 忠实度未运行，--min-score 无法校验", file=sys.stderr)
            failed = True
        elif report["faithfulness"]["summary"]["score_avg"] < args.min_score:
            print(
                f"  ✗ 忠实度 {report['faithfulness']['summary']['score_avg']:.4f} < {args.min_score:.2f}",
                file=sys.stderr,
            )
            failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
