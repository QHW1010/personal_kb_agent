"""RAG 检索质量评估：Recall@K / MRR。

跑法：
    python -m scripts.evaluate                 # 默认 Recall@3 / Recall@5
    python -m scripts.evaluate --top-k 8       # 加深召回池
    python -m scripts.evaluate --json          # 给 CI 消费
    python -m scripts.evaluate --show-hits     # 打印每题命中的片段，排障用
    python -m scripts.evaluate --limit 10      # 只跑前 10 题，改指标时省时间

判分方式（故意做得严格）：
一个片段算「相关」必须同时满足两条 —— 来源文件名对得上，且正文里含这道题
的锚点原文。只对文件名会让同一篇文档里的任意片段都算命中，指标会虚高到
没有意义；只看锚点又会让别的文档碰巧复述同一句话时算错。

输出三套数字：
- 向量粗召回（search_vectors）—— 衡量 embedding 本身的能力；
- 全链路 top-5（retrieve，含重排）—— 衡量线上那条路径的上限；
- 线上 top-3（与 RAG_RERANK_TOP_N 一致）—— 用户真正拿到的东西。
前两者对比才能说明重排有没有价值；第三者用来对齐忠实度评估（提示词里就这 3 条）。

题目定义在 scripts/eval_set.py，与忠实度评估共用同一批题。样本量与随机基线
（约 K / 切片数）会一起打印在报告里：题目少、语料薄的时候 Recall 必然顶到
天花板，那种数字只能当烟测基线，不能当 benchmark。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# 题目正文放在 scripts/eval_set.py：检索与忠实度两个维度必须用同一批题，
# 否则「检索涨了、忠实度掉了」这类结论根本无从对比。
from scripts.eval_set import EVAL_SET, EvalCase, by_topic  # noqa: E402

DEFAULT_DEPTH = 5


def _meta(hit: Any) -> dict[str, Any]:
    return getattr(hit, "metadata", None) or {}


def is_relevant(hit: Any, case: EvalCase) -> bool:
    """来源 + 锚点双重校验，避免同篇文档的无关片段混进命中。"""
    if str(_meta(hit).get("source", "")) != case.source:
        return False
    return case.anchor in (getattr(hit, "text", "") or "")


def first_relevant_rank(hits: list[Any], case: EvalCase) -> int:
    """返回第一个相关片段的排名（1 起）；没命中返回 0。"""
    for rank, hit in enumerate(hits, start=1):
        if is_relevant(hit, case):
            return rank
    return 0


def recall_at(ranks: list[int], k: int) -> float:
    if not ranks:
        return 0.0
    return sum(1 for r in ranks if 0 < r <= k) / len(ranks)


def mrr(ranks: list[int]) -> float:
    if not ranks:
        return 0.0
    return sum(1.0 / r for r in ranks if r > 0) / len(ranks)


def _display_width(text: str) -> int:
    """中文按两格算，否则表格一定错位。"""
    return sum(2 if unicodedata.east_asian_width(ch) in {"W", "F"} else 1 for ch in text)


def _fit(text: str, width: int) -> str:
    """截断到指定显示宽度并右侧补空格。"""
    if _display_width(text) <= width:
        return text + " " * (width - _display_width(text))
    out, used = "", 0
    for ch in text:
        step = 2 if unicodedata.east_asian_width(ch) in {"W", "F"} else 1
        if used + step > width - 1:
            break
        out += ch
        used += step
    out += "…"
    used += 1
    return out + " " * max(0, width - used)


@dataclass
class ArmResult:
    """一条评分臂（向量粗召回 / 全链路 / 线上配置）的逐题结果。"""

    name: str
    # 这条臂最多能返回几条。线上臂只返回 3 条，给它算 Recall@5 是假的。
    capacity: int = 0
    ranks: list[int] = field(default_factory=list)
    timings_ms: list[float] = field(default_factory=list)

    def summary(self, depths: list[int]) -> dict[str, Any]:
        usable = [k for k in depths if not self.capacity or k <= self.capacity]
        return {
            "arm": self.name,
            **{f"recall@{k}": round(recall_at(self.ranks, k), 4) for k in usable},
            "depths": usable,
            "mrr": round(mrr(self.ranks), 4),
            "avg_ms": round(sum(self.timings_ms) / len(self.timings_ms), 1) if self.timings_ms else 0.0,
        }


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


async def evaluate(
    *, depth: int, show_hits: bool, limit: int | None = None, top_n: int | None = None
) -> dict[str, Any]:
    from sqlalchemy import func, select

    from app.core.config import get_settings
    from app.db.models import Document
    from app.db.session import get_session_factory
    from app.rag.embeddings import get_embedder
    from app.rag.retriever import retrieve, search_vectors
    from app.rag.vectorstore import current_collection_name, get_vector_store

    cfg = get_settings()
    # 线上只把重排后的前 N 条塞进提示词，所以必须单独有一条「线上」臂：
    # 用 top-5 算出的 MRR 会把排在第 4、5 位的命中算成 0.2、0.25，
    # 而线上根本看不到这两条，那个数字是虚高的。
    top_n = top_n or cfg.rag_rerank_top_n

    factory = get_session_factory()
    async with factory() as session:
        documents = await session.scalar(select(func.count()).select_from(Document)) or 0
    chunks = await get_vector_store().count()

    cases = EVAL_SET[:limit] if limit else list(EVAL_SET)

    vector_arm = ArmResult(name="向量粗召回", capacity=depth)
    full_arm = ArmResult(name=f"全链路 top-{depth}", capacity=depth)
    prod_arm = ArmResult(name=f"线上 top-{top_n}", capacity=top_n)
    rows: list[dict[str, Any]] = []

    for case in cases:
        started = time.perf_counter()
        vector_hits = await search_vectors(case.question, top_k=depth)
        vector_ms = (time.perf_counter() - started) * 1000

        started = time.perf_counter()
        full_hits = await retrieve(case.question, top_k=max(depth, cfg.rag_top_k), top_n=depth)
        full_ms = (time.perf_counter() - started) * 1000

        prod_hits = full_hits[:top_n] if top_n <= depth else await retrieve(
            case.question, top_k=cfg.rag_top_k, top_n=top_n
        )

        rank_vector = first_relevant_rank(vector_hits, case)
        rank_full = first_relevant_rank(full_hits, case)
        rank_prod = first_relevant_rank(prod_hits, case)
        vector_arm.ranks.append(rank_vector)
        full_arm.ranks.append(rank_full)
        prod_arm.ranks.append(rank_prod)
        vector_arm.timings_ms.append(vector_ms)
        full_arm.timings_ms.append(full_ms)
        prod_arm.timings_ms.append(full_ms)

        rows.append(
            {
                "question": case.question,
                "topic": case.topic,
                "expected": case.anchor,
                "note": case.note,
                "rank_vector": rank_vector,
                "rank_full": rank_full,
                "rank_prod": rank_prod,
                "vector_hits": [
                    {"source": _meta(h).get("source"), "score": round(h.score, 4)} for h in vector_hits
                ],
                "full_hits": [
                    {"source": _meta(h).get("source"), "score": round(h.score, 4)} for h in full_hits
                ],
            }
        )

        if show_hits:
            print(f"\n── {case.question}")
            print(f"   期望锚点: {case.anchor}")
            for label, hits, rank in (
                ("向量", vector_hits, rank_vector),
                ("全链路", full_hits, rank_full),
                (f"线上 top-{top_n}", prod_hits, rank_prod),
            ):
                print(f"   [{label}] 首个相关片段排名: {rank or '未命中'}")
                for i, hit in enumerate(hits[:depth], start=1):
                    flag = "✅" if is_relevant(hit, case) else "  "
                    preview = (getattr(hit, "text", "") or "").strip().replace("\n", " ")[:56]
                    print(f"     {flag} #{i} {_meta(hit).get('source')} {hit.score:.4f} {preview}")

    depths = sorted({3, 5, depth})
    return {
        "corpus": {
            "documents": documents,
            "chunks": chunks,
            "embedder": f"{cfg.embedding_provider}/{cfg.embedding_model}",
            "dim": get_embedder().dim,
            "collection": current_collection_name(),
        },
        "questions": len(cases),
        "topics": by_topic(),
        # 随机命中一条的概率 K/切片数；按线上取的前 N 条算，才是真实下限
        "random_baseline_recall": round(min(1.0, top_n / chunks), 4) if chunks else 0.0,
        "depth": depth,
        "top_n": top_n,
        "arms": [
            vector_arm.summary(depths),
            full_arm.summary(depths),
            prod_arm.summary(depths),
        ],
        "rows": rows,
    }


def _mark(ok: bool) -> str:
    return "✅" if ok else "—"


def topic_stats(rows: list[dict[str, Any]], depth: int) -> list[dict[str, Any]]:
    """按分组汇总，41 题逐行打印没人看，分组表才看得出哪一类在退化。"""
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(row.get("topic") or "未分组", []).append(row)

    stats: list[dict[str, Any]] = []
    for topic, items in groups.items():
        ranks_vector = [int(i["rank_vector"]) for i in items]
        ranks_full = [int(i["rank_full"]) for i in items]
        stats.append(
            {
                "topic": topic,
                "n": len(items),
                f"recall@{3}_vector": recall_at(ranks_vector, 3),
                f"recall@{3}": recall_at(ranks_full, 3),
                f"recall@{depth}": recall_at(ranks_full, depth),
                "mrr": mrr(ranks_full),
            }
        )
    return stats


def _print_report(report: dict[str, Any], depths: list[int], show_rows: bool = False) -> None:
    corpus = report["corpus"]
    print("═" * 78)
    print("  RAG 检索质量评估")
    print("═" * 78)
    print(f"  语料      : {corpus['documents']} 篇文档 / {corpus['chunks']} 个切片")
    print(f"  Embedder  : {corpus['embedder']} (dim={corpus['dim']})")
    print(f"  Collection: {corpus['collection']}")
    print(f"  题量      : {report['questions']}   召回深度: {report['depth']}")
    print("─" * 78)

    depth = int(report["depth"])
    if show_rows:
        header = (
            "  " + _fit("#", 3) + _fit("问题", 33) + " "
            + _fit("R@3向量", 9) + _fit("R@3全链", 9) + _fit(f"R@{depth}全链", 9)
            + "首次命中排名"
        )
        print(header)
        for index, row in enumerate(report["rows"], start=1):
            rv, rf = row["rank_vector"], row["rank_full"]
            line = (
                "  " + _fit(str(index), 3) + _fit(row["question"], 33) + " "
                + _fit(_mark(0 < rv <= 3), 9) + _fit(_mark(0 < rf <= 3), 9)
                + _fit(_mark(0 < rf <= depth), 9)
                + f"向量#{rv or '-'} / 全链路#{rf or '-'}"
            )
            print(line)
        print("─" * 78)
    else:
        header = (
            "  " + _fit("分组", 14) + _fit("题量", 6)
            + _fit("R@3向量", 9) + _fit("R@3全链", 9) + _fit(f"R@{depth}全链", 10)
            + "MRR全链"
        )
        print(header)
        for stat in topic_stats(report["rows"], depth):
            print(
                "  " + _fit(stat["topic"], 14) + _fit(str(stat["n"]), 6)
                + _fit(f"{stat['recall@3_vector']:.2f}", 9)
                + _fit(f"{stat['recall@3']:.2f}", 9)
                + _fit(f"{stat[f'recall@{depth}']:.2f}", 10)
                + f"{stat['mrr']:.3f}"
            )
        print("─" * 78)

    for arm in report["arms"]:
        usable = arm.get("depths") or depths
        parts = [f"Recall@{k}={arm[f'recall@{k}']:.2f}" for k in usable]
        print(
            "  " + _fit(arm["arm"], 18)
            + "  ".join(parts)
            + f"  MRR={arm['mrr']:.3f}  {arm['avg_ms']:.0f}ms/题"
        )
    print("═" * 78)
    print(
        f"  注：{report['questions']} 题 / {corpus['chunks']} 切片，"
        f"随机命中基线约 Recall@{report['top_n']}={report['random_baseline_recall']:.2f}；"
        "题目与语料同源，绝对值只用于回归对比。"
    )
    print()


async def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="RAG 检索质量评估（Recall@K / MRR）")
    parser.add_argument("--top-k", type=int, default=DEFAULT_DEPTH, help="召回深度（默认 5）")
    parser.add_argument("--json", action="store_true", help="只输出 JSON，不打印表格")
    parser.add_argument("--show-hits", action="store_true", help="打印每题命中的片段")
    parser.add_argument("--rows", action="store_true", help="逐题打印，而不是只打分组汇总")
    parser.add_argument("--limit", type=int, default=None, help="只跑前 N 题")
    parser.add_argument(
        "--min-recall",
        type=float,
        default=None,
        metavar="R",
        help="Recall@3（全链路）低于阈值时以非零码退出，便于接 CI",
    )
    args = parser.parse_args(argv)

    started = time.perf_counter()
    await _bootstrap()
    report = await evaluate(depth=args.top_k, show_hits=args.show_hits, limit=args.limit)
    elapsed = (time.perf_counter() - started) * 1000

    depths = sorted({3, 5, args.top_k})
    if args.json:
        print(json.dumps({**report, "elapsed_ms": round(elapsed)}, ensure_ascii=False, indent=2))
    else:
        _print_report(report, depths, show_rows=args.rows)
        print(f"  耗时 {elapsed:.0f} ms")

    if args.min_recall is not None:
        # 用「线上」这条臂判定：它才是用户真正拿到的东西
        prod = next(a for a in report["arms"] if a["arm"].startswith("线上"))
        if prod["recall@3"] < args.min_recall:
            print(f"  ✗ 线上 Recall@3={prod['recall@3']:.2f} 低于阈值 {args.min_recall:.2f}")
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
