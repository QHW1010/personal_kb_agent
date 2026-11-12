"""RAG 检索质量评估：Recall@K / MRR。

跑法：
    python -m scripts.evaluate                 # 默认 Recall@3 / Recall@5
    python -m scripts.evaluate --top-k 8       # 加深召回池
    python -m scripts.evaluate --json          # 给 CI 消费
    python -m scripts.evaluate --show-hits     # 打印每题命中的片段，排障用

判分方式（故意做得严格）：
一个片段算「相关」必须同时满足两条 —— 来源文件名对得上，且正文里含这道题
的锚点原文。只对文件名会让同一篇文档里的任意片段都算命中，指标会虚高到
没有意义；只看锚点又会让别的文档碰巧复述同一句话时算错。

输出两套数字：
- 向量粗召回（search_vectors）—— 衡量 embedding 本身的能力；
- 全链路（retrieve，含重排）—— 衡量线上真正用的那条路径。
两者对比才能说明重排到底有没有价值。

注意样本量：5 道题足以发现「某一类问题完全检索不到」这种结构性缺陷，
但不足以支撑统计显著性结论。它是烟测基线，不是 benchmark；题目增删都会
让数字跳动，写进 README 时必须同时标注题量。
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


@dataclass(frozen=True)
class EvalCase:
    question: str
    source: str
    anchor: str
    note: str = ""


# 5 道题覆盖 docs/kb-core-design.md 的五个切片（简介 / 技术栈 / 部署 / Agent 架构 / 工程约束）
EVAL_SET: list[EvalCase] = [
    EvalCase(
        question="这个系统的项目定位是什么？",
        source="kb-core-design.md",
        anchor="生产级的个人知识库问答 Agent",
        note="第一章 项目简介",
    ),
    EvalCase(
        question="后端 Web 框架为什么要求异步？",
        source="kb-core-design.md",
        anchor="FastAPI（要求异步，自动生成 OpenAPI 文档）",
        note="第二章 技术栈",
    ),
    EvalCase(
        question="这个项目怎么做部署和测试？",
        source="kb-core-design.md",
        anchor="Docker Compose 一键启动",
        note="第二章 部署与测试",
    ),
    EvalCase(
        question="SQL 子图里的 fetch_schema 节点为什么严禁硬编码表结构？",
        source="kb-core-design.md",
        anchor="动态读取当前数据库 schema",
        note="第三章 六节点子图",
    ),
    EvalCase(
        question="对话上下文只保留最近 N 轮是为了防止什么问题？",
        source="kb-core-design.md",
        anchor="防止上下文窗口爆炸",
        note="第四章 工程约束",
    ),
]

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
    """一条评分臂（向量粗召回 / 全链路）的逐题结果。"""

    name: str
    ranks: list[int] = field(default_factory=list)
    timings_ms: list[float] = field(default_factory=list)

    def summary(self, depths: list[int]) -> dict[str, Any]:
        return {
            "arm": self.name,
            **{f"recall@{k}": round(recall_at(self.ranks, k), 4) for k in depths},
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


async def evaluate(*, depth: int, show_hits: bool) -> dict[str, Any]:
    from sqlalchemy import func, select

    from app.core.config import get_settings
    from app.db.models import Document
    from app.db.session import get_session_factory
    from app.rag.embeddings import get_embedder
    from app.rag.retriever import retrieve, search_vectors
    from app.rag.vectorstore import current_collection_name, get_vector_store

    cfg = get_settings()
    factory = get_session_factory()
    async with factory() as session:
        documents = await session.scalar(select(func.count()).select_from(Document)) or 0
    chunks = await get_vector_store().count()

    vector_arm = ArmResult(name="向量粗召回")
    full_arm = ArmResult(name="全链路(含重排)")
    rows: list[dict[str, Any]] = []

    for case in EVAL_SET:
        started = time.perf_counter()
        vector_hits = await search_vectors(case.question, top_k=depth)
        vector_ms = (time.perf_counter() - started) * 1000

        started = time.perf_counter()
        full_hits = await retrieve(case.question, top_k=max(depth, cfg.rag_top_k), top_n=depth)
        full_ms = (time.perf_counter() - started) * 1000

        rank_vector = first_relevant_rank(vector_hits, case)
        rank_full = first_relevant_rank(full_hits, case)
        vector_arm.ranks.append(rank_vector)
        full_arm.ranks.append(rank_full)
        vector_arm.timings_ms.append(vector_ms)
        full_arm.timings_ms.append(full_ms)

        rows.append(
            {
                "question": case.question,
                "expected": case.anchor,
                "note": case.note,
                "rank_vector": rank_vector,
                "rank_full": rank_full,
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
        "questions": len(EVAL_SET),
        "depth": depth,
        "arms": [vector_arm.summary(depths), full_arm.summary(depths)],
        "rows": rows,
    }


def _mark(ok: bool) -> str:
    return "✅" if ok else "—"


def _print_report(report: dict[str, Any], depths: list[int]) -> None:
    corpus = report["corpus"]
    print("═" * 78)
    print("  RAG 检索质量评估")
    print("═" * 78)
    print(f"  语料      : {corpus['documents']} 篇文档 / {corpus['chunks']} 个切片")
    print(f"  Embedder  : {corpus['embedder']} (dim={corpus['dim']})")
    print(f"  Collection: {corpus['collection']}")
    print(f"  题量      : {report['questions']}   召回深度: {report['depth']}")
    print("─" * 78)
    header = (
        "  " + _fit("#", 3) + _fit("问题", 33) + " "
        + _fit("R@3向量", 9) + _fit("R@3全链", 9) + _fit("R@5全链", 9)
        + "首次命中排名"
    )
    print(header)
    print("─" * 78)
    for index, row in enumerate(report["rows"], start=1):
        rv, rf = row["rank_vector"], row["rank_full"]
        line = (
            "  " + _fit(str(index), 3) + _fit(row["question"], 33) + " "
            + _fit(_mark(0 < rv <= 3), 9) + _fit(_mark(0 < rf <= 3), 9) + _fit(_mark(0 < rf <= 5), 9)
            + f"向量#{rv or '-'} / 全链路#{rf or '-'}"
        )
        print(line)

    print("─" * 78)
    for arm in report["arms"]:
        parts = [f"Recall@{k}={arm[f'recall@{k}']:.2f}" for k in depths]
        print(
            "  " + _fit(arm["arm"], 18)
            + "  ".join(parts)
            + f"  MRR={arm['mrr']:.3f}  {arm['avg_ms']:.0f}ms/题"
        )
    print("═" * 78)
    print("  注：5 题 / 9 切片属烟测基线，Recall 容易顶到天花板；MRR 更能反映重排的作用。")
    print()


async def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="RAG 检索质量评估（Recall@K / MRR）")
    parser.add_argument("--top-k", type=int, default=DEFAULT_DEPTH, help="召回深度（默认 5）")
    parser.add_argument("--json", action="store_true", help="只输出 JSON，不打印表格")
    parser.add_argument("--show-hits", action="store_true", help="打印每题命中的片段")
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
    report = await evaluate(depth=args.top_k, show_hits=args.show_hits)
    elapsed = (time.perf_counter() - started) * 1000

    depths = sorted({3, 5, args.top_k})
    if args.json:
        print(json.dumps({**report, "elapsed_ms": round(elapsed)}, ensure_ascii=False, indent=2))
    else:
        _print_report(report, depths)
        print(f"  耗时 {elapsed:.0f} ms")

    if args.min_recall is not None:
        full = next(a for a in report["arms"] if a["arm"].startswith("全链路"))
        if full["recall@3"] < args.min_recall:
            print(f"  ✗ Recall@3={full['recall@3']:.2f} 低于阈值 {args.min_recall:.2f}")
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
