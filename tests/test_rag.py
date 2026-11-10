"""RAG 链路：切块 / 召回 / 重排 / 引用生成。"""

from __future__ import annotations

from httpx import AsyncClient

from app.rag.embeddings import HashingEmbedder, tokenize
from app.rag.ingest import chunk_text
from app.rag.reranker import LexicalReranker
from app.rag.retriever import hits_to_citations, retrieve
from app.rag.vectorstore import Hit


def test_chunk_text_short_doc_is_single_chunk() -> None:
    assert chunk_text("很短的一段内容。") == ["很短的一段内容。"]


def test_chunk_text_splits_long_doc() -> None:
    doc = "\n".join(f"第{i}段：" + "内容" * 60 for i in range(10))
    chunks = chunk_text(doc, size=200, overlap=20)
    assert len(chunks) > 1
    assert all(len(c) <= 400 for c in chunks)


def test_tokenize_handles_cjk_bigrams() -> None:
    tokens = tokenize("知识库 RAG")
    assert "知" in tokens
    assert "知识" in tokens
    assert "rag" in tokens


def test_hashing_embedder_is_deterministic_and_normalized() -> None:
    embedder = HashingEmbedder(dim=64)
    a = embedder.embed_one("向量检索与重排序")
    b = embedder.embed_one("向量检索与重排序")
    assert a == b
    norm = sum(v * v for v in a) ** 0.5
    assert abs(norm - 1.0) < 1e-6
    assert embedder.embed_one("完全不同的话题") != a


def test_lexical_reranker_prefers_relevant_chunk() -> None:
    reranker = LexicalReranker()
    hits = [
        Hit("c1", "部署使用 Docker Compose 启动三个容器", {}, 0.80),
        Hit("c2", "重排序采用词项覆盖率与向量分融合打分", {}, 0.55),
    ]
    ordered = reranker.rerank("重排序 打分", hits, top_n=2)
    assert ordered[0].chunk_id == "c2"


async def test_retrieve_returns_relevant_demo_chunk() -> None:
    hits = await retrieve("重排序是怎么工作的", top_k=5, top_n=3)
    assert hits, "演示知识库应至少命中一个片段"
    assert any("重排" in h.text or "排序" in h.text for h in hits)


async def test_citations_are_well_formed() -> None:
    hits = await retrieve("Docker Compose 部署", top_k=3, top_n=2)
    citations = hits_to_citations(hits)
    assert citations
    for c in citations:
        assert c.chunk_id
        assert 0.0 <= c.score <= 1.0
        assert c.snippet


async def test_ingest_then_search_finds_new_document(client: AsyncClient, auth_headers: dict) -> None:
    unique = "量子纠缠蒸馏协议"
    ingest = await client.post(
        "/api/v1/knowledge/ingest",
        headers=auth_headers,
        json={
            "title": "内部实验记录",
            "content": f"本记录描述了{unique}的实现细节，用于验证入库后能否被检索命中。",
            "source": "test",
        },
    )
    assert ingest.status_code == 200
    assert ingest.json()["chunks"] >= 1

    search = await client.post(
        "/api/v1/knowledge/search",
        headers=auth_headers,
        json={"query": unique, "top_k": 3},
    )
    assert search.status_code == 200
    results = search.json()["results"]
    assert results
    assert any(unique in r["snippet"] for r in results)


async def test_knowledge_stats(client: AsyncClient) -> None:
    resp = await client.get("/api/v1/knowledge/stats")
    assert resp.status_code == 200
    body = resp.json()
    assert body["documents"] >= 4  # 演示语料至少 4 篇
    assert body["chunks"] >= 4
    assert body["backend"] == "memory"
    assert body["reranker"] == "lexical"
