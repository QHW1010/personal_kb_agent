"""Citation 兼容性与结构校验：新增定位字段不能破坏旧数据。"""

from __future__ import annotations

from app.rag.retriever import hits_to_citations
from app.rag.vectorstore import Hit
from app.schemas.agent import AgentAnswer, Citation


def _legacy_payload() -> dict:
    """模拟阶段一存进 chat_messages.citations 的旧结构（没有定位字段）。"""
    return {
        "chunk_id": "abc123",
        "document_id": "doc-1",
        "title": "旧文档",
        "source": "demo",
        "snippet": "旧片段",
        "score": 0.42,
    }


def test_citation_accepts_legacy_payload() -> None:
    citation = Citation(**_legacy_payload())
    assert citation.chunk_id == "abc123"
    assert citation.paragraph_index is None
    assert citation.char_start is None
    assert citation.char_end is None
    assert citation.section == ""


def test_citation_accepts_new_fields() -> None:
    citation = Citation(
        **_legacy_payload(),
        paragraph_index=12,
        char_start=1024,
        char_end=1400,
        section="设计文档 > 检索链路",
    )
    assert citation.paragraph_index == 12
    assert citation.char_end - citation.char_start == 376


def test_agent_answer_accepts_legacy_citations() -> None:
    """历史消息整条反序列化不能因为少了定位字段而失败。"""
    answer = AgentAnswer(answer="历史答案", route="rag", citations=[_legacy_payload()])
    assert len(answer.citations) == 1
    assert answer.citations[0].paragraph_index is None


def test_agent_answer_still_dedups_by_chunk_id() -> None:
    dup = _legacy_payload()
    answer = AgentAnswer(
        answer="去重测试",
        route="rag",
        citations=[dup, {**dup, "snippet": "重复片段"}, {**dup, "chunk_id": "other"}],
    )
    assert [c.chunk_id for c in answer.citations] == ["abc123", "other"]


def test_hits_to_citations_carries_offsets() -> None:
    hit = Hit(
        chunk_id="c1",
        text="切片正文",
        metadata={
            "document_id": "doc-9",
            "title": "核心设计文档",
            "source": "kb-core-design.md",
            "paragraph_index": 4,
            "char_start": 120,
            "char_end": 128,
            "section": "核心设计文档 > 三、架构",
        },
        score=0.87,
    )
    citation = hits_to_citations([hit])[0]
    assert citation.source == "kb-core-design.md"
    assert citation.paragraph_index == 4
    assert citation.char_start == 120
    assert citation.char_end == 128
    assert citation.section == "核心设计文档 > 三、架构"
    assert citation.score == 0.87


def test_hits_to_citations_tolerates_missing_metadata() -> None:
    """旧集合里的向量没有定位字段，不能因此 500。"""
    citation = hits_to_citations([Hit(chunk_id="c2", text="只有正文", metadata={}, score=-0.1)])[0]
    assert citation.paragraph_index is None
    assert citation.score == 0.0  # 负数被 clamp 到 0，满足 0..1 约束
