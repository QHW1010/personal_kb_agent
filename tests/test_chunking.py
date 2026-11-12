"""切分器测试：偏移不变量、结构完整性、段落号、确定性 chunk_id。"""

from __future__ import annotations

import pytest

from app.rag.chunking import chunk_text, normalize_content, split_markdown
from app.rag.vectorstore import collection_name_for, make_chunk

SAMPLE = """# 核心设计文档

## 一、项目简介

本系统是一个生产级的个人知识库问答 Agent，旨在通过大语言模型和检索增强生成技术，实现基于用户私有文档的精准问答。

## 二、技术栈

- 后端：FastAPI
- 编排：LangGraph

| 组件 | 说明 |
| --- | --- |
| Chroma | 向量库 |
| Redis | 缓存 |

### 代码示例

```python
def main():
    return 1
```

## 三、工程约束

所有 Agent 的输出必须经过 Pydantic 模式验证后才能返回用户。
"""


def test_empty_and_whitespace_doc_returns_empty() -> None:
    assert split_markdown("") == []
    assert split_markdown("   \n\n  \t ") == []


def test_offsets_are_consistent_with_source_text() -> None:
    content = normalize_content(SAMPLE)
    chunks = split_markdown(content, size=120, overlap=30)
    assert chunks
    for chunk in chunks:
        assert content[chunk.char_start : chunk.char_end] == chunk.text
        assert chunk.char_start < chunk.char_end


def test_chunk_windows_are_ordered_and_overlapping() -> None:
    content = normalize_content(SAMPLE)
    chunks = split_markdown(content, size=80, overlap=10)
    starts = [c.char_start for c in chunks]
    assert starts == sorted(starts)
    # overlap 在正常装箱路径也要生效，否则跨块语义会断。
    # 注意：只在同类型（正文→正文）之间重叠；正文/表格/代码交界处刻意不重叠，
    # 免得把代码糊进正文里。
    assert any(chunks[i + 1].char_start < chunks[i].char_end for i in range(len(chunks) - 1))


def test_code_fence_is_not_split() -> None:
    content = normalize_content(SAMPLE)
    chunks = split_markdown(content, size=120, overlap=30)
    code_hits = [c for c in chunks if "def main" in c.text]
    assert len(code_hits) == 1, "代码块被切碎了"
    assert code_hits[0].kind == "code"
    assert "```" in code_hits[0].text
    assert "return 1" in code_hits[0].text, "代码块尾部丢失"


def test_table_is_not_split() -> None:
    content = normalize_content(SAMPLE)
    chunks = split_markdown(content, size=120, overlap=30)
    table_hits = [c for c in chunks if "Chroma" in c.text and "Redis" in c.text]
    assert table_hits, "表格被切断，Chroma 与 Redis 不在同一片"
    assert any(c.kind == "table" for c in table_hits)


def test_section_path_follows_headings() -> None:
    content = normalize_content(SAMPLE)
    chunks = split_markdown(content, size=80, overlap=10)
    sections = {c.section for c in chunks}
    assert any("核心设计文档" in s for s in sections)
    assert any("二、技术栈" in s for s in sections)
    # 更深层的标题应该拼成路径，而不是覆盖掉一级标题
    assert any(s.count(">") >= 2 for s in sections)


def test_paragraph_index_is_monotonic() -> None:
    chunks = split_markdown(normalize_content(SAMPLE), size=120, overlap=30)
    indexes = [c.paragraph_index for c in chunks]
    assert indexes == sorted(indexes)
    assert indexes[0] == 0


def test_heading_is_attached_to_its_body() -> None:
    """纯标题切片是垃圾切片：能命中标题却拿不到正文，答案只能靠运气。"""
    content = normalize_content(SAMPLE)
    chunks = split_markdown(content, size=120, overlap=30)
    for heading in ("## 一、项目简介", "## 二、技术栈", "## 三、工程约束"):
        holders = [c for c in chunks if heading in c.text]
        assert holders, f"标题 {heading} 没有出现在任何切片里"
        assert any(len(c.text.strip()) > len(heading) + 10 for c in holders), (
            f"标题 {heading} 被单独切成了一片"
        )


def test_chunk_text_alias_matches_split_markdown() -> None:
    content = normalize_content(SAMPLE)
    assert chunk_text(content, size=150, overlap=20) == [
        c.text for c in split_markdown(content, size=150, overlap=20)
    ]


def test_short_doc_returns_single_chunk() -> None:
    assert chunk_text("很短的一段内容。") == ["很短的一段内容。"]


def test_long_single_paragraph_is_hard_split() -> None:
    content = "啊" * 1000
    chunks = split_markdown(content, size=200, overlap=20)
    assert len(chunks) > 1
    assert all(len(c.text) <= 200 for c in chunks)
    for chunk in chunks:
        assert content[chunk.char_start : chunk.char_end] == chunk.text


def test_normalize_content_handles_crlf_and_bom() -> None:
    assert normalize_content("\ufeffa\r\nb\rc") == "a\nb\nc"


def test_make_chunk_id_is_deterministic() -> None:
    kw = {"document_id": "doc1", "title": "T", "source": "docs/a.md", "index": 0}
    a = make_chunk("同一段文本", **kw)
    b = make_chunk("同一段文本", **kw)
    assert a.chunk_id == b.chunk_id
    assert make_chunk("同一段文本", **{**kw, "index": 1}).chunk_id != a.chunk_id
    assert make_chunk("另一段文本", **kw).chunk_id != a.chunk_id


def test_make_chunk_carries_paragraph_metadata() -> None:
    chunk = make_chunk(
        "内容",
        document_id="d",
        title="标题",
        source="docs/x.md",
        index=3,
        paragraph_index=7,
        char_start=100,
        char_end=102,
        section="A > B",
        kind="code",
    )
    assert chunk.metadata["paragraph_index"] == 7
    assert chunk.metadata["char_start"] == 100
    assert chunk.metadata["char_end"] == 102
    assert chunk.metadata["section"] == "A > B"
    assert chunk.metadata["kind"] == "code"
    # Chroma 不接受 None，所有元数据值必须是标量
    assert all(isinstance(v, (str, int, float, bool)) for v in chunk.metadata.values())


@pytest.mark.parametrize(
    ("provider", "model", "dim", "expected"),
    [
        ("hashing", None, 384, "kb_documents"),
        ("fastembed", "BAAI/bge-small-zh-v1.5", 512, "kb_documents_fastembed_baai_bge_small_zh_v1_5_512"),
    ],
)
def test_collection_name_versioning(provider: str, model: str | None, dim: int, expected: str) -> None:
    assert collection_name_for("kb_documents", provider, model, dim) == expected


def test_collection_name_keeps_old_index_untouched() -> None:
    """hashing 必须沿用基础名，否则旧索引就"消失"了。"""
    assert collection_name_for("kb_documents", "hashing", "BAAI/bge-small-zh-v1.5", 512) == "kb_documents"


def test_collection_name_fits_chroma_limits() -> None:
    name = collection_name_for("kb_documents", "fastembed", "some/very-long-model-name-" * 3, 1024)
    assert 3 <= len(name) <= 63
    assert name[0].isalnum() and name[-1].isalnum()
