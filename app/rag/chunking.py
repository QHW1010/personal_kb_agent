"""文档切分：结构感知（标题 / 代码块 / 表格）+ 长度约束 + 段落级溯源。

设计取舍（与 README 措辞保持一致，不美化）：
这里做的是「结构感知切分」，不是向量语义切分。用 Markdown 的标题、围栏代码块、
表格作为语义边界，块内再受字符长度约束，跨块用 overlap 兜底。它确定、可单测、
不依赖 embedding，对结构化文档的召回已经足够；真正的语义切分需要相邻句向量
相似度，成本和不可解释性都更高，本阶段不做。

核心不变量（单元测试守着）：
    content[chunk.char_start:chunk.char_end] == chunk.text
因此调用方必须把「同一份规范化文本」既用于切分、又用于落库，否则偏移会错位。
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.core.config import get_settings

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_FENCE_PREFIXES = ("```", "~~~")


@dataclass(frozen=True, slots=True)
class TextChunk:
    """一个切片及其在原文中的精确位置。"""

    text: str
    paragraph_index: int  # 起点所在语义段落序号（0 基），代码块/表格各算一段
    char_start: int
    char_end: int
    section: str = ""  # 标题路径，如 "核心设计文档 > 检索链路"
    kind: str = "text"  # text | code | table


@dataclass(frozen=True, slots=True)
class _Block:
    start: int
    end: int
    paragraph_index: int
    kind: str
    section: str
    is_heading: bool = False  # 该块只有一行标题，正文在下一块


def normalize_content(content: str) -> str:
    """统一换行与 BOM。切分与落库必须共用这一份结果。"""
    text = content.replace("\r\n", "\n").replace("\r", "\n")
    if text.startswith("\ufeff"):
        text = text[1:]
    return text.strip()


def _join_section(stack: list[tuple[int, str]]) -> str:
    return " > ".join(title for _, title in stack)


def _parse_blocks(text: str) -> list[_Block]:
    """把文本切成语义段落块：标题开新块、围栏代码块整块、连续表格行整块。"""
    blocks: list[_Block] = []
    stack: list[tuple[int, str]] = []

    offsets: list[tuple[int, int, str]] = []
    cursor = 0
    for line in text.splitlines(keepends=True):
        offsets.append((cursor, cursor + len(line), line))
        cursor += len(line)

    cur_start: int | None = None
    cur_kind = "text"
    cur_para = 0
    cur_section = ""
    cur_is_heading = False

    def flush(end_off: int) -> None:
        nonlocal cur_start, cur_is_heading
        if cur_start is not None and end_off > cur_start:
            raw = text[cur_start:end_off]
            if raw.strip():
                blocks.append(
                    _Block(cur_start, end_off, cur_para, cur_kind, cur_section, cur_is_heading)
                )
        cur_start = None
        cur_is_heading = False

    i = 0
    total = len(offsets)
    while i < total:
        start, _end, line = offsets[i]
        stripped = line.strip()

        # 围栏代码块：从开围栏到闭围栏整体作为一块，内部绝不断开
        if stripped.startswith(_FENCE_PREFIXES):
            flush(start)
            fence = stripped[:3]
            code_start = start
            code_section = _join_section(stack)
            i += 1
            last_end = _end
            while i < total:
                _s2, e2, line2 = offsets[i]
                last_end = e2
                i += 1
                if line2.strip().startswith(fence):
                    break
            if text[code_start:last_end].strip():
                blocks.append(
                    _Block(code_start, last_end, len(blocks), "code", code_section)
                )
            continue

        # 空行：段落边界
        if not stripped:
            flush(start)
            i += 1
            continue

        heading = _HEADING_RE.match(stripped)
        if heading:
            level = len(heading.group(1))
            title = heading.group(2).strip()
            flush(start)
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
            cur_start = start
            cur_kind = "text"
            cur_para = len(blocks)
            cur_section = _join_section(stack)
            cur_is_heading = True
            i += 1
            continue

        # 连续表格行视为一块
        if stripped.startswith("|"):
            if cur_kind != "table" or cur_start is None:
                flush(start)
                cur_start = start
                cur_kind = "table"
                cur_para = len(blocks)
                cur_section = _join_section(stack)
            i += 1
            continue

        # 普通正文行
        if cur_start is None or cur_kind != "text":
            flush(start)
            cur_start = start
            cur_kind = "text"
            cur_para = len(blocks)
            cur_section = _join_section(stack)
        cur_is_heading = False
        i += 1

    flush(len(text))
    return _attach_headings(blocks)


def _attach_headings(blocks: list[_Block]) -> list[_Block]:
    """把只有标题的块并入紧随其后的正文块。

    标题单独成片是典型垃圾切片：检索能命中"## 四、不可违反的工程约束"，
    但片子里一条约束都没有，答案只能靠重排碰运气。这里直接把标题挂到正文头部，
    位置信息沿用标题自己的（para/section 都指向该章节起点）。
    """
    merged: list[_Block] = []
    i = 0
    total = len(blocks)
    while i < total:
        block = blocks[i]
        if not block.is_heading:
            merged.append(block)
            i += 1
            continue

        j = i
        while j + 1 < total and blocks[j + 1].is_heading:
            j += 1
        if j + 1 < total:
            nxt = blocks[j + 1]
            merged.append(
                _Block(block.start, nxt.end, block.paragraph_index, nxt.kind, block.section, False)
            )
            i = j + 2
            continue
        # 文末孤立的标题，保留原样
        merged.append(block)
        i = j + 1
    return merged


_SENTENCE_ENDS = "\n。！？；"


def _snap_start(text: str, start: int, limit: int) -> int:
    """把切片起点前移到下一个句子边界。

    overlap 回退出来的起点常常落在词中间（比如从 `ute_sql` 开始），
    切片读起来是断的、引用也难看。这里向前找最近的句末/换行，最多走到 limit。
    """
    for i in range(start, min(limit, len(text))):
        if text[i] in _SENTENCE_ENDS:
            return i + 1
    return start


def _hard_split(
    text: str, start: int, end: int, size: int, overlap: int
) -> list[tuple[int, int, str]]:
    """单块超长时的滑窗硬切，优先在换行处断开。"""
    out: list[tuple[int, int, str]] = []
    pos = start
    while pos < end:
        stop = min(pos + size, end)
        broke_at_boundary = False
        if stop < end:
            newline = text.rfind("\n", pos + 1, stop)
            if newline > pos + size // 2:
                stop = newline + 1
                broke_at_boundary = True
        out.append((pos, stop, text[pos:stop]))
        if stop >= end:
            break
        if broke_at_boundary:
            # 已经在换行处断开，本身就干净，不需要 overlap 再造一个碎片
            pos = stop
        else:
            # 不得不在句子中间硬切 → 回退 overlap 并前移到下一个句子边界
            pos = max(stop - overlap, pos + 1)
            snapped = _snap_start(text, pos, stop)
            if snapped < stop:
                pos = snapped
    return out


def split_markdown(
    content: str, *, size: int | None = None, overlap: int | None = None
) -> list[TextChunk]:
    """把 Markdown/TXT 切成带段落号与字符区间的切片。"""
    cfg = get_settings()
    size = max(50, size or cfg.rag_chunk_size)
    raw_overlap = overlap if overlap is not None else cfg.rag_chunk_overlap
    overlap = max(0, min(raw_overlap, size // 2))

    text = normalize_content(content)
    if not text:
        return []

    if len(text) <= size:
        blocks = _parse_blocks(text)
        section = blocks[0].section if blocks else ""
        kind = blocks[0].kind if blocks else "text"
        return [
            TextChunk(
                text=text,
                paragraph_index=0,
                char_start=0,
                char_end=len(text),
                section=section,
                kind=kind,
            )
        ]

    blocks = _parse_blocks(text)
    if not blocks:
        return []

    chunks: list[TextChunk] = []
    pos = 0
    bi = 0
    total = len(blocks)
    emitted_end = -1

    while pos < len(text):
        while bi < total and blocks[bi].end <= pos:
            bi += 1
        if bi >= total:
            break

        block = blocks[bi]
        start = max(pos, block.start)
        if start > block.start:
            # 起点落在块中间（overlap 造成的）→ 前移到句子边界，别从词中间切
            snapped = _snap_start(text, start, block.end)
            if snapped < block.end:
                start = snapped
        kind0 = block.kind

        # 单块就超长 → 硬切（代码块也保不住完整性，但优先在换行处断）
        if block.end - start > size:
            for s, e, body in _hard_split(text, start, block.end, size, overlap):
                chunks.append(
                    TextChunk(body, block.paragraph_index, s, e, block.section, block.kind)
                )
                emitted_end = e
            pos = block.end
            bi += 1
            continue

        # 段落装箱：只合并同结构类型的相邻块（正文配正文、表格/代码各自独立），
        # 否则代码块会被糊进正文，溯源和召回都会变糊
        j = bi
        end = block.end
        while (
            j + 1 < total
            and blocks[j + 1].kind == kind0
            and blocks[j + 1].end - start <= size
        ):
            j += 1
            end = blocks[j].end

        if end <= emitted_end:
            # 这段内容已经完全被上一片覆盖（overlap 回退造成），跳过，
            # 直接跳到下一个块的起点，避免产出纯重复的碎片
            if j + 1 < total:
                pos = blocks[j + 1].start
                continue
            break

        body = text[start:end]
        if body.strip():
            chunks.append(
                TextChunk(
                    text=body,
                    paragraph_index=block.paragraph_index,
                    char_start=start,
                    char_end=end,
                    section=block.section,
                    kind=kind0,
                )
            )
            emitted_end = end

        if j >= total - 1:
            break

        # 下一片起点回退 overlap 字符，让跨块语义有重叠（正常路径也生效）
        next_start = end - overlap
        pos = max(next_start, start + 1) if next_start > start else end

    return chunks


def chunk_text(text: str, *, size: int | None = None, overlap: int | None = None) -> list[str]:
    """兼容壳：保留旧的「只要文本列表」接口（既有测试与调用方在用）。"""
    return [c.text for c in split_markdown(text, size=size, overlap=overlap)]
