"""Groups pages into sections by their running header, then splits any
section too long for one chunk on paragraph boundaries.

Grouping by (chapter_num, section_title) over *consecutive* pages mirrors
how the manual itself is organized (confirmed against the table of
contents): most sections are 1-2 pages, but some (e.g. "아웃사이드 미러 및
룸미러") run over 20 pages and need sub-splitting.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

MAX_CHUNK_CHARS = 1200
OVERLAP_CHARS = 150
# A boxed 경고/주의 call-out becomes its own block/line, always starting
# with the marker word followed by a space. "주의" also appears constantly
# mid-sentence in ordinary prose ("주의하십시오") — matching only a
# line-leading whole word avoids flagging those (69% of chunks with a naive
# substring match vs. this line-leading check).
_SAFETY_LINE_RE = re.compile(r"^(경고|주의)\b")


def _contains_safety_warning(text: str) -> bool:
    return any(_SAFETY_LINE_RE.match(line.strip()) for line in text.split("\n"))


@dataclass
class PageGroup:
    chapter_num: int | None
    chapter_title: str | None
    section_title: str | None
    pdf_page_start: int
    pdf_page_end: int
    printed_page_start: int | None
    printed_page_end: int | None
    text: str
    # pdf_page_index of any diagram-only page (see layout_text.py) among
    # this group's own constituent pages — precise because it comes
    # straight from the pages that built this group, not a range lookup
    # that could pick up unrelated pages. Note: when a group gets split
    # into multiple chunks (see _split_on_budget), every resulting piece
    # inherits the *whole* group's diagram pages, same as they already
    # inherit the whole group's pdf_page_start/end — imprecise for the
    # rare long (20+ page) sections, accurate for the common 1-2 page case.
    diagram_pdf_pages: list[int]


@dataclass
class Chunk:
    chunk_id: str
    chapter_num: int | None
    chapter_title: str | None
    section_title: str | None
    pdf_page_start: int
    pdf_page_end: int
    printed_page_start: int | None
    printed_page_end: int | None
    text: str
    char_count: int
    contains_safety_warning: bool
    diagram_pdf_pages: list[int]


def _load_pages(pages_path: Path) -> list[dict]:
    with pages_path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def group_pages(pages: list[dict]) -> list[PageGroup]:
    groups: list[PageGroup] = []
    cur_pages: list[dict] = []
    cur_key: tuple | None = None

    def flush() -> None:
        if not cur_pages:
            return
        printed_pages = [p["printed_page"] for p in cur_pages if p["printed_page"] is not None]
        texts = [p["text"] for p in cur_pages if not p["is_diagram_page"] and p["text"].strip()]
        diagram_pages = [p["pdf_page_index"] for p in cur_pages if p["is_diagram_page"]]
        groups.append(
            PageGroup(
                chapter_num=cur_pages[0]["chapter_num"],
                chapter_title=cur_pages[0]["chapter_title"],
                section_title=cur_pages[0]["section_title"],
                pdf_page_start=cur_pages[0]["pdf_page_index"],
                pdf_page_end=cur_pages[-1]["pdf_page_index"],
                printed_page_start=printed_pages[0] if printed_pages else None,
                printed_page_end=printed_pages[-1] if printed_pages else None,
                text="\n\n".join(texts),
                diagram_pdf_pages=diagram_pages,
            )
        )

    for page in pages:
        key = (page["chapter_num"], page["section_title"])
        if key != cur_key:
            flush()
            cur_pages = [page]
            cur_key = key
        else:
            cur_pages.append(page)
    flush()
    return groups


def _split_on_budget(text: str, max_chars: int, overlap_chars: int) -> list[str]:
    paragraphs = [p for p in text.split("\n\n") if p.strip()]
    # Some pages (e.g. the TOC) have no blank-line breaks at all, so a single
    # "paragraph" can itself blow the budget. Fall back to line breaks there.
    paragraphs = [
        line
        for para in paragraphs
        for line in ([para] if len(para) <= max_chars else para.split("\n"))
        if line.strip()
    ]
    if not paragraphs:
        return []

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for para in paragraphs:
        if current and current_len + len(para) + 2 > max_chars:
            chunks.append("\n\n".join(current))
            # carry the tail of the previous chunk forward for continuity
            tail = chunks[-1][-overlap_chars:]
            current = [tail, para]
            current_len = len(tail) + len(para)
        else:
            current.append(para)
            current_len += len(para) + 2

    if current:
        chunks.append("\n\n".join(current))
    return chunks


def build_chunks(groups: list[PageGroup]) -> list[Chunk]:
    chunks: list[Chunk] = []
    for group in groups:
        if not group.text.strip():
            continue
        pieces = (
            [group.text]
            if len(group.text) <= MAX_CHUNK_CHARS
            else _split_on_budget(group.text, MAX_CHUNK_CHARS, OVERLAP_CHARS)
        )
        title_slug = (group.section_title or group.chapter_title or "misc").strip()
        for idx, piece in enumerate(pieces):
            chunk_id = f"ch{group.chapter_num}-{title_slug}-{group.pdf_page_start}-{idx}"
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    chapter_num=group.chapter_num,
                    chapter_title=group.chapter_title,
                    section_title=group.section_title,
                    pdf_page_start=group.pdf_page_start,
                    pdf_page_end=group.pdf_page_end,
                    printed_page_start=group.printed_page_start,
                    printed_page_end=group.printed_page_end,
                    text=piece,
                    char_count=len(piece),
                    contains_safety_warning=_contains_safety_warning(piece),
                    diagram_pdf_pages=group.diagram_pdf_pages,
                )
            )
    return chunks


def save_chunks(chunks: list[Chunk], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")


if __name__ == "__main__":
    from filante_rag.config.settings import get_settings

    settings = get_settings()
    pages = _load_pages(settings.processed_dir / "pages.jsonl")
    groups = group_pages(pages)
    chunks = build_chunks(groups)
    out_path = settings.processed_dir / "chunks.jsonl"
    save_chunks(chunks, out_path)

    lens = [c.char_count for c in chunks]
    print(f"{len(groups)} sections -> {len(chunks)} chunks, wrote {out_path}")
    print(f"chunk chars: min={min(lens)} mean={sum(lens)/len(lens):.0f} max={max(lens)}")
    print(f"safety-flagged chunks: {sum(c.contains_safety_warning for c in chunks)}/{len(chunks)}")
    print(f"chunks with a diagram page: {sum(bool(c.diagram_pdf_pages) for c in chunks)}/{len(chunks)}")
