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
    # Per-page text, kept alongside the flattened `text` above so a group
    # that gets split into multiple chunks (see _split_on_budget) can tell
    # which of its own pages actually contributed to each resulting piece,
    # instead of every piece inheriting the whole group's pages/diagrams —
    # a real bug when a long section mixes an early page's diagram with a
    # much later page's unrelated warning (e.g. "스마트키 사용 방법 및 주의
    # 사항": the key-fob button diagram on page 1 of the section has
    # nothing to do with the "don't leave a pet in the car" warning three
    # pages later, but both used to get the same image attached).
    page_texts: list[tuple[int, str]]
    # pdf_page_index of pages worth showing a picture of (see
    # layout_text.py), split by why they were flagged — see the comment
    # in group_pages() for why these can't be merged into one list.
    diagram_only_pages: list[int]
    legend_photo_pages: list[int]


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
        page_texts = [
            (p["pdf_page_index"], p["text"])
            for p in cur_pages
            if not p["is_diagram_page"] and p["text"].strip()
        ]
        # Kept as two separate lists, not one merged set — is_diagram_page
        # pages contribute no text of their own (see page_texts above), so
        # a piece can only be linked to one via its adjacent text page.
        # has_legend_photo pages keep their own text, so they only belong
        # on a piece that actually contains that exact page (see
        # _diagram_pages_for) — applying the same neighbor-adjacency
        # fallback to them let one page's photo bleed onto a neighboring
        # piece that never mentioned it.
        diagram_only_pages = [p["pdf_page_index"] for p in cur_pages if p["is_diagram_page"]]
        legend_photo_pages = [p["pdf_page_index"] for p in cur_pages if p["has_legend_photo"]]
        groups.append(
            PageGroup(
                chapter_num=cur_pages[0]["chapter_num"],
                chapter_title=cur_pages[0]["chapter_title"],
                section_title=cur_pages[0]["section_title"],
                pdf_page_start=cur_pages[0]["pdf_page_index"],
                pdf_page_end=cur_pages[-1]["pdf_page_index"],
                printed_page_start=printed_pages[0] if printed_pages else None,
                printed_page_end=printed_pages[-1] if printed_pages else None,
                text="\n\n".join(t for _, t in page_texts),
                page_texts=page_texts,
                diagram_only_pages=diagram_only_pages,
                legend_photo_pages=legend_photo_pages,
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


def _split_on_budget(
    page_texts: list[tuple[int, str]], max_chars: int, overlap_chars: int
) -> list[tuple[str, list[int]]]:
    # Paragraphs, each tagged with the page it came from, so a resulting
    # piece can report exactly which pages it draws on (see PageGroup.page_texts).
    paragraphs: list[tuple[int, str]] = []
    for page_index, page_text in page_texts:
        # Some pages (e.g. the TOC) have no blank-line breaks at all, so a
        # single "paragraph" can itself blow the budget. Fall back to line
        # breaks there.
        for para in page_text.split("\n\n"):
            if not para.strip():
                continue
            if len(para) <= max_chars:
                paragraphs.append((page_index, para))
            else:
                paragraphs.extend((page_index, line) for line in para.split("\n") if line.strip())
    if not paragraphs:
        return []

    pieces: list[tuple[str, list[int]]] = []
    current: list[str] = []
    current_pages: list[int] = []
    current_len = 0

    for page_index, para in paragraphs:
        if current and current_len + len(para) + 2 > max_chars:
            pieces.append(("\n\n".join(current), _dedupe(current_pages)))
            # carry the tail of the previous piece forward for continuity —
            # it's still text from current_pages' last page, so that page
            # stays attributed to the new piece too.
            tail = pieces[-1][0][-overlap_chars:]
            current = [tail, para]
            current_pages = [current_pages[-1], page_index]
            current_len = len(tail) + len(para)
        else:
            current.append(para)
            current_pages.append(page_index)
            current_len += len(para) + 2

    if current:
        pieces.append(("\n\n".join(current), _dedupe(current_pages)))
    return pieces


def _dedupe(values: list[int]) -> list[int]:
    seen: dict[int, None] = {}
    for v in values:
        seen[v] = None
    return list(seen)


def _diagram_pages_for(
    piece_pages: list[int], diagram_only_pages: list[int], legend_photo_pages: list[int]
) -> list[int]:
    text_pages = set(piece_pages)
    # has_legend_photo pages keep their own text (see page_texts), so only
    # a piece that actually contains that page belongs with its photo —
    # no inference needed, and none wanted (a neighboring piece that never
    # touches the page shouldn't get its picture).
    direct = [d for d in legend_photo_pages if d in text_pages]
    # is_diagram_page pages contribute no text of their own, so the only
    # way to link one to a piece is via its immediately adjacent page —
    # confirmed as the legend's actual location across every such page in
    # this manual (see layout_text.py).
    neighbors = {p - 1 for p in text_pages} | {p + 1 for p in text_pages}
    adjacent = [d for d in diagram_only_pages if d in neighbors]
    return direct + adjacent


def build_chunks(groups: list[PageGroup]) -> list[Chunk]:
    chunks: list[Chunk] = []
    for group in groups:
        if not group.text.strip():
            continue
        pieces = (
            [(group.text, [p for p, _ in group.page_texts])]
            if len(group.text) <= MAX_CHUNK_CHARS
            else _split_on_budget(group.page_texts, MAX_CHUNK_CHARS, OVERLAP_CHARS)
        )
        title_slug = (group.section_title or group.chapter_title or "misc").strip()
        for idx, (piece, piece_pages) in enumerate(pieces):
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
                    diagram_pdf_pages=_diagram_pages_for(
                        piece_pages, group.diagram_only_pages, group.legend_photo_pages
                    ),
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
