"""Builds a per-page metadata map from the manual's running footer.

Every body page in this manual ends with a footer in one of two directions
(depending on left/right page layout), e.g.:

    "4 - 1. 환영합니다"          -> printed page 4, chapter "1. 환영합니다"
    "1. 환영합니다 - 5"          -> printed page 5, chapter "1. 환영합니다"

Parsing this directly is far more reliable than parsing the table of
contents (which would require matching dotted leaders and handling
multiple sections per page). Front-matter pages (cover, QR codes, TOC)
have no such footer and simply map to `None`.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import fitz

_PATTERN_PAGE_FIRST = re.compile(r"^(\d+)\s*-\s*(\d+)\.\s*(.+)$")
_PATTERN_PAGE_LAST = re.compile(r"^(\d+)\.\s*(.+?)\s*-\s*(\d+)$")


@dataclass(frozen=True)
class PageInfo:
    pdf_page_index: int  # 0-based index into the PDF
    printed_page: int | None  # page number as printed in the manual
    chapter_num: int | None
    chapter_title: str | None


def _match_footer(line: str) -> tuple[int, int, str] | None:
    """Returns (printed_page, chapter_num, chapter_title) or None."""
    line = line.strip()
    if m := _PATTERN_PAGE_FIRST.match(line):
        printed_page, chapter_num, chapter_title = m.groups()
        return int(printed_page), int(chapter_num), chapter_title.strip()
    if m := _PATTERN_PAGE_LAST.match(line):
        chapter_num, chapter_title, printed_page = m.groups()
        return int(printed_page), int(chapter_num), chapter_title.strip()
    return None


def extract_page_info(page: fitz.Page, pdf_page_index: int) -> PageInfo:
    lines = [l for l in page.get_text("text").splitlines() if l.strip()]
    # The footer is always among the last few lines; scan from the bottom.
    for line in reversed(lines[-5:]):
        if match := _match_footer(line):
            printed_page, chapter_num, chapter_title = match
            return PageInfo(pdf_page_index, printed_page, chapter_num, chapter_title)
    return PageInfo(pdf_page_index, None, None, None)


def build_page_map(pdf_path: Path) -> list[PageInfo]:
    doc = fitz.open(pdf_path)
    try:
        return [extract_page_info(doc[i], i) for i in range(doc.page_count)]
    finally:
        doc.close()


def save_page_map(page_map: list[PageInfo], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps([asdict(p) for p in page_map], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    from filante_rag.config.settings import get_settings

    settings = get_settings()
    page_map = build_page_map(settings.raw_pdf_path)
    save_page_map(page_map, settings.processed_dir / "page_map.json")

    matched = sum(1 for p in page_map if p.printed_page is not None)
    print(f"Matched footer on {matched}/{len(page_map)} pages")
    unmatched = [p.pdf_page_index for p in page_map if p.printed_page is None]
    print(f"Unmatched (front matter / section dividers): {unmatched[:20]}...")
