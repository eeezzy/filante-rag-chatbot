"""Combines the footer page-map, layout-aware text, and diagram-page flag
into one record per PDF page, saved as data/processed/pages.jsonl.

This is the single source of truth the chunker reads from.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import fitz

from filante_rag.ingestion.layout_text import analyze_page, extract_page_content
from filante_rag.ingestion.page_map import extract_page_info


@dataclass
class PageRecord:
    pdf_page_index: int
    printed_page: int | None
    chapter_num: int | None
    chapter_title: str | None
    section_title: str | None
    text: str
    is_diagram_page: bool
    diagram_reason: str | None


def build_pages(pdf_path: Path) -> list[PageRecord]:
    doc = fitz.open(pdf_path)
    try:
        records = []
        for i in range(doc.page_count):
            page = doc[i]
            info = extract_page_info(page, i)
            complexity = analyze_page(page, i)
            section_title, text = extract_page_content(page)
            records.append(
                PageRecord(
                    pdf_page_index=i,
                    printed_page=info.printed_page,
                    chapter_num=info.chapter_num,
                    chapter_title=info.chapter_title,
                    section_title=section_title,
                    text=text,
                    is_diagram_page=complexity.is_diagram_page,
                    diagram_reason=complexity.reason,
                )
            )
        return records
    finally:
        doc.close()


def save_pages(records: list[PageRecord], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")


if __name__ == "__main__":
    from filante_rag.config.settings import get_settings

    settings = get_settings()
    records = build_pages(settings.raw_pdf_path)
    out_path = settings.processed_dir / "pages.jsonl"
    save_pages(records, out_path)
    print(f"Wrote {len(records)} pages to {out_path}")

    no_chapter = [r.pdf_page_index for r in records if r.chapter_title is None]
    print(f"Pages with no chapter/footer match (front/back matter): {no_chapter}")
