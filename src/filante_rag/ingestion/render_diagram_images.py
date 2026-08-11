"""Renders the PDF pages referenced by chunks' diagram_pdf_pages to PNG
files served as static assets by the frontend (see the "images" ADR in
Phase 11 chat history: Netlify's CDN serves them faster and doesn't add
load/cold-start risk to a backend that scales to zero — these are static
and never change independent of a re-ingestion run).

Dev-time/ingestion-time only — never runs inside the deployed backend
container, unlike everything under config/settings.py, so it's fine to
resolve paths relative to this repo the simple way.
"""

from __future__ import annotations

import json
from pathlib import Path

import fitz

REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = REPO_ROOT / "frontend" / "public" / "manual-images"
ZOOM = 2.0  # ~144 DPI — matches the resolution used for ingestion's own page renders


def collect_diagram_pages(chunks_path: Path) -> set[int]:
    pages: set[int] = set()
    with chunks_path.open(encoding="utf-8") as f:
        for line in f:
            chunk = json.loads(line)
            pages.update(chunk["diagram_pdf_pages"])
    return pages


def render_pages(pdf_path: Path, pages: set[int], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    doc = fitz.open(pdf_path)
    try:
        for pdf_page_index in sorted(pages):
            page = doc[pdf_page_index]
            pix = page.get_pixmap(matrix=fitz.Matrix(ZOOM, ZOOM))
            out_path = output_dir / f"page-{pdf_page_index}.png"
            pix.save(out_path)
    finally:
        doc.close()


if __name__ == "__main__":
    from filante_rag.config.settings import get_settings

    settings = get_settings()
    pages = collect_diagram_pages(settings.processed_dir / "chunks.jsonl")
    print(f"Rendering {len(pages)} diagram pages to {OUTPUT_DIR}")
    render_pages(settings.raw_pdf_path, pages, OUTPUT_DIR)

    total_bytes = sum(f.stat().st_size for f in OUTPUT_DIR.glob("*.png"))
    print(f"Wrote {len(pages)} images, {total_bytes / 1e6:.1f}MB total")
