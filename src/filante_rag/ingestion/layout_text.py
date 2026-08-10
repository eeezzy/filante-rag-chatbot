"""Column-aware text extraction, plus a heuristic that tags pages whose
content is a diagram/photo with numbered callouts and little else.

For this manual, every such page's numbers are legended on the immediately
adjacent page (confirmed across all 30 pages this heuristic flags), so the
tag is used by the chunker to merge a diagram-only page into its neighbor's
chunk rather than triggering any extra extraction step.
"""

from __future__ import annotations

from dataclasses import dataclass

import fitz

# Tuned against samples from this manual: a normal 2-3 column body page vs.
# a callout-diagram page (e.g. "차량 외부") that's mostly a single image.
IMAGE_AREA_RATIO_THRESHOLD = 0.35
SPARSE_BLOCK_COUNT = 5
COLUMN_BAND_GAP = 15.0  # pts; gap between blocks' x-ranges to start a new band

# Chrome regions, consistent across every body page in this manual: a running
# section-title header at the top-left, a decorative chapter-number tab in
# the top-right margin, and the footer parsed separately by page_map.py.
HEADER_Y1_MAX = 32.0
FOOTER_Y0_MIN = 390.0
TAB_X0_MIN = 545.0


@dataclass
class Block:
    x0: float
    y0: float
    x1: float
    y1: float
    text: str


@dataclass
class ComplexityInfo:
    pdf_page_index: int
    block_count: int
    image_area_ratio: float
    is_diagram_page: bool
    reason: str | None


def _get_blocks(page: fitz.Page) -> list[Block]:
    raw = page.get_text("blocks")
    blocks = []
    for x0, y0, x1, y1, text, *_rest in raw:
        # A block is one semantic unit (line/paragraph/row); PyMuPDF
        # sometimes inserts spurious internal newlines within it that
        # don't represent real paragraph breaks, so collapse all internal
        # whitespace to single spaces. Real line/paragraph structure is
        # reintroduced later, only *between* blocks.
        clean = " ".join(text.split())
        if clean:
            blocks.append(Block(x0, y0, x1, y1, clean))
    return blocks


def _image_area_ratio(page: fitz.Page) -> float:
    page_area = page.rect.width * page.rect.height
    if page_area == 0:
        return 0.0
    total = 0.0
    for img in page.get_images(full=True):
        xref = img[0]
        for rect in page.get_image_rects(xref):
            total += rect.width * rect.height
    return total / page_area


def _cluster_columns(blocks: list[Block]) -> list[list[Block]]:
    """Greedily group blocks into left-to-right column bands by x-overlap."""
    if not blocks:
        return []
    by_x = sorted(blocks, key=lambda b: b.x0)
    bands: list[list[Block]] = [[by_x[0]]]
    band_x1 = by_x[0].x1
    for block in by_x[1:]:
        if block.x0 <= band_x1 + COLUMN_BAND_GAP:
            bands[-1].append(block)
            band_x1 = max(band_x1, block.x1)
        else:
            bands.append([block])
            band_x1 = block.x1
    return bands


def analyze_page(page: fitz.Page, pdf_page_index: int) -> ComplexityInfo:
    blocks = _get_blocks(page)
    ratio = _image_area_ratio(page)

    reason = None
    if ratio > IMAGE_AREA_RATIO_THRESHOLD:
        reason = f"image covers {ratio:.0%} of page"
    elif blocks and len(blocks) <= SPARSE_BLOCK_COUNT and ratio > 0.1:
        reason = f"only {len(blocks)} text blocks alongside imagery (ratio {ratio:.0%})"

    return ComplexityInfo(
        pdf_page_index=pdf_page_index,
        block_count=len(blocks),
        image_area_ratio=ratio,
        is_diagram_page=reason is not None,
        reason=reason,
    )


def _is_footer_or_tab(block: Block) -> bool:
    if block.y0 >= FOOTER_Y0_MIN:
        return True
    if block.x0 >= TAB_X0_MIN and block.text.isdigit() and len(block.text) <= 2:
        return True
    return False


def extract_page_content(page: fitz.Page) -> tuple[str | None, str]:
    """Returns (section_title, body_text), with header/footer/tab chrome
    stripped out of the body so it doesn't get repeated once pages are
    concatenated within a section by the chunker."""
    blocks = _get_blocks(page)

    header_blocks = [b for b in blocks if b.y1 <= HEADER_Y1_MAX]
    section_title = header_blocks[0].text if header_blocks else None

    body_blocks = [
        b for b in blocks if b.y1 > HEADER_Y1_MAX and not _is_footer_or_tab(b)
    ]
    bands = _cluster_columns(body_blocks)
    bands.sort(key=lambda band: min(b.x0 for b in band))
    lines: list[str] = []
    for band in bands:
        for block in sorted(band, key=lambda b: b.y0):
            lines.append(block.text)
    return section_title, "\n".join(lines)


if __name__ == "__main__":
    import json

    from filante_rag.config.settings import get_settings

    settings = get_settings()
    doc = fitz.open(settings.raw_pdf_path)
    results = [analyze_page(doc[i], i) for i in range(doc.page_count)]
    doc.close()

    flagged = [r for r in results if r.is_diagram_page]
    print(f"Diagram-only pages (merged into neighbor at chunk time): {len(flagged)}/{len(results)}")
    for r in flagged:
        print(f"  page {r.pdf_page_index}: {r.reason}")

    out = settings.processed_dir / "page_complexity.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps([r.__dict__ for r in results], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
