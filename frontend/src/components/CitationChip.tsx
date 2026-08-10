import type { CitedSource } from "../types";

function pageLabel(source: CitedSource): string {
  if (source.printed_page_start == null) return "";
  if (source.printed_page_start === source.printed_page_end) {
    return `P.${source.printed_page_start}`;
  }
  return `P.${source.printed_page_start}-${source.printed_page_end}`;
}

export function CitationChip({ source }: { source: CitedSource }) {
  return (
    <span
      className="citation-chip"
      title={source.section_title ?? undefined}
    >
      <span className="citation-chip__page">{pageLabel(source)}</span>
      <span>{source.section_title}</span>
    </span>
  );
}
