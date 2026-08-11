import type { CitedSource } from "../types";

// Diagram pages are static and never change independent of a re-ingestion
// run, so they're pre-rendered PNGs served straight from Netlify's CDN
// (frontend/public/manual-images/) rather than round-tripped through the
// backend — see render_diagram_images.py.
function diagramPages(sources: CitedSource[]): { pdfPage: number; caption: string | null }[] {
  const seen = new Map<number, string | null>();
  for (const source of sources) {
    for (const pdfPage of source.diagram_pdf_pages) {
      if (!seen.has(pdfPage)) seen.set(pdfPage, source.section_title);
    }
  }
  return [...seen.entries()].map(([pdfPage, caption]) => ({ pdfPage, caption }));
}

export function DiagramImages({ sources, alt }: { sources: CitedSource[]; alt: string }) {
  const pages = diagramPages(sources);
  if (pages.length === 0) return null;

  return (
    <div className="diagram-images">
      {pages.map(({ pdfPage, caption }) => (
        <figure className="diagram-image" key={pdfPage}>
          <img src={`/manual-images/page-${pdfPage}.png`} alt={caption ?? alt} loading="lazy" />
          {caption && <figcaption>{caption}</figcaption>}
        </figure>
      ))}
    </div>
  );
}
