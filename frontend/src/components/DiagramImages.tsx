import { useEffect, useState } from "react";
import { createPortal } from "react-dom";
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

function Lightbox({
  pdfPage,
  caption,
  alt,
  onClose,
}: {
  pdfPage: number;
  caption: string | null;
  alt: string;
  onClose: () => void;
}) {
  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") onClose();
    };
    document.addEventListener("keydown", onKeyDown);
    return () => document.removeEventListener("keydown", onKeyDown);
  }, [onClose]);

  return createPortal(
    <div className="lightbox" onClick={onClose}>
      <figure className="lightbox__frame" onClick={(event) => event.stopPropagation()}>
        <img src={`/manual-images/page-${pdfPage}.png`} alt={caption ?? alt} />
        {caption && <figcaption>{caption}</figcaption>}
      </figure>
      <button className="lightbox__close" onClick={onClose} aria-label="Close">
        ×
      </button>
    </div>,
    document.body,
  );
}

export function DiagramImages({ sources, alt }: { sources: CitedSource[]; alt: string }) {
  const pages = diagramPages(sources);
  const [expanded, setExpanded] = useState<number | null>(null);
  if (pages.length === 0) return null;

  const expandedPage = pages.find((p) => p.pdfPage === expanded);

  return (
    <div className="diagram-images">
      {pages.map(({ pdfPage, caption }) => (
        <figure className="diagram-image" key={pdfPage}>
          <button
            type="button"
            className="diagram-image__button"
            onClick={() => setExpanded(pdfPage)}
            aria-label={caption ?? alt}
          >
            <img src={`/manual-images/page-${pdfPage}.png`} alt={caption ?? alt} loading="lazy" />
          </button>
          {caption && <figcaption>{caption}</figcaption>}
        </figure>
      ))}
      {expandedPage && (
        <Lightbox
          pdfPage={expandedPage.pdfPage}
          caption={expandedPage.caption}
          alt={alt}
          onClose={() => setExpanded(null)}
        />
      )}
    </div>
  );
}
