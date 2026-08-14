export type Language = "ko" | "en" | "fr";

export interface CitedSource {
  number: number;
  section_title: string | null;
  printed_page_start: number | null;
  printed_page_end: number | null;
  diagram_pdf_pages: number[];
}

export type ChatEvent =
  | { type: "session"; session_id: string }
  | { type: "delta"; text: string }
  | {
      type: "done";
      text: string;
      sources: CitedSource[];
      has_safety_warning: boolean;
    };

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  text: string;
  sources: CitedSource[];
  hasSafetyWarning: boolean;
  isStreaming: boolean;
}
