import ReactMarkdown from "react-markdown";
import type { ChatMessage, Language } from "../types";
import { CitationChip } from "./CitationChip";
import { DiagramImages } from "./DiagramImages";
import { LOCALES } from "../locales";

// The citation footer chips already surface source/page info, so the
// inline [출처N]/[Source N] markers (needed server-side to parse *which*
// sources were used) are noise in the rendered prose — strip them from
// what's displayed. Matches both languages' marker formats (see the
// matching regex in streaming_generator.py).
const CITATION_MARKER_RE = /\[(?:출처|[Ss]ource)\s*\d+\]/g;

export function ChatMessageView({
  message,
  language,
}: {
  message: ChatMessage;
  language: Language;
}) {
  const isAssistant = message.role === "assistant";
  const classes = [
    "message",
    isAssistant ? "message--assistant" : "message--user",
    message.hasSafetyWarning ? "message--safety" : "",
  ]
    .filter(Boolean)
    .join(" ");

  const displayText = message.text.replace(CITATION_MARKER_RE, "");

  return (
    <div className={classes}>
      <div className="message__bubble">
        {isAssistant && message.hasSafetyWarning && (
          <span className="message__safety-tag">{LOCALES[language].safetyTag}</span>
        )}
        <div className={`message__text${isAssistant ? "" : " message__text--plain"}`}>
          {isAssistant ? (
            <ReactMarkdown>{displayText}</ReactMarkdown>
          ) : (
            displayText
          )}
          {message.isStreaming && <span className="cursor" aria-hidden="true" />}
        </div>
        {isAssistant && !message.isStreaming && (
          <DiagramImages sources={message.sources} alt={LOCALES[language].diagramAlt} />
        )}
        {message.sources.length > 0 && (
          <div className="citations">
            {message.sources.map((source) => (
              <CitationChip key={source.number} source={source} />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
