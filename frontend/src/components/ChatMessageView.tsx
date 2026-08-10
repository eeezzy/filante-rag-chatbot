import ReactMarkdown from "react-markdown";
import type { ChatMessage } from "../types";
import { CitationChip } from "./CitationChip";

// The citation footer chips already surface source/page info, so the
// inline [출처N] markers (needed server-side to parse *which* sources were
// used) are noise in the rendered prose — strip them from what's displayed.
const CITATION_MARKER_RE = /\[출처\s*\d+\]/g;

export function ChatMessageView({ message }: { message: ChatMessage }) {
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
          <span className="message__safety-tag">⚠ 안전 관련 안내 포함</span>
        )}
        <div className="message__text">
          {isAssistant ? (
            <ReactMarkdown>{displayText}</ReactMarkdown>
          ) : (
            displayText
          )}
          {message.isStreaming && <span className="cursor" aria-hidden="true" />}
        </div>
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
