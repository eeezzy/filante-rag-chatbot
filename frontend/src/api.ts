import type { ChatEvent } from "./types";

// Matches uvicorn's own default port, so `uvicorn ...:app --workers 1` with
// no --port flag works against this default with zero extra config.
const API_BASE = import.meta.env.VITE_API_URL ?? "http://127.0.0.1:8000";
// Must match the backend's API_SHARED_SECRET (see config/settings.py).
// This is baked into the built JS bundle, so it's a soft gate against
// casual/bot traffic, not a real secret — anyone can extract it from the
// deployed site. It still stops the common case: random crawlers hitting
// a public URL and burning API budget.
const API_TOKEN = import.meta.env.VITE_API_TOKEN ?? "";

/**
 * The backend sends SSE-formatted frames over a POST response body (native
 * EventSource only supports GET, so we read the stream manually via fetch
 * and split on the blank-line frame boundary ourselves).
 */
export async function* streamChat(
  message: string,
  sessionId: string | null,
  signal?: AbortSignal,
): AsyncGenerator<ChatEvent> {
  const response = await fetch(`${API_BASE}/chat/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json", "X-API-Key": API_TOKEN },
    body: JSON.stringify({ message, session_id: sessionId }),
    signal,
  });

  if (!response.ok || !response.body) {
    throw new Error(`Chat request failed: ${response.status}`);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    let boundary = buffer.indexOf("\n\n");
    while (boundary !== -1) {
      const frame = buffer.slice(0, boundary);
      buffer = buffer.slice(boundary + 2);
      const line = frame.split("\n").find((l) => l.startsWith("data: "));
      if (line) {
        yield JSON.parse(line.slice("data: ".length)) as ChatEvent;
      }
      boundary = buffer.indexOf("\n\n");
    }
  }
}
