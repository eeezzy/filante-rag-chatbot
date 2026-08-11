import { useRef, useState } from "react";
import { streamChat } from "./api";
import { Composer } from "./components/Composer";
import { Header } from "./components/Header";
import { ChatMessageView } from "./components/ChatMessageView";
import type { ChatMessage } from "./types";
import "./styles.css";

let nextId = 0;
const newId = () => `msg-${nextId++}`;

export default function App() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [isStreaming, setIsStreaming] = useState(false);
  const sessionIdRef = useRef<string | null>(null);

  function updateLast(patch: Partial<ChatMessage>) {
    setMessages((prev) => {
      const next = [...prev];
      next[next.length - 1] = { ...next[next.length - 1], ...patch };
      return next;
    });
  }

  async function handleSend(text: string) {
    const userMessage: ChatMessage = {
      id: newId(),
      role: "user",
      text,
      sources: [],
      hasSafetyWarning: false,
      isStreaming: false,
    };
    const assistantMessage: ChatMessage = {
      id: newId(),
      role: "assistant",
      text: "",
      sources: [],
      hasSafetyWarning: false,
      isStreaming: true,
    };
    setMessages((prev) => [...prev, userMessage, assistantMessage]);
    setIsStreaming(true);

    try {
      for await (const event of streamChat(text, sessionIdRef.current)) {
        if (event.type === "session") {
          sessionIdRef.current = event.session_id;
        } else if (event.type === "delta") {
          setMessages((prev) => {
            const next = [...prev];
            const last = next[next.length - 1];
            next[next.length - 1] = { ...last, text: last.text + event.text };
            return next;
          });
        } else if (event.type === "done") {
          updateLast({
            text: event.text,
            sources: event.sources,
            hasSafetyWarning: event.has_safety_warning,
            isStreaming: false,
          });
        }
      }
    } catch (err) {
      console.error("[App] handleSend error:", err);
      updateLast({
        text: "연결 중 오류가 발생했습니다. 잠시 후 다시 시도해 주십시오.",
        isStreaming: false,
      });
    } finally {
      setIsStreaming(false);
    }
  }

  return (
    <div className="app">
      <Header />
      <div className="messages">
        {messages.length === 0 && (
          <p className="messages__empty">
            FILANTE 차량 사용설명서 내용을 바탕으로 답변합니다.
            <br />
            궁금한 점을 자유롭게 물어보세요.
          </p>
        )}
        {messages.map((message) => (
          <ChatMessageView key={message.id} message={message} />
        ))}
      </div>
      <Composer disabled={isStreaming} onSend={handleSend} />
    </div>
  );
}
