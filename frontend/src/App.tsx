import { useRef, useState } from "react";
import { streamChat } from "./api";
import { Composer } from "./components/Composer";
import { Header } from "./components/Header";
import { ChatMessageView } from "./components/ChatMessageView";
import type { ChatMessage, Language } from "./types";
import { LOCALES } from "./locales";
import "./styles.css";

let nextId = 0;
const newId = () => `msg-${nextId++}`;

export default function App() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [isStreaming, setIsStreaming] = useState(false);
  const [language, setLanguage] = useState<Language>("ko");
  const sessionIdRef = useRef<string | null>(null);
  const locale = LOCALES[language];

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
      for await (const event of streamChat(text, sessionIdRef.current, language)) {
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
        text: locale.connectionError,
        isStreaming: false,
      });
    } finally {
      setIsStreaming(false);
    }
  }

  return (
    <div className="app">
      <Header language={language} onLanguageChange={setLanguage} />
      <div className="messages">
        {messages.length === 0 && (
          <p className="messages__empty">
            {locale.emptyState.split("\n").map((line, i) => (
              <span key={i}>
                {i > 0 && <br />}
                {line}
              </span>
            ))}
          </p>
        )}
        {messages.map((message) => (
          <ChatMessageView key={message.id} message={message} language={language} />
        ))}
      </div>
      <Composer disabled={isStreaming} language={language} onSend={handleSend} />
    </div>
  );
}
