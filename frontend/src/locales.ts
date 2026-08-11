import type { Language } from "./types";

// UI chrome strings, keyed the same way as the backend's prompt packs
// (src/filante_rag/generation/prompts/*.yaml) — same principle: adding a
// language is a config addition, not a code change to components.
export interface Locale {
  subtitle: string;
  emptyState: string;
  placeholder: string;
  inputLabel: string;
  sendLabel: string;
  safetyTag: string;
  connectionError: string;
}

export const LOCALES: Record<Language, Locale> = {
  ko: {
    subtitle: "차량 사용설명서 어시스턴트",
    emptyState: "FILANTE 차량 사용설명서 내용을 바탕으로 답변합니다.\n궁금한 점을 자유롭게 물어보세요.",
    placeholder: "차량에 대해 궁금한 점을 입력하세요...",
    inputLabel: "질문 입력",
    sendLabel: "질문 보내기",
    safetyTag: "⚠ 안전 관련 안내 포함",
    connectionError: "연결 중 오류가 발생했습니다. 잠시 후 다시 시도해 주십시오.",
  },
  en: {
    subtitle: "Vehicle Manual Assistant",
    emptyState: "Answers are based on the FILANTE owner's manual.\nAsk anything about your vehicle.",
    placeholder: "Ask a question about your vehicle...",
    inputLabel: "Question input",
    sendLabel: "Send question",
    safetyTag: "⚠ Contains safety-related guidance",
    connectionError: "A connection error occurred. Please try again in a moment.",
  },
};
