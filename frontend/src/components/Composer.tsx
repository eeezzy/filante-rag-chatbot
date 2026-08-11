import { useState, type KeyboardEvent } from "react";
import type { Language } from "../types";
import { LOCALES } from "../locales";

interface ComposerProps {
  disabled: boolean;
  language: Language;
  onSend: (message: string) => void;
}

export function Composer({ disabled, language, onSend }: ComposerProps) {
  const [value, setValue] = useState("");
  const locale = LOCALES[language];

  function submit() {
    const trimmed = value.trim();
    if (!trimmed || disabled) return;
    onSend(trimmed);
    setValue("");
  }

  function handleKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      submit();
    }
  }

  return (
    <form
      className="composer"
      onSubmit={(event) => {
        event.preventDefault();
        submit();
      }}
    >
      <textarea
        className="composer__input"
        placeholder={locale.placeholder}
        rows={1}
        value={value}
        onChange={(event) => setValue(event.target.value)}
        onKeyDown={handleKeyDown}
        aria-label={locale.inputLabel}
      />
      <button
        type="submit"
        className="composer__send"
        disabled={disabled || value.trim().length === 0}
        aria-label={locale.sendLabel}
      >
        →
      </button>
    </form>
  );
}
