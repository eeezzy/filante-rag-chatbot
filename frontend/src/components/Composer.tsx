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
    // The Enter that confirms an IME composition (Korean, Japanese, etc.)
    // also fires a keydown with key "Enter" — submitting on that keystroke
    // clears the textarea before the composed character is committed, so
    // it's left behind. isComposing (and the legacy keyCode 229 fallback
    // some browsers still use) distinguishes the two.
    if (event.key === "Enter" && !event.shiftKey && !event.nativeEvent.isComposing && event.keyCode !== 229) {
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
