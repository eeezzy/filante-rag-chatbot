import { useState, type KeyboardEvent } from "react";

interface ComposerProps {
  disabled: boolean;
  onSend: (message: string) => void;
}

export function Composer({ disabled, onSend }: ComposerProps) {
  const [value, setValue] = useState("");

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
        placeholder="차량에 대해 궁금한 점을 입력하세요..."
        rows={1}
        value={value}
        onChange={(event) => setValue(event.target.value)}
        onKeyDown={handleKeyDown}
        aria-label="질문 입력"
      />
      <button
        type="submit"
        className="composer__send"
        disabled={disabled || value.trim().length === 0}
        aria-label="질문 보내기"
      >
        →
      </button>
    </form>
  );
}
