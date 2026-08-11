import type { Language } from "../types";
import { LOCALES } from "../locales";

interface HeaderProps {
  language: Language;
  onLanguageChange: (language: Language) => void;
}

export function Header({ language, onLanguageChange }: HeaderProps) {
  return (
    <header className="header">
      <span className="header__mark">
        FILANTE<span>.</span>
      </span>
      <span className="header__subtitle">{LOCALES[language].subtitle}</span>
      <div className="header__lang-toggle" role="group" aria-label="Language / 언어">
        {(["ko", "en"] as const).map((code) => (
          <button
            key={code}
            type="button"
            className={`header__lang-btn${language === code ? " header__lang-btn--active" : ""}`}
            onClick={() => onLanguageChange(code)}
            aria-pressed={language === code}
          >
            {code === "ko" ? "한국어" : "EN"}
          </button>
        ))}
      </div>
    </header>
  );
}
