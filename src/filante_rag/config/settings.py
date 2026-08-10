"""Central configuration: paths, model choices, and per-language settings.

Keeping every language-specific string (embedding model, prompts, locale)
behind `LanguageConfig` is what lets us add a new language later by adding a
config entry instead of touching pipeline code.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[3]

# pydantic-settings reads .env into our own Settings object only — it never
# touches the real process environment. Libraries that read os.environ
# directly (e.g. Langfuse's get_client(), which auto-configures from
# LANGFUSE_* env vars) need this explicit load to see .env values at all.
load_dotenv(REPO_ROOT / ".env")


class LanguageConfig(BaseModel):
    code: str
    embedding_model: str
    prompt_template_path: Path
    system_locale: str


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    anthropic_api_key: str = ""

    # security — see api/main.py. Required in any publicly reachable
    # deployment; requests are rejected fail-closed if unset.
    api_shared_secret: str = ""
    # comma-separated list of allowed frontend origins for CORS; "*" (the
    # local-dev default) must be replaced before any non-local deployment.
    allowed_origins: str = "*"

    # generation
    generation_model: str = "claude-sonnet-5"

    # data paths
    raw_pdf_path: Path = REPO_ROOT / "data" / "raw" / "FILANTE_manualfull_2603.pdf"
    processed_dir: Path = REPO_ROOT / "data" / "processed"
    eval_dir: Path = REPO_ROOT / "data" / "eval"

    # vector store (Qdrant local on-disk mode — no server needed for dev;
    # swap to QdrantClient(url=...) for a real deployment, see vector_store.py)
    qdrant_path: Path = REPO_ROOT / "data" / "qdrant"
    qdrant_collection: str = "filante_manual"

    default_language: str = "ko"

    @property
    def languages(self) -> dict[str, LanguageConfig]:
        prompts_dir = REPO_ROOT / "src" / "filante_rag" / "generation" / "prompts"
        return {
            "ko": LanguageConfig(
                code="ko",
                embedding_model="BAAI/bge-m3",
                prompt_template_path=prompts_dir / "ko.yaml",
                system_locale="ko-KR",
            ),
            # Add "en": LanguageConfig(...) here to extend — no other
            # pipeline code needs to change; retrieval stays cross-lingual
            # because the embedding model above is multilingual.
        }

    @property
    def allowed_origins_list(self) -> list[str]:
        return [origin.strip() for origin in self.allowed_origins.split(",") if origin.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
