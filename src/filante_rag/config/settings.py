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

# The filante_rag package's own directory — NOT the repo root. Computing
# this via parents[N] traversal from an installed package's __file__ only
# happens to work for an editable install (pip install -e ., which just
# links back to src/filante_rag); a real install (as used in the Docker
# image, deliberately non-editable — see Dockerfile) copies the package to
# site-packages, where that traversal lands somewhere like
# /usr/local/lib/python3.12 instead of the repo root. Caught by actually
# running the built container, not just building it: it started, then
# crashed on first request with the prompt YAML "not found" at a path that
# didn't exist. PACKAGE_ROOT only ever needs to locate files that ship
# *inside* the package (e.g. prompts/), which this resolves correctly
# under either install mode; external data (data/processed, data/qdrant)
# uses plain CWD-relative paths below instead, since it deliberately lives
# outside the installed package.
PACKAGE_ROOT = Path(__file__).resolve().parent.parent

# pydantic-settings reads .env into our own Settings object only — it never
# touches the real process environment. Libraries that read os.environ
# directly (e.g. Langfuse's get_client(), which auto-configures from
# LANGFUSE_* env vars) need this explicit load to see .env values at all.
# Deployed environments (Fly.io secrets, etc.) set real env vars directly
# and have no .env file — load_dotenv() is a no-op if the path is missing,
# not an error.
load_dotenv(Path.cwd() / ".env")


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

    # data paths — relative to the current working directory (repo root in
    # local dev, /app in the Docker image, which COPYs data/processed and
    # data/qdrant to the same relative location under WORKDIR /app).
    raw_pdf_path: Path = Path("data/raw/FILANTE_manualfull_2603.pdf")
    processed_dir: Path = Path("data/processed")
    eval_dir: Path = Path("data/eval")

    # vector store (Qdrant local on-disk mode — no server needed for dev;
    # swap to QdrantClient(url=...) for a real deployment, see vector_store.py)
    qdrant_path: Path = Path("data/qdrant")
    qdrant_collection: str = "filante_manual"

    default_language: str = "ko"

    @property
    def languages(self) -> dict[str, LanguageConfig]:
        prompts_dir = PACKAGE_ROOT / "generation" / "prompts"
        return {
            "ko": LanguageConfig(
                code="ko",
                embedding_model="BAAI/bge-m3",
                prompt_template_path=prompts_dir / "ko.yaml",
                system_locale="ko-KR",
            ),
            # Same embedding_model as "ko" — the underlying index is never
            # re-embedded per language. BGE-M3 is cross-lingual, so an
            # English query already retrieves the right Korean chunks
            # against the *same* collection; only the prompt pack (which
            # instructs Claude to read Korean sources, write English
            # answers) differs.
            "en": LanguageConfig(
                code="en",
                embedding_model="BAAI/bge-m3",
                prompt_template_path=prompts_dir / "en.yaml",
                system_locale="en-US",
            ),
        }

    @property
    def allowed_origins_list(self) -> list[str]:
        return [origin.strip() for origin in self.allowed_origins.split(",") if origin.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
