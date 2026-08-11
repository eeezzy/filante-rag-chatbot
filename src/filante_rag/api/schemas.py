from __future__ import annotations

import uuid

from pydantic import BaseModel, Field, field_validator

from filante_rag.config.settings import get_settings

# A generous cap for a manual-lookup question, not a hard technical limit —
# mainly guards against pasted walls of text driving up token cost/latency
# per request.
MAX_MESSAGE_LENGTH = 2000


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=MAX_MESSAGE_LENGTH)
    session_id: str | None = None
    # None -> StreamingPipeline falls back to its configured default
    # language. Validated against config/settings.py's `languages` dict
    # rather than hardcoded here, so adding a language is still a
    # config-only change — this file doesn't need to know the list.
    language: str | None = None

    @field_validator("language")
    @classmethod
    def _language_must_be_configured(cls, value: str | None) -> str | None:
        if value is not None and value not in get_settings().languages:
            raise ValueError(f"Unsupported language: {value!r}")
        return value

    @field_validator("session_id")
    @classmethod
    def _session_id_must_be_uuid(cls, value: str | None) -> str | None:
        # Session IDs are dict keys in the in-memory ConversationStore
        # (see session_store.py) — accepting arbitrary client-supplied
        # strings would let a client grow that store with unbounded junk
        # keys. Requiring UUID shape keeps it to what the server itself
        # issues (see api/main.py's session_id = uuid.uuid4() default).
        if value is not None:
            try:
                uuid.UUID(value)
            except ValueError as exc:
                raise ValueError("session_id must be a valid UUID") from exc
        return value
