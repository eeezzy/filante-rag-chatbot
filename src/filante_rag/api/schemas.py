from __future__ import annotations

import uuid

from pydantic import BaseModel, Field, field_validator

# A generous cap for a manual-lookup question, not a hard technical limit —
# mainly guards against pasted walls of text driving up token cost/latency
# per request.
MAX_MESSAGE_LENGTH = 2000


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=MAX_MESSAGE_LENGTH)
    session_id: str | None = None

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
