"""In-memory, per-process conversation history.

Deliberately simple: a real deployment with multiple workers/replicas would
need this backed by Redis (or similar) so sessions survive across
processes — swapping the storage is a one-class change since nothing else
touches session state directly.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from threading import Lock

MAX_TURNS = 6  # user+assistant pairs kept per session, bounds prompt growth
# Even with session_id validated as UUID-shaped (see api/schemas.py), a
# client could still send many distinct valid UUIDs to grow this dict
# unbounded — cap total sessions and evict the least-recently-used.
MAX_SESSIONS = 1000


@dataclass
class ConversationStore:
    _sessions: OrderedDict[str, list[dict]] = field(default_factory=OrderedDict)
    _lock: Lock = field(default_factory=Lock)

    def get_history(self, session_id: str) -> list[dict]:
        with self._lock:
            history = self._sessions.get(session_id)
            if history is None:
                return []
            self._sessions.move_to_end(session_id)
            return list(history)

    def append(self, session_id: str, role: str, content: str) -> None:
        with self._lock:
            if session_id not in self._sessions and len(self._sessions) >= MAX_SESSIONS:
                self._sessions.popitem(last=False)  # evict least-recently-used

            history = self._sessions.setdefault(session_id, [])
            self._sessions.move_to_end(session_id)
            history.append({"role": role, "content": content})
            overflow = len(history) - MAX_TURNS * 2
            if overflow > 0:
                del history[:overflow]
