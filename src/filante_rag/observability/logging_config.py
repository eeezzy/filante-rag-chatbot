"""JSON-structured logging, so log lines can be grepped/parsed as data
(e.g. `jq 'select(.event=="generation_completed")'`) instead of scraped as
free text. No external dependency — this is the zero-cost first layer of
observability; Langfuse tracing sits on top of it for request-level traces.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone

_BASE_RECORD_KEYS = set(logging.LogRecord("", 0, "", 0, "", None, None).__dict__.keys())


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in _BASE_RECORD_KEYS:
                payload[key] = value
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


def setup_logging(level: int = logging.INFO) -> None:
    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level)
