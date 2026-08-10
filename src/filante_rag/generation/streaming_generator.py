"""Plain-text streaming generation with inline [출처N] citation markers,
parsed out via regex once the stream completes.

This is a separate path from ClaudeGenerator's forced tool-use: a forced
tool call returns one JSON blob at the end, not incremental text, so it
can't drive a token-by-token streaming UI. Trading the tool-call's
reliability guarantees for streamability means citation extraction has to
happen a cheaper way — asking the model to mark citations inline and
regex-parsing them back out, rather than a second structured call.
"""

from __future__ import annotations

import logging
import re
import time
from collections.abc import AsyncIterator
from typing import Any

import anthropic

from filante_rag.generation.generator import format_sources
from filante_rag.retrieval.vector_store import SearchResult

logger = logging.getLogger(__name__)

_CITATION_RE = re.compile(r"\[출처\s*(\d+)\]")


def parse_cited_sources(text: str) -> list[int]:
    return sorted({int(n) for n in _CITATION_RE.findall(text)})


class StreamingGenerator:
    def __init__(self, client: anthropic.AsyncAnthropic, model: str, system_prompt: str) -> None:
        self.client = client
        self.model = model
        self._system_prompt = system_prompt

    async def stream(
        self,
        query: str,
        sources: list[SearchResult],
        history: list[dict],
        parent_span: Any,
        session_id: str | None = None,
    ) -> AsyncIterator[str]:
        context = format_sources(sources)
        messages = [
            *history,
            {"role": "user", "content": f"출처:\n{context}\n\n질문: {query}"},
        ]
        span = parent_span.start_observation(
            name="generation", as_type="generation", model=self.model, input=messages
        )
        start = time.monotonic()
        parts: list[str] = []
        try:
            async with self.client.messages.stream(
                model=self.model,
                max_tokens=4096,
                system=self._system_prompt,
                messages=messages,
            ) as stream:
                async for text in stream.text_stream:
                    parts.append(text)
                    yield text
                final_message = await stream.get_final_message()

            logger.info(
                "generation_completed",
                extra={
                    "event": "generation_completed",
                    "session_id": session_id,
                    "model": self.model,
                    "input_tokens": final_message.usage.input_tokens,
                    "output_tokens": final_message.usage.output_tokens,
                    "latency_ms": round((time.monotonic() - start) * 1000),
                },
            )
            span.update(
                output="".join(parts),
                usage_details={
                    "input": final_message.usage.input_tokens,
                    "output": final_message.usage.output_tokens,
                },
            )
        except Exception as exc:
            span.update(level="ERROR", status_message=str(exc))
            raise
        finally:
            span.end()
