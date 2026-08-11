"""Orchestrates the interactive, multi-turn, streaming request path:
condense follow-up -> retrieve -> guardrail gate -> stream generation ->
record turn in session history.

Kept separate from RAGPipeline (used by eval/CLI): streaming needs an
async-iterator API that yields incremental events, not a single return
value, and needs session state RAGPipeline has no notion of.

Langfuse tracing here uses the explicit start_observation()/.end() API
rather than start_as_current_observation()'s implicit context-manager
form — the latter relies on OTel's ambient context propagation, which was
found (via a live test against Langfuse Cloud) to silently drop nested
spans across `await` boundaries in this asyncio setup: 0/2 observations
landed with the context-manager form vs. 2/2 with explicit parent/child
span objects passed directly. Filed as a real, reproducible finding, not
an assumption.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Literal

from langfuse import get_client

from filante_rag.generation.guardrails import any_safety_warning, has_sufficient_context
from filante_rag.generation.query_condenser import QueryCondenser
from filante_rag.generation.session_store import ConversationStore
from filante_rag.generation.streaming_generator import StreamingGenerator, parse_cited_sources
from filante_rag.retrieval.retriever import Retriever

logger = logging.getLogger(__name__)
langfuse_client = get_client()


@dataclass
class CitedSource:
    number: int
    section_title: str | None
    printed_page_start: int | None
    printed_page_end: int | None


@dataclass
class StreamEvent:
    type: Literal["delta", "done"]
    text: str = ""
    cited_sources: list[CitedSource] = field(default_factory=list)
    has_safety_warning: bool = False


@dataclass
class StreamingPipeline:
    retriever: Retriever
    condenser: QueryCondenser
    generator: StreamingGenerator
    session_store: ConversationStore
    # Keyed by language code (see config/settings.py's `languages`), not a
    # single fixed dict — the retriever/condenser/generator are already
    # language-agnostic (shared Claude client, cross-lingual embeddings),
    # so per-request language support is *only* a matter of picking which
    # prompt pack's strings to use for that request.
    prompt_packs: dict[str, dict]
    default_language: str = "ko"

    async def ask_stream(
        self, session_id: str, message: str, language: str | None = None, top_k: int = 5
    ) -> AsyncIterator[StreamEvent]:
        request_start = time.monotonic()
        prompt_pack = self.prompt_packs.get(
            language or self.default_language, self.prompt_packs[self.default_language]
        )
        root_span = langfuse_client.start_observation(
            name="chat_request",
            as_type="span",
            input=message,
            # Native session-grouping via update(session_id=...) was tested
            # and did not persist in this SDK version — kept in metadata
            # instead, still fully filterable/visible in the dashboard.
            metadata={"session_id": session_id},
        )
        try:
            history = self.session_store.get_history(session_id)

            condense_start = time.monotonic()
            standalone_query = await self.condenser.condense(history, message, root_span)
            if history:
                logger.info(
                    "condense_completed",
                    extra={
                        "event": "condense_completed",
                        "session_id": session_id,
                        "original_message": message,
                        "standalone_query": standalone_query,
                        "latency_ms": round((time.monotonic() - condense_start) * 1000),
                    },
                )

            # Retriever is sync/CPU-bound (BGE-M3 embedding + Qdrant search)
            # — run off the event loop so one request doesn't stall every
            # other concurrent request for the ~0.5-1s this takes.
            retrieval_span = root_span.start_observation(
                name="retrieval", as_type="retriever", input=standalone_query
            )
            retrieval_start = time.monotonic()
            retrieval = await asyncio.to_thread(self.retriever.search, standalone_query, top_k)
            results = retrieval.results
            retrieval_span.update(
                output=[r.chunk_id for r in results],
                metadata={"dense_relevance_score": retrieval.dense_relevance_score},
            )
            retrieval_span.end()
            logger.info(
                "retrieval_completed",
                extra={
                    "event": "retrieval_completed",
                    "session_id": session_id,
                    "query": standalone_query,
                    "dense_relevance_score": retrieval.dense_relevance_score,
                    "result_count": len(results),
                    "top_chunk_ids": [r.chunk_id for r in results[:5]],
                    "latency_ms": round((time.monotonic() - retrieval_start) * 1000),
                },
            )

            if not has_sufficient_context(retrieval.dense_relevance_score):
                logger.info(
                    "guardrail_rejected",
                    extra={
                        "event": "guardrail_rejected",
                        "session_id": session_id,
                        "query": standalone_query,
                        "dense_relevance_score": retrieval.dense_relevance_score,
                    },
                )
                answer = prompt_pack["no_context_message"].strip()
                yield StreamEvent(type="delta", text=answer)
                self.session_store.append(session_id, "user", message)
                self.session_store.append(session_id, "assistant", answer)
                root_span.update(output=answer, metadata={"guardrail_rejected": True})
                yield StreamEvent(type="done", text=answer)
                return

            parts: list[str] = []
            try:
                async for delta in self.generator.stream(
                    standalone_query,
                    results,
                    history,
                    root_span,
                    system_prompt=prompt_pack["streaming_system_prompt"],
                    sources_label=prompt_pack["sources_label"],
                    question_label=prompt_pack["question_label"],
                    session_id=session_id,
                ):
                    parts.append(delta)
                    yield StreamEvent(type="delta", text=delta)
            except Exception:
                # Deltas already sent can't be recalled — append an honest
                # note rather than pretending the response completed cleanly.
                logger.exception(
                    "generation_failed", extra={"event": "generation_failed", "session_id": session_id}
                )
                error_text = prompt_pack["error_message"].strip()
                root_span.update(output=error_text, level="ERROR")
                yield StreamEvent(type="delta", text=error_text)
                yield StreamEvent(type="done", text=error_text)
                return

            full_text = "".join(parts)

            # Based on what the answer actually *cited*, not everything
            # retrieval happened to surface — checking all retrieved
            # chunks (the original approach) meant an unrelated warning
            # elsewhere in the same section (e.g. asking how to turn on
            # the AC, retrieving a nearby chunk that also contains a
            # "don't block the vents" caution the answer never mentions)
            # would still trigger the disclaimer. That's not "safer", it's
            # crying wolf — the disclaimer stops meaning anything once it
            # shows up on ordinary questions too.
            cited_numbers = [n for n in parse_cited_sources(full_text) if 1 <= n <= len(results)]
            cited_results = [results[n - 1] for n in cited_numbers]
            warning = any_safety_warning(cited_results)

            if warning:
                disclaimer = prompt_pack["safety_disclaimer"]
                yield StreamEvent(type="delta", text=disclaimer)
                full_text += disclaimer

            self.session_store.append(session_id, "user", message)
            self.session_store.append(session_id, "assistant", full_text)

            cited_sources = [
                CitedSource(
                    number=n,
                    section_title=r.payload.get("section_title"),
                    printed_page_start=r.payload.get("printed_page_start"),
                    printed_page_end=r.payload.get("printed_page_end"),
                )
                for n, r in zip(cited_numbers, cited_results)
            ]
            root_span.update(
                output=full_text,
                metadata={"has_safety_warning": warning, "cited_source_count": len(cited_sources)},
            )
            logger.info(
                "request_completed",
                extra={
                    "event": "request_completed",
                    "session_id": session_id,
                    "has_safety_warning": warning,
                    "cited_source_count": len(cited_sources),
                    "total_latency_ms": round((time.monotonic() - request_start) * 1000),
                },
            )
            yield StreamEvent(
                type="done",
                text=full_text,
                cited_sources=cited_sources,
                has_safety_warning=warning,
            )
        finally:
            # No per-request flush() — that forces a blocking synchronous
            # export and would add latency to every response. The SDK's
            # background thread batches and exports continuously on its
            # own; flush() belongs at process shutdown (see api/main.py),
            # not in the request path.
            root_span.end()
