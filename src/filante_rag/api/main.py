"""FastAPI app exposing the streaming, multi-turn chat pipeline over SSE.

Local-mode Qdrant (see vector_store.py) is single-process, so this must
run with a single worker (`uvicorn ... --workers 1`) — a real deployment
would move Qdrant to a server and could then scale workers freely.
"""

from __future__ import annotations

import asyncio
import json
import secrets
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import anthropic
from fastapi import Depends, FastAPI, Header, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from langfuse import get_client as get_langfuse_client
from starlette.middleware.base import BaseHTTPMiddleware

from filante_rag.api.schemas import ChatRequest
from filante_rag.config.settings import get_settings
from filante_rag.generation.guardrails import load_prompt_pack
from filante_rag.generation.query_condenser import QueryCondenser
from filante_rag.generation.session_store import ConversationStore
from filante_rag.generation.streaming_generator import StreamingGenerator
from filante_rag.generation.streaming_pipeline import StreamingPipeline
from filante_rag.observability.logging_config import setup_logging
from filante_rag.retrieval.embedder import BGEM3Embedder
from filante_rag.retrieval.retriever import Retriever
from filante_rag.retrieval.vector_store import QdrantVectorStore

setup_logging()

MAX_BODY_BYTES = 10_000  # generous for a short chat message + session_id


async def verify_api_key(x_api_key: str = Header(default="")) -> None:
    settings = get_settings()
    if not settings.api_shared_secret:
        # Fail closed: an unset secret is a deployment misconfiguration,
        # not an invitation to run the endpoint open.
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Server is missing API_SHARED_SECRET configuration",
        )
    if not secrets.compare_digest(x_api_key, settings.api_shared_secret):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or missing API key")


KEEPALIVE_INTERVAL_SECONDS = 5.0


async def _with_keepalive(source: AsyncIterator[str]) -> AsyncIterator[str]:
    """Injects an SSE comment (ignored by clients) if `source` goes quiet
    for more than KEEPALIVE_INTERVAL_SECONDS — e.g. during condense +
    retrieval, before the first generated token exists to send. Paired
    with the leading padding comment in event_source(): padding gets a
    proxy to start flushing, regular small writes are what keeps it
    flushing instead of re-buffering during a quiet stretch.
    """
    aiter = source.__aiter__()
    pending: asyncio.Task | None = None
    try:
        while True:
            if pending is None:
                pending = asyncio.ensure_future(aiter.__anext__())
            try:
                item = await asyncio.wait_for(asyncio.shield(pending), timeout=KEEPALIVE_INTERVAL_SECONDS)
            except asyncio.TimeoutError:
                yield ": keepalive\n\n"
                continue
            except StopAsyncIteration:
                return
            pending = None
            yield item
    finally:
        if pending is not None:
            pending.cancel()


class BodySizeLimitMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > MAX_BODY_BYTES:
            return JSONResponse({"detail": "Request body too large"}, status_code=413)
        return await call_next(request)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    lang_config = settings.languages[settings.default_language]

    # Embedding + vector search stay sync (see streaming_pipeline.py, run
    # via asyncio.to_thread); only the Claude-calling components need the
    # async client so the event loop stays free during streaming.
    embedder = BGEM3Embedder(lang_config.embedding_model)
    vector_store = QdrantVectorStore.from_local_path(
        settings.qdrant_path, settings.qdrant_collection, embedder.dense_dim
    )
    retriever = Retriever(embedder=embedder, vector_store=vector_store)

    async_client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)
    # One prompt pack per configured language (see config/settings.py) —
    # the retriever/condenser/generator below are shared across all of
    # them; only the prompt strings vary per request.
    prompt_packs = {
        code: load_prompt_pack(cfg.prompt_template_path) for code, cfg in settings.languages.items()
    }
    condenser = QueryCondenser(client=async_client)
    generator = StreamingGenerator(client=async_client, model=settings.generation_model)

    app.state.pipeline = StreamingPipeline(
        retriever=retriever,
        condenser=condenser,
        generator=generator,
        session_store=ConversationStore(),
        prompt_packs=prompt_packs,
        default_language=settings.default_language,
    )
    yield
    # Langfuse batches/exports traces on a background thread; flush once
    # here so nothing buffered is lost on shutdown (not done per-request —
    # see streaming_pipeline.py for why).
    get_langfuse_client().flush()


app = FastAPI(title="FILANTE RAG API", lifespan=lifespan)

app.add_middleware(BodySizeLimitMiddleware)

# "*" (the local-dev default) must be replaced via ALLOWED_ORIGINS before
# any non-local deployment — see config/settings.py.
app.add_middleware(
    CORSMiddleware,
    allow_origins=get_settings().allowed_origins_list,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/chat/stream", dependencies=[Depends(verify_api_key)])
async def chat_stream(req: ChatRequest) -> StreamingResponse:
    pipeline: StreamingPipeline = app.state.pipeline
    session_id = req.session_id or str(uuid.uuid4())

    async def event_source() -> AsyncIterator[str]:
        # Cloud Run's front-end proxy has been observed (empirically, via
        # repeated live tests) to intermittently buffer the *entire*
        # response — even with Cache-Control: no-transform set — and only
        # flush it once at the end or after a long delay, defeating SSE
        # entirely for roughly 2 of 3 requests in testing. A leading
        # padding comment is a well-established workaround for exactly
        # this class of proxy-buffering behavior: many proxies only start
        # actively flushing once enough bytes have been written, so
        # padding past that threshold up front forces streaming to begin
        # immediately instead of waiting to accumulate a full buffer.
        # SSE comment lines (leading ":") are ignored by clients.
        yield f": {' ' * 2048}\n\n"
        yield f"data: {json.dumps({'type': 'session', 'session_id': session_id})}\n\n"
        async for event in pipeline.ask_stream(session_id, req.message, language=req.language):
            payload = {"type": event.type, "text": event.text}
            if event.type == "done":
                payload["sources"] = [
                    {
                        "number": s.number,
                        "section_title": s.section_title,
                        "printed_page_start": s.printed_page_start,
                        "printed_page_end": s.printed_page_end,
                    }
                    for s in event.cited_sources
                ]
                payload["has_safety_warning"] = event.has_safety_warning
            yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        _with_keepalive(event_source()),
        media_type="text/event-stream",
        # Cloud Run's front-end transparently gzip-compresses responses
        # when the client sends Accept-Encoding (every browser, always) —
        # confirmed live: curl without that header streamed fine, curl
        # *with* it (matching what a browser sends) hung for 60s with zero
        # bytes received, reproducing the exact hang seen in the browser.
        # Compression needs to buffer to be effective, which defeats SSE's
        # whole "flush immediately" premise. `no-transform` is the
        # standards-compliant instruction to intermediate proxies not to
        # alter the response body at all.
        headers={"Cache-Control": "no-cache, no-transform", "X-Accel-Buffering": "no"},
    )
