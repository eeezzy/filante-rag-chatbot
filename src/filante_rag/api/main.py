"""FastAPI app exposing the streaming, multi-turn chat pipeline over SSE.

Local-mode Qdrant (see vector_store.py) is single-process, so this must
run with a single worker (`uvicorn ... --workers 1`) — a real deployment
would move Qdrant to a server and could then scale workers freely.
"""

from __future__ import annotations

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
    prompt_pack = load_prompt_pack(lang_config.prompt_template_path)
    condenser = QueryCondenser(client=async_client)
    generator = StreamingGenerator(
        client=async_client,
        model=settings.generation_model,
        system_prompt=prompt_pack["streaming_system_prompt"],
    )

    app.state.pipeline = StreamingPipeline(
        retriever=retriever,
        condenser=condenser,
        generator=generator,
        session_store=ConversationStore(),
        prompt_pack=prompt_pack,
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
        yield f"data: {json.dumps({'type': 'session', 'session_id': session_id})}\n\n"
        async for event in pipeline.ask_stream(session_id, req.message):
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

    return StreamingResponse(event_source(), media_type="text/event-stream")
