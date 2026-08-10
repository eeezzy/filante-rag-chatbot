"""Claude wrapper that turns retrieved chunks + a question into a grounded
answer. Uses a forced tool call (rather than parsing free text) so we get
an explicit list of which sources were actually used — the same
structured-output pattern used during ingestion. Answerability is derived
from whether "answer" is non-empty rather than a separate boolean field;
see the comment in `generate()` for why.
"""

from __future__ import annotations

from dataclasses import dataclass

import anthropic
from tenacity import retry, stop_after_attempt, wait_exponential

from filante_rag.retrieval.vector_store import SearchResult

_TOOL_NAME = "answer_with_citations"

_TOOL_SCHEMA = {
    "name": _TOOL_NAME,
    "description": "Record the answer to the user's question, grounded in the provided sources.",
    "input_schema": {
        "type": "object",
        "properties": {
            "answer": {
                "type": "string",
                "description": (
                    "Korean answer grounded in the sources. Must be exactly an empty "
                    "string if the sources are insufficient to answer — never explain "
                    "the refusal here, an empty string is the refusal signal."
                ),
            },
            "cited_source_numbers": {
                "type": "array",
                "items": {"type": "integer"},
                "description": "The [출처 N] numbers actually used to compose the answer",
            },
        },
        "required": ["answer", "cited_source_numbers"],
    },
}


@dataclass
class GenerationResult:
    answerable: bool
    answer: str
    cited_source_numbers: list[int]


def format_sources(sources: list[SearchResult]) -> str:
    blocks = []
    for i, r in enumerate(sources, start=1):
        p = r.payload
        pages = f"{p['printed_page_start']}-{p['printed_page_end']}"
        blocks.append(f"[출처 {i}] {p['section_title']} (매뉴얼 {pages}페이지)\n{p['text']}")
    return "\n\n".join(blocks)


class ClaudeGenerator:
    def __init__(self, client: anthropic.Anthropic, model: str, system_prompt: str) -> None:
        self.client = client
        self.model = model
        self._system_prompt = system_prompt

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=2, max=20))
    def generate(self, query: str, sources: list[SearchResult]) -> GenerationResult:
        context = format_sources(sources)
        message = self.client.messages.create(
            model=self.model,
            # Claude emits "answer" before "answerable"/"cited_source_numbers"
            # in the tool call regardless of schema property order, so a
            # verbose answer hitting a tight max_tokens truncates the JSON
            # before those trailing fields arrive — silently, not as an
            # error. Generous headroom here, not tighter parsing, is the fix.
            max_tokens=4096,
            system=self._system_prompt,
            tools=[_TOOL_SCHEMA],
            tool_choice={"type": "tool", "name": _TOOL_NAME},
            messages=[
                {
                    "role": "user",
                    "content": f"출처:\n{context}\n\n질문: {query}",
                }
            ],
        )
        if message.stop_reason == "max_tokens":
            # Retry rather than silently treating a cut-off response as
            # "unanswerable" — that would misreport content that exists in
            # the manual as missing from it.
            raise RuntimeError("Generation truncated at max_tokens; retrying")

        tool_use = next(b for b in message.content if b.type == "tool_use")
        payload = tool_use.input

        # Measured empirically: even with tool_choice forcing this schema,
        # a separate "answerable" boolean was missing from the payload in
        # 5/8 identical repeated calls (stop_reason was "tool_use", not
        # truncation — Claude just treated that field as droppable). The
        # "answer" string was present and correct in all 8, so answerability
        # is derived from it instead of relying on a second field that
        # isn't reliably produced.
        answer = payload.get("answer", "").strip()
        answerable = bool(answer)

        return GenerationResult(
            answerable=answerable,
            answer=answer,
            cited_source_numbers=payload.get("cited_source_numbers", []) if answerable else [],
        )
