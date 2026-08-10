"""Ties retrieval, guardrails, and generation into the single entry point
the CLI/API will call. Keeping this orchestration in one place (rather than
inline in an API route) means the API layer, a CLI, and eval scripts can
all share the exact same request path.
"""

from __future__ import annotations

from dataclasses import dataclass

import anthropic

from filante_rag.config.settings import Settings, get_settings
from filante_rag.generation.generator import ClaudeGenerator
from filante_rag.generation.guardrails import (
    any_safety_warning,
    has_sufficient_context,
    load_prompt_pack,
)
from filante_rag.retrieval.embedder import BGEM3Embedder
from filante_rag.retrieval.retriever import Retriever
from filante_rag.retrieval.vector_store import QdrantVectorStore, SearchResult


@dataclass
class AnswerResult:
    query: str
    answer: str
    answerable: bool
    sources: list[SearchResult]
    has_safety_warning: bool


@dataclass
class RAGPipeline:
    retriever: Retriever
    generator: ClaudeGenerator
    prompt_pack: dict

    def ask(self, query: str, top_k: int = 5) -> AnswerResult:
        retrieval = self.retriever.search(query, top_k=top_k)
        results = retrieval.results

        if not has_sufficient_context(retrieval.dense_relevance_score):
            return AnswerResult(
                query=query,
                answer=self.prompt_pack["no_context_message"].strip(),
                answerable=False,
                sources=[],
                has_safety_warning=False,
            )

        try:
            generated = self.generator.generate(query, results)
        except Exception:
            # A real API/generation failure is not the same claim as "the
            # manual doesn't cover this" — conflating them would misinform
            # the user about what the manual actually contains.
            return AnswerResult(
                query=query,
                answer=self.prompt_pack["error_message"].strip(),
                answerable=False,
                sources=[],
                has_safety_warning=False,
            )

        if not generated.answerable:
            return AnswerResult(
                query=query,
                answer=self.prompt_pack["no_context_message"].strip(),
                answerable=False,
                sources=[],
                has_safety_warning=False,
            )

        cited = [
            results[n - 1] for n in generated.cited_source_numbers if 1 <= n <= len(results)
        ] or results

        warning = any_safety_warning(cited)
        answer_text = generated.answer
        if warning:
            answer_text += self.prompt_pack["safety_disclaimer"]

        return AnswerResult(
            query=query,
            answer=answer_text,
            answerable=True,
            sources=cited,
            has_safety_warning=warning,
        )


def build_default_pipeline(settings: Settings | None = None, language: str = "ko") -> RAGPipeline:
    settings = settings or get_settings()
    lang_config = settings.languages[language]

    embedder = BGEM3Embedder(lang_config.embedding_model)
    vector_store = QdrantVectorStore.from_local_path(
        settings.qdrant_path, settings.qdrant_collection, embedder.dense_dim
    )
    retriever = Retriever(embedder=embedder, vector_store=vector_store)

    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    prompt_pack = load_prompt_pack(lang_config.prompt_template_path)
    generator = ClaudeGenerator(
        client=client, model=settings.generation_model, system_prompt=prompt_pack["system_prompt"]
    )

    return RAGPipeline(retriever=retriever, generator=generator, prompt_pack=prompt_pack)
