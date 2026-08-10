"""Thin query-time wrapper: embed the query, search the vector store.

Kept separate from `Embedder`/`VectorStore` so a later upgrade (reranking,
metadata filtering) changes only this file.
"""

from __future__ import annotations

from dataclasses import dataclass

from filante_rag.retrieval.embedder import Embedder
from filante_rag.retrieval.vector_store import SearchResult, VectorStore


@dataclass
class RetrievalResponse:
    results: list[SearchResult]
    # Dense-only cosine similarity of the top result — NOT the same scale as
    # `results[i].score` when hybrid fusion is used. RRF fusion scores are
    # rank-based, so a single strong lexical match can score higher than a
    # genuinely relevant semantic match (measured: an off-topic "오늘 날씨
    # 어때?" scored 0.83 post-fusion, higher than a legitimately relevant
    # query's 0.5). The "is this even relevant" guardrail needs the
    # calibrated cosine-similarity scale, so it's tracked separately here
    # rather than reusing whatever scale `results` happens to be in.
    dense_relevance_score: float


@dataclass
class Retriever:
    embedder: Embedder
    vector_store: VectorStore

    def search(self, query: str, top_k: int = 5) -> RetrievalResponse:
        embedded = self.embedder.embed_query(query)
        results = self.vector_store.search(embedded.dense, embedded.sparse, top_k=top_k)
        dense_only = self.vector_store.search(embedded.dense, None, top_k=1)
        relevance = dense_only[0].score if dense_only else 0.0
        return RetrievalResponse(results=results, dense_relevance_score=relevance)
