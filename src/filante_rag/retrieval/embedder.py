"""Embedding interface + a self-hosted BGE-M3 implementation.

`Embedder` is a Protocol (structural typing) rather than an ABC: any object
with these methods works, so swapping in a different model or a managed API
later means writing one new class, not touching the retriever or indexer.

BGE-M3 produces a dense vector *and* a sparse lexical-weight vector from the
same forward pass, so we capture both now even though the first retriever
pass only uses dense similarity — the sparse vectors are stored alongside so
hybrid (dense+sparse) search is a query-time change later, not a re-index.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass
class EmbeddingResult:
    dense: list[float]
    sparse: dict[int, float]  # token_id -> lexical weight


class Embedder(Protocol):
    dense_dim: int

    def embed_documents(self, texts: list[str]) -> list[EmbeddingResult]: ...
    def embed_query(self, text: str) -> EmbeddingResult: ...


class BGEM3Embedder:
    """Self-hosted, runs on CPU. No API key, no per-token cost."""

    dense_dim = 1024

    def __init__(self, model_name: str = "BAAI/bge-m3") -> None:
        from FlagEmbedding import BGEM3FlagModel

        self._model = BGEM3FlagModel(model_name, use_fp16=False)

    def _encode(self, texts: list[str]) -> list[EmbeddingResult]:
        out = self._model.encode(texts, return_dense=True, return_sparse=True)
        results = []
        for dense_vec, lexical_weights in zip(out["dense_vecs"], out["lexical_weights"]):
            sparse = {int(token_id): weight for token_id, weight in lexical_weights.items()}
            results.append(EmbeddingResult(dense=dense_vec.tolist(), sparse=sparse))
        return results

    def embed_documents(self, texts: list[str]) -> list[EmbeddingResult]:
        return self._encode(texts)

    def embed_query(self, text: str) -> EmbeddingResult:
        return self._encode([text])[0]
