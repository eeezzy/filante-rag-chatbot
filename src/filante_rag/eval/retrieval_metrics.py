"""Recall@k and MRR for the retriever against the golden set.

These are cheap (no LLM calls, just vector search) and deterministic, so
they're the first thing to check when retrieval quality regresses after a
chunking or embedding change.
"""

from __future__ import annotations

from dataclasses import dataclass

from filante_rag.retrieval.retriever import Retriever

K_VALUES = (1, 3, 5)


@dataclass
class RetrievalMetrics:
    recall_at_k: dict[int, float]
    mrr: float
    per_example_rank: list[int | None]  # None if not found within max(K_VALUES)


def evaluate_retrieval(golden_set: list[dict], retriever: Retriever) -> RetrievalMetrics:
    max_k = max(K_VALUES)
    ranks: list[int | None] = []

    for example in golden_set:
        results = retriever.search(example["question"], top_k=max_k).results
        retrieved_ids = [r.chunk_id for r in results]
        try:
            rank = retrieved_ids.index(example["expected_chunk_id"]) + 1
        except ValueError:
            rank = None
        ranks.append(rank)

    n = len(golden_set)
    recall_at_k = {
        k: sum(1 for r in ranks if r is not None and r <= k) / n for k in K_VALUES
    }
    mrr = sum(1 / r for r in ranks if r is not None) / n

    return RetrievalMetrics(recall_at_k=recall_at_k, mrr=mrr, per_example_rank=ranks)
