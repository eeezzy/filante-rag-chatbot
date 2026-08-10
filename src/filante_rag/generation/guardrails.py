"""Guardrail logic kept separate from the generator so thresholds and
disclaimer wording can be tuned/tested without touching the Claude call.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from filante_rag.retrieval.vector_store import SearchResult

# Empirically calibrated against dense cosine similarity: off-topic queries
# (weather, recipes, movies) top out around 0.46 against this corpus, while
# genuinely relevant queries in our sanity check ranged 0.59-0.80. 0.5 sits
# cleanly between them.
#
# This must be checked against RetrievalResponse.dense_relevance_score, NOT
# a hybrid/fused SearchResult.score — RRF fusion scores are rank-based, not
# a calibrated similarity scale, and mixing the two silently breaks this
# gate (measured: an off-topic query scored 0.83 post-fusion, well above a
# genuinely relevant query's 0.5).
MIN_RELEVANCE_SCORE = 0.5


def has_sufficient_context(dense_relevance_score: float) -> bool:
    return dense_relevance_score >= MIN_RELEVANCE_SCORE


def any_safety_warning(results: list[SearchResult]) -> bool:
    return any(r.payload.get("contains_safety_warning") for r in results)


def load_prompt_pack(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))
