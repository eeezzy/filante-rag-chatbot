"""Vector store interface + a Qdrant implementation.

Qdrant runs in local on-disk mode by default (`QdrantClient(path=...)`) —
no Docker, no server, no account needed, which keeps this a zero-infra dev
setup. Swapping to a real deployment later (Docker container or Qdrant
Cloud) is a one-line change to `QdrantClient(url=...)` in `from_settings`;
nothing else in the codebase touches the client directly.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Protocol

from qdrant_client import QdrantClient, models

_ID_NAMESPACE = uuid.UUID("2f3c9c1a-2b0e-4a7e-9c3b-6a1b1a5b2b2f")
_DENSE_VECTOR_NAME = "dense"
_SPARSE_VECTOR_NAME = "sparse"
# Each of dense/sparse search fetches this many candidates before RRF fusion
# narrows to top_k — standard practice so fusion has enough to work with.
_PREFETCH_MULTIPLIER = 4


@dataclass
class VectorRecord:
    chunk_id: str
    dense: list[float]
    sparse: dict[int, float]
    payload: dict


@dataclass
class SearchResult:
    chunk_id: str
    score: float
    payload: dict


class VectorStore(Protocol):
    def upsert(self, records: list[VectorRecord]) -> None: ...
    def search(
        self, dense_vector: list[float], sparse_vector: dict[int, float] | None, top_k: int
    ) -> list[SearchResult]: ...
    def count(self) -> int: ...


def _point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(_ID_NAMESPACE, chunk_id))


class QdrantVectorStore:
    def __init__(self, client: QdrantClient, collection_name: str, dense_dim: int) -> None:
        self._client = client
        self._collection = collection_name
        self._ensure_collection(dense_dim)

    @classmethod
    def from_local_path(cls, path, collection_name: str, dense_dim: int) -> "QdrantVectorStore":
        return cls(QdrantClient(path=str(path)), collection_name, dense_dim)

    def _ensure_collection(self, dense_dim: int) -> None:
        if self._client.collection_exists(self._collection):
            return
        self._client.create_collection(
            collection_name=self._collection,
            vectors_config={
                _DENSE_VECTOR_NAME: models.VectorParams(
                    size=dense_dim, distance=models.Distance.COSINE
                )
            },
            sparse_vectors_config={_SPARSE_VECTOR_NAME: models.SparseVectorParams()},
        )

    def upsert(self, records: list[VectorRecord]) -> None:
        points = [
            models.PointStruct(
                id=_point_id(r.chunk_id),
                vector={
                    _DENSE_VECTOR_NAME: r.dense,
                    _SPARSE_VECTOR_NAME: models.SparseVector(
                        indices=list(r.sparse.keys()), values=list(r.sparse.values())
                    ),
                },
                payload={**r.payload, "chunk_id": r.chunk_id},
            )
            for r in records
        ]
        self._client.upsert(collection_name=self._collection, points=points)

    def search(
        self,
        dense_vector: list[float],
        sparse_vector: dict[int, float] | None = None,
        top_k: int = 5,
    ) -> list[SearchResult]:
        if sparse_vector:
            # Hybrid: fetch top candidates from each of dense (semantic
            # similarity) and sparse (exact lexical match) search, then fuse
            # with Reciprocal Rank Fusion. This is what catches exact
            # keyword/part-number lookups (e.g. a specific fuse name) that
            # dense-only similarity can miss — confirmed by eval.
            prefetch_limit = top_k * _PREFETCH_MULTIPLIER
            hits = self._client.query_points(
                collection_name=self._collection,
                prefetch=[
                    models.Prefetch(
                        query=dense_vector, using=_DENSE_VECTOR_NAME, limit=prefetch_limit
                    ),
                    models.Prefetch(
                        query=models.SparseVector(
                            indices=list(sparse_vector.keys()),
                            values=list(sparse_vector.values()),
                        ),
                        using=_SPARSE_VECTOR_NAME,
                        limit=prefetch_limit,
                    ),
                ],
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                limit=top_k,
            ).points
        else:
            hits = self._client.query_points(
                collection_name=self._collection,
                query=dense_vector,
                using=_DENSE_VECTOR_NAME,
                limit=top_k,
            ).points

        return [
            SearchResult(chunk_id=h.payload["chunk_id"], score=h.score, payload=h.payload)
            for h in hits
        ]

    def count(self) -> int:
        return self._client.count(self._collection).count
