"""Embeds chunks.jsonl with BGE-M3 and upserts into the local Qdrant store.

Batched (16 chunks/call) since encoding is the bottleneck on CPU; batching
amortizes the model's fixed per-call overhead versus one chunk at a time.
"""

from __future__ import annotations

import json
from pathlib import Path

from tqdm import tqdm

from filante_rag.config.settings import get_settings
from filante_rag.retrieval.embedder import BGEM3Embedder
from filante_rag.retrieval.vector_store import QdrantVectorStore, VectorRecord

BATCH_SIZE = 16


def load_chunks(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def main() -> None:
    settings = get_settings()
    chunks = load_chunks(settings.processed_dir / "chunks.jsonl")
    print(f"Loaded {len(chunks)} chunks")

    embedder = BGEM3Embedder()
    store = QdrantVectorStore.from_local_path(
        settings.qdrant_path, settings.qdrant_collection, embedder.dense_dim
    )

    for i in tqdm(range(0, len(chunks), BATCH_SIZE), desc="embedding+indexing"):
        batch = chunks[i : i + BATCH_SIZE]
        embeddings = embedder.embed_documents([c["text"] for c in batch])
        records = [
            VectorRecord(chunk_id=c["chunk_id"], dense=e.dense, sparse=e.sparse, payload=c)
            for c, e in zip(batch, embeddings)
        ]
        store.upsert(records)

    print(f"Indexed {store.count()} points into collection '{settings.qdrant_collection}'")


if __name__ == "__main__":
    main()
