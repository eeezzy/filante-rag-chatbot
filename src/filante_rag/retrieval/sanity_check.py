"""Manual retrieval sanity check against a handful of realistic queries
spanning different chapters, to eyeball quality before wiring up generation.
"""

from __future__ import annotations

from filante_rag.config.settings import get_settings
from filante_rag.retrieval.embedder import BGEM3Embedder
from filante_rag.retrieval.retriever import Retriever
from filante_rag.retrieval.vector_store import QdrantVectorStore

QUERIES = [
    "타이어 공기압은 얼마로 유지해야 하나요?",
    "스마트키 배터리가 방전되면 어떻게 하나요?",
    "동승석에 어린이용 보조시트를 장착해도 되나요?",
    "엔진 오일은 언제 교환해야 하나요?",
    "사고가 났을 때 어떻게 해야 하나요?",
]


def main() -> None:
    settings = get_settings()
    embedder = BGEM3Embedder()
    store = QdrantVectorStore.from_local_path(
        settings.qdrant_path, settings.qdrant_collection, embedder.dense_dim
    )
    retriever = Retriever(embedder=embedder, vector_store=store)

    for query in QUERIES:
        print("=" * 70)
        print("Q:", query)
        for r in retriever.search(query, top_k=3).results:
            p = r.payload
            print(
                f"  [{r.score:.3f}] {p['section_title']} "
                f"(printed p.{p['printed_page_start']}-{p['printed_page_end']})"
            )
            print("   ", p["text"][:150].replace("\n", " "))


if __name__ == "__main__":
    main()
