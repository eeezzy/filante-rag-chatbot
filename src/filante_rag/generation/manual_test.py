"""End-to-end manual test of the full RAG pipeline: retrieval + guardrails +
Claude generation. Covers an answerable question, a safety-critical one
(checks the disclaimer gets appended), and an off-topic one (checks the
no-context guardrail fires instead of hallucinating).
"""

from __future__ import annotations

from filante_rag.generation.pipeline import build_default_pipeline

QUERIES = [
    "타이어 공기압은 얼마로 유지해야 하나요?",
    "동승석에 어린이용 보조시트를 장착해도 되나요?",
    "오늘 날씨 어때?",
]


def main() -> None:
    pipeline = build_default_pipeline()
    for query in QUERIES:
        print("=" * 70)
        print("Q:", query)
        result = pipeline.ask(query)
        print(f"answerable={result.answerable} safety_warning={result.has_safety_warning}")
        print("A:", result.answer)
        if result.sources:
            print("sources:", [s.payload["section_title"] for s in result.sources])


if __name__ == "__main__":
    main()
