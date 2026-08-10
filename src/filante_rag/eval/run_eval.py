"""Full eval run: retrieval metrics (cheap, deterministic) + generation
metrics (LLM-judged) against the golden set, via the real RAGPipeline —
not the idealized ground-truth chunk — so generation scores reflect what
the system actually retrieves and cites, imperfections included.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from tqdm import tqdm

from filante_rag.config.settings import get_settings
from filante_rag.eval.judge import Judgment, judge_answer
from filante_rag.eval.retrieval_metrics import RetrievalMetrics, evaluate_retrieval
from filante_rag.generation.generator import format_sources
from filante_rag.generation.pipeline import build_default_pipeline


@dataclass
class GenerationEvalRow:
    question: str
    answerable: bool
    expected_answerable: bool  # golden set questions should all be answerable
    faithfulness_score: int
    relevancy_score: int
    contains_safety_warning: bool


def load_golden_set(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def evaluate_generation(golden_set: list[dict], pipeline) -> list[GenerationEvalRow]:
    client = pipeline.generator.client
    model = pipeline.generator.model
    rows = []
    for example in tqdm(golden_set, desc="judging generation"):
        result = pipeline.ask(example["question"])
        if not result.answerable:
            rows.append(
                GenerationEvalRow(
                    question=example["question"],
                    answerable=False,
                    expected_answerable=True,
                    faithfulness_score=1,
                    relevancy_score=1,
                    contains_safety_warning=example["contains_safety_warning"],
                )
            )
            continue

        sources_text = format_sources(result.sources)
        judgment: Judgment = judge_answer(client, model, example["question"], sources_text, result.answer)
        rows.append(
            GenerationEvalRow(
                question=example["question"],
                answerable=True,
                expected_answerable=True,
                faithfulness_score=judgment.faithfulness_score,
                relevancy_score=judgment.relevancy_score,
                contains_safety_warning=example["contains_safety_warning"],
            )
        )
    return rows


def summarize(retrieval: RetrievalMetrics, generation: list[GenerationEvalRow]) -> dict:
    n = len(generation)
    answerable_rate = sum(r.answerable for r in generation) / n
    avg_faithfulness = sum(r.faithfulness_score for r in generation) / n
    avg_relevancy = sum(r.relevancy_score for r in generation) / n

    safety_rows = [r for r in generation if r.contains_safety_warning]
    safety_faithfulness = (
        sum(r.faithfulness_score for r in safety_rows) / len(safety_rows) if safety_rows else None
    )

    return {
        "n_examples": n,
        "retrieval": {
            "recall_at_k": retrieval.recall_at_k,
            "mrr": retrieval.mrr,
        },
        "generation": {
            "answerable_rate": answerable_rate,
            "avg_faithfulness": avg_faithfulness,
            "avg_relevancy": avg_relevancy,
            "avg_faithfulness_safety_only": safety_faithfulness,
            "n_safety_examples": len(safety_rows),
        },
    }


def main() -> None:
    settings = get_settings()
    golden_set = load_golden_set(settings.eval_dir / "golden_set.jsonl")
    pipeline = build_default_pipeline()

    retrieval = evaluate_retrieval(golden_set, pipeline.retriever)
    generation = evaluate_generation(golden_set, pipeline)
    summary = summarize(retrieval, generation)

    print(json.dumps(summary, ensure_ascii=False, indent=2))

    report_path = settings.eval_dir / "eval_report.json"
    report_path.write_text(
        json.dumps(
            {
                "summary": summary,
                "generation_rows": [asdict(r) for r in generation],
                "retrieval_ranks": retrieval.per_example_rank,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Wrote full report to {report_path}")


if __name__ == "__main__":
    main()
