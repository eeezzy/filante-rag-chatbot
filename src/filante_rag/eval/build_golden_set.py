"""Generates a golden QA set for retrieval/generation evaluation.

For each sampled chunk, asks Claude for a natural question that *this*
specific passage (and not just any page) answers, plus a concise reference
answer — the standard synthetic-QA-from-context approach for building RAG
eval sets when no human-labeled set exists yet.

Sampling is stratified by chapter (proportional to each chapter's chunk
count) rather than a flat random sample, so a large chapter like "주행"
doesn't crowd out small ones like "서비스 제도" in the eval set.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import anthropic
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm import tqdm

from filante_rag.config.settings import get_settings

TARGET_TOTAL = 40
SEED = 42

_TOOL_NAME = "record_qa_pair"
_TOOL_SCHEMA = {
    "name": _TOOL_NAME,
    "description": "Record one evaluation question-answer pair for a manual passage.",
    "input_schema": {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": (
                    "A natural, realistic Korean question a vehicle owner might ask, "
                    "answerable specifically from this passage — avoid questions so "
                    "generic they could be answered by many other pages."
                ),
            },
            "reference_answer": {
                "type": "string",
                "description": "A concise, factually correct Korean answer, based only on this passage.",
            },
        },
        "required": ["question", "reference_answer"],
    },
}

_SYSTEM_PROMPT = (
    "당신은 RAG 챗봇 평가를 위한 질문-답변 쌍을 생성하는 어시스턴트입니다. "
    "주어진 차량 사용설명서 발췌문을 바탕으로, 실제 차량 소유자가 물어볼 법한 "
    "자연스러운 한국어 질문 하나와 그에 대한 간결한 정답을 생성하십시오. "
    "질문은 반드시 이 발췌문의 구체적인 내용에 근거해야 합니다."
)


@dataclass
class GoldenExample:
    question: str
    reference_answer: str
    expected_chunk_id: str
    chapter_num: int | None
    section_title: str | None
    contains_safety_warning: bool


def load_chunks(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def stratified_sample(chunks: list[dict], target_total: int, seed: int = SEED) -> list[dict]:
    by_chapter: dict[int | None, list[dict]] = defaultdict(list)
    for c in chunks:
        by_chapter[c["chapter_num"]].append(c)

    rng = random.Random(seed)
    total = len(chunks)
    sample = []
    for group in by_chapter.values():
        n = max(1, round(target_total * len(group) / total))
        sample.extend(rng.sample(group, min(n, len(group))))
    return sample


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=2, max=20))
def generate_qa_pair(client: anthropic.Anthropic, model: str, chunk: dict) -> tuple[str, str]:
    message = client.messages.create(
        model=model,
        max_tokens=512,
        system=_SYSTEM_PROMPT,
        tools=[_TOOL_SCHEMA],
        tool_choice={"type": "tool", "name": _TOOL_NAME},
        messages=[{"role": "user", "content": f"발췌문:\n{chunk['text']}"}],
    )
    tool_use = next(b for b in message.content if b.type == "tool_use")
    payload = tool_use.input
    # Same defensive lesson as the generator: don't assume every field
    # survives the tool call, verify and skip rather than crash.
    return payload.get("question", "").strip(), payload.get("reference_answer", "").strip()


def main() -> None:
    settings = get_settings()
    chunks = load_chunks(settings.processed_dir / "chunks.jsonl")
    sample = stratified_sample(chunks, TARGET_TOTAL)
    print(f"Sampled {len(sample)} chunks across {len(set(c['chapter_num'] for c in sample))} chapters")

    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    examples = []
    for chunk in tqdm(sample, desc="generating QA pairs"):
        question, answer = generate_qa_pair(client, settings.generation_model, chunk)
        if not question or not answer:
            continue
        examples.append(
            GoldenExample(
                question=question,
                reference_answer=answer,
                expected_chunk_id=chunk["chunk_id"],
                chapter_num=chunk["chapter_num"],
                section_title=chunk["section_title"],
                contains_safety_warning=chunk["contains_safety_warning"],
            )
        )

    out_path = settings.eval_dir / "golden_set.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for e in examples:
            f.write(json.dumps(asdict(e), ensure_ascii=False) + "\n")
    print(f"Wrote {len(examples)} golden examples to {out_path}")


if __name__ == "__main__":
    main()
