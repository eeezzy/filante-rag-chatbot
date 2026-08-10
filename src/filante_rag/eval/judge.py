"""LLM-as-judge for generation quality: faithfulness (no hallucinated
claims beyond the cited sources) and relevancy (does the answer address
the question). Same forced tool-call pattern as the rest of the codebase.

Uses integer scores rather than a boolean pass/fail — we've seen this model
occasionally drop a lone boolean field from a tool call while adjacent
string/array fields survive (see generator.py), so scores default to the
worst case (1) rather than silently upgrading a missing field, which could
mask a systemic parsing problem in aggregate results.
"""

from __future__ import annotations

from dataclasses import dataclass

import anthropic
from tenacity import retry, stop_after_attempt, wait_exponential

_TOOL_NAME = "record_judgment"
_TOOL_SCHEMA = {
    "name": _TOOL_NAME,
    "description": "Record a faithfulness and relevancy judgment for a generated answer.",
    "input_schema": {
        "type": "object",
        "properties": {
            "faithfulness_score": {
                "type": "integer",
                "description": (
                    "1-5: 생성된 답변의 모든 주장이 제공된 출처로 뒷받침되는 정도. "
                    "5=출처에 없는 내용(환각)이 전혀 없음, 1=대부분 근거 없는 내용."
                ),
            },
            "relevancy_score": {
                "type": "integer",
                "description": (
                    "1-5: 생성된 답변이 질문에 실제로 답하는 정도. "
                    "5=질문에 정확하고 완전하게 답함, 1=질문과 무관함."
                ),
            },
            "reasoning": {
                "type": "string",
                "description": "판단 근거를 한두 문장으로 설명",
            },
        },
        "required": ["faithfulness_score", "relevancy_score", "reasoning"],
    },
}

_SYSTEM_PROMPT = (
    "당신은 RAG 챗봇의 답변 품질을 평가하는 엄격한 평가자입니다. "
    "[질문], [출처], [생성된 답변]을 보고 충실성(faithfulness)과 "
    "관련성(relevancy)을 각각 1~5점으로 평가하십시오. "
    "출처에 없는 내용을 답변이 지어냈다면 충실성 점수를 낮게 주십시오."
)


@dataclass
class Judgment:
    faithfulness_score: int
    relevancy_score: int
    reasoning: str


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=2, max=20))
def judge_answer(
    client: anthropic.Anthropic,
    model: str,
    question: str,
    sources_text: str,
    generated_answer: str,
) -> Judgment:
    message = client.messages.create(
        model=model,
        max_tokens=512,
        system=_SYSTEM_PROMPT,
        tools=[_TOOL_SCHEMA],
        tool_choice={"type": "tool", "name": _TOOL_NAME},
        messages=[
            {
                "role": "user",
                "content": (
                    f"[질문]\n{question}\n\n[출처]\n{sources_text}\n\n"
                    f"[생성된 답변]\n{generated_answer}"
                ),
            }
        ],
    )
    tool_use = next(b for b in message.content if b.type == "tool_use")
    payload = tool_use.input

    def _score(key: str) -> int:
        value = payload.get(key)
        return value if isinstance(value, int) and 1 <= value <= 5 else 1

    return Judgment(
        faithfulness_score=_score("faithfulness_score"),
        relevancy_score=_score("relevancy_score"),
        reasoning=payload.get("reasoning", ""),
    )
