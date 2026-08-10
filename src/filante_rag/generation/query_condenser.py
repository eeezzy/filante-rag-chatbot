"""Rewrites a follow-up question into a standalone query using conversation
history, so retrieval embeds something complete rather than a bare
pronoun reference (e.g. "그건 얼마나 자주 확인해야 해?" — "how often should
THAT be checked?").

Uses Haiku rather than the main generation model: this is a cheap
rewriting task, not knowledge-intensive generation, so a smaller/faster
model keeps the added latency (this runs *before* retrieval, blocking the
start of the stream) as small as possible.
"""

from __future__ import annotations

from typing import Any

import anthropic

CONDENSER_MODEL = "claude-haiku-4-5"

_SYSTEM_PROMPT = (
    "대화 기록을 참고하여, 사용자의 마지막 질문을 대화 맥락 없이도 이해할 수 있는 "
    "독립적인 한국어 질문으로 다시 작성하십시오.\n\n"
    "규칙:\n"
    "- 마지막 질문이 이전 대화의 대상을 가리키는 지시어(그것, 그거, 이거 등)나 "
    "생략된 주어를 포함하는 경우에만, 이전 대화를 참고해 그 지시어/주어를 "
    "명확하게 채워 넣으십시오.\n"
    "- 마지막 질문이 이전 대화와 관련 없는 새로운 주제라면, 이전 대화를 절대 "
    "참고하지 말고 마지막 질문을 그대로 반환하십시오. 새로운 주제를 이전 "
    "대화의 주제와 섞지 마십시오.\n"
    "- 질문의 의미나 주제를 절대 바꾸지 마십시오.\n"
    "- 다시 작성된 질문만 출력하고 다른 설명은 하지 마십시오.\n\n"
    "예시 1:\n"
    "[대화 기록] user: 타이어 공기압은 얼마로 유지해야 하나요?\n"
    "[마지막 질문] 그럼 그건 얼마나 자주 확인해야 해?\n"
    "[출력] 타이어 공기압은 얼마나 자주 확인해야 하나요?\n\n"
    "예시 2 (관련 없는 새 주제 — 이전 맥락을 섞지 않음):\n"
    "[대화 기록] user: 타이어 공기압은 얼마로 유지해야 하나요?\n"
    "[마지막 질문] 오늘 날씨 어때?\n"
    "[출력] 오늘 날씨 어때?"
)


class QueryCondenser:
    def __init__(self, client: anthropic.AsyncAnthropic, model: str = CONDENSER_MODEL) -> None:
        self.client = client
        self.model = model

    async def condense(self, history: list[dict], new_message: str, parent_span: Any) -> str:
        if not history:
            return new_message

        span = parent_span.start_observation(
            name="condense",
            as_type="generation",
            model=self.model,
            input={"history": history, "new_message": new_message},
        )
        try:
            history_text = "\n".join(f"{turn['role']}: {turn['content']}" for turn in history)
            message = await self.client.messages.create(
                model=self.model,
                max_tokens=256,
                system=_SYSTEM_PROMPT,
                messages=[
                    {
                        "role": "user",
                        "content": f"[대화 기록]\n{history_text}\n\n[마지막 질문]\n{new_message}",
                    }
                ],
            )
            text = "".join(b.text for b in message.content if b.type == "text").strip()
            result = text or new_message
            span.update(
                output=result,
                usage_details={
                    "input": message.usage.input_tokens,
                    "output": message.usage.output_tokens,
                },
            )
            return result
        finally:
            span.end()
