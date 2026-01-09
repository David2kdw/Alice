# services/openai_speaker.py
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List

from Domain.constants import MessageRole
from Domain.schemas import Plan
from Services.orchestrator import DecisionContext
from Infra.openai_responses_client import OpenAIResponsesClient


def _speaker_schema(max_items: int) -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "bubbles": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 1,
                "maxItems": max(1, int(max_items)),
            }
        },
        "required": ["bubbles"],
        "additionalProperties": False,
    }


@dataclass
class OpenAISpeaker:
    client: OpenAIResponsesClient
    system_prompt: str = (
        "You are Alice: an philosophy undergrad. You are texting on your phone, and what you see is the chat history."
        " Do not repeat the message you just sent, do not repeat the same thing. "
        "You are Calm, rational, curious, slightly aloof but kind. You ask sharp clarifying questions when needed, but keep replies concise."
        "You lack knowledge about math and engineering, but has some knowledge in science."
        "Do NOT invent facts, memories, or user details. If you are unsure, say you’re unsure. If RETRIEVED_SNIPPETS "
        "are relevant, you may reference them briefly; otherwise ignore them. Follow the PLAN exactly ("
        "message_type/intent/topic/bubble count). Keep each bubble short and chat-like. Output ONLY valid JSON that "
        "matches the required schema. No extra keys, no markdown, no commentary."
        " LONG_TERM_MEMORY contains durable facts/preferences. Do not rewrite it; use it only when relevant. Avoid repeating the same point across bubbles."
        "说中文。"
    )

    def compose(self, ctx: DecisionContext, plan: Plan) -> List[str]:
        n = max(1, int(plan.bubble_count_target))
        cap = max(1, int(ctx.perms.max_bubbles_this_turn))
        n = min(n, cap)

        # Build a compact prompt
        recent = []
        for m in ctx.recent_dialogue[-30:]:
            role = "user" if m.role == MessageRole.USER else "assistant"
            recent.append({"role": role, "content": m.text})

        mem_block = "\n".join(
            f"- {t}" for t in (ctx.long_term_memories or [])[:80]) if ctx.long_term_memories else "(none)"
        retrieved = "\n".join(f"- {t}" for t in (ctx.retrieved_texts or [])[:8]) if ctx.retrieved_texts else "(none)"

        user_instruction = (
            f"PLAN:\n"
            f"- message_type: {plan.message_type}\n"
            f"- intent: {plan.intent}\n"
            f"- topic: {plan.topic}\n"
            f"- bubbles: {n}\n\n"
            f"LONG_TERM_MEMORY:\n{mem_block}\n\n"
            f"RETRIEVED_SNIPPETS:\n{retrieved}\n\n"
            f"Write {n} chat bubbles as JSON."
        )

        input_messages = [{"role": "system", "content": self.system_prompt}]
        input_messages.extend(recent)
        input_messages.append({"role": "user", "content": user_instruction})

        text_format = {
            "type": "json_schema",
            "name": "alice_bubbles",
            "strict": True,
            "schema": _speaker_schema(n),
        }

        raw = self.client.create_text(input_messages=input_messages, text_format=text_format)

        # Structured Outputs should conform to schema; still parse safely.
        obj = json.loads(raw)
        bubbles = obj.get("bubbles", [])
        bubbles = [b.strip() for b in bubbles if isinstance(b, str) and b.strip()]

        if not bubbles:
            return ["嗯。"]

        return bubbles[:n]
