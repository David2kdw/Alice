# services/llm_director.py
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

from Domain.constants import MessageRole, SessionMode
from Domain.schemas import Plan, plan_json_schema
from Services.orchestrator import DecisionContext
from Infra.openai_responses_client import OpenAIResponsesClient


@dataclass
class LLMDirector:
    client: OpenAIResponsesClient

    system_prompt: str = (
        "You are the Director for a mobile chat agent named Alice; output ONLY strict Plan JSON (all fields, "
        "no extras). IMPORTANT: bubble_count_target means how many separate chat bubbles/messages Alice will send in "
        "this single turn (like texting); it is normal to send multiple short bubbles in one turn. "
        "HARD RULES: if allowed_to_speak=false or user_batch_open=true => STAY_SILENT; if "
        "action=SPEAK then bubble_count_target must be an integer 1..max_bubbles_hard; be concise, don’t invent "
        "memories, use RETRIEVED_SNIPPETS only if relevant."
        "When WAITING, it is the situation where the user hasn't respond Alice yet. "
        "When ACTIVE, it is the situation where Alice hasn't respond user yet."
        "Be smart about when to stay silent. Don't always try to find something to talk about."
        "Be creative, you are THE DIRECTOR"
    )

    def decide(self, ctx: DecisionContext) -> Plan:
        perms = ctx.perms
        cap = max(1, int(perms.max_bubbles_this_turn))

        def silent(intent: str) -> Plan:
            # Keep bubble_count_target >= 1 to avoid schema/validation edge-cases
            return Plan(
                action="STAY_SILENT",
                speak_mode="idle",
                message_type="none",
                intent=intent,
                topic="",
                max_bubbles_hard=cap,
                bubble_count_target=1,
                length="short",
                use_memory_ids=[],
                cooldown_seconds=0,
                confidence=0.0,
            )

        def fallback_after_error(err: str) -> Plan:
            # If user just finished, at least reply once; otherwise stay silent.
            if perms.user_turn_end_ready:
                return Plan(
                    action="SPEAK",
                    speak_mode="active" if perms.mode == SessionMode.ACTIVE else "idle",
                    message_type="reply",
                    intent=f"fallback_after_error:{err}",
                    topic="ack",
                    max_bubbles_hard=cap,
                    bubble_count_target=1,
                    length="short",
                    use_memory_ids=[],
                    cooldown_seconds=30,
                    confidence=0.2,
                )
            return silent(f"fallback_after_error:{err}")

        # ---- Hard local guards (save cost + guarantee invariants) ----
        if not perms.allowed_to_speak:
            return silent(f"not_allowed:{perms.reason}")

        if perms.user_batch_open:
            return silent("user_batch_open")

        # Never self-start before first user message (same as RuleDirectorV0)
        has_user_spoken = any(m.role == MessageRole.USER for m in ctx.recent_dialogue)
        if not has_user_spoken:
            return silent("await_first_user_message")

        # Build compact dialogue window
        recent_msgs: List[Dict[str, str]] = []
        for m in ctx.recent_dialogue[-20:]:
            role = "user" if m.role == MessageRole.USER else "assistant"
            recent_msgs.append({"role": role, "content": m.text})

        retrieved = "\n".join(f"- {t}" for t in (ctx.retrieved_texts or [])[:6]) or "(none)"

        state_hint = (
            "NOW_STATE:\n"
            f"- mode: {perms.mode.value}\n"
            f"- allowed_to_speak: {perms.allowed_to_speak}\n"
            f"- user_batch_open: {perms.user_batch_open}\n"
            f"- user_turn_end_ready: {perms.user_turn_end_ready}\n"
            f"- reason: {perms.reason}\n"
            f"- max_bubbles_hard: {cap}\n"
        )

        user_instruction = (
            state_hint
            + "\nRETRIEVED_SNIPPETS:\n"
            + retrieved
            + "\n\nReturn a Plan JSON."
        )

        input_messages: List[Dict[str, str]] = [{"role": "system", "content": self.system_prompt}]
        input_messages.extend(recent_msgs)
        input_messages.append({"role": "user", "content": user_instruction})

        text_format: Dict[str, Any] = {
            "type": "json_schema",
            "name": "alice_plan",
            "strict": True,
            "schema": plan_json_schema(),
        }

        # ---- Call LLM + parse/validate with fallback ----
        try:
            raw = self.client.create_text(input_messages=input_messages, text_format=text_format)
            obj = json.loads(raw)
            plan = Plan.model_validate(obj)
        except Exception as e:
            return fallback_after_error(repr(e))

        # ---- Enforce caps and normalize fields regardless of what LLM said ----
        plan = plan.model_copy(update={"max_bubbles_hard": cap})

        if plan.action == "STAY_SILENT":
            # normalize to safe silent defaults
            return plan.model_copy(update={"bubble_count_target": 1, "cooldown_seconds": 0})

        # SPEAK: clamp bubble target
        n = int(plan.bubble_count_target)
        n = max(1, min(n, cap))
        plan = plan.model_copy(update={"bubble_count_target": n})

        # Speak-mode sanity:
        # - IDLE initiation should be idle
        if perms.mode == SessionMode.IDLE and plan.message_type == "idle_initiation":
            plan = plan.model_copy(update={"speak_mode": "idle"})
        # - WAITING followup should not be idle
        if perms.mode == SessionMode.WAITING and plan.message_type == "followup":
            plan = plan.model_copy(update={"speak_mode": "active"})

        return plan
