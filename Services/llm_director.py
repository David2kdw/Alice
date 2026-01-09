# services/llm_director.py
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import List, Dict, Any

from Domain.constants import MessageRole, SessionMode
from Domain.schemas import Plan, MemoryOps, plan_json_schema
from Services.orchestrator import DecisionContext
from Infra.openai_responses_client import OpenAIResponsesClient


@dataclass
class LLMDirector:
    client: OpenAIResponsesClient

    # NOTE:
    # - OpenAI strict json_schema requires ALL keys present (including memory_ops keys).
    # - We therefore treat "memory_ops.action=NONE" as the default safe state.
    system_prompt: str = (
        "You are the Director for a mobile chat agent named Alice. Output ONLY strict Plan JSON (all fields, no extras). "
        "Interpret bubble_count_target as the number of separate chat bubbles/messages Alice will send in this single turn (like texting). "
        "HARD RULES: if allowed_to_speak=false OR user_batch_open=true => action=STAY_SILENT. "
        "If action=SPEAK then bubble_count_target must be an integer 1..max_bubbles_hard and <= max_bubbles_hard. "
        "Do not invent memories or facts; use RETRIEVED_SNIPPETS only if relevant.\n\n"
        "IMPORTANT: The Plan includes memory_ops (long-term memory decision). "
        "DEFAULT memory_ops.action=NONE. Only set memory_ops.action=WRITE if the information is likely useful for weeks/months, "
        "stable, and safe to store. Store ONLY concise content (1–2 short sentences), no transient chat, no repeated paraphrases, "
        "no sensitive/private data. Prefer durable user or Alice preferences, constraints, long-term goals, ongoing projects, or repeated patterns. "
        "If WRITE: title and content must be non-empty and short; tags small list; importance 0..1.\n\n"
        "When memory_ops.action=WRITE, avoid duplicates of LONG_TERM_MEMORY; if already present, choose NONE.\n"
        "Be smart about when to stay silent; do not spam proactive followups. Prefer STAY_SILENT unless there is clear value."
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
                memory_ops=MemoryOps(),  # explicit safe default
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
                    memory_ops=MemoryOps(),  # never write memory on fallback
                )
            return silent(f"fallback_after_error:{err}")

        # ---- Hard local guards (save cost + guarantee invariants) ----
        if not perms.allowed_to_speak:
            return silent(f"not_allowed:{perms.reason}")

        if perms.user_batch_open:
            return silent("user_batch_open")

        # Never self-start before first user message
        has_user_spoken = any(m.role == MessageRole.USER for m in ctx.recent_dialogue)
        if not has_user_spoken:
            return silent("await_first_user_message")

        # Build compact dialogue window
        recent_msgs: List[Dict[str, str]] = []
        for m in ctx.recent_dialogue[-20:]:
            role = "user" if m.role == MessageRole.USER else "assistant"
            recent_msgs.append({"role": role, "content": m.text})

        mem_block = "\n".join(f"- {t}" for t in (ctx.long_term_memories or [])[:200]) or "(none)"
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
                + "\nLONG_TERM_MEMORY:\n"
                + mem_block
                + "\n\nRETRIEVED_SNIPPETS:\n"
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

        # ---- memory_ops sanitation (Director-controlled, but we still guard) ----
        ops = plan.memory_ops or MemoryOps()

        def _clamp01(x: float) -> float:
            try:
                v = float(x)
            except Exception:
                return 0.0
            return 0.0 if v < 0.0 else (1.0 if v > 1.0 else v)

        # Safety policy:
        # - Never write memory if STAY_SILENT (reduces accidental writes on proactive/idle ticks).
        # - Never write memory on fallback-like message_types (optional, conservative).
        if plan.action != "SPEAK":
            ops = MemoryOps()
        else:
            if ops.action == "WRITE":
                title = (ops.title or "").strip()
                content = (ops.content or "").strip()

                # If missing essentials, downgrade to NONE
                if not title or not content:
                    ops = MemoryOps()
                else:
                    # Hard length caps to keep memory concise and safe to feed every turn
                    title = title.replace("\n", " ").strip()[:60]
                    content = content.replace("\n", " ").strip()[:280]

                    tags = list(ops.tags or [])
                    # Keep tags small and clean
                    cleaned_tags: List[str] = []
                    for t in tags:
                        tt = str(t).strip()
                        if not tt:
                            continue
                        tt = tt.replace("\n", " ")[:20]
                        if tt not in cleaned_tags:
                            cleaned_tags.append(tt)
                        if len(cleaned_tags) >= 6:
                            break

                    ops = MemoryOps(
                        action="WRITE",
                        title=title,
                        content=content,
                        tags=cleaned_tags,
                        importance=_clamp01(ops.importance),
                    )
            else:
                ops = MemoryOps()

        plan = plan.model_copy(update={"memory_ops": ops})

        # ---- Action-specific normalization ----
        if plan.action == "STAY_SILENT":
            return plan.model_copy(update={"bubble_count_target": 1, "cooldown_seconds": 0})

        # SPEAK: clamp bubble target
        n = int(plan.bubble_count_target)
        n = max(1, min(n, cap))
        plan = plan.model_copy(update={"bubble_count_target": n})

        # Speak-mode sanity
        if perms.mode == SessionMode.IDLE and plan.message_type == "idle_initiation":
            plan = plan.model_copy(update={"speak_mode": "idle"})
        if perms.mode == SessionMode.WAITING and plan.message_type == "followup":
            plan = plan.model_copy(update={"speak_mode": "active"})

        return plan
