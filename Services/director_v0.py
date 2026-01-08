# alice/services/director_v0.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from Domain.constants import SessionMode
from Domain.schemas import Plan
from Services.orchestrator import DecisionContext
from Domain.constants import MessageRole

@dataclass
class RuleDirectorV0:
    """
    Deterministic Director v0.

    Goals:
    - If user finished a batch (user_turn_end_ready): reply.
    - Else if WAITING and allowed_to_speak: send followup (bounded by SessionManager permissions).
    - Else if IDLE and allowed_to_speak: idle initiation.
    - Else: stay silent.

    Notes:
    - Orchestrator already gates on perms.allowed_to_speak.
    - We still respect perms.max_bubbles_this_turn via max_bubbles_hard.
    """

    # default bubble targets (will be capped by max_bubbles_hard downstream)
    reply_bubbles: int = 2
    followup_bubbles: int = 1
    idle_bubbles: int = 1

    # default cooldowns (seconds)
    reply_cooldown_s: int = 0
    followup_cooldown_s: int = 45
    idle_cooldown_s: int = 90

    def decide(self, ctx: DecisionContext) -> Plan:
        perms = ctx.perms
        mode = perms.mode
        cap = max(1, int(perms.max_bubbles_this_turn))

        # Helper to make plans consistently
        def speak(
            *,
            speak_mode: str,
            message_type: str,
            intent: str,
            topic: str,
            bubble_target: int,
            cooldown_seconds: int,
            confidence: float,
            length: str = "short",
        ) -> Plan:
            return Plan(
                action="SPEAK",
                speak_mode=speak_mode,
                message_type=message_type,
                intent=intent,
                topic=topic,
                max_bubbles_hard=cap,
                bubble_count_target=max(1, min(int(bubble_target), cap)),
                length=length,
                use_memory_ids=[],
                cooldown_seconds=int(cooldown_seconds),
                confidence=float(confidence),
            )

        def silent(reason: str) -> Plan:
            return Plan(
                action="STAY_SILENT",
                speak_mode="idle",
                message_type="none",
                intent=reason,
                topic="",
                max_bubbles_hard=cap,
                bubble_count_target=0,
                length="short",
                use_memory_ids=[],
                cooldown_seconds=0,
                confidence=0.0,
            )

        has_user_spoken = any(m.role == MessageRole.USER for m in ctx.recent_dialogue)
        if not has_user_spoken:
            return silent("await_first_user_message")

        # 1) User finished batch -> reply (highest priority)
        if perms.user_turn_end_ready:
            return speak(
                speak_mode="active" if mode == SessionMode.ACTIVE else "idle",
                message_type="reply",
                intent="respond after user finished batch",
                topic="reply",
                bubble_target=self.reply_bubbles,
                cooldown_seconds=self.reply_cooldown_s,
                confidence=0.9,
                length="short",
            )

        # 2) Proactive followup in WAITING
        if mode == SessionMode.WAITING:
            return speak(
                speak_mode="active",
                message_type="followup",
                intent="follow up while waiting for user reply",
                topic="followup",
                bubble_target=self.followup_bubbles,
                cooldown_seconds=self.followup_cooldown_s,
                confidence=0.6,
                length="short",
            )

        # 3) Idle initiation in IDLE
        if mode == SessionMode.IDLE:
            return speak(
                speak_mode="idle",
                message_type="idle_initiation",
                intent="start a light conversation while idle",
                topic="idle",
                bubble_target=self.idle_bubbles,
                cooldown_seconds=self.idle_cooldown_s,
                confidence=0.4,
                length="short",
            )

        # 4) ACTIVE but not user_turn_end_ready: usually stay silent (user still typing or no reason)
        if mode == SessionMode.ACTIVE:
            return silent("active_no_turn_end_ready")

        # Fallback
        return silent("no_action")
