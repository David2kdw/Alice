# Services/director.py
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional

from Domain.constants import SessionMode
from Domain.schemas import Plan


@dataclass
class DirectorConfig:
    """
    Rule-based Director behavior knobs.

    Probabilities are only used when the mode allows proactive speech
    (i.e., permissions.allowed_to_speak is already True).
    """
    # When responding to a finished user batch
    reply_bubbles_default: int = 2
    reply_cooldown_seconds: int = 60

    # ACTIVE proactive speech
    active_proactive_probability: float = 0.6
    active_proactive_bubbles: int = 1
    active_proactive_cooldown_seconds: int = 90

    # WAITING follow-up
    waiting_followup_probability: float = 0.8
    waiting_followup_bubbles: int = 1
    waiting_followup_cooldown_seconds: int = 120

    # IDLE initiation
    idle_initiation_probability: float = 0.25
    idle_initiation_bubbles: int = 1
    idle_initiation_cooldown_seconds: int = 300


class RuleDirector:
    """
    Minimal rule-based Director.

    Contract:
    - Only produces a Plan (no user-facing text).
    - Must obey ctx.perms.max_bubbles_this_turn via max_bubbles_hard + bubble_count_target.
    """

    def __init__(self, cfg: Optional[DirectorConfig] = None, *, seed: Optional[int] = None):
        self.cfg = cfg or DirectorConfig()
        self._rng = random.Random(seed)

    def decide(self, ctx) -> Plan:
        perms = ctx.perms
        mode = perms.mode

        # Orchestrator calls Director only when allowed_to_speak is True,
        # but we still guard here for safety.
        if not perms.allowed_to_speak:
            return self._plan_silent(perms, reason="not_allowed")

        # Highest priority: user finished batch -> reply.
        if perms.user_turn_end_ready:
            bubble_target = min(self.cfg.reply_bubbles_default, perms.max_bubbles_this_turn)
            bubble_target = max(1, bubble_target)
            return Plan(
                action="SPEAK",
                speak_mode="active" if mode == SessionMode.ACTIVE else "idle",
                message_type="reply",
                intent="respond after user finished batch",
                topic="reply_to_user",
                max_bubbles_hard=perms.max_bubbles_this_turn,
                bubble_count_target=bubble_target,
                length="short",
                use_memory_ids=[],
                cooldown_seconds=self.cfg.reply_cooldown_seconds,
                confidence=0.7,
            )

        # Mode-specific proactive behavior
        if mode == SessionMode.WAITING:
            # Follow-up in WAITING should be rare-ish and short
            if self._rng.random() > self.cfg.waiting_followup_probability:
                return self._plan_silent(perms, reason="waiting_skip_followup")
            bubble_target = min(self.cfg.waiting_followup_bubbles, perms.max_bubbles_this_turn)
            bubble_target = max(1, bubble_target)
            return Plan(
                action="SPEAK",
                speak_mode="followup",
                message_type="followup",
                intent="light follow-up while waiting for user reply",
                topic="nudge",
                max_bubbles_hard=perms.max_bubbles_this_turn,
                bubble_count_target=bubble_target,
                length="short",
                use_memory_ids=[],
                cooldown_seconds=self.cfg.waiting_followup_cooldown_seconds,
                confidence=0.5,
            )

        if mode == SessionMode.ACTIVE:
            if self._rng.random() > self.cfg.active_proactive_probability:
                return self._plan_silent(perms, reason="active_skip_proactive")
            bubble_target = min(self.cfg.active_proactive_bubbles, perms.max_bubbles_this_turn)
            bubble_target = max(1, bubble_target)
            return Plan(
                action="SPEAK",
                speak_mode="active",
                message_type="share",
                intent="casual companion speech in active chat",
                topic="small_talk",
                max_bubbles_hard=perms.max_bubbles_this_turn,
                bubble_count_target=bubble_target,
                length="short",
                use_memory_ids=[],
                cooldown_seconds=self.cfg.active_proactive_cooldown_seconds,
                confidence=0.4,
            )

        if mode == SessionMode.IDLE:
            if self._rng.random() > self.cfg.idle_initiation_probability:
                return self._plan_silent(perms, reason="idle_skip_initiation")
            bubble_target = min(self.cfg.idle_initiation_bubbles, perms.max_bubbles_this_turn)
            bubble_target = max(1, bubble_target)
            return Plan(
                action="SPEAK",
                speak_mode="idle",
                message_type="check_in",
                intent="initiate chat lightly in idle mode",
                topic="check_in",
                max_bubbles_hard=perms.max_bubbles_this_turn,
                bubble_count_target=bubble_target,
                length="short",
                use_memory_ids=[],
                cooldown_seconds=self.cfg.idle_initiation_cooldown_seconds,
                confidence=0.3,
            )

        return self._plan_silent(perms, reason="unknown_mode")

    def _plan_silent(self, perms, reason: str) -> Plan:
        return Plan(
            action="STAY_SILENT",
            speak_mode="idle",
            message_type="none",
            intent=reason,
            topic="",
            max_bubbles_hard=perms.max_bubbles_this_turn,
            bubble_count_target=1,
            length="short",
            use_memory_ids=[],
            cooldown_seconds=60,
            confidence=0.0,
        )
