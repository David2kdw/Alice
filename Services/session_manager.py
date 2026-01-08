# alice/services/session_manager.py
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional, Union

from Domain.constants import SessionMode
from Domain.models import (
    AliceMessageEvent,
    ChatState,
    TickEvent,
    UserMessageEvent,
)


# ----------------------------
# Config + Permissions
# ----------------------------

@dataclass(frozen=True)
class SessionConfig:
    """
    Pure timing/limits config for session logic.

    Notes:
    - user_batch_end_silence_seconds: user "batch" ends when no new user message arrives for this long.
    - detach_after_seconds: user considered detached if no user message for this long => IDLE.
    - active pacing: used as *guidance* and throttling; Director still decides content.
    """
    user_batch_end_silence_seconds: int = 5
    detach_after_seconds: int = 300

    # ACTIVE pacing / throttle
    active_min_gap_seconds: int = 0
    active_max_gap_seconds: int = 30
    active_max_bubbles_per_turn: int = 5
    active_max_followups_without_user_reply: int = 1
    active_followup_after_seconds: int = 20

    # IDLE initiation throttle (daily reset is handled elsewhere; see notes below)
    idle_daily_initiation_cap: int = 4
    idle_default_max_bubbles_per_turn: int = 2

    # Generic hard caps (safety)
    hard_max_bubbles_per_turn: int = 5


@dataclass(frozen=True)
class SpeakPermissions:
    """
    Output of SessionManager.permissions(): what is allowed *right now*.
    Director should obey these constraints.
    """
    allowed_to_speak: bool
    mode: SessionMode

    # Conversation constraints / guidance
    max_bubbles_this_turn: int
    min_gap_seconds: int
    max_gap_seconds: int

    # Signals for orchestrator/director
    user_turn_end_ready: bool
    user_batch_open: bool
    reason: str


Event = Union[UserMessageEvent, AliceMessageEvent, TickEvent]


# ----------------------------
# Session Manager
# ----------------------------

class SessionManager:
    """
    Manages conversation/session state with three core concepts:

    1) User batching:
       - user may send multiple messages quickly.
       - batch ends when silence >= user_batch_end_silence_seconds.
       - once batch ends -> user_turn_end_ready=True (Alice may take a turn).

    2) Session mode:
       - ACTIVE: user engaged recently (<= detach_after_seconds)
       - WAITING: Alice spoke; waiting for user reply / optional followup
       - IDLE: user detached (>= detach_after_seconds); low-frequency initiations

    3) Throttling:
       - next_allowed_speak_ts is a hard cooldown gate (set by orchestrator using plan.cooldown_seconds).
       - ACTIVE has min/max gap guidance and follow-up constraints.

    Important: This module is rule-based and must NOT call LLM or storage.
    """

    def __init__(self, cfg: Optional[SessionConfig] = None):
        self.cfg = cfg or SessionConfig()

    # ---------
    # Helpers
    # ---------

    def _within_detach_window(self, chat: ChatState, now: datetime) -> bool:
        if chat.user_last_ts is None:
            return False
        return (now - chat.user_last_ts).total_seconds() < self.cfg.detach_after_seconds

    def _should_detach(self, chat: ChatState, now: datetime) -> bool:
        if chat.user_last_ts is None:
            return True
        return (now - chat.user_last_ts).total_seconds() >= self.cfg.detach_after_seconds

    def _cooldown_allows(self, chat: ChatState, now: datetime) -> bool:
        if chat.next_allowed_speak_ts is None:
            return True
        return now >= chat.next_allowed_speak_ts

    def _time_since(self, ts: Optional[datetime], now: datetime) -> Optional[float]:
        if ts is None:
            return None
        return (now - ts).total_seconds()

    def _close_user_batch_if_needed(self, chat: ChatState, now: datetime) -> ChatState:
        """If batch is open and silence >= 30s, close it and mark user_turn_end_ready."""
        if not chat.user_batch_open or chat.user_batch_last_ts is None:
            return chat

        silence = (now - chat.user_batch_last_ts).total_seconds()
        if silence < self.cfg.user_batch_end_silence_seconds:
            return chat

        # Close batch -> Alice may take a turn
        return ChatState(
            mode=chat.mode,
            user_last_ts=chat.user_last_ts,
            alice_last_ts=chat.alice_last_ts,
            user_batch_open=False,
            user_batch_last_ts=chat.user_batch_last_ts,
            user_turn_end_ready=True,
            next_allowed_speak_ts=chat.next_allowed_speak_ts,
            followups_since_user_reply=chat.followups_since_user_reply,
            idle_initiations_today=chat.idle_initiations_today,
            active_turns_since_user_msg=chat.active_turns_since_user_msg,
        )

    # ---------
    # Public API
    # ---------

    def update(
        self,
        event: Event,
        chat: ChatState,
        now: datetime,
        *,
        cooldown_seconds: Optional[int] = None,
        is_idle_initiation: bool = False,
        is_followup: bool = False,
    ) -> ChatState:
        """
        Update ChatState based on an event.

        Parameters
        ----------
        event:
          - UserMessageEvent(text, ts): user sent a message.
          - AliceMessageEvent(messages, ts): Alice sent one "turn" (possibly multi-bubble).
          - TickEvent(ts): time passed; used to close user batches and detach.

        now:
          - current time (authoritative clock). We still respect event.ts for storing timestamps.

        cooldown_seconds:
          - Optional. If provided with AliceMessageEvent, sets next_allowed_speak_ts = event.ts + cooldown.
          - Recommended: orchestrator passes plan.cooldown_seconds here.

        is_idle_initiation:
          - If True and AliceMessageEvent occurs while in IDLE, increments idle_initiations_today.

        is_followup:
          - If True and AliceMessageEvent occurs while user hasn't replied, increments followups_since_user_reply.
        """
        cfg = self.cfg

        # 1) Tick: close user batch, detach if needed
        if isinstance(event, TickEvent):
            chat2 = self._close_user_batch_if_needed(chat, now)

            if self._should_detach(chat2, now):
                # Detach => IDLE. Also stop waiting on unfinished turn.
                return ChatState(
                    mode=SessionMode.IDLE,
                    user_last_ts=chat2.user_last_ts,
                    alice_last_ts=chat2.alice_last_ts,
                    user_batch_open=False,
                    user_batch_last_ts=chat2.user_batch_last_ts,
                    user_turn_end_ready=False,
                    next_allowed_speak_ts=chat2.next_allowed_speak_ts,
                    followups_since_user_reply=chat2.followups_since_user_reply,
                    idle_initiations_today=chat2.idle_initiations_today,
                    active_turns_since_user_msg=0,  # reset; user is detached
                )

            # If not detached, keep ACTIVE/WAITING as-is; batch close may flip readiness.
            # If user engaged recently, ensure mode is ACTIVE unless Alice is waiting.
            if self._within_detach_window(chat2, now) and chat2.mode == SessionMode.IDLE:
                # Rare: state was IDLE but user is actually within window (e.g., restored state).
                chat2 = ChatState(
                    mode=SessionMode.ACTIVE,
                    user_last_ts=chat2.user_last_ts,
                    alice_last_ts=chat2.alice_last_ts,
                    user_batch_open=chat2.user_batch_open,
                    user_batch_last_ts=chat2.user_batch_last_ts,
                    user_turn_end_ready=chat2.user_turn_end_ready,
                    next_allowed_speak_ts=chat2.next_allowed_speak_ts,
                    followups_since_user_reply=chat2.followups_since_user_reply,
                    idle_initiations_today=chat2.idle_initiations_today,
                    active_turns_since_user_msg=chat2.active_turns_since_user_msg,
                )
            return chat2

        # 2) User message: open/extend batch, set ACTIVE, clear followup counter
        if isinstance(event, UserMessageEvent):
            ts = event.ts

            return ChatState(
                mode=SessionMode.ACTIVE,
                user_last_ts=ts,
                alice_last_ts=chat.alice_last_ts,
                user_batch_open=True,
                user_batch_last_ts=ts,
                user_turn_end_ready=False,  # user is still speaking / batch not ended
                next_allowed_speak_ts=None,
                followups_since_user_reply=0,  # user replied => reset followup count
                idle_initiations_today=chat.idle_initiations_today,
                active_turns_since_user_msg=0,  # reset recent active turns on user input
            )

        # 3) Alice message: clear user_turn_end_ready, move to WAITING (unless user batch reopened later)
        if isinstance(event, AliceMessageEvent):
            ts = event.ts

            next_allowed = chat.next_allowed_speak_ts
            if cooldown_seconds is not None:
                next_allowed = ts + timedelta(seconds=max(0, int(cooldown_seconds)))

            if is_followup:
                followups = chat.followups_since_user_reply + 1
            else:
                followups = 0

            idle_inits = chat.idle_initiations_today
            if is_idle_initiation:
                idle_inits += 1

            # After Alice speaks, we're typically waiting for the user.
            # If user is detached already, we are still IDLE.
            new_mode = SessionMode.WAITING
            if self._should_detach(chat, now):
                new_mode = SessionMode.IDLE
            inc_active = 0 if self._should_detach(chat, now) else 1
            return ChatState(
                mode=new_mode,
                user_last_ts=chat.user_last_ts,
                alice_last_ts=ts,
                user_batch_open=False,              # user batch must be closed by definition if Alice spoke
                user_batch_last_ts=chat.user_batch_last_ts,
                user_turn_end_ready=False,          # Alice has taken the turn
                next_allowed_speak_ts=next_allowed,
                followups_since_user_reply=followups,
                idle_initiations_today=idle_inits,
                active_turns_since_user_msg=chat.active_turns_since_user_msg + inc_active,
            )

        # Fallback (shouldn't happen)
        return chat

    def permissions(self, chat: ChatState, now: datetime) -> SpeakPermissions:
        """
        Compute whether Alice is allowed to speak *now*, and under what constraints.

        Rules summary:
        - Never interject while user_batch_open.
        - If user_turn_end_ready == True, Alice is allowed to take a turn (subject to cooldown).
        - ACTIVE: may proactively speak with pacing (1–3 min), subject to cooldown and min gap.
        - WAITING: may send at most limited follow-up(s) if user hasn't replied, subject to cooldown.
        - IDLE: may initiate chat subject to cooldown and daily cap (daily reset handled elsewhere).
        """
        cfg = self.cfg

        # Apply time-based maintenance: close user batch if needed
        chat2 = self._close_user_batch_if_needed(chat, now)

        # Detach check
        if self._should_detach(chat2, now):
            mode = SessionMode.IDLE
        else:
            mode = chat2.mode
            # If user is within detach window and we aren't waiting on them, treat as ACTIVE
            if mode == SessionMode.IDLE and self._within_detach_window(chat2, now):
                mode = SessionMode.ACTIVE

        # If user is batching, do not speak.
        if chat2.user_batch_open:
            return SpeakPermissions(
                allowed_to_speak=False,
                mode=mode,
                max_bubbles_this_turn=1,
                min_gap_seconds=cfg.active_min_gap_seconds,
                max_gap_seconds=cfg.active_max_gap_seconds,
                user_turn_end_ready=chat2.user_turn_end_ready,
                user_batch_open=True,
                reason="user_batch_open: do not interject while user is sending a batch",
            )

        # Cooldown gate
        if not self._cooldown_allows(chat2, now):
            return SpeakPermissions(
                allowed_to_speak=False,
                mode=mode,
                max_bubbles_this_turn=1,
                min_gap_seconds=cfg.active_min_gap_seconds,
                max_gap_seconds=cfg.active_max_gap_seconds,
                user_turn_end_ready=chat2.user_turn_end_ready,
                user_batch_open=False,
                reason="cooldown: next_allowed_speak_ts not reached",
            )

        # If the user turn ended, Alice may respond (this is the primary "reply" trigger)
        if chat2.user_turn_end_ready:
            max_bubbles = cfg.active_max_bubbles_per_turn if mode == SessionMode.ACTIVE else cfg.idle_default_max_bubbles_per_turn
            max_bubbles = min(max_bubbles, cfg.hard_max_bubbles_per_turn)
            return SpeakPermissions(
                allowed_to_speak=True,
                mode=mode,
                max_bubbles_this_turn=max_bubbles,
                min_gap_seconds=cfg.active_min_gap_seconds,
                max_gap_seconds=cfg.active_max_gap_seconds,
                user_turn_end_ready=True,
                user_batch_open=False,
                reason="user_turn_end_ready: user finished batch; Alice may take a turn",
            )

        # Mode-specific proactive rules
        if mode == SessionMode.ACTIVE:
            # Proactive pacing: require at least active_min_gap since last Alice message
            dt_alice = self._time_since(chat2.alice_last_ts, now)
            if dt_alice is not None and dt_alice < cfg.active_min_gap_seconds:
                return SpeakPermissions(
                    allowed_to_speak=False,
                    mode=mode,
                    max_bubbles_this_turn=cfg.active_max_bubbles_per_turn,
                    min_gap_seconds=cfg.active_min_gap_seconds,
                    max_gap_seconds=cfg.active_max_gap_seconds,
                    user_turn_end_ready=False,
                    user_batch_open=False,
                    reason="ACTIVE: min_gap not reached since last Alice message",
                )

            return SpeakPermissions(
                allowed_to_speak=True,
                mode=mode,
                max_bubbles_this_turn=min(cfg.active_max_bubbles_per_turn, cfg.hard_max_bubbles_per_turn),
                min_gap_seconds=cfg.active_min_gap_seconds,
                max_gap_seconds=cfg.active_max_gap_seconds,
                user_turn_end_ready=False,
                user_batch_open=False,
                reason="ACTIVE: proactive allowed (pacing gates satisfied)",
            )

        if mode == SessionMode.WAITING:
            # Follow-up logic: only after a delay, and only limited times without user reply
            if chat2.followups_since_user_reply >= cfg.active_max_followups_without_user_reply:
                return SpeakPermissions(
                    allowed_to_speak=False,
                    mode=mode,
                    max_bubbles_this_turn=1,
                    min_gap_seconds=cfg.active_min_gap_seconds,
                    max_gap_seconds=cfg.active_max_gap_seconds,
                    user_turn_end_ready=False,
                    user_batch_open=False,
                    reason="WAITING: followup limit reached without user reply",
                )

            dt_alice = self._time_since(chat2.alice_last_ts, now)
            if dt_alice is None or dt_alice < cfg.active_followup_after_seconds:
                return SpeakPermissions(
                    allowed_to_speak=False,
                    mode=mode,
                    max_bubbles_this_turn=1,
                    min_gap_seconds=cfg.active_min_gap_seconds,
                    max_gap_seconds=cfg.active_max_gap_seconds,
                    user_turn_end_ready=False,
                    user_batch_open=False,
                    reason="WAITING: followup delay not reached",
                )

            return SpeakPermissions(
                allowed_to_speak=True,
                mode=mode,
                max_bubbles_this_turn=1,  # follow-up should be short
                min_gap_seconds=cfg.active_min_gap_seconds,
                max_gap_seconds=cfg.active_max_gap_seconds,
                user_turn_end_ready=False,
                user_batch_open=False,
                reason="WAITING: followup allowed",
            )

        # IDLE: initiation budget (daily reset is external; we only enforce cap against current counter)
        if mode == SessionMode.IDLE:
            if chat2.idle_initiations_today >= cfg.idle_daily_initiation_cap:
                return SpeakPermissions(
                    allowed_to_speak=False,
                    mode=mode,
                    max_bubbles_this_turn=cfg.idle_default_max_bubbles_per_turn,
                    min_gap_seconds=cfg.active_min_gap_seconds,
                    max_gap_seconds=cfg.active_max_gap_seconds,
                    user_turn_end_ready=False,
                    user_batch_open=False,
                    reason="IDLE: daily initiation cap reached",
                )

            return SpeakPermissions(
                allowed_to_speak=True,
                mode=mode,
                max_bubbles_this_turn=min(cfg.idle_default_max_bubbles_per_turn, cfg.hard_max_bubbles_per_turn),
                min_gap_seconds=cfg.active_min_gap_seconds,
                max_gap_seconds=cfg.active_max_gap_seconds,
                user_turn_end_ready=False,
                user_batch_open=False,
                reason="IDLE: initiation allowed (subject to Director deciding to speak)",
            )

        # Default deny (shouldn't be hit)
        return SpeakPermissions(
            allowed_to_speak=False,
            mode=mode,
            max_bubbles_this_turn=1,
            min_gap_seconds=cfg.active_min_gap_seconds,
            max_gap_seconds=cfg.active_max_gap_seconds,
            user_turn_end_ready=False,
            user_batch_open=False,
            reason="default deny",
        )
