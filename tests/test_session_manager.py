# tests/test_session_manager.py
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from Services.session_manager import SessionConfig, SessionManager
from Domain.constants import SessionMode
from Domain.models import AliceMessageEvent, ChatState, TickEvent, UserMessageEvent


# ----------------------------
# Helpers
# ----------------------------

def t0() -> datetime:
    return datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)


def make_sm(**overrides) -> SessionManager:
    """
    Build SessionManager with small, explicit defaults and allow overrides safely.
    (Avoid passing the same kw twice -> use dict-update.)
    """
    params = {
        "user_batch_end_silence_seconds": 30,
        "detach_after_seconds": 300,

        "active_min_gap_seconds": 60,
        "active_max_gap_seconds": 180,
        "active_max_bubbles_per_turn": 3,
        "active_max_followups_without_user_reply": 1,
        "active_followup_after_seconds": 90,

        "idle_daily_initiation_cap": 4,
        "idle_default_max_bubbles_per_turn": 2,

        "hard_max_bubbles_per_turn": 5,
    }
    params.update(overrides)
    return SessionManager(SessionConfig(**params))


def chat(**kwargs) -> ChatState:
    """
    Make a ChatState with sane defaults; override only what matters in each test.
    """
    return ChatState(**kwargs)


# ============================================================
# update(): TickEvent / UserMessageEvent / AliceMessageEvent
# ============================================================

def test_update_user_message_opens_batch_sets_active_and_resets_counters():
    sm = make_sm()
    now = t0()

    # Arrange: start in any mode; counters non-zero
    c0 = chat(
        mode=SessionMode.IDLE,
        alice_last_ts=now - timedelta(seconds=10),
        next_allowed_speak_ts=now + timedelta(seconds=999),
        followups_since_user_reply=2,
        idle_initiations_today=3,
        active_turns_since_user_msg=7,
    )

    # Act
    c1 = sm.update(UserMessageEvent(text="hi", ts=now), c0, now)

    # Assert
    assert c1.mode == SessionMode.ACTIVE
    assert c1.user_last_ts == now
    assert c1.user_batch_open is True
    assert c1.user_batch_last_ts == now
    assert c1.user_turn_end_ready is False

    # User input clears cooldown + resets followup counter and active-turn counter
    assert c1.next_allowed_speak_ts is None
    assert c1.followups_since_user_reply == 0
    assert c1.active_turns_since_user_msg == 0

    # Idle initiation counter is not reset by user input
    assert c1.idle_initiations_today == 3


def test_update_tick_closes_user_batch_after_silence_and_marks_turn_end_ready():
    sm = make_sm(user_batch_end_silence_seconds=30)
    base = t0()

    # Arrange: user batch is open; last user msg at base
    c0 = chat(
        mode=SessionMode.ACTIVE,
        user_last_ts=base,
        user_batch_open=True,
        user_batch_last_ts=base,
        user_turn_end_ready=False,
    )

    # Act: 31 seconds later
    now = base + timedelta(seconds=31)
    c1 = sm.update(TickEvent(now), c0, now)

    # Assert: batch closed + ready to respond
    assert c1.user_batch_open is False
    assert c1.user_turn_end_ready is True


def test_update_tick_does_not_close_batch_before_silence_threshold():
    sm = make_sm(user_batch_end_silence_seconds=30)
    base = t0()

    c0 = chat(
        mode=SessionMode.ACTIVE,
        user_last_ts=base,
        user_batch_open=True,
        user_batch_last_ts=base,
        user_turn_end_ready=False,
    )

    # Act: only 29 seconds later
    now = base + timedelta(seconds=29)
    c1 = sm.update(TickEvent(now), c0, now)

    assert c1.user_batch_open is True
    assert c1.user_turn_end_ready is False


def test_update_tick_detaches_after_detach_seconds_and_resets_active_turns():
    sm = make_sm(detach_after_seconds=300)
    base = t0()

    # Arrange: last user msg long ago -> should detach
    c0 = chat(
        mode=SessionMode.WAITING,
        user_last_ts=base,
        alice_last_ts=base + timedelta(seconds=1),
        user_batch_open=False,
        user_turn_end_ready=True,      # even if it was ready, detach should clear it
        active_turns_since_user_msg=5,
        followups_since_user_reply=1,
    )

    # Act: exactly 300s later => detach (>=)
    now = base + timedelta(seconds=300)
    c1 = sm.update(TickEvent(now), c0, now)

    assert c1.mode == SessionMode.IDLE
    assert c1.user_turn_end_ready is False
    assert c1.user_batch_open is False
    assert c1.active_turns_since_user_msg == 0


def test_update_tick_restores_idle_to_active_if_user_is_within_detach_window():
    sm = make_sm(detach_after_seconds=300)
    base = t0()

    # Arrange: state says IDLE but user_last_ts is recent (restore edge case)
    c0 = chat(
        mode=SessionMode.IDLE,
        user_last_ts=base,
        user_batch_open=False,
    )

    # Act: 10 seconds later (within detach window)
    now = base + timedelta(seconds=10)
    c1 = sm.update(TickEvent(now), c0, now)

    assert c1.mode == SessionMode.ACTIVE


def test_update_alice_message_moves_to_waiting_sets_cooldown_and_increments_active_turns():
    sm = make_sm(detach_after_seconds=300)
    base = t0()

    # Arrange: user is engaged recently -> not detached
    c0 = chat(
        mode=SessionMode.ACTIVE,
        user_last_ts=base,
        user_batch_open=False,
        user_turn_end_ready=True,  # Alice is replying
        active_turns_since_user_msg=0,
    )

    # Act: Alice speaks at base+5, with cooldown 60s
    ts = base + timedelta(seconds=5)
    c1 = sm.update(
        AliceMessageEvent(messages=["a1", "a2"], ts=ts),
        c0,
        now=ts,
        cooldown_seconds=60,
        is_idle_initiation=False,
        is_followup=False,
    )

    # Assert
    assert c1.mode == SessionMode.WAITING
    assert c1.alice_last_ts == ts
    assert c1.user_turn_end_ready is False
    assert c1.user_batch_open is False
    assert c1.next_allowed_speak_ts == ts + timedelta(seconds=60)
    assert c1.active_turns_since_user_msg == 1
    assert c1.followups_since_user_reply == 0


def test_update_alice_message_in_idle_stays_idle_and_does_not_increment_active_turns():
    sm = make_sm(detach_after_seconds=300)
    base = t0()

    # Arrange: user is detached (user_last_ts too old)
    c0 = chat(
        mode=SessionMode.IDLE,
        user_last_ts=base,
        active_turns_since_user_msg=10,
    )

    now = base + timedelta(seconds=301)  # detached now
    c1 = sm.update(
        AliceMessageEvent(messages=["hello"], ts=now),
        c0,
        now=now,
        cooldown_seconds=0,
        is_idle_initiation=True,
        is_followup=False,
    )

    assert c1.mode == SessionMode.IDLE
    # active_turns_since_user_msg should not increase when detached
    assert c1.active_turns_since_user_msg == 10
    # idle initiation increments counter
    assert c1.idle_initiations_today == c0.idle_initiations_today + 1


def test_update_followup_increments_followups_counter_otherwise_resets():
    sm = make_sm()
    base = t0()

    c0 = chat(
        mode=SessionMode.WAITING,
        user_last_ts=base,
        followups_since_user_reply=0,
    )

    ts1 = base + timedelta(seconds=10)
    c1 = sm.update(
        AliceMessageEvent(messages=["f1"], ts=ts1),
        c0,
        now=ts1,
        is_followup=True,
    )
    assert c1.followups_since_user_reply == 1

    ts2 = base + timedelta(seconds=20)
    c2 = sm.update(
        AliceMessageEvent(messages=["normal"], ts=ts2),
        c1,
        now=ts2,
        is_followup=False,
    )
    assert c2.followups_since_user_reply == 0


# ============================================================
# permissions(): allowed_to_speak gates and constraints
# ============================================================

def test_permissions_denies_while_user_batch_open():
    sm = make_sm()
    now = t0()

    c0 = chat(
        mode=SessionMode.ACTIVE,
        user_last_ts=now,
        user_batch_open=True,
        user_batch_last_ts=now,
    )

    perms = sm.permissions(c0, now)
    assert perms.allowed_to_speak is False
    assert perms.user_batch_open is True
    assert "user_batch_open" in perms.reason


def test_permissions_denies_when_in_cooldown():
    sm = make_sm()
    now = t0()

    c0 = chat(
        mode=SessionMode.ACTIVE,
        user_last_ts=now,
        user_batch_open=False,
        next_allowed_speak_ts=now + timedelta(seconds=10),
    )

    perms = sm.permissions(c0, now)
    assert perms.allowed_to_speak is False
    assert "cooldown" in perms.reason


def test_permissions_allows_reply_when_user_turn_end_ready_and_caps_bubbles_by_mode_and_hard_cap():
    # Make active max bubbles > hard cap to verify cap behavior
    sm = make_sm(active_max_bubbles_per_turn=10, hard_max_bubbles_per_turn=5, idle_default_max_bubbles_per_turn=2)
    now = t0()

    # Case A: ACTIVE reply -> capped by hard_max
    c_active = chat(
        mode=SessionMode.ACTIVE,
        user_last_ts=now,
        user_turn_end_ready=True,
        user_batch_open=False,
    )
    p1 = sm.permissions(c_active, now)
    assert p1.allowed_to_speak is True
    assert p1.user_turn_end_ready is True
    assert p1.max_bubbles_this_turn == 5  # hard cap wins

    # Case B: IDLE reply -> uses idle_default (and still <= hard cap)
    c_idle = chat(
        mode=SessionMode.IDLE,
        user_last_ts=now - timedelta(seconds=999),  # detached
        user_turn_end_ready=True,
        user_batch_open=False,
    )
    p2 = sm.permissions(c_idle, now)
    assert p2.allowed_to_speak is True
    assert p2.mode == SessionMode.IDLE
    assert p2.max_bubbles_this_turn == 2


def test_permissions_active_proactive_denies_if_min_gap_not_reached():
    sm = make_sm(active_min_gap_seconds=60)
    now = t0()

    c0 = chat(
        mode=SessionMode.ACTIVE,
        user_last_ts=now,
        alice_last_ts=now - timedelta(seconds=30),
        user_batch_open=False,
        user_turn_end_ready=False,
    )

    perms = sm.permissions(c0, now)
    assert perms.allowed_to_speak is False
    assert "min_gap" in perms.reason


def test_permissions_active_proactive_allows_if_min_gap_reached():
    sm = make_sm(active_min_gap_seconds=60, active_max_bubbles_per_turn=3, hard_max_bubbles_per_turn=5)
    now = t0()

    c0 = chat(
        mode=SessionMode.ACTIVE,
        user_last_ts=now,
        alice_last_ts=now - timedelta(seconds=61),
        user_batch_open=False,
        user_turn_end_ready=False,
    )

    perms = sm.permissions(c0, now)
    assert perms.allowed_to_speak is True
    assert perms.max_bubbles_this_turn == 3
    assert perms.mode == SessionMode.ACTIVE


def test_permissions_waiting_denies_if_followup_limit_reached():
    sm = make_sm(active_max_followups_without_user_reply=1)
    now = t0()

    c0 = chat(
        mode=SessionMode.WAITING,
        user_last_ts=now,
        alice_last_ts=now - timedelta(seconds=999),
        followups_since_user_reply=1,  # already at limit
        user_batch_open=False,
    )

    perms = sm.permissions(c0, now)
    assert perms.allowed_to_speak is False
    assert "limit" in perms.reason


def test_permissions_waiting_denies_if_followup_delay_not_reached_or_no_alice_last_ts():
    sm = make_sm(active_followup_after_seconds=90)
    now = t0()

    # No alice_last_ts => cannot measure delay => deny
    c_none = chat(mode=SessionMode.WAITING, user_last_ts=now, alice_last_ts=None)
    p_none = sm.permissions(c_none, now)
    assert p_none.allowed_to_speak is False
    assert "delay" in p_none.reason

    # Too soon since last Alice => deny
    c_soon = chat(mode=SessionMode.WAITING, user_last_ts=now, alice_last_ts=now - timedelta(seconds=89))
    p_soon = sm.permissions(c_soon, now)
    assert p_soon.allowed_to_speak is False
    assert "delay" in p_soon.reason


def test_permissions_waiting_allows_followup_after_delay_and_caps_to_one_bubble():
    sm = make_sm(active_followup_after_seconds=90)
    now = t0()

    c0 = chat(
        mode=SessionMode.WAITING,
        user_last_ts=now,
        alice_last_ts=now - timedelta(seconds=90),
        followups_since_user_reply=0,
        user_batch_open=False,
    )

    perms = sm.permissions(c0, now)
    assert perms.allowed_to_speak is True
    assert perms.max_bubbles_this_turn == 1
    assert perms.mode == SessionMode.WAITING


def test_permissions_idle_denies_if_daily_cap_reached_else_allows():
    sm = make_sm(idle_daily_initiation_cap=2, idle_default_max_bubbles_per_turn=2)
    now = t0()

    c_cap = chat(
        mode=SessionMode.IDLE,
        user_last_ts=now - timedelta(seconds=999),
        idle_initiations_today=2,
    )
    p_cap = sm.permissions(c_cap, now)
    assert p_cap.allowed_to_speak is False
    assert "cap" in p_cap.reason

    c_ok = chat(
        mode=SessionMode.IDLE,
        user_last_ts=now - timedelta(seconds=999),
        idle_initiations_today=1,
    )
    p_ok = sm.permissions(c_ok, now)
    assert p_ok.allowed_to_speak is True
    assert p_ok.max_bubbles_this_turn == 2


def test_permissions_closes_batch_internally_after_silence_and_allows_reply():
    """
    permissions() internally calls _close_user_batch_if_needed(chat, now).
    So even if orchestrator doesn't send a TickEvent, permissions can flip
    (user_batch_open -> False, user_turn_end_ready -> True) after enough silence.
    """
    sm = make_sm(user_batch_end_silence_seconds=30)
    base = t0()

    c0 = chat(
        mode=SessionMode.ACTIVE,
        user_last_ts=base,
        user_batch_open=True,
        user_batch_last_ts=base,
        user_turn_end_ready=False,
    )

    # Act: query permissions after 31s silence
    now = base + timedelta(seconds=31)
    perms = sm.permissions(c0, now)

    # Assert: batch considered closed => allowed to respond
    assert perms.allowed_to_speak is True
    assert perms.user_batch_open is False
    assert perms.user_turn_end_ready is True
