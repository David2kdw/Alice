# tests/test_orchestrator.py
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Sequence

import pytest

import Services.orchestrator as orch_mod
from Services.session_manager import SessionConfig, SessionManager
from Services.orchestrator import DecisionContext, Orchestrator
from Domain.constants import MessageRole, SessionMode
from Domain.models import AffectState, ChatState, Message, StateBundle, WorldState
from Domain.schemas import Plan


# ----------------------------
# Global test hygiene
# ----------------------------

@pytest.fixture(autouse=True)
def _no_real_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Orchestrator uses time.sleep() inside bubble delays.
    Patch it to a no-op so tests never block.
    """
    monkeypatch.setattr(orch_mod.time, "sleep", lambda _s: None)


def t0() -> datetime:
    return datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)


# ----------------------------
# Stubs / fakes
# ----------------------------

@dataclass
class FakeClock:
    now_value: datetime

    def now(self) -> datetime:
        return self.now_value


class InMemoryStateStore:
    def __init__(self, init: Optional[StateBundle] = None):
        self._bundle = init

    def load(self) -> Optional[StateBundle]:
        return self._bundle

    def save(self, bundle: StateBundle) -> None:
        self._bundle = bundle


class TranscriptSink:
    def __init__(self):
        self.messages: List[Message] = []

    def append(self, messages: Sequence[Message]) -> None:
        self.messages.extend(list(messages))


class GuardDirector:
    """Fails the test if orchestrator calls director.decide()."""
    def decide(self, ctx: DecisionContext) -> Plan:
        raise AssertionError("Director must not be called in this scenario.")


class FixedDirector:
    """
    Returns a fixed SPEAK/STAY_SILENT plan that obeys ctx.perms.max_bubbles_this_turn.
    """

    def __init__(
        self,
        *,
        action: str = "SPEAK",
        bubbles: int = 2,
        speak_mode: str = "active",
        message_type: str = "reply",
        cooldown: int = 0,
    ):
        self.action = action
        self.bubbles = bubbles
        self.speak_mode = speak_mode
        self.message_type = message_type
        self.cooldown = cooldown

    def decide(self, ctx: DecisionContext) -> Plan:
        cap = int(ctx.perms.max_bubbles_this_turn)

        if self.action == "STAY_SILENT":
            return Plan(
                action="STAY_SILENT",
                speak_mode="idle",
                message_type="none",
                intent="test_silent",
                topic="",
                max_bubbles_hard=cap,
                bubble_count_target=1,
                length="short",
                use_memory_ids=[],
                cooldown_seconds=0,
                confidence=1.0,
            )

        return Plan(
            action="SPEAK",
            speak_mode=self.speak_mode,
            message_type=self.message_type,
            intent="test_speak",
            topic="test",
            max_bubbles_hard=cap,
            bubble_count_target=max(1, min(int(self.bubbles), cap)),
            length="short",
            use_memory_ids=[],
            cooldown_seconds=int(self.cooldown),
            confidence=1.0,
        )


class RaisingDirector:
    def decide(self, ctx: DecisionContext) -> Plan:
        raise RuntimeError("boom")


class ListSpeaker:
    def __init__(self, texts: Sequence[str]):
        self.texts = list(texts)

    def compose(self, ctx: DecisionContext, plan: Plan) -> List[str]:
        n = max(1, int(plan.bubble_count_target))
        return self.texts[:n]


class GreedySpeaker:
    """Ignores plan and returns too many texts (to test orchestrator cap enforcement)."""
    def __init__(self, texts: Sequence[str]):
        self.texts = list(texts)

    def compose(self, ctx: DecisionContext, plan: Plan) -> List[str]:
        return list(self.texts)


# ----------------------------
# Helpers
# ----------------------------

def make_sm(**overrides) -> SessionManager:
    # Use a dict so overrides can safely replace defaults without "multiple values" errors.
    params = {
        "user_batch_end_silence_seconds": 30,
        "detach_after_seconds": 300,

        # Make tests permissive; we pass explicit `now` anyway.
        "active_min_gap_seconds": 0,
        "active_max_gap_seconds": 180,
        "active_max_bubbles_per_turn": 3,
        "active_max_followups_without_user_reply": 1,
        "active_followup_after_seconds": 0,

        "idle_daily_initiation_cap": 999,
        "idle_default_max_bubbles_per_turn": 2,

        "hard_max_bubbles_per_turn": 5,
    }
    params.update(overrides)

    cfg = SessionConfig(**params)
    return SessionManager(cfg)



def seed_bundle(chat: ChatState, now: datetime) -> StateBundle:
    return StateBundle(
        world=WorldState(now=now),
        affect=AffectState(),
        chat=chat,
        last_tick_time=now,
        outbox=[],
    )


def make_orch(
    *,
    sm: SessionManager,
    store: InMemoryStateStore,
    director,
    speaker,
    clock: FakeClock,
    transcript: TranscriptSink,
    poll_user_message,
    bubble_delay_ms: int = 0,
    working_memory_max_messages: int = 50,
) -> Orchestrator:
    return Orchestrator(
        session_manager=sm,
        director=director,
        speaker=speaker,
        state_store=store,
        clock=clock,
        transcript_sink=transcript,
        poll_user_message=poll_user_message,
        working_memory_max_messages=working_memory_max_messages,
        bubble_delay_min_ms=bubble_delay_ms,
        bubble_delay_max_ms=bubble_delay_ms,
    )


# ----------------------------
# Tests
# ----------------------------

def test_user_message_is_buffered_no_reply_and_opens_batch():
    sm = make_sm()
    now = t0()
    clock = FakeClock(now)
    store = InMemoryStateStore(None)
    transcript = TranscriptSink()

    orch = make_orch(
        sm=sm,
        store=store,
        director=FixedDirector(action="SPEAK", bubbles=2),
        speaker=ListSpeaker(["a1", "a2"]),
        clock=clock,
        transcript=transcript,
        poll_user_message=lambda: None,
        bubble_delay_ms=0,
    )

    out = orch.step(user_input="hi", now=now)
    assert out.messages == []
    assert "awaiting batch end" in (out.debug.get("note", "") or "")

    saved = store.load()
    assert saved is not None
    assert saved.chat.user_batch_open is True
    assert saved.chat.user_turn_end_ready is False
    assert saved.chat.mode == SessionMode.ACTIVE

    # Transcript got the user message
    assert [m.role for m in transcript.messages] == [MessageRole.USER]
    assert transcript.messages[0].text == "hi"


def test_user_batch_open_blocks_speaking_until_silence_closes_batch():
    sm = make_sm()
    base = t0()
    clock = FakeClock(base)
    store = InMemoryStateStore(None)
    transcript = TranscriptSink()

    orch = make_orch(
        sm=sm,
        store=store,
        director=FixedDirector(action="SPEAK", bubbles=2),
        speaker=ListSpeaker(["a1", "a2"]),
        clock=clock,
        transcript=transcript,
        poll_user_message=lambda: None,
        bubble_delay_ms=0,
    )

    orch.step(user_input="msg1", now=base)

    # Only 1s later => still batching => should not speak
    out = orch.step(user_input=None, now=base + timedelta(seconds=1))
    assert out.messages == []
    assert "user_batch_open" in (out.debug.get("reason", "") or "")


def test_batch_closes_after_silence_then_replies():
    sm = make_sm()
    base = t0()
    clock = FakeClock(base)

    store = InMemoryStateStore(None)
    transcript = TranscriptSink()

    orch = make_orch(
        sm=sm,
        store=store,
        director=FixedDirector(action="SPEAK", bubbles=2, cooldown=0),
        speaker=ListSpeaker(["r1", "r2", "r3"]),
        clock=clock,
        transcript=transcript,
        poll_user_message=lambda: None,
        bubble_delay_ms=0,
    )

    # User sends a message => orchestrator buffers, no reply
    orch.step(user_input="hello", now=base)

    # After 31s silence, TickEvent should close batch and enable reply
    now2 = base + timedelta(seconds=31)
    out = orch.step(user_input=None, now=now2)

    assert [m.text for m in out.messages] == ["r1", "r2"]

    saved = store.load()
    assert saved is not None
    assert saved.chat.user_batch_open is False
    assert saved.chat.user_turn_end_ready is False
    assert saved.chat.mode == SessionMode.WAITING

    # Transcript has user + alice messages (order: user then alice bubbles)
    assert transcript.messages[0].role == MessageRole.USER
    assert [m.text for m in transcript.messages if m.role == MessageRole.ALICE] == ["r1", "r2"]


def test_cooldown_denies_speaking_and_director_not_called():
    sm = make_sm()
    now = t0()
    clock = FakeClock(now)

    chat = ChatState(
        mode=SessionMode.ACTIVE,
        user_last_ts=now,
        alice_last_ts=now - timedelta(seconds=10),
        user_batch_open=False,
        user_batch_last_ts=now - timedelta(seconds=40),
        user_turn_end_ready=True,
        next_allowed_speak_ts=now + timedelta(seconds=60),  # cooldown in effect
        followups_since_user_reply=0,
        idle_initiations_today=0,
        active_turns_since_user_msg=0,
    )
    store = InMemoryStateStore(seed_bundle(chat, now))
    transcript = TranscriptSink()

    orch = make_orch(
        sm=sm,
        store=store,
        director=GuardDirector(),
        speaker=ListSpeaker(["should_not_send"]),
        clock=clock,
        transcript=transcript,
        poll_user_message=lambda: None,
        bubble_delay_ms=0,
    )

    out = orch.step(user_input=None, now=now)
    assert out.messages == []
    assert "cooldown" in (out.debug.get("reason", "") or "")


def test_orchestrator_stay_silent_clears_user_turn_end_ready():
    sm = make_sm()
    now = t0()
    clock = FakeClock(now)

    chat = ChatState(
        mode=SessionMode.ACTIVE,
        user_last_ts=now,
        alice_last_ts=None,
        user_batch_open=False,
        user_batch_last_ts=now,
        user_turn_end_ready=True,
        next_allowed_speak_ts=None,
        followups_since_user_reply=0,
        idle_initiations_today=0,
        active_turns_since_user_msg=0,
    )
    store = InMemoryStateStore(seed_bundle(chat, now))
    transcript = TranscriptSink()

    orch = make_orch(
        sm=sm,
        store=store,
        director=FixedDirector(action="STAY_SILENT"),
        speaker=ListSpeaker(["a1", "a2"]),
        clock=clock,
        transcript=transcript,
        poll_user_message=lambda: None,
        bubble_delay_ms=0,
    )

    out = orch.step(user_input=None, now=now)
    assert out.messages == []

    saved = store.load()
    assert saved is not None
    assert saved.chat.user_turn_end_ready is False


def test_director_exception_falls_back_to_speak_once_when_ready():
    sm = make_sm()
    now = t0()
    clock = FakeClock(now)

    chat = ChatState(
        mode=SessionMode.ACTIVE,
        user_last_ts=now,
        alice_last_ts=None,
        user_batch_open=False,
        user_batch_last_ts=now,
        user_turn_end_ready=True,
        next_allowed_speak_ts=None,
        followups_since_user_reply=0,
        idle_initiations_today=0,
        active_turns_since_user_msg=0,
    )
    store = InMemoryStateStore(seed_bundle(chat, now))
    transcript = TranscriptSink()

    orch = make_orch(
        sm=sm,
        store=store,
        director=RaisingDirector(),
        speaker=ListSpeaker(["fallback1", "fallback2"]),
        clock=clock,
        transcript=transcript,
        poll_user_message=lambda: None,
        bubble_delay_ms=0,
    )

    out = orch.step(user_input=None, now=now)
    assert [m.text for m in out.messages] == ["fallback1"]

    saved = store.load()
    assert saved is not None
    assert saved.chat.user_turn_end_ready is False


def test_speaker_overproduces_is_capped_by_permissions():
    # Force max bubbles to 2
    sm = make_sm(active_max_bubbles_per_turn=2)
    now = t0()
    clock = FakeClock(now)

    chat = ChatState(
        mode=SessionMode.ACTIVE,
        user_last_ts=now,
        alice_last_ts=None,
        user_batch_open=False,
        user_batch_last_ts=now,
        user_turn_end_ready=True,
        next_allowed_speak_ts=None,
        followups_since_user_reply=0,
        idle_initiations_today=0,
        active_turns_since_user_msg=0,
    )
    store = InMemoryStateStore(seed_bundle(chat, now))
    transcript = TranscriptSink()

    orch = make_orch(
        sm=sm,
        store=store,
        director=FixedDirector(action="SPEAK", bubbles=2),
        speaker=GreedySpeaker(["a1", "a2", "a3", "a4"]),
        clock=clock,
        transcript=transcript,
        poll_user_message=lambda: None,
        bubble_delay_ms=0,
    )

    out = orch.step(user_input=None, now=now)
    assert [m.text for m in out.messages] == ["a1", "a2"]


def test_per_bubble_timestamps_monotonic_and_alice_last_ts_is_last_bubble():
    sm = make_sm()
    base = t0()
    clock = FakeClock(base)

    chat = ChatState(
        mode=SessionMode.ACTIVE,
        user_last_ts=base,
        alice_last_ts=None,
        user_batch_open=False,
        user_batch_last_ts=base,
        user_turn_end_ready=True,
        next_allowed_speak_ts=None,
        followups_since_user_reply=0,
        idle_initiations_today=0,
        active_turns_since_user_msg=0,
    )
    store = InMemoryStateStore(seed_bundle(chat, base))
    transcript = TranscriptSink()

    orch = make_orch(
        sm=sm,
        store=store,
        director=FixedDirector(action="SPEAK", bubbles=3),
        speaker=ListSpeaker(["b1", "b2", "b3"]),
        clock=clock,
        transcript=transcript,
        poll_user_message=lambda: None,
        bubble_delay_ms=100,  # 0.1s between bubbles (sleep patched to no-op)
    )

    out = orch.step(user_input=None, now=base)
    ts = [m.ts for m in out.messages]
    assert len(ts) == 3
    assert ts[0] <= ts[1] <= ts[2]

    # Expect ~0.1s increments (tolerance for float->timedelta rounding)
    d01 = (ts[1] - ts[0]).total_seconds()
    d12 = (ts[2] - ts[1]).total_seconds()
    assert 0.09 <= d01 <= 0.11
    assert 0.09 <= d12 <= 0.11

    saved = store.load()
    assert saved is not None
    assert saved.chat.alice_last_ts == ts[-1]


def test_interrupt_cancels_remaining_bubbles_and_opens_new_batch():
    sm = make_sm()
    now = t0()
    clock = FakeClock(now)

    chat = ChatState(
        mode=SessionMode.ACTIVE,
        user_last_ts=now,
        alice_last_ts=None,
        user_batch_open=False,
        user_batch_last_ts=now,
        user_turn_end_ready=True,
        next_allowed_speak_ts=None,
        followups_since_user_reply=0,
        idle_initiations_today=0,
        active_turns_since_user_msg=0,
    )
    store = InMemoryStateStore(seed_bundle(chat, now))
    transcript = TranscriptSink()

    director = FixedDirector(action="SPEAK", bubbles=3, cooldown=0)
    speaker = ListSpeaker(["a1", "a2", "a3"])

    # With bubble_delay_ms=0, poll_user_message is called:
    # - once between bubble 1 and 2
    # - once between bubble 2 and 3
    calls = {"n": 0}

    def poll_user_message() -> Optional[str]:
        calls["n"] += 1
        if calls["n"] == 2:
            return "user_interrupt"
        return None

    orch = make_orch(
        sm=sm,
        store=store,
        director=director,
        speaker=speaker,
        clock=clock,
        transcript=transcript,
        poll_user_message=poll_user_message,
        bubble_delay_ms=0,
    )

    out = orch.step(user_input=None, now=now)

    assert [m.text for m in out.messages] == ["a1", "a2"]

    saved = store.load()
    assert saved is not None
    assert saved.chat.user_batch_open is True
    assert saved.chat.user_turn_end_ready is False
    assert saved.chat.mode == SessionMode.ACTIVE

    texts = [m.text for m in transcript.messages]
    assert "a1" in texts and "a2" in texts
    assert "user_interrupt" in texts
    assert "a3" not in texts


def test_waiting_followup_increments_followups_since_user_reply():
    sm = make_sm(active_followup_after_seconds=0, active_max_followups_without_user_reply=2)
    now = t0()
    clock = FakeClock(now)

    chat = ChatState(
        mode=SessionMode.WAITING,
        user_last_ts=now - timedelta(seconds=10),   # still within detach window
        alice_last_ts=now - timedelta(seconds=100), # enough time passed for followup
        user_batch_open=False,
        user_batch_last_ts=None,
        user_turn_end_ready=False,
        next_allowed_speak_ts=None,
        followups_since_user_reply=0,
        idle_initiations_today=0,
        active_turns_since_user_msg=1,
    )
    store = InMemoryStateStore(seed_bundle(chat, now))
    transcript = TranscriptSink()

    orch = make_orch(
        sm=sm,
        store=store,
        director=FixedDirector(action="SPEAK", bubbles=2),  # WAITING perms caps to 1 bubble anyway
        speaker=ListSpeaker(["f1", "f2"]),
        clock=clock,
        transcript=transcript,
        poll_user_message=lambda: None,
        bubble_delay_ms=0,
    )

    out = orch.step(user_input=None, now=now)
    assert [m.text for m in out.messages] == ["f1"]

    saved = store.load()
    assert saved is not None
    assert saved.chat.followups_since_user_reply == 1
    assert saved.chat.mode == SessionMode.WAITING


def test_idle_initiation_increments_idle_initiations_today():
    sm = make_sm(idle_daily_initiation_cap=10)
    now = t0()
    clock = FakeClock(now)

    chat = ChatState(
        mode=SessionMode.IDLE,
        user_last_ts=now - timedelta(seconds=999),  # detached
        alice_last_ts=None,
        user_batch_open=False,
        user_batch_last_ts=None,
        user_turn_end_ready=False,
        next_allowed_speak_ts=None,
        followups_since_user_reply=0,
        idle_initiations_today=0,
        active_turns_since_user_msg=0,
    )
    store = InMemoryStateStore(seed_bundle(chat, now))
    transcript = TranscriptSink()

    orch = make_orch(
        sm=sm,
        store=store,
        director=FixedDirector(action="SPEAK", bubbles=2),
        speaker=ListSpeaker(["i1", "i2"]),
        clock=clock,
        transcript=transcript,
        poll_user_message=lambda: None,
        bubble_delay_ms=0,
    )

    out = orch.step(user_input=None, now=now)
    assert [m.text for m in out.messages] == ["i1", "i2"]

    saved = store.load()
    assert saved is not None
    assert saved.chat.idle_initiations_today == 1
    assert saved.chat.mode == SessionMode.IDLE


def test_active_proactive_speaks_when_allowed_and_not_turn_end_ready():
    sm = make_sm(active_min_gap_seconds=0)
    now = t0()
    clock = FakeClock(now)

    chat = ChatState(
        mode=SessionMode.ACTIVE,
        user_last_ts=now - timedelta(seconds=5),
        alice_last_ts=now - timedelta(seconds=100),
        user_batch_open=False,
        user_batch_last_ts=None,
        user_turn_end_ready=False,
        next_allowed_speak_ts=None,
        followups_since_user_reply=0,
        idle_initiations_today=0,
        active_turns_since_user_msg=0,
    )
    store = InMemoryStateStore(seed_bundle(chat, now))
    transcript = TranscriptSink()

    orch = make_orch(
        sm=sm,
        store=store,
        director=FixedDirector(action="SPEAK", bubbles=2, speak_mode="active", message_type="share"),
        speaker=ListSpeaker(["p1", "p2"]),
        clock=clock,
        transcript=transcript,
        poll_user_message=lambda: None,
        bubble_delay_ms=0,
    )

    out = orch.step(user_input=None, now=now)
    assert [m.text for m in out.messages] == ["p1", "p2"]

    saved = store.load()
    assert saved is not None
    assert saved.chat.mode == SessionMode.WAITING
