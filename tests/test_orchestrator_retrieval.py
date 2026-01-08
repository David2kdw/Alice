# tests/test_orchestrator_retrieval.py
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Sequence, Tuple, Dict, Any

import pytest

from Services.session_manager import SessionConfig, SessionManager
from Services.orchestrator import Orchestrator, DecisionContext
from Domain.constants import MessageRole, SessionMode
from Domain.models import Message, StateBundle, WorldState, AffectState, ChatState
from Domain.schemas import Plan


# ----------------------------
# Test helpers / stubs
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


class DummyTranscriptSink:
    def __init__(self):
        self.messages: List[Message] = []

    def append(self, messages: Sequence[Message]) -> None:
        self.messages.extend(list(messages))


class CapturingDirector:
    """
    Captures the last DecisionContext it received.
    Returns a simple SPEAK plan.
    """
    def __init__(self, *, bubbles: int = 1, cooldown: int = 0):
        self.last_ctx: Optional[DecisionContext] = None
        self.calls = 0
        self.bubbles = bubbles
        self.cooldown = cooldown

    def decide(self, ctx: DecisionContext) -> Plan:
        self.last_ctx = ctx
        self.calls += 1
        cap = ctx.perms.max_bubbles_this_turn
        return Plan(
            action="SPEAK",
            speak_mode="active",
            message_type="reply",
            intent="test",
            topic="test",
            max_bubbles_hard=cap,
            bubble_count_target=min(self.bubbles, cap),
            length="short",
            use_memory_ids=[],
            cooldown_seconds=self.cooldown,
            confidence=1.0,
        )


class DummySpeaker:
    def __init__(self, texts: Sequence[str]):
        self.texts = list(texts)
        self.calls = 0
        self.last_ctx: Optional[DecisionContext] = None

    def compose(self, ctx: DecisionContext, plan: Plan) -> List[str]:
        self.calls += 1
        self.last_ctx = ctx
        n = max(1, int(plan.bubble_count_target))
        return self.texts[:n]


class FakeTranscriptIndex:
    def __init__(self, return_texts: Tuple[str, ...] = ("HIT1", "HIT2")):
        self.return_texts = tuple(return_texts)
        self.calls: List[Tuple[str, Dict[str, Any]]] = []

    def retrieve_texts(self, query: str, **kwargs) -> Tuple[str, ...]:
        self.calls.append((query, dict(kwargs)))
        return self.return_texts


def make_sm(**overrides) -> SessionManager:
    params = {
        "user_batch_end_silence_seconds": 30,
        "detach_after_seconds": 300,

        # tests: permissive timing
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



def t0() -> datetime:
    return datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)


def make_orch(
    *,
    sm: SessionManager,
    store: InMemoryStateStore,
    director,
    speaker,
    transcript_index: Optional[FakeTranscriptIndex],
    clock: FakeClock,
) -> Orchestrator:
    # Make delays 0 so tests don't sleep.
    return Orchestrator(
        session_manager=sm,
        director=director,
        speaker=speaker,
        state_store=store,
        clock=clock,
        transcript_sink=DummyTranscriptSink(),
        poll_user_message=lambda: None,
        working_memory_max_messages=50,
        bubble_delay_min_ms=0,
        bubble_delay_max_ms=0,
        typing_base_delay_ms=0,
        typing_ms_per_char=0.0,
        typing_jitter_ms=0,
        transcript_index=transcript_index,
        retrieval_top_k_hits=7,
        retrieval_context_window=2,
        retrieval_max_chars=1234,
    )


# ----------------------------
# Tests
# ----------------------------

def test_index_not_called_when_not_allowed_to_speak_user_batch_open():
    sm = make_sm()
    now = t0()
    clock = FakeClock(now)

    # Seed: user is still batching (silence < 30s), so permissions should deny speaking
    chat = ChatState(
        mode=SessionMode.ACTIVE,
        user_last_ts=now - timedelta(seconds=5),
        alice_last_ts=None,
        user_batch_open=True,
        user_batch_last_ts=now - timedelta(seconds=10),  # silence=10s < 30 => still open
        user_turn_end_ready=False,
        next_allowed_speak_ts=None,
        followups_since_user_reply=0,
        idle_initiations_today=0,
        active_turns_since_user_msg=0,
    )
    bundle = StateBundle(
        world=WorldState(now=now),
        affect=AffectState(),
        chat=chat,
        last_tick_time=now,
        outbox=[],
    )
    store = InMemoryStateStore(bundle)

    index = FakeTranscriptIndex()
    director = CapturingDirector()
    speaker = DummySpeaker(["a1"])

    orch = make_orch(
        sm=sm,
        store=store,
        director=director,
        speaker=speaker,
        transcript_index=index,
        clock=clock,
    )

    out = orch.step(user_input=None, now=now)
    assert out.messages == []

    # Not allowed => must not call index/director/speaker
    assert index.calls == []
    assert director.calls == 0
    assert speaker.calls == 0


def test_index_called_and_ctx_contains_retrieved_texts_on_reply_after_batch_close():
    sm = make_sm()
    base = t0()
    clock = FakeClock(base)

    store = InMemoryStateStore(None)
    index = FakeTranscriptIndex(("R1", "R2"))
    director = CapturingDirector(bubbles=1, cooldown=0)
    speaker = DummySpeaker(["a1"])

    orch = make_orch(
        sm=sm,
        store=store,
        director=director,
        speaker=speaker,
        transcript_index=index,
        clock=clock,
    )

    # 1) User sends message -> orchestrator buffers and returns early (no retrieval yet)
    out1 = orch.step(user_input="hello sqlite", now=base)
    assert out1.messages == []
    assert index.calls == []  # must not retrieve on user_input path

    # 2) After >=30s silence -> batch closes -> allowed to speak -> retrieval should happen
    now2 = base + timedelta(seconds=31)
    out2 = orch.step(user_input=None, now=now2)
    assert [m.text for m in out2.messages] == ["a1"]

    # Index called exactly once with last USER message as query
    assert len(index.calls) == 1
    query, kwargs = index.calls[0]
    assert query == "hello sqlite"
    assert kwargs["top_k_hits"] == 7
    assert kwargs["context_window"] == 2
    assert kwargs["max_chars"] == 1234

    # Director saw retrieved_texts in ctx
    assert director.last_ctx is not None
    assert director.last_ctx.retrieved_texts == ("R1", "R2")

    # Speaker also receives same ctx
    assert speaker.last_ctx is not None
    assert speaker.last_ctx.retrieved_texts == ("R1", "R2")


def test_retrieval_query_prefers_last_user_message_even_after_alice_spoke_followup():
    # Allow exactly one followup without user reply, and no followup delay
    sm = make_sm(active_max_followups_without_user_reply=1, active_followup_after_seconds=0)
    base = t0()
    clock = FakeClock(base)

    store = InMemoryStateStore(None)
    index = FakeTranscriptIndex(("CTX",))
    director = CapturingDirector(bubbles=1, cooldown=0)
    speaker = DummySpeaker(["a1"])  # reply/followup content doesn't matter

    orch = make_orch(
        sm=sm,
        store=store,
        director=director,
        speaker=speaker,
        transcript_index=index,
        clock=clock,
    )

    # User message
    orch.step(user_input="remember this", now=base)

    # Reply after batch closes
    orch.step(user_input=None, now=base + timedelta(seconds=31))
    assert len(index.calls) == 1
    assert index.calls[0][0] == "remember this"

    # Now we're in WAITING; followup is allowed immediately (active_followup_after_seconds=0)
    orch.step(user_input=None, now=base + timedelta(seconds=32))
    assert len(index.calls) == 2

    # Even though the last message in recent buffer is Alice, query should still be last USER message
    assert index.calls[1][0] == "remember this"
