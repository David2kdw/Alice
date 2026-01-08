# tests/test_integration_pipeline.py
from __future__ import annotations

import inspect
import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pytest

from Domain.constants import MessageRole, SessionMode
from Services.session_manager import SessionConfig, SessionManager
from Services.orchestrator import Orchestrator
from Services.director_v0 import RuleDirectorV0
from Services.speaker_v0 import RuleSpeakerV0

from Infra.sqlite_state_store import SQLiteStateStore
from Infra.jsonl_transcript_sink import JSONLTranscriptSink
from Infra.jsonl_transcript_reader import JSONLTranscriptReader
from Services.sklearn_transcript_index import SklearnTranscriptIndex


# ----------------------------
# Helpers
# ----------------------------

def t0() -> datetime:
    return datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)


def make_sm(**overrides) -> SessionManager:
    params = {
        "user_batch_end_silence_seconds": 30,
        "detach_after_seconds": 300,

        # make tests fast
        "active_min_gap_seconds": 0,
        "active_max_gap_seconds": 180,
        "active_max_bubbles_per_turn": 3,
        "active_max_followups_without_user_reply": 2,
        "active_followup_after_seconds": 0,  # allow followup immediately in WAITING

        "idle_daily_initiation_cap": 999,
        "idle_default_max_bubbles_per_turn": 2,

        "hard_max_bubbles_per_turn": 5,
    }
    params.update(overrides)
    return SessionManager(SessionConfig(**params))


class SpyIndex:
    """Wrap a real transcript index but record calls (so we can assert retrieval wiring)."""

    def __init__(self, inner: Any):
        self.inner = inner
        self.calls: List[Tuple[str, Dict[str, Any]]] = []

    def retrieve_texts(self, query: str, **kwargs) -> Tuple[str, ...]:
        self.calls.append((query, dict(kwargs)))
        return self.inner.retrieve_texts(query, **kwargs)


def build_orchestrator(**kwargs) -> Orchestrator:
    """
    Build Orchestrator while being robust to minor signature differences:
    only pass kwargs that __init__ actually accepts.
    """
    sig = inspect.signature(Orchestrator.__init__)
    allowed = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return Orchestrator(**allowed)


def read_jsonl(path: Path) -> List[dict]:
    out: List[dict] = []
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


# ----------------------------
# Integration tests
# ----------------------------

def test_pipeline_user_to_reply_writes_transcript_and_calls_index(tmp_path: Path):
    db_path = tmp_path / "state.sqlite3"
    log_path = tmp_path / "transcript.jsonl"

    store = SQLiteStateStore(str(db_path))
    sink = JSONLTranscriptSink(str(log_path))
    reader = JSONLTranscriptReader(str(log_path))

    real_index = SklearnTranscriptIndex(reader=reader, max_index_messages=500)
    spy_index = SpyIndex(real_index)

    sm = make_sm()
    # Cooldowns 0 so we can step immediately in tests
    director = RuleDirectorV0(reply_bubbles=2, reply_cooldown_s=0, followup_cooldown_s=0, idle_cooldown_s=0)
    speaker = RuleSpeakerV0()

    orch = build_orchestrator(
        session_manager=sm,
        director=director,
        speaker=speaker,
        state_store=store,
        transcript_sink=sink,
        transcript_index=spy_index,         # <-- key integration: real index wrapped by spy
        retrieval_top_k_hits=6,
        retrieval_context_window=1,
        retrieval_max_chars=1200,

        # avoid sleeps in tests
        bubble_delay_min_ms=0,
        bubble_delay_max_ms=0,
        typing_base_delay_ms=0,
        typing_ms_per_char=0.0,
        typing_jitter_ms=0,
        poll_user_message=lambda: None,

        debug=False,
    )

    base = t0()

    # 1) User message -> buffered, written to transcript, no reply yet
    out1 = orch.step(user_input="sqlite transcript index", now=base)
    assert out1.messages == []

    lines1 = read_jsonl(log_path)
    assert len(lines1) == 1
    assert lines1[0]["role"] == MessageRole.USER.value
    assert "sqlite" in lines1[0]["text"]

    # 2) After >=30s silence -> batch closes -> orchestrator replies and calls index
    out2 = orch.step(user_input=None, now=base + timedelta(seconds=31))
    assert len(out2.messages) >= 1
    assert all(m.role == MessageRole.ALICE for m in out2.messages)

    # Transcript now contains user + alice bubbles
    lines2 = read_jsonl(log_path)
    assert len(lines2) == 1 + len(out2.messages)

    # Retrieval wiring: index must have been called once with the last user message as query
    # (If your Orchestrator build doesn't support transcript_index yet, this will fail fast => good.)
    assert len(spy_index.calls) == 1
    q, kw = spy_index.calls[0]
    assert q == "sqlite transcript index"
    assert kw.get("top_k_hits") == 6
    assert kw.get("context_window") == 1
    assert kw.get("max_chars") == 1200

    store.close()


def test_pipeline_restart_loads_sqlite_and_followup_increments_counter(tmp_path: Path):
    db_path = tmp_path / "state.sqlite3"
    log_path = tmp_path / "transcript.jsonl"

    # First run
    store1 = SQLiteStateStore(str(db_path))
    sink1 = JSONLTranscriptSink(str(log_path))
    reader1 = JSONLTranscriptReader(str(log_path))
    index1 = SpyIndex(SklearnTranscriptIndex(reader=reader1, max_index_messages=500))

    sm = make_sm(active_followup_after_seconds=0, active_max_followups_without_user_reply=2)
    director = RuleDirectorV0(reply_bubbles=1, reply_cooldown_s=0, followup_cooldown_s=0, idle_cooldown_s=0)
    speaker = RuleSpeakerV0()

    orch1 = build_orchestrator(
        session_manager=sm,
        director=director,
        speaker=speaker,
        state_store=store1,
        transcript_sink=sink1,
        transcript_index=index1,
        bubble_delay_min_ms=0,
        bubble_delay_max_ms=0,
        typing_base_delay_ms=0,
        typing_ms_per_char=0.0,
        typing_jitter_ms=0,
        poll_user_message=lambda: None,
        debug=False,
    )

    base = t0()
    orch1.step(user_input="hello", now=base)
    out_reply = orch1.step(user_input=None, now=base + timedelta(seconds=31))
    assert len(out_reply.messages) >= 1

    # After reply, state in sqlite should be WAITING
    b1 = store1.load()
    assert b1 is not None
    assert b1.chat.mode == SessionMode.WAITING
    assert b1.chat.followups_since_user_reply == 0  # reply resets followups counter

    store1.close()

    # "Restart": new store + new orchestrator
    store2 = SQLiteStateStore(str(db_path))
    sink2 = JSONLTranscriptSink(str(log_path))  # append-only, safe
    reader2 = JSONLTranscriptReader(str(log_path))
    index2 = SpyIndex(SklearnTranscriptIndex(reader=reader2, max_index_messages=500))

    orch2 = build_orchestrator(
        session_manager=sm,
        director=director,
        speaker=speaker,
        state_store=store2,
        transcript_sink=sink2,
        transcript_index=index2,
        bubble_delay_min_ms=0,
        bubble_delay_max_ms=0,
        typing_base_delay_ms=0,
        typing_ms_per_char=0.0,
        typing_jitter_ms=0,
        poll_user_message=lambda: None,
        debug=False,
    )

    # In WAITING, followup is allowed immediately (we set active_followup_after_seconds=0)
    out_followup = orch2.step(user_input=None, now=base + timedelta(seconds=32))
    assert len(out_followup.messages) >= 1

    b2 = store2.load()
    assert b2 is not None
    assert b2.chat.mode == SessionMode.WAITING
    assert b2.chat.followups_since_user_reply == 1  # <-- integration assertion

    store2.close()
