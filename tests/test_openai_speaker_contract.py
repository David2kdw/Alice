# tests/test_openai_speaker_contract.py
from __future__ import annotations

import json
import sys
import types
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import pytest


# ----------------------------
# Import helpers (robust to your folder naming)
# ----------------------------

def _import_symbol(module_paths: List[str], name: str):
    last_err: Optional[BaseException] = None
    for mp in module_paths:
        try:
            mod = __import__(mp, fromlist=[name])
            return getattr(mod, name)
        except BaseException as e:
            last_err = e
            continue
    raise ImportError(f"Cannot import {name} from any of: {module_paths}. Last error: {last_err!r}")


# If your project doesn't have openai installed yet, keep tests importable.
# This prevents ImportError if Infra.openai_responses_client imports `from openai import OpenAI`.
try:
    import openai  # noqa: F401
except Exception:
    dummy = types.ModuleType("openai")

    class OpenAI:  # minimal stub
        def __init__(self, *args, **kwargs): ...
    dummy.OpenAI = OpenAI
    sys.modules["openai"] = dummy


# Project imports (try a few common path variants)
OpenAISpeaker = _import_symbol(
    ["Services.openai_speaker", "services.openai_speaker", "alice.Services.openai_speaker", "alice.services.openai_speaker"],
    "OpenAISpeaker",
)
DecisionContext = _import_symbol(
    ["Services.orchestrator", "services.orchestrator", "alice.Services.orchestrator", "alice.services.orchestrator"],
    "DecisionContext",
)
SpeakPermissions = _import_symbol(
    ["Services.session_manager", "services.session_manager", "alice.Services.session_manager", "alice.services.session_manager"],
    "SpeakPermissions",
)

Message = _import_symbol(
    ["Domain.models", "domain.models", "alice.Domain.models", "alice.domain.models"],
    "Message",
)
WorldState = _import_symbol(
    ["Domain.models", "domain.models", "alice.Domain.models", "alice.domain.models"],
    "WorldState",
)
AffectState = _import_symbol(
    ["Domain.models", "domain.models", "alice.Domain.models", "alice.domain.models"],
    "AffectState",
)
ChatState = _import_symbol(
    ["Domain.models", "domain.models", "alice.Domain.models", "alice.domain.models"],
    "ChatState",
)

Plan = _import_symbol(
    ["Domain.schemas", "domain.schemas", "alice.Domain.schemas", "alice.domain.schemas"],
    "Plan",
)

MessageRole = _import_symbol(
    ["Domain.constants", "domain.constants", "alice.Domain.constants", "alice.domain.constants"],
    "MessageRole",
)
SessionMode = _import_symbol(
    ["Domain.constants", "domain.constants", "alice.Domain.constants", "alice.domain.constants"],
    "SessionMode",
)


# ----------------------------
# Fakes
# ----------------------------

@dataclass
class FakeClient:
    raw: str
    calls: List[Dict[str, Any]]

    def __init__(self, raw: str):
        self.raw = raw
        self.calls = []

    def create_text(self, *, input_messages: List[Dict[str, str]], text_format: Optional[Dict[str, Any]] = None) -> str:
        self.calls.append({"input_messages": input_messages, "text_format": text_format})
        return self.raw


def _t0() -> datetime:
    return datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)


def _mk_ctx(
    *,
    max_bubbles: int = 3,
    recent: Optional[List[Message]] = None,
    retrieved: Tuple[str, ...] = (),
    mode: Any = None,
) -> Any:
    now = _t0()
    perms = SpeakPermissions(
        allowed_to_speak=True,
        mode=mode or SessionMode.ACTIVE,
        max_bubbles_this_turn=max_bubbles,
        min_gap_seconds=0,
        max_gap_seconds=180,
        user_turn_end_ready=True,
        user_batch_open=False,
        reason="test",
    )
    ctx = DecisionContext(
        now=now,
        world=WorldState(now=now),
        affect=AffectState(),
        chat=ChatState(mode=mode or SessionMode.ACTIVE),
        perms=perms,
        recent_dialogue=tuple(recent or []),
        retrieved_texts=retrieved,
    )
    return ctx


def _mk_plan(*, target: int) -> Any:
    return Plan(
        action="SPEAK",
        speak_mode="active",
        message_type="reply",
        intent="test_intent",
        topic="test_topic",
        max_bubbles_hard=5,
        bubble_count_target=target,
        length="short",
        use_memory_ids=[],
        cooldown_seconds=0,
        confidence=1.0,
    )


# ----------------------------
# Tests
# ----------------------------

def test_schema_max_items_equals_target_and_is_capped_by_permissions():
    # plan wants 5 bubbles but perms cap is 2 => n=2
    fake = FakeClient(raw=json.dumps({"bubbles": ["a", "b", "c"]}))
    speaker = OpenAISpeaker(client=fake)

    recent = [
        Message(role=MessageRole.USER, text="hi", ts=_t0()),
        Message(role=MessageRole.ALICE, text="yo", ts=_t0()),
    ]
    ctx = _mk_ctx(max_bubbles=2, recent=recent)
    plan = _mk_plan(target=5)

    out = speaker.compose(ctx, plan)
    assert out == ["a", "b"]  # capped to n=2

    assert len(fake.calls) == 1
    call = fake.calls[0]
    fmt = call["text_format"]
    assert fmt["type"] == "json_schema"
    assert fmt["strict"] is True
    assert fmt["schema"]["properties"]["bubbles"]["maxItems"] == 2  # n after cap


def test_input_messages_roles_and_order():
    fake = FakeClient(raw=json.dumps({"bubbles": ["ok"]}))
    speaker = OpenAISpeaker(client=fake, system_prompt="SYS")

    recent = [
        Message(role=MessageRole.USER, text="u1", ts=_t0()),
        Message(role=MessageRole.ALICE, text="a1", ts=_t0()),
    ]
    ctx = _mk_ctx(max_bubbles=3, recent=recent)
    plan = _mk_plan(target=1)

    speaker.compose(ctx, plan)

    msgs = fake.calls[0]["input_messages"]
    assert msgs[0]["role"] == "system"
    assert msgs[0]["content"] == "SYS"

    # recent dialogue mapped
    assert msgs[1]["role"] == "user" and msgs[1]["content"] == "u1"
    assert msgs[2]["role"] == "assistant" and msgs[2]["content"] == "a1"

    # final instruction is a user message
    assert msgs[-1]["role"] == "user"
    assert "PLAN:" in msgs[-1]["content"]
    assert "Write 1 chat bubbles" in msgs[-1]["content"]


def test_retrieved_snippets_injected_and_limited_to_8():
    retrieved = tuple(f"SNIP{i}" for i in range(10))  # 10 snippets
    fake = FakeClient(raw=json.dumps({"bubbles": ["ok"]}))
    speaker = OpenAISpeaker(client=fake)

    ctx = _mk_ctx(max_bubbles=3, recent=[], retrieved=retrieved)
    plan = _mk_plan(target=1)
    speaker.compose(ctx, plan)

    instr = fake.calls[0]["input_messages"][-1]["content"]
    # first 8 appear
    for i in range(8):
        assert f"- SNIP{i}" in instr
    # last 2 should NOT
    assert "SNIP8" not in instr
    assert "SNIP9" not in instr


def test_filters_blank_and_non_string_and_fallback_when_empty():
    fake = FakeClient(raw=json.dumps({"bubbles": ["   ", "", 123, None]}))
    speaker = OpenAISpeaker(client=fake)

    ctx = _mk_ctx(max_bubbles=3, recent=[], retrieved=())
    plan = _mk_plan(target=2)

    out = speaker.compose(ctx, plan)
    assert out == ["嗯。"]  # fallback


def test_truncates_to_n_preserves_order_and_strips():
    fake = FakeClient(raw=json.dumps({"bubbles": ["  a  ", "b", "c", "d"]}))
    speaker = OpenAISpeaker(client=fake)

    ctx = _mk_ctx(max_bubbles=10, recent=[], retrieved=())
    plan = _mk_plan(target=3)

    out = speaker.compose(ctx, plan)
    assert out == ["a", "b", "c"]


def test_invalid_json_raises():
    fake = FakeClient(raw="not json")
    speaker = OpenAISpeaker(client=fake)

    ctx = _mk_ctx(max_bubbles=3, recent=[], retrieved=())
    plan = _mk_plan(target=1)

    with pytest.raises(json.JSONDecodeError):
        speaker.compose(ctx, plan)
