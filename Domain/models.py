# alice/Domain/models.py
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

from Domain.constants import SessionMode, MessageRole, MemoryType
from Domain.utils import isoformat, parse_datetime



# ----------------------------
# Core runtime states
# ----------------------------

@dataclass(frozen=True)
class WorldState:
    """
    External world snapshot (coarse-grained; enough to justify actions).
    Keep it small and stable; you can extend later.
    """
    now: datetime
    daypart: str = "unknown"     # e.g. morning/noon/afternoon/evening/night/late_night
    place: str = "room"          # room/outside/shop/transit/quiet
    activity: str = "idle"       # lying/walking/reading/scrolling/...
    progress: float = 0.0        # 0..1 or 0..100; you choose convention
    weather: str = "abstract"    # hot/muggy/breezy/light_rain/...

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["now"] = isoformat(self.now)
        return d

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "WorldState":
        return WorldState(
            now=parse_datetime(d["now"]),
            daypart=d.get("daypart", "unknown"),
            place=d.get("place", "room"),
            activity=d.get("activity", "idle"),
            progress=float(d.get("progress", 0.0)),
            weather=d.get("weather", "abstract"),
        )


@dataclass(frozen=True)
class AffectState:
    valence: float = 0.0
    arousal: float = 0.2
    closeness: float = 0.3
    curiosity: float = 0.3
    fatigue: float = 0.2

    energy: int = 100          # 0..100
    mood_seed: str = "neutral" # e.g. neutral/calm/happy/irritated/empty

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "AffectState":
        return AffectState(
            valence=float(d.get("valence", 0.0)),
            arousal=float(d.get("arousal", 0.2)),
            closeness=float(d.get("closeness", 0.3)),
            curiosity=float(d.get("curiosity", 0.3)),
            fatigue=float(d.get("fatigue", 0.2)),
            energy=int(d.get("energy", 100)),
            mood_seed=str(d.get("mood_seed", "neutral")),
        )



@dataclass(frozen=True)
class ChatState:
    """
    Conversation/session state maintained by SessionManager.
    Stores timing, mode, budgets/cooldowns, and user-batch status.
    """

    # ChatState serialization fields explanation:
    # - mode: Current session mode decided by SessionManager: IDLE / ACTIVE / WAITING.
    # - user_last_ts: Timestamp of the user's most recent message.
    #   Used to detect detach (e.g., user silent >= 5 min => IDLE).
    # - alice_last_ts: Timestamp of Alice's most recent sent message batch.
    #   Used for cooldown/pacing and WAITING timers.
    # - user_batch_open: Whether we are currently collecting a "user batch" (multiple user messages).
    #   While open, Alice should not reply/interject.
    # - user_batch_last_ts: Timestamp of the last user message within the current batch.
    #   Batch ends when now - user_batch_last_ts >= batch_end_silence_seconds (e.g., 30s).
    # - user_turn_end_ready: A flag indicating the user's batch has ended and Alice may take a turn.
    #   Should be cleared after Alice processes the turn (speaks or decides to stay silent).
    # - next_allowed_speak_ts: Earliest time Alice is allowed to speak again (cooldown / active pacing).
    # - followups_since_user_reply: Number of short follow-up messages Alice sent without a user reply.
    #   Reset to 0 when the user replies.
    # - idle_initiations_today: Count of proactive "chat initiations" in IDLE mode for daily budgeting.
    # - active_turns_since_user_msg: Count of Alice speaking turns in a since last user message

    mode: SessionMode = SessionMode.IDLE

    # Timing
    user_last_ts: Optional[datetime] = None
    alice_last_ts: Optional[datetime] = None

    # User batching: user messages are considered a "batch" until silence >= batch_end_silence_seconds
    user_batch_open: bool = False
    user_batch_last_ts: Optional[datetime] = None  # last message within current batch
    user_turn_end_ready: bool = False              # becomes True once batch closes (a "turn boundary")

    # Active pacing / cooldown
    next_allowed_speak_ts: Optional[datetime] = None
    followups_since_user_reply: int = 0

    # Budgets (you can interpret these as counters with a reset policy in services)
    idle_initiations_today: int = 0
    active_turns_since_user_msg: int = 0

    def to_dict(self) -> Dict[str, Any]:
        def dt(x: Optional[datetime]) -> Optional[str]:
            return isoformat(x) if x is not None else None

        return {
            "mode": self.mode.value,
            "user_last_ts": dt(self.user_last_ts),
            "alice_last_ts": dt(self.alice_last_ts),
            "user_batch_open": self.user_batch_open,
            "user_batch_last_ts": dt(self.user_batch_last_ts),
            "user_turn_end_ready": self.user_turn_end_ready,
            "next_allowed_speak_ts": dt(self.next_allowed_speak_ts),
            "followups_since_user_reply": self.followups_since_user_reply,
            "idle_initiations_today": self.idle_initiations_today,
            "active_turns_since_user_msg": self.active_turns_since_user_msg,
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "ChatState":
        def p(x: Optional[str]) -> Optional[datetime]:
            return parse_datetime(x) if x else None

        return ChatState(
            mode=SessionMode(d.get("mode", SessionMode.IDLE.value)),
            user_last_ts=p(d.get("user_last_ts")),
            alice_last_ts=p(d.get("alice_last_ts")),
            user_batch_open=bool(d.get("user_batch_open", False)),
            user_batch_last_ts=p(d.get("user_batch_last_ts")),
            user_turn_end_ready=bool(d.get("user_turn_end_ready", False)),
            next_allowed_speak_ts=p(d.get("next_allowed_speak_ts")),
            followups_since_user_reply=int(d.get("followups_since_user_reply", 0)),
            idle_initiations_today=int(d.get("idle_initiations_today", 0)),
            active_turns_since_user_msg=int(d.get("active_turns_since_user_msg", 0)),
        )


# ----------------------------
# Dialogue messages & outputs
# ----------------------------

@dataclass(frozen=True)
class Message:
    """
    A single chat bubble.
    'role' distinguishes user/alice/system.
    """
    role: MessageRole
    text: str
    ts: datetime

    def to_dict(self) -> Dict[str, Any]:
        return {"role": self.role.value, "text": self.text, "ts": isoformat(self.ts)}

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Message":
        return Message(
            role=MessageRole(d["role"]),
            text=d["text"],
            ts=parse_datetime(d["ts"]),
        )


@dataclass(frozen=True)
class TurnOutput:
    """
    Output of a single orchestrator step.
    """
    messages: List[Message] = field(default_factory=list)
    debug: Dict[str, Any] = field(default_factory=dict)


# ----------------------------
# Memory records
# ----------------------------

@dataclass(frozen=True)
class MemoryRecord:
    """
    A persisted memory record.
    Embedding is typically stored in FAISS, not necessarily here,
    but keeping it optional helps in tests or small-scale experiments.
    """
    id: str
    type: MemoryType
    text: str

    created_at: datetime
    last_accessed_at: Optional[datetime] = None

    importance: float = 0.5  # 0..1
    tags: Tuple[str, ...] = field(default_factory=tuple)
    source: str = ""         # e.g. "chat:<conv_id>:<turn_id>" or "tick:<timestamp>"

    def to_dict(self) -> Dict[str, Any]:
        def dt(x: Optional[datetime]) -> Optional[str]:
            return isoformat(x) if x else None

        return {
            "id": self.id,
            "type": self.type.value,
            "text": self.text,
            "created_at": isoformat(self.created_at),
            "last_accessed_at": dt(self.last_accessed_at),
            "importance": self.importance,
            "tags": list(self.tags),
            "source": self.source,
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "MemoryRecord":
        return MemoryRecord(
            id=d["id"],
            type=MemoryType(d["type"]),
            text=d["text"],
            created_at=parse_datetime(d["created_at"]),
            last_accessed_at=parse_datetime(d["last_accessed_at"]) if d.get("last_accessed_at") else None,
            importance=float(d.get("importance", 0.5)),
            tags=tuple(d.get("tags", [])),
            source=d.get("source", ""),
        )


# ----------------------------
# Events (for SessionManager / Orchestrator)
# ----------------------------

@dataclass(frozen=True)
class UserMessageEvent:
    text: str
    ts: datetime


@dataclass(frozen=True)
class AliceMessageEvent:
    messages: Sequence[str]  # bubble texts
    ts: datetime


@dataclass(frozen=True)
class TickEvent:
    """
    Time progression without user input (e.g., for background pacing or offline simulation).
    """
    ts: datetime

@dataclass(frozen=True)
class UserAwayEvent:
    ts: datetime



# ----------------------------
# State bundle (for persistence)
# ----------------------------

@dataclass(frozen=True)
class StateBundle:
    """
    Everything needed to resume: world + affect + chat + memory pointers + outbox + last_tick_time.
    Memory content itself typically resides in SQLite+FAISS; here we track runtime state.
    """
    world: WorldState
    affect: AffectState
    chat: ChatState
    last_tick_time: datetime
    outbox: List[Message] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "world": self.world.to_dict(),
            "affect": self.affect.to_dict(),
            "chat": self.chat.to_dict(),
            "last_tick_time": isoformat(self.last_tick_time),
            "outbox": [m.to_dict() for m in self.outbox],
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "StateBundle":
        return StateBundle(
            world=WorldState.from_dict(d["world"]),
            affect=AffectState.from_dict(d["affect"]),
            chat=ChatState.from_dict(d["chat"]),
            last_tick_time=parse_datetime(d["last_tick_time"]),
            outbox=[Message.from_dict(x) for x in d.get("outbox", [])],
        )
