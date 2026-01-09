# alice/services/orchestrator.py
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import (
    Callable, List, Optional, Protocol, Sequence, Tuple, Set, Dict, Any, Union
)

from Domain.constants import MessageRole, SessionMode
from Domain.models import (
    AffectState,
    AliceMessageEvent,
    ChatState,
    Message,
    StateBundle,
    TickEvent,
    TurnOutput,
    UserMessageEvent,
    WorldState,
)
from Domain.schemas import Plan
from Infra.jsonl_memory_store import JsonlMemoryStore
from Services.session_manager import SessionManager, SpeakPermissions

import random
import time


# ----------------------------
# Protocols (ports)
# ----------------------------

class IClock(Protocol):
    def now(self) -> datetime: ...


class IStateStore(Protocol):
    """Persist/restore StateBundle. Implementation can be in-memory now, SQLite later."""
    def load(self) -> Optional[StateBundle]: ...
    def save(self, bundle: StateBundle) -> None: ...


class ITranscriptSink(Protocol):
    """
    Store transcript messages (source of truth).
    You can implement this later; orchestrator only calls append().
    """
    def append(self, messages: Sequence[Message]) -> None: ...


class IDirector(Protocol):
    def decide(self, ctx: "DecisionContext") -> Plan: ...


class ISpeaker(Protocol):
    def compose(self, ctx: "DecisionContext", plan: Plan) -> List[str]: ...


class IMemoryWriter(Protocol):
    def apply(self, ctx: "DecisionContext", interaction: "Interaction", ops: "MemoryOps") -> None: ...



class ITranscriptIndex(Protocol):
    def retrieve_texts(
        self,
        query: str,
        *,
        top_k_hits: int = 6,
        context_window: int = 1,
        roles: Optional[Set[MessageRole]] = None,
        start_ts: Optional[datetime] = None,
        end_ts: Optional[datetime] = None,
        max_chars: int = 1600,
    ) -> Tuple[str, ...]: ...


# ----------------------------
# Context objects (services-level)
# ----------------------------

@dataclass(frozen=True)
class DecisionContext:
    now: datetime
    world: WorldState
    affect: AffectState
    chat: ChatState
    perms: SpeakPermissions

    # minimal working memory
    recent_dialogue: Tuple[Message, ...] = ()
    long_term_memories: Tuple[str, ...] = ()
    # retrieval results (semantic memories + transcript chunks) later
    retrieved_texts: Tuple[str, ...] = ()


@dataclass(frozen=True)
class Interaction:
    """A single orchestrator step interaction snapshot (for memory writing later)."""
    user_input: Optional[str]
    sent_texts: Tuple[str, ...]
    plan: Optional[Plan]


# ----------------------------
# Orchestrator
# ----------------------------

PollIncoming = Union[str, Tuple[str, datetime]]  # NEW: allow timestamped user inputs


class Orchestrator:
    """
    Main loop (single-step) that wires:
      SessionManager -> Director -> Speaker -> (later MemoryWriter) -> StateStore.

    Notes about batching:
    - User messages open a batch (user_batch_open=True).
    - Batch closes only after 30s silence, which is handled by TickEvent.
    - In a REPL, you may want to call step(None) after waiting, or provide a "force close"
      helper outside orchestrator for dev convenience (not included here on purpose).
    """

    def __init__(
        self,
        *,
        session_manager: SessionManager,
        director: IDirector,
        speaker: ISpeaker,
        state_store: IStateStore,
        clock: Optional[IClock] = None,
        transcript_sink: Optional[ITranscriptSink] = None,
        memory_writer: Optional[IMemoryWriter] = None,
        # poll hook: allow mid-send user interruption
        poll_user_message: Optional[Callable[[], Optional[PollIncoming]]] = None,  # CHANGED
        # UI hook: called immediately when a bubble is actually sent
        on_bubble_sent: Optional[Callable[[str, datetime], None]] = None,

        # keep a small working memory buffer for context (optional)
        working_memory_max_messages: int = 60,
        bubble_delay_min_ms: int = 20000,
        bubble_delay_max_ms: int = 50000,
        typing_base_delay_ms: int = 120,
        typing_ms_per_char: float = 18.0,
        typing_jitter_ms: int = 120,
        transcript_index: Optional[ITranscriptIndex] = None,
        retrieval_top_k_hits: int = 6,
        retrieval_context_window: int = 1,
        retrieval_max_chars: int = 1600,

        # DEBUG
        debug: bool = False,
        debug_print: Optional[Callable[[str], None]] = None,
        debug_max_retrieved_lines: int = 6,
        debug_max_retrieved_chars: int = 140,

        # --- NEW: director throttling to avoid token explosion on frequent ticks ---
        director_throttle_active_s: int = 60,
        director_throttle_waiting_s: int = 60,
        director_throttle_idle_s: int = 300,

        memory_store: Optional[JsonlMemoryStore] = None,
        memory_context_limit: int = 200,
        memory_context_max_chars: int = 4000,

    ):
        self.sm = session_manager
        self.director = director
        self.speaker = speaker
        self.state_store = state_store
        self.clock = clock
        self.transcript_sink = transcript_sink
        self.memory_writer = memory_writer
        self.poll_user_message = poll_user_message or (lambda: None)
        self.on_bubble_sent = on_bubble_sent

        self.working_memory_max_messages = max(0, int(working_memory_max_messages))

        # in-memory working memory (recent dialogue); persisted transcript is separate
        self._recent: List[Message] = []

        self.bubble_delay_min_ms = int(bubble_delay_min_ms)
        self.bubble_delay_max_ms = int(bubble_delay_max_ms)
        if self.bubble_delay_min_ms < 0:
            self.bubble_delay_min_ms = 0
        if self.bubble_delay_max_ms < self.bubble_delay_min_ms:
            self.bubble_delay_max_ms = self.bubble_delay_min_ms

        self.typing_base_delay_ms = int(typing_base_delay_ms)
        self.typing_ms_per_char = float(typing_ms_per_char)
        self.typing_jitter_ms = int(typing_jitter_ms)

        self.transcript_index = transcript_index
        self.retrieval_top_k_hits = int(retrieval_top_k_hits)
        self.retrieval_context_window = int(retrieval_context_window)
        self.retrieval_max_chars = int(retrieval_max_chars)

        self.debug = bool(debug)
        self.debug_print = debug_print or print
        self.debug_max_retrieved_lines = int(debug_max_retrieved_lines)
        self.debug_max_retrieved_chars = int(debug_max_retrieved_chars)

        # --- NEW: next time we are allowed to call Director for proactive (non-reply) checks ---
        self._next_proactive_director_ts: Optional[datetime] = None
        self.director_throttle_active_s = int(director_throttle_active_s)
        self.director_throttle_waiting_s = int(director_throttle_waiting_s)
        self.director_throttle_idle_s = int(director_throttle_idle_s)

        self.memory_store = memory_store
        self.memory_context_limit = int(memory_context_limit)
        self.memory_context_max_chars = int(memory_context_max_chars)

    # ---------
    # Public API
    # ---------

    def step(self, user_input: Optional[str] = None, now: Optional[datetime] = None) -> TurnOutput:
        """
        Execute one step:
          - tick maintenance (close batch/detach)
          - optional user input event
          - permissions -> maybe decide/speak
          - update state + return sent messages (as Message objects)

        Returns TurnOutput with messages actually sent this step (Alice only).
        """
        now = now or (self.clock.now() if self.clock else datetime.now(timezone.utc))
        bundle = self.state_store.load() or self._default_state(now)

        # 0) Tick maintenance first: closes batches / detaches
        chat = self.sm.update(TickEvent(now), bundle.chat, now)

        # 1) Handle new user input (opens/extends batch)
        if user_input is not None and user_input.strip() != "":
            # NEW: user said something -> allow next proactive decision later from scratch
            self._next_proactive_director_ts = None

            user_msg = Message(role=MessageRole.USER, text=user_input, ts=now)
            self._append_recent(user_msg)
            if self.transcript_sink:
                self.transcript_sink.append([user_msg])

            chat = self.sm.update(UserMessageEvent(text=user_input, ts=now), chat, now)

            new_bundle = StateBundle(
                world=bundle.world,
                affect=bundle.affect,
                chat=chat,
                last_tick_time=now,
                outbox=bundle.outbox,
            )
            self.state_store.save(new_bundle)

            self._dbg(f"user_input buffered; len={len(user_input)} awaiting batch end")
            return TurnOutput(
                messages=[],
                debug={
                    "note": "user_message_received; awaiting batch end",
                    "mode": getattr(chat.mode, "value", str(chat.mode)),
                },
            )

        # 2) Decide permissions at current time
        perms = self.sm.permissions(chat, now)

        self._dbg(
            "perms "
            f"allowed={getattr(perms,'allowed_to_speak',None)} "
            f"mode={getattr(getattr(perms,'mode',None),'value',getattr(perms,'mode',None))} "
            f"reason={getattr(perms,'reason',None)} "
            f"batch_open={getattr(perms,'user_batch_open',None)} "
            f"ready={getattr(perms,'user_turn_end_ready',None)} "
            f"max_bubbles={getattr(perms,'max_bubbles_this_turn',None)}"
        )

        # 3) If not allowed, just persist ticked state (skip retrieval)
        if not perms.allowed_to_speak:
            new_bundle = StateBundle(
                world=bundle.world,
                affect=bundle.affect,
                chat=chat,
                last_tick_time=now,
                outbox=bundle.outbox,
            )
            self.state_store.save(new_bundle)
            return TurnOutput(messages=[], debug={"reason": perms.reason, "mode": perms.mode.value})

        # --- NEW: throttle Director calls for proactive (non-reply) evaluations ---
        if not perms.user_turn_end_ready:
            if self._next_proactive_director_ts is not None and now < self._next_proactive_director_ts:
                new_bundle = StateBundle(
                    world=bundle.world,
                    affect=bundle.affect,
                    chat=chat,
                    last_tick_time=now,
                    outbox=bundle.outbox,
                )
                self.state_store.save(new_bundle)
                return TurnOutput(
                    messages=[],
                    debug={
                        "note": "director_throttled",
                        "mode": perms.mode.value,
                        "until": self._next_proactive_director_ts.isoformat(),
                    },
                )

        # 3.5) Transcript retrieval (optional)
        retrieval_query = ""
        retrieved_texts: Tuple[str, ...] = ()
        retrieved_preview: List[str] = []

        if self.transcript_index is not None:
            retrieval_query = self._build_retrieval_query()
            if retrieval_query:
                try:
                    retrieved_texts = self.transcript_index.retrieve_texts(
                        retrieval_query,
                        top_k_hits=self.retrieval_top_k_hits,
                        context_window=self.retrieval_context_window,
                        max_chars=self.retrieval_max_chars,
                    )
                except Exception as e:
                    self._dbg(f"retrieval error: {e!r}")
                    retrieved_texts = ()

        for s in retrieved_texts[: self.debug_max_retrieved_lines]:
            s2 = (s or "").strip().replace("\n", " ")
            if len(s2) > self.debug_max_retrieved_chars:
                s2 = s2[: self.debug_max_retrieved_chars].rstrip() + "…"
            retrieved_preview.append(s2)

        self._dbg(
            f"retrieval query={retrieval_query!r} hits={len(retrieved_texts)} preview={retrieved_preview}"
        )

        long_term_memories: Tuple[str, ...] = ()
        if self.memory_store is not None:
            try:
                mem_snips = self.memory_store.format_for_context(
                    limit=self.memory_context_limit,
                    max_chars=self.memory_context_max_chars,
                )
                long_term_memories = tuple(mem_snips)
            except Exception as e:
                self._dbg(f"memory_store read error: {e!r}")
                long_term_memories = ()

        # Build context for downstream decision
        ctx = DecisionContext(
            now=now,
            world=bundle.world,
            affect=bundle.affect,
            chat=chat,
            perms=perms,
            recent_dialogue=tuple(self._recent[-self.working_memory_max_messages:]) if self.working_memory_max_messages else (),
            long_term_memories=long_term_memories,
            retrieved_texts=retrieved_texts,
        )

        # 4) Allowed to speak -> Director decides Plan
        plan: Optional[Plan] = None
        try:
            plan = self.director.decide(ctx)
        except Exception as e:
            plan = self._fallback_plan(ctx, error=str(e))
            self._dbg(f"director exception -> fallback plan; error={e!r}")

        self._dbg(
            f"plan action={plan.action} type={plan.message_type} speak_mode={plan.speak_mode} "
            f"bubble_target={plan.bubble_count_target} cooldown={plan.cooldown_seconds} "
            f"topic={plan.topic!r} intent={plan.intent!r}"
        )

        # schedule next proactive check if silent + proactive
        if plan.action == "STAY_SILENT" and not perms.user_turn_end_ready:
            if perms.mode == SessionMode.ACTIVE:
                dt = self.director_throttle_active_s
            elif perms.mode == SessionMode.WAITING:
                dt = self.director_throttle_waiting_s
            else:
                dt = self.director_throttle_idle_s
            self._next_proactive_director_ts = now + timedelta(seconds=max(1, int(dt)))

        # If Director says stay silent, clear readiness
        if plan.action == "STAY_SILENT":
            chat = self._clear_user_turn_end_ready(chat)
            new_bundle = StateBundle(
                world=bundle.world,
                affect=bundle.affect,
                chat=chat,
                last_tick_time=now,
                outbox=bundle.outbox,
            )
            self.state_store.save(new_bundle)
            return TurnOutput(
                messages=[],
                debug={
                    "note": "plan=STAY_SILENT",
                    "mode": perms.mode.value,
                    "retrieval_query": retrieval_query,
                    "retrieved_count": len(retrieved_texts),
                    "retrieved_preview": retrieved_preview,
                    "plan": {
                        "action": plan.action,
                        "message_type": plan.message_type,
                        "speak_mode": plan.speak_mode,
                        "bubble_count_target": plan.bubble_count_target,
                        "cooldown_seconds": plan.cooldown_seconds,
                        "topic": plan.topic,
                        "intent": plan.intent,
                    },
                    "next_proactive_director_ts": (
                        self._next_proactive_director_ts.isoformat()
                        if self._next_proactive_director_ts is not None
                        else None
                    ),
                },
            )

        # 5) Speaker composes bubbles (cap by permissions hard limit)
        texts = self.speaker.compose(ctx, plan) or []
        cap = max(1, int(perms.max_bubbles_this_turn))
        texts_before = len(texts)
        texts = [t for t in texts if t and t.strip()][:cap]
        self._dbg(f"speaker produced={texts_before} after_cap={len(texts)} cap={cap}")

        # 6) Strategy 1: send one by one; stop if user interrupts mid-turn
        #    NEW: persist transcript per-bubble via emit callback.
        sent_msgs: List[Message] = []

        def emit_bubble(text: str, ts: datetime) -> None:
            # Persist immediately (one bubble at a time)
            if self.on_bubble_sent is not None:
                try:
                    self.on_bubble_sent(text, ts)
                except Exception:
                    pass

            m = Message(role=MessageRole.ALICE, text=text, ts=ts)
            sent_msgs.append(m)
            self._append_recent(m)
            if self.transcript_sink:
                self.transcript_sink.append([m])

            self._dbg(
                f"emit_bubble ts={ts.isoformat()} chars={len(text)} "
                f"preview={(text[:60].replace(chr(10), ' ') + ('…' if len(text) > 60 else ''))!r}"
            )

        # Send bubbles (streaming to UI should happen inside emit_bubble or via on_bubble_sent there)
        sent_pairs, interrupted_user_text, interrupted_ts = self._send_with_interrupt(
            texts,
            now,
            emit=emit_bubble,  # ✅ key
        )
        sent_texts = [t for (t, _) in sent_pairs]

        self._dbg(
            f"send sent_pairs={len(sent_pairs)} interrupted={interrupted_user_text is not None} "
            f"interrupted_ts={(interrupted_ts.isoformat() if interrupted_ts else None)}"
        )

        # NOTE:
        # - Do NOT append transcript again based on sent_pairs; it would duplicate writes.
        # - sent_msgs is already populated in-order, one-by-one as bubbles were sent.

        # 7) Update chat state with AliceMessageEvent
        if sent_pairs:
            last_sent_ts = sent_pairs[-1][1]
            is_followup = (perms.mode == SessionMode.WAITING and not perms.user_turn_end_ready)
            is_idle_initiation = (perms.mode == SessionMode.IDLE and not perms.user_turn_end_ready)

            chat = self.sm.update(
                AliceMessageEvent(messages=sent_texts, ts=last_sent_ts),
                chat,
                last_sent_ts,
                cooldown_seconds=plan.cooldown_seconds,
                is_idle_initiation=is_idle_initiation,
                is_followup=is_followup,
            )
        else:
            chat = self._clear_user_turn_end_ready(chat)

        # 8) If interrupted: process the user message as new UserMessageEvent
        if interrupted_user_text is not None and interrupted_user_text.strip() != "":
            self._next_proactive_director_ts = None

            it_ts = interrupted_ts or now
            user_msg = Message(role=MessageRole.USER, text=interrupted_user_text, ts=it_ts)
            self._append_recent(user_msg)
            if self.transcript_sink:
                self.transcript_sink.append([user_msg])
            chat = self.sm.update(UserMessageEvent(text=interrupted_user_text, ts=it_ts), chat, it_ts)

        # 9) Persist bundle
        new_bundle = StateBundle(
            world=bundle.world,
            affect=bundle.affect,
            chat=chat,
            last_tick_time=now,
            outbox=bundle.outbox,
        )
        self.state_store.save(new_bundle)

        # memory writing hook (Director-controlled)
        if self.memory_writer and plan is not None:
            try:
                ops = plan.memory_ops
                # 建议：只在确实发出了 bubble 时才允许写长期记忆（更安全）
                if getattr(ops, "action", "NONE") == "WRITE" and sent_texts:
                    # user_input 在这个分支通常是 None（因为用户输入在前面被 buffering return 了）
                    # 所以用 retrieval_query（最后一条 user 话）当作 interaction.user_input 更合理
                    self.memory_writer.apply(
                        ctx,
                        Interaction(
                            user_input=(retrieval_query or None),
                            sent_texts=tuple(sent_texts),
                            plan=plan,
                        ),
                        ops,
                    )
            except Exception as e:
                self._dbg(f"memory_writer error: {e!r}")

        plan_debug: Dict[str, Any] = {
            "action": plan.action,
            "message_type": plan.message_type,
            "speak_mode": plan.speak_mode,
            "bubble_count_target": plan.bubble_count_target,
            "cooldown_seconds": plan.cooldown_seconds,
            "topic": plan.topic,
            "intent": plan.intent,
        }

        return TurnOutput(
            messages=sent_msgs,
            debug={
                "mode": perms.mode.value,
                "reason": perms.reason,
                "sent_count": len(sent_msgs),
                "interrupted": interrupted_user_text is not None,

                "retrieval_query": retrieval_query,
                "retrieved_count": len(retrieved_texts),
                "retrieved_preview": retrieved_preview,

                "plan": plan_debug,
                "next_proactive_director_ts": (
                    self._next_proactive_director_ts.isoformat()
                    if self._next_proactive_director_ts is not None
                    else None
                ),
            },
        )

    # ---------
    # Internals
    # ---------

    def _default_state(self, now: datetime) -> StateBundle:
        world = WorldState(now=now)
        affect = AffectState()
        chat = ChatState(mode=SessionMode.IDLE)
        return StateBundle(world=world, affect=affect, chat=chat, last_tick_time=now, outbox=[])

    def _append_recent(self, msg: Message) -> None:
        if self.working_memory_max_messages <= 0:
            return
        self._recent.append(msg)
        overflow = len(self._recent) - self.working_memory_max_messages
        if overflow > 0:
            del self._recent[:overflow]

    def _clear_user_turn_end_ready(self, chat: ChatState) -> ChatState:
        """Clear the 'ready' flag so we don't repeatedly respond to the same closed batch."""
        if not chat.user_turn_end_ready and not chat.user_batch_open:
            return chat
        return ChatState(
            mode=chat.mode,
            user_last_ts=chat.user_last_ts,
            alice_last_ts=chat.alice_last_ts,
            user_batch_open=False,
            user_batch_last_ts=chat.user_batch_last_ts,
            user_turn_end_ready=False,
            next_allowed_speak_ts=chat.next_allowed_speak_ts,
            followups_since_user_reply=chat.followups_since_user_reply,
            idle_initiations_today=chat.idle_initiations_today,
            active_turns_since_user_msg=chat.active_turns_since_user_msg,
        )

    def _typing_delay_ms(self, text: str) -> int:
        """Delay before sending the next bubble. Simulates typing."""
        n = len(text or "")
        delay = self.typing_base_delay_ms + self.typing_ms_per_char * n

        delay += 120 * (text.count("。") + text.count("."))
        delay += 80 * (text.count("，") + text.count(","))
        delay += 150 * (text.count("？") + text.count("?") + text.count("！") + text.count("!"))

        if self.typing_jitter_ms > 0:
            delay += random.randint(-self.typing_jitter_ms, self.typing_jitter_ms)

        delay_i = int(delay)
        delay_i = max(self.bubble_delay_min_ms, delay_i)
        delay_i = min(self.bubble_delay_max_ms, delay_i)
        return delay_i

    def _normalize_incoming(
        self, incoming: Optional[PollIncoming]
    ) -> Tuple[Optional[str], Optional[datetime]]:
        """Accept either str or (str, datetime)."""
        if incoming is None:
            return None, None
        if isinstance(incoming, tuple) and len(incoming) == 2:
            text, ts = incoming
            if isinstance(text, str) and isinstance(ts, datetime):
                return text, ts
        if isinstance(incoming, str):
            return incoming, None
        # Unknown shape: treat as no message
        return None, None

    def _send_with_interrupt(
            self,
            texts: List[str],
            base_now: datetime,
            *,
            emit: Optional[Callable[[str, datetime], None]] = None,  # NEW
    ) -> Tuple[List[Tuple[str, datetime]], Optional[str], Optional[datetime]]:
        """
        Send bubble-by-bubble and stop if user interrupts mid-turn.

        Returns:
          - sent_pairs: list of (text, ts) for each bubble actually sent
          - interrupted_user_text: user text if interruption happened, else None
          - interrupted_ts: timestamp when interruption was detected
              - if poll provides typed_ts, use that
              - else fallback to (base_now + elapsed)
        """
        sent: List[Tuple[str, datetime]] = []

        poll_interval_s = 0.05
        elapsed_s = 0.0

        for i, t in enumerate(texts):
            ts = base_now + timedelta(seconds=elapsed_s)
            sent.append((t, ts))

            # NEW: emit immediately (caller can persist transcript per-bubble)
            if emit is not None:
                try:
                    emit(t, ts)
                except Exception:
                    pass
            else:
                # backward compatible: still stream to UI immediately
                if self.on_bubble_sent is not None:
                    try:
                        self.on_bubble_sent(t, ts)
                    except Exception:
                        pass

            # After last bubble, don't wait, but still allow immediate interrupt capture
            if i == len(texts) - 1:
                incoming = self.poll_user_message()
                incoming_text, incoming_ts = self._normalize_incoming(incoming)
                if incoming_text is not None:
                    return sent, incoming_text, (incoming_ts or (base_now + timedelta(seconds=elapsed_s)))
                return sent, None, None

            # delay before next bubble (typing model based on NEXT bubble text)
            delay_ms = self._typing_delay_ms(texts[i + 1])
            remaining = delay_ms / 1000.0

            # During delay, keep polling so user can interrupt
            while remaining > 0:
                incoming = self.poll_user_message()
                incoming_text, incoming_ts = self._normalize_incoming(incoming)
                if incoming_text is not None:
                    return sent, incoming_text, (incoming_ts or (base_now + timedelta(seconds=elapsed_s)))

                step = poll_interval_s if remaining > poll_interval_s else remaining
                time.sleep(step)
                remaining -= step
                elapsed_s += step

            # One more poll right before next bubble
            incoming = self.poll_user_message()
            incoming_text, incoming_ts = self._normalize_incoming(incoming)
            if incoming_text is not None:
                return sent, incoming_text, (incoming_ts or (base_now + timedelta(seconds=elapsed_s)))

        return sent, None, None

    def _fallback_plan(self, ctx: DecisionContext, error: str) -> Plan:
        """Minimal fallback when Director fails."""
        if ctx.perms.user_turn_end_ready:
            return Plan(
                action="SPEAK",
                speak_mode="active" if ctx.perms.mode == SessionMode.ACTIVE else "idle",
                message_type="reply",
                intent="respond after user finished batch",
                topic="ack",
                max_bubbles_hard=ctx.perms.max_bubbles_this_turn,
                bubble_count_target=1,
                length="short",
                use_memory_ids=[],
                cooldown_seconds=60,
                confidence=0.2,
            )
        return Plan(
            action="STAY_SILENT",
            speak_mode="idle",
            message_type="none",
            intent=f"director_error:{error}",
            topic="",
            max_bubbles_hard=ctx.perms.max_bubbles_this_turn,
            bubble_count_target=1,
            length="short",
            use_memory_ids=[],
            cooldown_seconds=60,
            confidence=0.0,
        )

    def _build_retrieval_query(self) -> str:
        """Pick a query string for transcript retrieval."""
        for m in reversed(self._recent):
            if m.role == MessageRole.USER and m.text.strip():
                return m.text.strip()
        if self._recent and self._recent[-1].text.strip():
            return self._recent[-1].text.strip()
        return ""

    def _dbg(self, msg: str) -> None:
        if self.debug:
            self.debug_print(f"[orch] {msg}")
