# app/repl_streaming_cleanprompt.py
from __future__ import annotations

import json
import queue
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional, Any, Dict, Tuple

from Services.orchestrator import Orchestrator
from Services.session_manager import SessionConfig, SessionManager

from Infra.openai_responses_client import OpenAIResponsesClient
from Services.openai_speaker import OpenAISpeaker
from Services.llm_director import LLMDirector
from Infra.jsonl_transcript_sink import JSONLTranscriptSink
from Infra.sqlite_state_store import SQLiteStateStore
from Services.sklearn_transcript_index import SklearnTranscriptIndex
from Infra.jsonl_transcript_reader import JSONLTranscriptReader
from Infra.jsonl_memory_store import JsonlMemoryStore
from Services.director_memory_writer import DirectorMemoryWriter

UserItem = Tuple[str, datetime]  # (text, typed_ts)


# ----------------------------
# Minimal infra (in-memory only)
# ----------------------------

@dataclass
class InMemoryStateStore:
    bundle: Optional[object] = None

    def load(self):
        return self.bundle

    def save(self, bundle):
        self.bundle = bundle


class TranscriptSink:
    def __init__(self):
        self.messages = []

    def append(self, messages):
        self.messages.extend(list(messages))


class SystemClock:
    def now(self) -> datetime:
        return datetime.now(timezone.utc)


# ----------------------------
# Pretty helpers
# ----------------------------

def _fmt_dt(dt: Any) -> str:
    if dt is None:
        return "None"
    try:
        return dt.isoformat()
    except Exception:
        return str(dt)


def _chat_state_summary(bundle: Any) -> str:
    if bundle is None:
        return "[state] bundle=None"
    chat = getattr(bundle, "chat", None)
    if chat is None:
        return "[state] chat=None"
    mode = getattr(getattr(chat, "mode", None), "value", getattr(chat, "mode", None))
    return (
        "[state] "
        f"mode={mode} "
        f"user_batch_open={getattr(chat, 'user_batch_open', None)} "
        f"user_turn_end_ready={getattr(chat, 'user_turn_end_ready', None)} "
        f"user_last_ts={_fmt_dt(getattr(chat, 'user_last_ts', None))} "
        f"user_batch_last_ts={_fmt_dt(getattr(chat, 'user_batch_last_ts', None))} "
        f"alice_last_ts={_fmt_dt(getattr(chat, 'alice_last_ts', None))} "
        f"next_allowed_speak_ts={_fmt_dt(getattr(chat, 'next_allowed_speak_ts', None))} "
        f"followups_since_user_reply={getattr(chat, 'followups_since_user_reply', None)} "
        f"idle_initiations_today={getattr(chat, 'idle_initiations_today', None)} "
        f"active_turns_since_user_msg={getattr(chat, 'active_turns_since_user_msg', None)}"
    )


# ----------------------------
# REPL
# ----------------------------

def main() -> None:
    print("Alice REPL (streaming + true interrupt + clean prompt)")
    print("Commands: /quit  /help  /debug on|off  /state on|off  /ticks on|off  /dump")
    print()

    cfg = SessionConfig()
    sm = SessionManager(cfg)

    client = OpenAIResponsesClient(model="gpt-4o")
    client2 = OpenAIResponsesClient(model="gpt-5")
    director = LLMDirector(client=client)
    speaker = OpenAISpeaker(client=client)

    reader = JSONLTranscriptReader(path="transcript.jsonl")
    store = InMemoryStateStore()
    transcript = JSONLTranscriptSink(path="transcript.jsonl")
    index = SklearnTranscriptIndex(
        reader=reader,
        max_index_messages=1200,
        analyzer="char",
        ngram_range=(2, 4),
        recency_boost=0.15,
        min_similarity=0.0,
    )
    clock = SystemClock()
    mem_store = JsonlMemoryStore("memories.jsonl")
    mem_writer = DirectorMemoryWriter(store=mem_store, dbg=print)

    flags = {"debug": True, "state": False, "ticks": False}

    user_q: "queue.Queue[UserItem]" = queue.Queue()
    stop = threading.Event()

    prompt_str = "> "
    lock = threading.Lock()
    prompt_active = {"value": False}  # mutable box for cross-thread visibility

    def ui_print(s: str) -> None:
        """
        Print without the input prompt prefix contaminating the output.
        If user is currently at the prompt, we:
          1) move to a new line
          2) print the message
          3) redraw the prompt
        """
        with lock:
            if prompt_active["value"]:
                print()  # break out of the current "> ..." line
            print(s)
            if prompt_active["value"]:
                print(prompt_str, end="", flush=True)

    def ui_print_multiline(s: str) -> None:
        for line in s.splitlines():
            ui_print(line)

    def show_help() -> None:
        ui_print("Commands:")
        ui_print("  /quit")
        ui_print("  /help")
        ui_print("  /debug on|off   (print TurnOutput.debug)")
        ui_print("  /state on|off   (print ChatState summary)")
        ui_print("  /ticks on|off   (also print debug/state on tick steps)")
        ui_print("  /dump           (print current bundle state immediately)")

    def handle_command(line: str) -> bool:
        s = line.strip()
        if not s.startswith("/"):
            return False
        parts = s.split()
        cmd = parts[0].lower()

        if cmd == "/quit":
            stop.set()
            return True
        if cmd == "/help":
            show_help()
            return True
        if cmd == "/dump":
            ui_print(_chat_state_summary(store.load()))
            return True

        def onoff(name: str) -> None:
            if len(parts) >= 2:
                v = parts[1].lower()
                if v in ("on", "true", "1"):
                    flags[name] = True
                elif v in ("off", "false", "0"):
                    flags[name] = False
            ui_print(f"[repl] {name}={flags[name]}")

        if cmd == "/debug":
            onoff("debug")
            return True
        if cmd == "/state":
            onoff("state")
            return True
        if cmd == "/ticks":
            onoff("ticks")
            return True

        ui_print(f"[repl] Unknown command: {s}")
        return True

    def poll_user_message() -> Optional[UserItem]:
        # Called from engine thread while Alice is "typing"
        try:
            line, ts = user_q.get_nowait()
        except queue.Empty:
            return None
        if handle_command(line):
            return None
        return (line, ts)

    def on_bubble_sent(text: str, ts: datetime) -> None:
        ui_print(f"Alice: {text}")

    orch = Orchestrator(
        session_manager=sm,
        director=director,
        speaker=speaker,
        state_store=store,
        clock=clock,

        transcript_sink=transcript,
        transcript_index=index,

        # ✅ long-term memory
        memory_store=mem_store,  # 用于 ctx.long_term_memories
        memory_writer=mem_writer,  # 用于 plan.memory_ops -> 落盘

        poll_user_message=poll_user_message,  # may return (text, typed_ts)
        on_bubble_sent=on_bubble_sent,  # streaming
        working_memory_max_messages=60,
        debug=True,
    )

    tick_interval = 2.0

    def after_step(out, *, is_tick: bool) -> None:
        # Don't print out.messages here (streamed already)
        if (not is_tick) or flags["ticks"]:
            if flags["debug"]:
                ui_print("[debug] " + json.dumps(out.debug or {}, ensure_ascii=False, indent=2, default=str))
            if flags["state"]:
                ui_print(_chat_state_summary(store.load()))

    def engine_loop() -> None:
        last_tick = time.time()
        try:
            while not stop.is_set():
                # user input takes priority
                try:
                    item = user_q.get_nowait()
                except queue.Empty:
                    item = None

                if item is not None:
                    line, typed_ts = item
                    if handle_command(line):
                        continue
                    out = orch.step(user_input=line, now=typed_ts)
                    after_step(out, is_tick=False)
                    continue

                # tick
                now = time.time()
                if now - last_tick >= tick_interval:
                    last_tick = now
                    out = orch.step(user_input=None)
                    if out.messages or flags["ticks"]:
                        after_step(out, is_tick=True)

                time.sleep(0.01)
        except KeyboardInterrupt:
            stop.set()

    engine = threading.Thread(target=engine_loop, daemon=True)
    engine.start()

    try:
        while not stop.is_set():
            # IMPORTANT: set prompt_active before input() so ui_print can redraw cleanly
            with lock:
                prompt_active["value"] = True
            try:
                line = input(prompt_str)
            except EOFError:
                stop.set()
                break
            finally:
                with lock:
                    prompt_active["value"] = False

            typed_ts = datetime.now(timezone.utc)
            user_q.put((line, typed_ts))

            if line.strip().lower() == "/quit":
                stop.set()
                break
    except KeyboardInterrupt:
        stop.set()
    finally:
        stop.set()
        engine.join(timeout=1.0)
        ui_print("Bye.")


if __name__ == "__main__":
    main()
