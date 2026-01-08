# alice/infra/jsonl_transcript_reader.py
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Sequence, Set

from Domain.constants import MessageRole
from Domain.models import Message
from Domain.utils import parse_datetime


@dataclass
class JSONLTranscriptReader:
    """
    Read transcript messages from an append-only JSONL file.

    Each line is expected to be a JSON object with at least:
      { "role": "...", "text": "...", "ts": "ISO-8601" }

    Extra keys are allowed and ignored.
    Corrupt lines are skipped (best-effort).
    """

    path: str

    # ----------------------------
    # Public APIs
    # ----------------------------

    def tail_messages(
        self,
        n: int = 200,
        *,
        roles: Optional[Set[MessageRole]] = None,
        start_ts: Optional[datetime] = None,
        end_ts: Optional[datetime] = None,
    ) -> List[Message]:
        """
        Return last N messages (chronological order) with optional filters.
        """
        if n <= 0:
            return []

        lines = self._tail_lines(Path(self.path), n)
        msgs: List[Message] = []
        for line in lines:
            m = self._parse_message_line(line)
            if m is None:
                continue
            if roles is not None and m.role not in roles:
                continue
            if start_ts is not None and m.ts < start_ts:
                continue
            if end_ts is not None and m.ts > end_ts:
                continue
            msgs.append(m)
        return msgs

    def iter_messages(
        self,
        *,
        roles: Optional[Set[MessageRole]] = None,
        start_ts: Optional[datetime] = None,
        end_ts: Optional[datetime] = None,
        limit: Optional[int] = None,
    ) -> Iterator[Message]:
        """
        Stream messages from the whole file (chronological).
        Use this when you really want full-scan.
        """
        p = Path(self.path)
        if not p.exists():
            return iter(())

        count = 0
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                m = self._parse_message_line(line)
                if m is None:
                    continue
                if roles is not None and m.role not in roles:
                    continue
                if start_ts is not None and m.ts < start_ts:
                    continue
                if end_ts is not None and m.ts > end_ts:
                    continue
                yield m
                count += 1
                if limit is not None and count >= limit:
                    break

    # ----------------------------
    # Internals
    # ----------------------------

    def _parse_message_line(self, line: str) -> Optional[Message]:
        line = line.strip()
        if not line:
            return None
        try:
            obj = json.loads(line)
            # Only keep required keys; tolerate extra meta keys.
            role = obj.get("role")
            text = obj.get("text")
            ts = obj.get("ts")
            if role is None or text is None or ts is None:
                return None
            # Message.from_dict expects {"role","text","ts"} exactly.
            return Message.from_dict({"role": role, "text": text, "ts": ts})
        except Exception:
            return None

    def _tail_lines(self, path: Path, n: int, *, block_size: int = 4096, max_bytes: int = 2_000_000) -> List[str]:
        """
        Efficiently read last N lines from a file without loading everything.

        max_bytes is a safety valve: if the tail section is enormous, we stop growing
        the buffer and just return as many trailing lines as we have.
        """
        if not path.exists():
            return []

        with path.open("rb") as f:
            f.seek(0, 2)
            end = f.tell()
            if end == 0:
                return []

            buf = b""
            pos = end
            # Keep pulling blocks from the end until we have enough newlines
            while pos > 0 and buf.count(b"\n") <= n:
                step = block_size if pos >= block_size else pos
                pos -= step
                f.seek(pos)
                chunk = f.read(step)
                buf = chunk + buf
                if len(buf) > max_bytes:
                    break

        lines = buf.splitlines()
        tail = lines[-n:] if n < len(lines) else lines
        # Decode each line safely
        return [ln.decode("utf-8", errors="replace") for ln in tail]
