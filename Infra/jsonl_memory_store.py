from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.isoformat()


@dataclass(frozen=True)
class MemoryRecord:
    """
    A single long-term memory record stored as one JSONL line.
    Keep it minimal and stable: these will be fed to Speaker later.
    """
    ts: str                 # ISO string
    title: str
    content: str
    tags: List[str]
    importance: float       # 0..1
    source_user: str = ""   # optional: last user utterance (short)
    source_alice: str = ""  # optional: what Alice said (short)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ts": self.ts,
            "title": self.title,
            "content": self.content,
            "tags": list(self.tags),
            "importance": float(self.importance),
            "source_user": self.source_user,
            "source_alice": self.source_alice,
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "MemoryRecord":
        return MemoryRecord(
            ts=str(d.get("ts", "")),
            title=str(d.get("title", "")),
            content=str(d.get("content", "")),
            tags=list(d.get("tags", []) or []),
            importance=float(d.get("importance", 0.0)),
            source_user=str(d.get("source_user", "")),
            source_alice=str(d.get("source_alice", "")),
        )


class JsonlMemoryStore:
    """
    Append-only JSONL store for long-term memories.
    - One memory record per line (JSON object).
    - Thread-safe within a single process.
    - Reads are tolerant of bad/corrupt lines (skips them).
    """

    def __init__(
        self,
        path: str | Path,
        *,
        encoding: str = "utf-8",
        fsync: bool = False,
    ) -> None:
        self.path = Path(path)
        self.encoding = encoding
        self.fsync = bool(fsync)
        self._lock = threading.Lock()

        if self.path.parent and str(self.path.parent) != ".":
            self.path.parent.mkdir(parents=True, exist_ok=True)

        if not self.path.exists():
            self.path.write_text("", encoding=self.encoding)

    def append(self, record: MemoryRecord) -> None:
        line = json.dumps(record.to_dict(), ensure_ascii=False, separators=(",", ":")) + "\n"
        with self._lock:
            with self.path.open("a", encoding=self.encoding, newline="\n") as f:
                f.write(line)
                f.flush()
                if self.fsync:
                    os.fsync(f.fileno())

    def append_many(self, records: Iterable[MemoryRecord]) -> None:
        lines = [
            json.dumps(r.to_dict(), ensure_ascii=False, separators=(",", ":")) + "\n"
            for r in records
        ]
        if not lines:
            return
        with self._lock:
            with self.path.open("a", encoding=self.encoding, newline="\n") as f:
                f.writelines(lines)
                f.flush()
                if self.fsync:
                    os.fsync(f.fileno())

    def iter_all(self) -> Iterable[MemoryRecord]:
        if not self.path.exists():
            return
        try:
            with self.path.open("r", encoding=self.encoding) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        d = json.loads(line)
                        if isinstance(d, dict):
                            yield MemoryRecord.from_dict(d)
                    except Exception:
                        continue
        except FileNotFoundError:
            return

    def list_recent(self, limit: int = 50) -> List[MemoryRecord]:
        if limit <= 0:
            return []
        items = list(self.iter_all())
        return items[-limit:]

    def format_for_context(self, limit: int = 30, *, max_chars: int = 1600) -> List[str]:
        recs = self.list_recent(limit=limit)
        out: List[str] = []
        used = 0
        for r in recs:
            s = f"[MEM] {r.title}: {r.content}".strip()
            if r.tags:
                s += f" (tags: {', '.join(r.tags)})"
            if r.source_user:
                s += f" | src_user: {r.source_user}"
            if r.source_alice:
                s += f" | src_alice: {r.source_alice}"

            if not s:
                continue
            if used + len(s) > max_chars:
                break
            out.append(s)
            used += len(s)
        return out

    def prune_keep_last(self, keep_last: int = 2000) -> int:
        if keep_last <= 0:
            keep_last = 0
        with self._lock:
            recs = list(self.iter_all())
            recs = recs[-keep_last:] if keep_last else []
            tmp = self.path.with_suffix(self.path.suffix + ".tmp")
            with tmp.open("w", encoding=self.encoding, newline="\n") as f:
                for r in recs:
                    f.write(json.dumps(r.to_dict(), ensure_ascii=False, separators=(",", ":")) + "\n")
                f.flush()
                if self.fsync:
                    os.fsync(f.fileno())
            tmp.replace(self.path)
            return len(recs)


def build_memory_record(
    *,
    title: str,
    content: str,
    tags: Optional[List[str]] = None,
    importance: float = 0.0,
    ts: Optional[datetime] = None,
    source_user: str = "",
    source_alice: str = "",
) -> MemoryRecord:
    ts = ts or _utc_now()
    return MemoryRecord(
        ts=_iso(ts),
        title=title.strip(),
        content=content.strip(),
        tags=list(tags or []),
        importance=float(importance),
        source_user=(source_user.strip()[:200] if source_user else ""),
        source_alice=(source_alice.strip()[:200] if source_alice else ""),
    )
