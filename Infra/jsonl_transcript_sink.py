# alice/infra/jsonl_transcript_sink.py
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Dict, Any

from Domain.models import Message


@dataclass
class JSONLTranscriptSink:
    """
    Append-only transcript sink using JSON Lines (one message per line).

    Each appended line is a JSON object:
      {
        "role": "...",
        "text": "...",
        "ts": "...",
        ... optional meta fields ...
      }

    This is designed to be the "source of truth" log.
    """

    path: str
    meta: Optional[Dict[str, Any]] = None  # optional static metadata added to every line
    fsync: bool = False  # set True if you want stronger durability per append

    def __post_init__(self) -> None:
        p = Path(self.path)
        p.parent.mkdir(parents=True, exist_ok=True)
        # Touch file so tooling sees it early
        p.touch(exist_ok=True)

    def append(self, messages: Sequence[Message]) -> None:
        if not messages:
            return

        p = Path(self.path)
        with p.open("a", encoding="utf-8") as f:
            for m in messages:
                obj = m.to_dict()
                if self.meta:
                    # meta fields won't overwrite message fields
                    for k, v in self.meta.items():
                        obj.setdefault(k, v)
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")

            f.flush()
            if self.fsync:
                import os
                os.fsync(f.fileno())
