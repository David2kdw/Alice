# alice/services/transcript_retriever.py
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Sequence, Set, Tuple

from Domain.constants import MessageRole
from Domain.models import Message
from Infra.jsonl_transcript_reader import JSONLTranscriptReader


def _tokenize(s: str) -> List[str]:
    # Simple tokenizer: English words + CJK continuous blocks
    s = (s or "").lower()
    # Split into alnum chunks OR CJK runs
    parts = re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]+", s)
    return [p for p in parts if p]


def _score(query: str, tokens: List[str], text: str) -> float:
    """
    Very simple relevance score:
    - exact substring match gets a big boost
    - token occurrences add smaller boosts
    """
    q = (query or "").strip().lower()
    t = (text or "").lower()

    if not q:
        return 0.0

    score = 0.0
    if q in t:
        score += 10.0

    for tok in tokens:
        # count occurrences (cap per token so spam doesn't dominate)
        c = t.count(tok)
        if c > 0:
            score += min(3.0, 0.8 * c)

    return score


def format_messages_for_context(messages: Sequence[Message], *, max_chars: int = 1600) -> Tuple[str, ...]:
    """
    Convert messages to compact strings for DecisionContext.retrieved_texts
    with a hard char budget.
    """
    out: List[str] = []
    used = 0
    for m in messages:
        prefix = "USER" if m.role == MessageRole.USER else "ALICE" if m.role == MessageRole.ALICE else "SYSTEM"
        line = f"{prefix}: {m.text}".strip()
        if not line:
            continue
        if used + len(line) + 1 > max_chars:
            break
        out.append(line)
        used += len(line) + 1
    return tuple(out)


@dataclass
class TranscriptRetriever:
    """
    Brute-force transcript retriever (no embeddings yet).

    Strategy:
    - tail last `max_scan_messages` messages
    - score each message for relevance to query
    - take top hits, then expand with `context_window` neighbors
    """

    reader: JSONLTranscriptReader
    max_scan_messages: int = 500
    top_k_hits: int = 6
    context_window: int = 1  # include +/- neighbors around each hit

    def retrieve_messages(
        self,
        query: str,
        *,
        now: Optional[datetime] = None,
        roles: Optional[Set[MessageRole]] = None,
        start_ts: Optional[datetime] = None,
        end_ts: Optional[datetime] = None,
    ) -> List[Message]:
        """
        Return a list of messages (chronological) suitable as "retrieved context".
        """
        query = (query or "").strip()
        if not query:
            return []

        candidates = self.reader.tail_messages(
            n=self.max_scan_messages,
            roles=roles,
            start_ts=start_ts,
            end_ts=end_ts,
        )
        if not candidates:
            return []

        tokens = _tokenize(query)

        # score
        scored: List[Tuple[float, int]] = []
        for idx, m in enumerate(candidates):
            s = _score(query, tokens, m.text)
            if s <= 0:
                continue
            # tiny recency bump (later in list is more recent)
            if len(candidates) > 1:
                s += 0.2 * (idx / (len(candidates) - 1))
            scored.append((s, idx))

        if not scored:
            return []

        scored.sort(key=lambda x: x[0], reverse=True)
        hit_indices = [idx for _, idx in scored[: self.top_k_hits]]

        # expand with neighbors
        picked: Set[int] = set()
        for idx in hit_indices:
            for j in range(idx - self.context_window, idx + self.context_window + 1):
                if 0 <= j < len(candidates):
                    picked.add(j)

        # keep chronological order
        result = [candidates[i] for i in sorted(picked)]
        return result

    def retrieve_texts(
        self,
        query: str,
        *,
        roles: Optional[Set[MessageRole]] = None,
        start_ts: Optional[datetime] = None,
        end_ts: Optional[datetime] = None,
        max_chars: int = 1600,
    ) -> Tuple[str, ...]:
        msgs = self.retrieve_messages(query, roles=roles, start_ts=start_ts, end_ts=end_ts)
        return format_messages_for_context(msgs, max_chars=max_chars)
