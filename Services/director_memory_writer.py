from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence

from Domain.schemas import MemoryOps, Plan
from Services.orchestrator import DecisionContext, Interaction
from Infra.jsonl_memory_store import JsonlMemoryStore, build_memory_record


def _norm(s: str) -> str:
    # normalize for dedup
    s = (s or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


_EMAIL_RE = re.compile(r"\b[\w\.-]+@[\w\.-]+\.\w+\b", re.IGNORECASE)
# very rough: long digit runs often mean phone / id; tune as needed
_LONG_DIGITS_RE = re.compile(r"\d{8,}")


@dataclass
class DirectorMemoryWriter:
    """
    Execute Director-chosen long-term memory writes into JsonlMemoryStore.

    This writer is intentionally conservative:
    - Only writes when ops.action == "WRITE"
    - Enforces short length caps (memories are always fed back to LLM)
    - Deduplicates against recent memories
    - Optional basic PII filters
    """

    store: JsonlMemoryStore

    # hard caps
    title_max_chars: int = 60
    content_max_chars: int = 280
    max_tags: int = 6
    tag_max_chars: int = 20

    # dedup
    dedup_recent: int = 300

    # safety filters
    block_emails: bool = True
    block_long_digit_runs: bool = True

    # debug hook (optional)
    dbg: Optional[Callable[[str], None]] = None

    # ---------
    # New API (preferred)
    # ---------
    def apply(self, ctx: DecisionContext, interaction: Interaction, ops: MemoryOps) -> bool:
        """
        Returns True if written, False if skipped.
        """
        if ops is None or getattr(ops, "action", "NONE") != "WRITE":
            return False

        title = self._clean_title(getattr(ops, "title", ""))
        content = self._clean_content(getattr(ops, "content", ""))
        tags = self._clean_tags(getattr(ops, "tags", []) or [])
        importance = self._clamp01(getattr(ops, "importance", 0.0))

        # Require essentials
        if not title or not content:
            self._log("skip: missing title/content")
            return False

        # Basic safety: avoid obvious PII in long-term memory
        if self._looks_sensitive(title) or self._looks_sensitive(content):
            self._log("skip: looks_sensitive")
            return False

        # Dedup against recent store contents
        if self._is_duplicate(title, content):
            self._log("skip: duplicate")
            return False

        # Source snippets: short evidence only
        source_user = self._pick_source_user(ctx, interaction)
        source_alice = self._pick_source_alice(interaction)

        rec = build_memory_record(
            title=title,
            content=content,
            tags=tags,
            importance=importance,
            ts=ctx.now,
            source_user=source_user,
            source_alice=source_alice,
        )
        self.store.append(rec)
        self._log(f"write: title={title!r} imp={importance:.2f} tags={tags}")
        return True

    # ---------
    # Backward-compat: old orchestrator Protocol extract(...)
    # ---------
    def extract(self, ctx: DecisionContext, interaction: Interaction) -> None:
        """
        Compatibility shim: if interaction.plan has memory_ops, execute it.
        Safe to keep even after you migrate orchestrator to call apply(..., ops).
        """
        plan: Optional[Plan] = interaction.plan
        if plan is None:
            return
        ops = getattr(plan, "memory_ops", None)
        if ops is None:
            return
        try:
            self.apply(ctx, interaction, ops)
        except Exception as e:
            self._log(f"error: {e!r}")

    # ---------
    # Internals
    # ---------
    def _log(self, msg: str) -> None:
        if self.dbg:
            try:
                self.dbg(f"[mem] {msg}")
            except Exception:
                pass

    def _clamp01(self, x: float) -> float:
        try:
            v = float(x)
        except Exception:
            return 0.0
        if v < 0.0:
            return 0.0
        if v > 1.0:
            return 1.0
        return v

    def _clean_title(self, s: str) -> str:
        s = (s or "").replace("\n", " ").strip()
        if len(s) > self.title_max_chars:
            s = s[: self.title_max_chars].rstrip()
        return s

    def _clean_content(self, s: str) -> str:
        s = (s or "").replace("\n", " ").strip()
        if len(s) > self.content_max_chars:
            s = s[: self.content_max_chars].rstrip()
        return s

    def _clean_tags(self, tags: Sequence[str]) -> List[str]:
        out: List[str] = []
        for t in tags:
            tt = (str(t) if t is not None else "").replace("\n", " ").strip()
            if not tt:
                continue
            if len(tt) > self.tag_max_chars:
                tt = tt[: self.tag_max_chars].rstrip()
            if tt not in out:
                out.append(tt)
            if len(out) >= self.max_tags:
                break
        return out

    def _looks_sensitive(self, s: str) -> bool:
        if not s:
            return False
        if self.block_emails and _EMAIL_RE.search(s):
            return True
        if self.block_long_digit_runs and _LONG_DIGITS_RE.search(s):
            return True
        return False

    def _is_duplicate(self, title: str, content: str) -> bool:
        key = _norm(title) + "||" + _norm(content)
        try:
            recent = self.store.list_recent(self.dedup_recent)
        except Exception:
            return False

        for r in recent:
            k2 = _norm(getattr(r, "title", "")) + "||" + _norm(getattr(r, "content", ""))
            if k2 == key:
                return True
        return False

    def _pick_source_user(self, ctx: DecisionContext, interaction: Interaction) -> str:
        # Prefer interaction.user_input if provided; else last USER message in recent_dialogue
        if interaction.user_input:
            return interaction.user_input.strip()[:200]
        for m in reversed(ctx.recent_dialogue):
            try:
                if m.role.name == "USER" or str(m.role) == "MessageRole.USER":
                    return (m.text or "").strip()[:200]
            except Exception:
                continue
        return ""

    def _pick_source_alice(self, interaction: Interaction) -> str:
        if not interaction.sent_texts:
            return ""
        # keep only first bubble as evidence
        return (interaction.sent_texts[0] or "").strip()[:200]
