# alice/Domain/utils.py
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


def clamp(x: float, lo: float, hi: float) -> float:
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x


def utc_now() -> datetime:
    """Domain-safe helper; infra can override time via IClock in services."""
    return datetime.now(timezone.utc)


def isoformat(dt: datetime) -> str:
    """Stable ISO string (timezone-aware recommended)."""
    return dt.isoformat()


def parse_datetime(s: str) -> datetime:
    """Parse ISO datetime string. Accepts timezone-aware strings."""
    # Python's fromisoformat supports many ISO forms.
    return datetime.fromisoformat(s)


def safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default
