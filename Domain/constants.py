# alice/Domain/constants.py
from __future__ import annotations

from enum import Enum


class SessionMode(str, Enum):
    """Conversation session mode (decided by SessionManager)."""
    IDLE = "IDLE"        # user detached; low-frequency initiation budget
    ACTIVE = "ACTIVE"    # in active chat; allow frequent proactive speech
    WAITING = "WAITING"  # Alice has spoken; waiting user response / follow-up logic


class MemoryType(str, Enum):
    """Type of memory record."""
    EPISODIC = "episodic"     # events / episodes
    SEMANTIC = "semantic"     # stable facts / summaries / preferences
    PROFILE = "profile"       # compact self/relationship summary (optional)


class MessageRole(str, Enum):
    """Role for a message in dialogue history."""
    USER = "user"
    ALICE = "alice"
    SYSTEM = "system"  # internal notes; usually not shown to user
