# alice/Domain/schemas.py
from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field, model_validator

from Domain.constants import MemoryType

from copy import deepcopy
from typing import Any, Dict


class Plan(BaseModel):
    """
    Director output contract.
    This is the *plan*, not the final user-facing text.

    Key idea:
    - SessionManager decides conversation mode and hard limits (permissions).
    - Director must obey those limits (e.g., max_bubbles_hard).
    """

    # Whether Alice should speak now.
    action: Literal["SPEAK", "STAY_SILENT"] = Field(
        ...,
        description="Whether Alice should speak now.",
    )

    # Where this decision belongs in the session system.
    # - active: high-frequency chat (1–3 min pacing)
    # - idle: low-frequency initiation budget
    # - followup: short extra message while waiting for user reply
    speak_mode: Literal["active", "idle", "followup"] = Field(
        "idle",
        description="Session-aware speak mode: active/idle/followup.",
    )

    # Category of message, free-form for now.
    message_type: str = Field(
        ...,
        description="Category of message: share/ask/check_in/diary/...",
    )

    # Why say it now; goal of the message.
    intent: str = Field(
        ...,
        description="Why say it now; goal of the message.",
    )

    # Concrete topic for this turn.
    topic: str = Field(
        ...,
        description="Concrete topic for this turn.",
    )

    # Hard limit for bubble count from SessionManager permissions.
    # Director must obey it.
    max_bubbles_hard: int = Field(
        3,
        ge=1,
        le=5,
        description="Hard cap of bubbles this turn (from SessionManager). Director must obey.",
    )

    # Target number of chat bubbles to send.
    bubble_count_target: int = Field(
        1,
        ge=1,
        le=10,
        description="Target number of chat bubbles to send. Must be <= max_bubbles_hard.",
    )

    # Overall verbosity target.
    length: Literal["short", "medium"] = Field(
        "short",
        description="Overall verbosity target.",
    )

    # Which memories to use (selected from retrieval results).
    use_memory_ids: List[str] = Field(
        default_factory=list,
        description="Memory record IDs to reference.",
    )

    # After speaking, wait this long before next proactive speak.
    cooldown_seconds: int = Field(
        60,
        ge=0,
        le=3600,
        description="After speaking, wait this long before next proactive speak.",
    )

    # Optional self-reported confidence for the plan.
    confidence: float = Field(
        0.7,
        ge=0.0,
        le=1.0,
        description="Optional self-reported confidence for the plan.",
    )

    @model_validator(mode="after")
    def _validate_bubble_count(self) -> "Plan":
        # Enforce bubble_count_target <= max_bubbles_hard
        if self.bubble_count_target > self.max_bubbles_hard:
            raise ValueError(
                f"bubble_count_target ({self.bubble_count_target}) must be <= max_bubbles_hard ({self.max_bubbles_hard})."
            )

        # If staying silent, bubble_count_target should effectively be 0/ignored.
        # We keep schema simple: enforce target=1 but downstream should ignore when STAY_SILENT.
        # Alternatively you can set it to 0 and widen constraints.
        return self


class MemoryRecordDraft(BaseModel):
    """
    Draft record to be persisted. ID is assigned by storage layer.
    """
    type: MemoryType = Field(..., description="episodic/semantic/profile")
    text: str = Field(..., min_length=1, description="Memory text content.")
    importance: float = Field(0.5, ge=0.0, le=1.0)
    tags: List[str] = Field(default_factory=list)
    source: str = Field("", description="Where it came from, e.g., chat turn id.")


class SemanticUpdateDraft(BaseModel):
    """
    Update to semantic/profile summary (optional, sparse).
    """
    key: str = Field(
        ...,
        description="Which semantic slot to update, e.g., 'user_preferences' or 'relationship_summary'.",
    )
    text: str = Field(..., description="New summary text (compact).")
    importance: float = Field(0.7, ge=0.0, le=1.0)


class MemoryWrite(BaseModel):
    """
    MemoryWriter output contract.
    """
    new_records: List[MemoryRecordDraft] = Field(default_factory=list)
    semantic_updates: List[SemanticUpdateDraft] = Field(default_factory=list)

    # For debugging or later ranking:
    notes: Optional[str] = Field(None, description="Optional internal notes (not shown to user).")


def _force_openai_strict(schema: Any) -> Any:
    """
    Patch a JSON schema to satisfy OpenAI strict json_schema constraints:
      - every object must set additionalProperties: false
      - every object with properties must set required that includes ALL property keys
        (OpenAI strict doesn't allow optional keys)
    """
    if isinstance(schema, dict):
        # If this node is an object schema
        if schema.get("type") == "object":
            # 1) no extra keys
            schema["additionalProperties"] = False

            # 2) required must include all properties keys
            props = schema.get("properties")
            if isinstance(props, dict) and props:
                schema["required"] = list(props.keys())

        # recurse into children
        for k, v in list(schema.items()):
            schema[k] = _force_openai_strict(v)
        return schema

    if isinstance(schema, list):
        return [_force_openai_strict(x) for x in schema]

    return schema


def plan_json_schema() -> dict:
    s = deepcopy(Plan.model_json_schema())
    return _force_openai_strict(s)


def memory_write_json_schema() -> dict:
    """Convenience: JSON schema for MemoryWrite for strict LLM output."""
    return MemoryWrite.model_json_schema()
