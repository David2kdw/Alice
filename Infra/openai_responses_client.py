# infra/openai_responses_client.py
from __future__ import annotations

import random
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

from openai import OpenAI


def _is_reasoning_model(model: str) -> bool:
    m = (model or "").lower()
    # gpt-5 / o* generally behave like "reasoning models"
    return m.startswith("gpt-5") or m.startswith("o")


def _extract_text_from_response(resp: Any) -> str:
    """
    Robustly extract visible assistant text from Responses API response.
    Prefer resp.output_text helper, but fall back to scanning resp.output.
    """
    # 1) happy path
    text = getattr(resp, "output_text", None)
    if isinstance(text, str) and text.strip():
        return text.strip()

    # 2) scan output items
    out = getattr(resp, "output", None)
    if not out:
        return ""

    chunks: List[str] = []

    for item in out:
        # openai-python usually gives objects with .type; be defensive for dicts too
        itype = getattr(item, "type", None) or (item.get("type") if isinstance(item, dict) else None)
        if itype != "message":
            continue

        content = getattr(item, "content", None) or (item.get("content") if isinstance(item, dict) else None)
        if not content:
            continue

        for c in content:
            ctype = getattr(c, "type", None) or (c.get("type") if isinstance(c, dict) else None)

            # Most common: {"type":"output_text","text":"..."}
            if ctype in ("output_text", "text"):
                t = getattr(c, "text", None) or (c.get("text") if isinstance(c, dict) else None)
                if isinstance(t, str) and t:
                    chunks.append(t)

            # Some SDK shapes can nest as {"type":"output_text","text":{"value":"..."}}; be defensive
            elif isinstance(c, dict):
                t = c.get("text")
                if isinstance(t, dict) and isinstance(t.get("value"), str):
                    chunks.append(t["value"])

    return "".join(chunks).strip()


@dataclass
class OpenAIResponsesClient:
    model: str
    max_output_tokens: int = 1500
    temperature: float = 0.7
    max_retries: int = 5
    reasoning_effort: str = "low"  # "low" | "medium" | "high" (reasoning models)

    def __post_init__(self) -> None:
        self._client = OpenAI()

    def create_text(
        self,
        *,
        input_messages: List[Dict[str, str]],
        text_format: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Returns visible assistant text. If none is produced, raise to trigger retry.
        If text_format is provided, it should be the content of `text={"format": ...}`.
        """
        attempt = 0
        while True:
            try:
                kwargs: Dict[str, Any] = dict(
                    model=self.model,
                    input=input_messages,
                    max_output_tokens=int(self.max_output_tokens),
                )

                # reasoning models: set reasoning effort; also avoid unsupported sampling params like temperature
                if _is_reasoning_model(self.model):
                    kwargs["reasoning"] = {"effort": self.reasoning_effort}
                else:
                    kwargs["temperature"] = float(self.temperature)

                if text_format is not None:
                    kwargs["text"] = {"format": text_format}

                resp = self._client.responses.create(**kwargs)

                text = _extract_text_from_response(resp)
                if not text:
                    # Important: do NOT return "" — this would later crash json.loads("")
                    raise ValueError("Empty visible output_text (response contained no message content).")
                return text

            except Exception:
                attempt += 1
                if attempt > self.max_retries:
                    raise
                sleep_s = min(8.0, (2 ** (attempt - 1)) * 0.25) + random.random() * 0.25
                time.sleep(sleep_s)
