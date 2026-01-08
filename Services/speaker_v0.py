# alice/services/speaker_v0.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List

from Domain.constants import MessageRole
from Domain.schemas import Plan
from Services.orchestrator import DecisionContext


@dataclass
class RuleSpeakerV0:
    """
    Very small speaker that demonstrates retrieval usage.
    - If retrieved_texts exists: quote one line as "memory"
    - Otherwise: generic ack
    """

    max_quote_chars: int = 120

    def compose(self, ctx: DecisionContext, plan: Plan) -> List[str]:
        n = max(1, int(plan.bubble_count_target))

        # pick last user message (if any)
        last_user = ""
        for m in reversed(ctx.recent_dialogue):
            if m.role == MessageRole.USER and m.text.strip():
                last_user = m.text.strip()
                break

        bubbles: List[str] = []

        # Bubble 1: memory cue if available
        if ctx.retrieved_texts:
            # pick first retrieved line as quote (already formatted "USER: ..." / "ALICE: ...")
            q = ctx.retrieved_texts[0].strip()
            if len(q) > self.max_quote_chars:
                q = q[: self.max_quote_chars].rstrip() + "…"
            bubbles.append(f"我记得我们之前提过：{q}")
        else:
            # if no retrieval, a short acknowledgement
            if last_user:
                bubbles.append(f"嗯，我看到你说：{last_user[:80]}{'…' if len(last_user) > 80 else ''}")
            else:
                bubbles.append("嗯。")

        # Bubble 2+: simple follow-up based on plan type
        while len(bubbles) < n:
            if plan.message_type == "followup":
                bubbles.append("要不要继续讲讲你现在想怎么推进？")
            elif plan.message_type == "reply":
                bubbles.append("我可以按这个思路继续往下拆任务。")
            else:
                bubbles.append("我在。")

        return bubbles[:n]
