# Services/speaker.py
from __future__ import annotations

from typing import List

from Domain.constants import MessageRole
from Domain.schemas import Plan


class TemplateSpeaker:
    """
    Minimal rule-based Speaker.

    Takes a Plan and produces 1..N short chat bubbles.
    Orchestrator will still enforce hard caps. :contentReference[oaicite:1]{index=1}
    """

    def compose(self, ctx, plan: Plan) -> List[str]:
        n = max(1, int(plan.bubble_count_target))

        last_user = self._last_user_text(ctx)
        mode = ctx.perms.mode.value

        # A tiny set of canned templates by message_type
        if plan.message_type == "reply":
            base = [
                "嗯嗯，我在听。",
                "你继续说也可以，我会跟着你的节奏。",
                "如果你想换个轻松点的话题也行。",
            ]
            if last_user:
                base.insert(0, f"关于你刚才说的「{self._shorten(last_user)}」：我懂。")

        elif plan.message_type == "followup":
            base = [
                "（小声）我还在这儿。",
                "不急，你忙完再回也行。",
                "我先待机一下。"
            ]

        elif plan.message_type == "check_in":
            base = [
                "在嘛？我刚刚发了会儿呆。",
                "你那边还顺利吗？",
                "如果你想聊点轻松的，我也可以。"
            ]

        else:  # share / small_talk / default
            base = [
                "我刚刚想到一个小事。",
                "今天的时间感有点奇怪……像永远的暑假那样。",
                "你现在是在做什么呢？"
            ]

        # Optionally adapt by length setting
        if plan.length == "medium":
            base = [b + "（要不要我展开讲讲？）" for b in base]

        # Pick first n bubbles
        return base[:n]

    def _last_user_text(self, ctx) -> str:
        # ctx.recent_dialogue is a tuple of Message
        for m in reversed(getattr(ctx, "recent_dialogue", ())):
            if m.role == MessageRole.USER and m.text:
                return m.text
        return ""

    def _shorten(self, s: str, max_len: int = 24) -> str:
        s = s.strip().replace("\n", " ")
        if len(s) <= max_len:
            return s
        return s[: max_len - 1] + "…"
