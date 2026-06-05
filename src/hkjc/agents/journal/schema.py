"""The multi-agent journal entry schema.

Three agents (modelling, strategist, trading) talk across sessions through this journal.
Beyond a durable finding, an entry carries WHO wrote it (``author``), WHO it is for
(``audience`` — the inbox routing), and WHICH conversation it belongs to (``channel`` +
``thread``), so a trading-agent observation can be addressed to the strategist and picked
up next session, and a chain of replies stays linked.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

# Controlled vocabularies — conventions the agents reuse (not hard-enforced).
AUTHORS = ("modelling", "strategist", "trading", "human")
CHANNELS = ("research", "strategy", "ops", "handoff")
KINDS = ("finding", "hypothesis", "result", "decision", "gotcha", "observation", "deadend")
STATUSES = ("open", "confirmed", "refuted", "superseded", "archived", "actioned")
ALL = "all"

META_KEYS = ("id", "author", "audience", "channel", "kind", "status", "thread",
             "tags", "refs", "created", "updated")
JSON_KEYS = ("audience", "tags", "refs")


@dataclass
class Entry:
    """One journal entry — durable, addressable, threadable."""

    id: str
    author: str = "modelling"
    audience: List[str] = field(default_factory=lambda: [ALL])
    channel: str = "research"
    kind: str = "finding"
    status: str = "open"
    thread: Optional[str] = None          # parent entry id (links a reply onto a prior entry)
    tags: List[str] = field(default_factory=list)
    refs: Dict[str, object] = field(default_factory=dict)   # race_id/date/venue/pool/constants
    created: str = ""
    updated: str = ""
    title: str = ""
    body: str = ""

    def summary(self, n: int = 200) -> str:
        flat = " ".join(self.body.split())
        return flat[:n] + ("…" if len(flat) > n else "")

    def addressed_to(self, agent: str) -> bool:
        """True if this entry is in ``agent``'s inbox (explicitly or via the ``all`` fan-out)."""
        aud = self.audience or [ALL]
        return agent in aud or ALL in aud

    def __str__(self) -> str:  # pragma: no cover - display helper
        to = ",".join(self.audience or [ALL])
        tags = (" [" + ", ".join(self.tags) + "]") if self.tags else ""
        return f"{self.id} {self.author}→{to} ({self.channel}/{self.kind}/{self.status}){tags}: {self.title}"
