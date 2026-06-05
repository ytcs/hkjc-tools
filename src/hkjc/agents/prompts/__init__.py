"""Static agent system prompts (markdown) + a loader.

The three agents' role definitions live as markdown next to this module so they are
diff-able and reviewable. Load one with :func:`load` to seed an agent session.
"""
from __future__ import annotations

from pathlib import Path

_DIR = Path(__file__).resolve().parent
AGENTS = ("modelling_agent", "betting_strategist", "trading_agent")


def path(agent: str) -> Path:
    """Filesystem path to an agent's prompt markdown (accepts e.g. ``"modelling"``)."""
    name = agent if agent.endswith("_agent") or agent in AGENTS else f"{agent}_agent"
    aliases = {"modelling": "modelling_agent", "strategist": "betting_strategist",
               "betting": "betting_strategist", "trading": "trading_agent"}
    name = aliases.get(agent, name)
    p = _DIR / f"{name}.md"
    if not p.exists():
        raise KeyError(f"no prompt for {agent!r} (have {AGENTS})")
    return p


def load(agent: str) -> str:
    """The system prompt text for an agent (``modelling`` | ``strategist`` | ``trading``)."""
    return path(agent).read_text(encoding="utf-8")
