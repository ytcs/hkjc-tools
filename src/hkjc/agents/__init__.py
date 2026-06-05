"""Agents layer — orchestration: the three agent prompts + the cross-session journal.

Nothing in the library imports ``agents``; it sits on top and drives the rest. Three agents
with clear separation of concerns communicate across sessions through one journal:

- **modelling** — owns the ABM and its calibration (physics + game-theory regularization;
  fit-the-market goodness-of-fit; the diagnose→recalibrate loop). Never bets.
- **strategist** — researches betting strategies on top of the model + pool structure.
  Consumes the model; never modifies it.
- **trading** — the live (paper) runtime: pulls odds, executes the chosen strategy, and feeds
  live observations back to the other two.

- :mod:`~hkjc.agents.journal` — the multi-agent journal (`Journal`, inbox routing) + `OpsLog`.
- :mod:`~hkjc.agents.prompts` — the three system prompts (:func:`~hkjc.agents.prompts.load`).
- :mod:`~hkjc.agents.state` — mutable per-agent working memory (`*_STATE.md`) under the journal root.
"""
from . import journal, prompts, state
from .journal import Entry, Journal, OpsLog
from .prompts import load as load_prompt

__all__ = ["journal", "prompts", "state", "Journal", "OpsLog", "Entry", "load_prompt"]
