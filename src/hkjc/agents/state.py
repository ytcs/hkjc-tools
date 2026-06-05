"""Per-agent living state — mutable working memory under the journal root.

Distinct from the append-only journal: ``*_STATE.md`` are documents an agent rewrites in
place (the modelling agent's current constants + residual queue, the strategist's active
strategies, the trading agent's bankroll model + known frictions). The journal records the
append-only *why* the state changed; the state file is the current *what*. Lives under
``{journal_root}/state/`` so it travels with the journal, not the installed package.
"""
from __future__ import annotations

from pathlib import Path

from .journal.store import DEFAULT_ROOT

STATE_DIR = DEFAULT_ROOT / "state"
_FILES = {"modelling": "MODEL_STATE.md", "model": "MODEL_STATE.md",
          "strategist": "STRATEGY_STATE.md", "strategy": "STRATEGY_STATE.md",
          "trading": "TRADING_STATE.md"}

_STUBS = {
    "STRATEGY_STATE.md": "# Strategy state\n\n_Active strategies, parameters, and open research_ "
                         "_questions for the betting strategist. (empty — to be populated.)_\n",
    "TRADING_STATE.md": "# Trading state\n\n_Active strategy, bankroll model, and known live_ "
                        "_frictions for the trading agent. Paper-trade only. (empty — to be populated.)_\n",
}


def path(agent: str, root: Path = STATE_DIR) -> Path:
    """Path to an agent's state doc (``modelling`` | ``strategist`` | ``trading``)."""
    try:
        return Path(root) / _FILES[agent]
    except KeyError:
        raise KeyError(f"no state file for {agent!r} (have {sorted(set(_FILES))})") from None


def read(agent: str, root: Path = STATE_DIR) -> str:
    """Read an agent's state doc; returns a stub if it hasn't been created yet."""
    p = path(agent, root)
    if p.exists():
        return p.read_text(encoding="utf-8")
    return _STUBS.get(p.name, f"# {p.stem}\n\n_(empty)_\n")


def write(agent: str, text: str, root: Path = STATE_DIR) -> Path:
    """Overwrite an agent's state doc (in-place living memory)."""
    p = path(agent, root)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    return p
