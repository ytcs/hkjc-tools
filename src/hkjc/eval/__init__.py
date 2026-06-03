"""Evaluation layer — the harness that judges everything else.

Nothing in the package imports ``eval``; ``eval`` reads data/model outputs and
scores them. Keeping the judge un-depended-on is deliberate: the objective
function of the research loop must not be reachable (and thus game-able) from
inside the things it evaluates.

- :mod:`~hkjc.eval.clv` — closing-line value: de-vig implied probabilities,
  opening/closing snapshots, odds drift (steam), and CLV of placed bets.
"""
from . import clv

__all__ = ["clv"]
