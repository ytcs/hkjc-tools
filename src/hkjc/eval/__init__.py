"""Evaluation layer — the harness that judges everything else.

Nothing in the package imports ``eval``; ``eval`` reads data/model outputs and
scores them. Keeping the judge un-depended-on is deliberate: the objective
function of the research loop must not be reachable (and thus game-able) from
inside the things it evaluates.

- :mod:`~hkjc.eval.calibration` — proper scoring of predicted probabilities
  against realized results (log-loss, Brier, reliability, ECE, RPS).
- :mod:`~hkjc.eval.walkforward` — walk-forward folds with an embargo and the
  sealed :class:`~hkjc.eval.walkforward.Lockbox`.
- :mod:`~hkjc.eval.selection` — multiple-testing control (Benjamini-Hochberg FDR,
  Deflated/Probabilistic Sharpe).
- :mod:`~hkjc.eval.harness` — fit-on-train / score-on-test walk-forward evaluation.
"""
from . import calibration, walkforward, selection, harness
from .walkforward import Fold, Lockbox, LockboxBreachError, walk_forward_folds, split_lockbox
from .harness import walk_forward_evaluate, score_predictions, aggregate

__all__ = [
    "calibration", "walkforward", "selection", "harness",
    "Fold", "Lockbox", "LockboxBreachError", "walk_forward_folds", "split_lockbox",
    "walk_forward_evaluate", "score_predictions", "aggregate",
]
