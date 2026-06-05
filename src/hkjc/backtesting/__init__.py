"""Backtesting layer — replay history, enforce no-lookahead, score model/odds strategies.

Nothing in the package imports ``backtesting``; it reads data/model outputs and scores
them. Keeping the judge un-depended-on is deliberate: the objective function of the
research loop must not be reachable (and thus game-able) from inside the things it
evaluates.

- :mod:`~hkjc.backtesting.temporal` — :class:`AsOfView`, the enforced no-lookahead data
  accessor: a frame wrapper that can only return rows strictly before a cutoff time.
- :mod:`~hkjc.backtesting.walkforward` — walk-forward folds with an embargo and the
  sealed :class:`~hkjc.backtesting.walkforward.Lockbox`.
- :mod:`~hkjc.backtesting.scoring` — proper scoring of predicted probabilities against
  realized results (log-loss, Brier, reliability, ECE, RPS).
- :mod:`~hkjc.backtesting.selection` — multiple-testing control (Benjamini-Hochberg FDR,
  Deflated/Probabilistic Sharpe).
- :mod:`~hkjc.backtesting.harness` — fit-on-train / score-on-test walk-forward evaluation.
- :mod:`~hkjc.backtesting.leakage` — :func:`assert_no_lookahead`, the adversarial
  future-perturbation check that turns no-lookahead from convention into a test.
"""
from . import harness, leakage, scoring, selection, temporal, walkforward
from .harness import aggregate, score_predictions, walk_forward_evaluate
from .leakage import assert_no_lookahead
from .temporal import AsOfView
from .walkforward import Fold, Lockbox, LockboxBreachError, split_lockbox, walk_forward_folds

__all__ = [
    "harness", "leakage", "scoring", "selection", "temporal", "walkforward",
    "Fold", "Lockbox", "LockboxBreachError", "walk_forward_folds", "split_lockbox",
    "walk_forward_evaluate", "score_predictions", "aggregate",
    "AsOfView", "assert_no_lookahead",
]
