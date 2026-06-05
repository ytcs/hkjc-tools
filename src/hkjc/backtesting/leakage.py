"""Adversarial no-lookahead check — turn leak-freedom from a convention into a test.

A correct fit→predict uses only information available before each race is run. The strong
way to verify that is adversarial: take a fit_predict, perturb the things it must NOT see
(the test races' realised outcomes, and any future-dated rows), re-run, and assert the
predictions did not move. If they move, the pipeline peeked at the future.

:func:`assert_no_lookahead` is meant to run in CI over a sample of walk-forward folds, and
as a unit test on any new feature/fit code.
"""
from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
import polars as pl

FitPredict = Callable[[pl.DataFrame, pl.DataFrame], pl.DataFrame]


def assert_no_lookahead(
    fit_predict: FitPredict,
    train: pl.DataFrame,
    test: pl.DataFrame,
    *,
    outcome_cols: Sequence[str] = ("won", "placed", "finish_pos", "result_win_odds"),
    prob_col: str = "p",
    key_cols: Sequence[str] = ("race_id", "runner"),
    tol: float = 1e-12,
) -> float:
    """Assert predictions are invariant to corrupting the test outcomes. Returns the max drift.

    Runs ``fit_predict(train, test)`` to get a baseline, then shuffles each present
    ``outcome_col`` in ``test`` (destroying the realised result) and re-runs. A leak-free
    model predicts from pre-race features only, so the ``prob_col`` must be unchanged
    within ``tol``; otherwise the prediction depended on the future and we raise.
    """
    key = [c for c in key_cols if c in test.columns]
    base = fit_predict(train, test).sort(key)
    corrupted = test
    for c in outcome_cols:
        if c in corrupted.columns:
            vals = corrupted[c].to_numpy().copy()
            new = np.roll(vals, 1)                       # deterministic permutation
            if np.array_equal(new, vals):               # constant / length<2 → try reverse
                new = vals[::-1]
            corrupted = corrupted.with_columns(pl.Series(c, new))
    after = fit_predict(train, corrupted).sort(key)
    drift = float(np.abs(base[prob_col].to_numpy() - after[prob_col].to_numpy()).max())
    if drift > tol:
        raise AssertionError(
            f"look-ahead detected: perturbing {tuple(outcome_cols)} changed "
            f"'{prob_col}' by up to {drift:.3g} (> tol {tol:g}). The pipeline reads the future.")
    return drift
