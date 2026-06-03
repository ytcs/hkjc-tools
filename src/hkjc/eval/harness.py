"""Walk-forward evaluation — the loop that turns a model into a score.

Ties the pieces together: split the timeline (``walkforward``), fit the model on
each train block and predict the strictly-later test block, then score the
predictions (``calibration``). The harness owns *splitting and scoring*; the model
owns *fit_predict*. That boundary is deliberate — the judge never reaches into the
model, and the model never sees the test outcomes.

A model is supplied as a ``fit_predict(train_df, test_df) -> preds_df`` callable:
it may train on ``train_df`` however it likes, and must return the ``test_df`` rows
augmented with a probability column. This keeps the harness model-agnostic — it
works for the Harville baseline, a results-trained model, or anything else.
"""
from __future__ import annotations

from typing import Callable, List, Optional, Sequence

import polars as pl

from . import calibration
from .walkforward import Fold

FitPredict = Callable[[pl.DataFrame, pl.DataFrame], pl.DataFrame]


def score_predictions(
    preds: pl.DataFrame,
    prob_col: str = "p",
    outcome_col: str = "won",
    race_col: Optional[str] = None,
    bins: int = 10,
) -> dict:
    """Calibration metrics for an already-built predictions frame.

    ``preds`` has one row per runner with a predicted probability ``prob_col`` and a
    0/1 ``outcome_col``. When ``race_col`` is given, also reports the multiclass
    per-race win log-loss.
    """
    if preds.height == 0:
        return {"n": 0}
    p = preds[prob_col].to_numpy()
    y = preds[outcome_col].to_numpy()
    report = calibration.calibration_report(p, y, bins=bins)
    if race_col is not None:
        report["race_win_log_loss"] = calibration.race_win_log_loss(
            preds, prob_col=prob_col, winner_col=outcome_col, race_col=race_col)
        report["n_races"] = int(preds[race_col].n_unique())
    return report


def walk_forward_evaluate(
    data: pl.DataFrame,
    folds: Sequence[Fold],
    fit_predict: FitPredict,
    date_col: str = "date",
    prob_col: str = "p",
    outcome_col: str = "won",
    race_col: Optional[str] = None,
    bins: int = 10,
) -> pl.DataFrame:
    """Fit-on-train / score-on-test across every fold.

    Returns one metrics row per fold (with ``fold``, ``n_train``, ``n_test``). Use
    :func:`aggregate` for the across-fold summary. Folds whose model errors or
    returns nothing are skipped (logged via a null-height row is avoided).
    """
    rows: List[dict] = []
    for fold in folds:
        train = data.filter(pl.col(date_col).is_in(fold.train))
        test = data.filter(pl.col(date_col).is_in(fold.test))
        if test.height == 0 or train.height == 0:
            continue
        preds = fit_predict(train, test)
        if preds is None or preds.height == 0:
            continue
        m = score_predictions(preds, prob_col=prob_col, outcome_col=outcome_col,
                              race_col=race_col, bins=bins)
        m = {"fold": fold.index, "n_train": train.height, "n_test": test.height, **m}
        rows.append(m)
    return pl.DataFrame(rows)


def aggregate(fold_metrics: pl.DataFrame, weight_col: str = "n_test") -> dict:
    """Collapse per-fold metrics into a single, test-size-weighted summary.

    Weighting by test size avoids letting a tiny final fold dominate. Returns the
    mean of each numeric metric plus totals.
    """
    if fold_metrics.height == 0:
        return {"folds": 0}
    w = fold_metrics[weight_col].to_numpy().astype(float)
    wsum = w.sum()
    out = {"folds": fold_metrics.height, "n_test_total": int(wsum)}
    skip = {"fold", "n_train", "n_test", "n_races", "n"}
    for col in fold_metrics.columns:
        if col in skip or fold_metrics[col].dtype not in (pl.Float64, pl.Float32):
            continue
        vals = fold_metrics[col].to_numpy().astype(float)
        mask = ~pl.Series(vals).is_nan().to_numpy()
        if mask.any():
            out[col] = float((vals[mask] * w[mask]).sum() / w[mask].sum())
    return out
