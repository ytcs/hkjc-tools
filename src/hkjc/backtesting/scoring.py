"""Calibration metrics — scoring predicted probabilities against realized results.

This is Layer 1 of the harness (see ``docs/AGENT_ARCHITECTURE.md`` §3.1): the
cheapest, densest way to judge a model. *Every* historical race yields a score,
so a badly-calibrated model is killed before it ever reaches CLV or P&L.

Two views of win probability:

- **One-vs-rest** (``log_loss`` / ``brier_score`` / reliability over the binary
  "did this runner win" indicator across all runner rows) — measures calibration:
  do horses given 20% actually win ~20%?
- **Multiclass** (``race_win_log_loss``) — the proper score for a per-race
  distribution: ``-mean(log p[winner])``. Rewards putting mass on the actual winner.

Pure functions: probabilities + outcomes in, scores out.
"""
from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
import polars as pl


def _arr(x) -> np.ndarray:
    return np.asarray(x, dtype=float)


def log_loss(p, y, eps: float = 1e-15) -> float:
    """Binary cross-entropy of probabilities ``p`` against 0/1 outcomes ``y``."""
    p = np.clip(_arr(p), eps, 1 - eps)
    y = _arr(y)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def brier_score(p, y) -> float:
    """Mean squared error between probabilities and 0/1 outcomes."""
    return float(np.mean((_arr(p) - _arr(y)) ** 2))


def reliability_table(p, y, bins: int = 10) -> pl.DataFrame:
    """Binned reliability: per probability bin, the count, mean predicted and
    mean observed frequency. A well-calibrated model has ``mean_pred ≈ mean_obs``.
    """
    p, y = _arr(p), _arr(y)
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    rows = []
    for b in range(bins):
        m = idx == b
        n = int(m.sum())
        rows.append({
            "bin": b, "lo": float(edges[b]), "hi": float(edges[b + 1]), "count": n,
            "mean_pred": float(p[m].mean()) if n else None,
            "mean_obs": float(y[m].mean()) if n else None,
        })
    return pl.DataFrame(rows)


def expected_calibration_error(p, y, bins: int = 10) -> float:
    """Count-weighted mean ``|mean_pred - mean_obs|`` across bins (lower = better)."""
    p, y = _arr(p), _arr(y)
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    n = len(p)
    if n == 0:
        return float("nan")
    ece = 0.0
    for b in range(bins):
        m = idx == b
        c = int(m.sum())
        if c:
            ece += c / n * abs(p[m].mean() - y[m].mean())
    return float(ece)


def ranked_probability_score(probs, outcome_idx: int) -> float:
    """RPS for one ordinal forecast over K ordered categories.

    ``probs`` is a length-K distribution (e.g. over finishing-position buckets);
    ``outcome_idx`` is the realized category. 0 = perfect, larger = worse. Penalizes
    forecasts by how far off the *cumulative* distribution is — order matters.
    """
    p = _arr(probs)
    k = len(p)
    obs = np.zeros(k)
    obs[outcome_idx] = 1.0
    cp, co = np.cumsum(p), np.cumsum(obs)
    return float(np.sum((cp - co) ** 2) / (k - 1)) if k > 1 else 0.0


def mean_ranked_probability_score(prob_rows: Iterable, outcome_idxs: Iterable[int]) -> float:
    vals = [ranked_probability_score(p, o) for p, o in zip(prob_rows, outcome_idxs)]
    return float(np.mean(vals)) if vals else float("nan")


def race_win_log_loss(
    df: pl.DataFrame,
    prob_col: str = "p",
    winner_col: str = "won",
    race_col: Optional[str] = None,
    eps: float = 1e-15,
) -> float:
    """Multiclass win log-loss: ``-mean(log p[winner])`` over races.

    ``df`` holds one row per runner with a predicted win probability ``prob_col``
    and a 0/1 ``winner_col``. Assumes one winning row per race (dead-heats aside).
    """
    win = df.filter(pl.col(winner_col) == 1)
    if win.height == 0:
        return float("nan")
    p = np.clip(win[prob_col].to_numpy().astype(float), eps, 1.0)
    return float(-np.mean(np.log(p)))


def calibration_report(p, y, bins: int = 10) -> dict:
    """Bundle the headline one-vs-rest calibration metrics into a dict."""
    return {
        "n": int(len(_arr(p))),
        "log_loss": log_loss(p, y),
        "brier": brier_score(p, y),
        "ece": expected_calibration_error(p, y, bins=bins),
        "base_rate": float(np.mean(_arr(y))) if len(_arr(y)) else float("nan"),
        "mean_pred": float(np.mean(_arr(p))) if len(_arr(p)) else float("nan"),
    }
