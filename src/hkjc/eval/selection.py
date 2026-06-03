"""Multiple-testing control — penalizing selection so we don't bank noise.

The agent will try thousands of hypotheses against <1000 races/year. Naïve
"best of N" *guarantees* phantom edges (``docs/AGENT_ARCHITECTURE.md`` §3.4). Two
complementary tools:

- **Benjamini-Hochberg FDR** — control the false-discovery rate across a batch of
  hypothesis tests (feature significance, A/B model comparisons).
- **Deflated Sharpe Ratio** (Bailey & López de Prado) — the finance-specific
  haircut: given that you picked the best of ``n_trials`` strategies, how likely is
  the observed Sharpe to be genuinely > 0? Needs the *number of trials* and the
  cross-sectional spread of their Sharpes, so the orchestrator must track the trial
  count honestly (e.g. from the ledger).

Pure functions over numbers; no state.
"""
from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np
from scipy import stats

_EULER_MASCHERONI = 0.5772156649015329


def benjamini_hochberg(pvalues: Sequence[float], alpha: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
    """Benjamini-Hochberg FDR control.

    Returns ``(reject, adjusted)`` aligned to the input order: ``reject`` is a bool
    mask of hypotheses significant at FDR ``alpha``; ``adjusted`` are the BH-adjusted
    p-values (q-values).
    """
    p = np.asarray(pvalues, dtype=float)
    n = p.size
    if n == 0:
        return np.zeros(0, dtype=bool), np.zeros(0, dtype=float)
    order = np.argsort(p)
    ranked = p[order]
    ranks = np.arange(1, n + 1)
    thresh = alpha * ranks / n
    below = ranked <= thresh
    reject_sorted = np.zeros(n, dtype=bool)
    if below.any():
        kmax = np.max(np.where(below)[0])
        reject_sorted[: kmax + 1] = True
    # BH-adjusted p-values: enforce monotonicity from the largest rank down.
    adj_sorted = np.minimum.accumulate((ranked * n / ranks)[::-1])[::-1]
    adj_sorted = np.clip(adj_sorted, 0.0, 1.0)
    reject = np.zeros(n, dtype=bool)
    adjusted = np.empty(n, dtype=float)
    reject[order] = reject_sorted
    adjusted[order] = adj_sorted
    return reject, adjusted


def sharpe_ratio(returns: Sequence[float], ddof: int = 1) -> float:
    """Per-period Sharpe of a return series (no annualization)."""
    r = np.asarray(returns, dtype=float)
    sd = r.std(ddof=ddof)
    return float(r.mean() / sd) if sd > 0 else float("nan")


def probabilistic_sharpe_ratio(
    sr: float, sr_benchmark: float, n: int, skew: float = 0.0, kurt: float = 3.0
) -> float:
    """P(true Sharpe > ``sr_benchmark``) given an estimate ``sr`` from ``n`` obs.

    ``skew``/``kurt`` are the return distribution's 3rd/4th moments (kurt=3 is
    Gaussian). Accounts for the fact that non-normal returns make a Sharpe estimate
    less reliable.
    """
    denom = np.sqrt(1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr ** 2)
    z = (sr - sr_benchmark) * np.sqrt(n - 1) / denom
    return float(stats.norm.cdf(z))


def expected_max_sharpe(n_trials: int, sr_std: float = 1.0) -> float:
    """Expected maximum Sharpe across ``n_trials`` independent strategies of true
    Sharpe 0, given the cross-sectional spread ``sr_std`` of their estimates.

    This is the bar a candidate must clear to be more than the luckiest of N tries.
    """
    if n_trials < 2:
        return 0.0
    g = _EULER_MASCHERONI
    z1 = stats.norm.ppf(1.0 - 1.0 / n_trials)
    z2 = stats.norm.ppf(1.0 - 1.0 / (n_trials * np.e))
    return float(sr_std * ((1.0 - g) * z1 + g * z2))


def deflated_sharpe_ratio(
    sr: float, n: int, n_trials: int, sr_trials_std: float,
    skew: float = 0.0, kurt: float = 3.0,
) -> float:
    """Deflated Sharpe Ratio: PSR against the expected-max-of-N benchmark.

    Probability the strategy's true Sharpe exceeds what you'd expect from the best
    of ``n_trials`` random tries. A DSR near 1 survives the multiple-testing
    haircut; near 0.5 or below means the result is consistent with luck.
    """
    sr_star = expected_max_sharpe(n_trials, sr_trials_std)
    return probabilistic_sharpe_ratio(sr, sr_star, n, skew=skew, kurt=kurt)


def min_track_record_length(
    sr: float, sr_benchmark: float = 0.0, skew: float = 0.0, kurt: float = 3.0, alpha: float = 0.05
) -> float:
    """Minimum number of observations for the Sharpe to beat ``sr_benchmark`` at
    confidence ``1 - alpha``. ``inf`` if ``sr <= sr_benchmark``."""
    if sr <= sr_benchmark:
        return float("inf")
    z = stats.norm.ppf(1.0 - alpha)
    return float(1.0 + (1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr ** 2) * (z / (sr - sr_benchmark)) ** 2)
