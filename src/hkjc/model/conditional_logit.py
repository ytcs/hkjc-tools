"""Results-trained win model — conditional logit (the circularity break).

The Harville baseline fits strengths *to the market*, so it can only reproduce the
market. This model instead fits to **realized winners** (McFadden's conditional
logit, the classic horse-racing discrete-choice model): each runner gets a latent
strength ``s_i = β·x_i`` linear in features, and the within-race win probability is
the softmax ``exp(s_i) / Σ_j exp(s_j)``. ``β`` is chosen to maximize the likelihood
of the horses that actually won.

Market-implied probability is just **one feature** in ``x``. If it's the only
feature, the fitted coefficient ≈ 1 reproduces the market; adding fundamentals
(draw, rating, pace, …) lets the model *diverge* from — and, where it's better
calibrated, *beat* — the market. Edge = divergence × superior calibration
(``docs/AGENT_ARCHITECTURE.md`` §0.1).

The negative log-likelihood is convex, so L-BFGS-B finds the global optimum. A
fitted model also yields per-runner ``strengths`` that drop straight into
:class:`~hkjc.model.plackett_luce.PlackettLuceModel` to simulate Place / Quinella.
"""
from __future__ import annotations

from typing import Callable, Optional, Sequence

import numpy as np
import polars as pl
from scipy.optimize import minimize

from ..core.contracts import RACE_KEYS


def _codes(race_index: Sequence) -> np.ndarray:
    """Map arbitrary race identifiers to contiguous integer group ids ``0..G-1``."""
    _, inv = np.unique(np.asarray(race_index), return_inverse=True)
    return inv.astype(np.int64)


class ConditionalLogit:
    """Conditional-logit win model fit by maximum likelihood of race winners.

    Parameters
    ----------
    l2:
        Ridge penalty on the standardized coefficients (keeps the fit stable when
        features are collinear, as market prob and fundamentals tend to be).
    max_iter:
        L-BFGS-B iteration cap.

    Notes
    -----
    There is deliberately **no intercept**: a constant added to every runner's
    strength cancels in the within-race softmax (it is unidentifiable). Features
    must therefore vary *across runners within a race*. Inputs are standardized
    internally (mean/std from the training rows), and missing values are
    mean-imputed — so a column that's null for some runners degrades gracefully.
    """

    def __init__(self, l2: float = 1e-4, max_iter: int = 200):
        self.l2 = float(l2)
        self.max_iter = int(max_iter)
        self.coef_: Optional[np.ndarray] = None
        self.success_: Optional[bool] = None
        self._mean: Optional[np.ndarray] = None
        self._std: Optional[np.ndarray] = None

    # ------------------------------------------------------------- scaling
    def _fit_scaler(self, X: np.ndarray) -> np.ndarray:
        self._mean = np.nanmean(X, axis=0)
        std = np.nanstd(X, axis=0)
        std[(std == 0) | ~np.isfinite(std)] = 1.0
        self._std = std
        return self._transform(X)

    def _transform(self, X: np.ndarray) -> np.ndarray:
        Xs = (X - self._mean) / self._std
        return np.nan_to_num(Xs, nan=0.0)  # mean-impute (0 after centering)

    # ---------------------------------------------------------------- fit
    def fit(self, X, race_index, won) -> "ConditionalLogit":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(won, dtype=np.float64)
        groups = _codes(race_index)
        G = int(groups.max()) + 1
        Xs = self._fit_scaler(X)
        k = Xs.shape[1]
        l2 = self.l2

        def objective(beta):
            s = Xs @ beta
            m = np.full(G, -np.inf)
            np.maximum.at(m, groups, s)
            es = np.exp(s - m[groups])
            denom = np.bincount(groups, weights=es, minlength=G)
            p = es / denom[groups]
            lse = m + np.log(denom)
            nll = float(lse.sum() - s[y == 1].sum()) + l2 * float(beta @ beta)
            grad = Xs.T @ (p - y) + 2.0 * l2 * beta
            return nll, grad

        res = minimize(objective, np.zeros(k), jac=True, method="L-BFGS-B",
                       options={"maxiter": self.max_iter})
        self.coef_ = res.x
        self.success_ = bool(res.success)
        return self

    # ------------------------------------------------------------ predict
    def _softmax(self, s: np.ndarray, groups: np.ndarray) -> np.ndarray:
        G = int(groups.max()) + 1
        m = np.full(G, -np.inf)
        np.maximum.at(m, groups, s)
        es = np.exp(s - m[groups])
        denom = np.bincount(groups, weights=es, minlength=G)
        return es / denom[groups]

    def predict_proba(self, X, race_index) -> np.ndarray:
        """Within-race win probabilities (sum to 1 per race)."""
        Xs = self._transform(np.asarray(X, dtype=np.float64))
        return self._softmax(Xs @ self.coef_, _codes(race_index))

    def strengths(self, X) -> np.ndarray:
        """Per-runner Plackett-Luce strengths ``exp(β·x)`` (for Place/Quinella sim)."""
        Xs = self._transform(np.asarray(X, dtype=np.float64))
        return np.exp(Xs @ self.coef_)


# --------------------------------------------------------- harness integration
def _design(frame: pl.DataFrame, feature_cols: Sequence[str], race_keys: Sequence[str],
            outcome_col: Optional[str]):
    X = frame.select([pl.col(c).cast(pl.Float64, strict=False) for c in feature_cols]).to_numpy()
    rk = frame.select(
        pl.concat_str([pl.col(k).cast(pl.Utf8) for k in race_keys], separator="|").alias("rk")
    )["rk"].to_numpy()
    y = frame[outcome_col].to_numpy() if (outcome_col and outcome_col in frame.columns) else None
    return X, rk, y


def conditional_logit_fit_predict(
    feature_cols: Sequence[str],
    l2: float = 1e-4,
    prob_col: str = "p",
    outcome_col: str = "won",
    race_keys: Sequence[str] = tuple(RACE_KEYS),
) -> Callable[[pl.DataFrame, pl.DataFrame], pl.DataFrame]:
    """Build a ``fit_predict(train, test)`` for :func:`hkjc.eval.harness.walk_forward_evaluate`.

    Fits a :class:`ConditionalLogit` on the training rows (races with a labelled
    winner) and returns the test rows with a per-runner win probability ``prob_col``.
    """
    feature_cols = list(feature_cols)
    race_keys = list(race_keys)

    def fit_predict(train: pl.DataFrame, test: pl.DataFrame) -> pl.DataFrame:
        # keep only training races that actually have a winner
        train = train.filter(pl.col(outcome_col).sum().over(race_keys) >= 1)
        Xtr, gtr, ytr = _design(train, feature_cols, race_keys, outcome_col)
        model = ConditionalLogit(l2=l2).fit(Xtr, gtr, ytr)
        Xte, gte, _ = _design(test, feature_cols, race_keys, None)
        p = model.predict_proba(Xte, gte)
        return test.with_columns(pl.Series(prob_col, p))

    return fit_predict
