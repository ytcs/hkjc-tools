"""Tests for the results-trained conditional-logit win model."""
import numpy as np
import polars as pl

from hkjc.backtesting import scoring as cal
from hkjc.backtesting import harness
from hkjc.backtesting import walkforward as wf
from hkjc.model.baselines import market_baseline_fit_predict, uniform_fit_predict
from hkjc.model.conditional_logit import (
    ConditionalLogit,
    _codes,
    _design,
    conditional_logit_fit_predict,
)

FCOLS = ["f0", "f1"]
RKEYS = ["date", "venue", "race_no"]


def _synth(n_races: int = 500, field: int = 8, beta=(1.5, -0.8), seed: int = 0) -> pl.DataFrame:
    """Races whose winner is drawn from the true conditional-logit law."""
    rng = np.random.default_rng(seed)
    beta = np.asarray(beta, dtype=float)
    rows = []
    for r in range(n_races):
        X = rng.normal(size=(field, len(beta)))
        s = X @ beta
        p = np.exp(s - s.max())
        p /= p.sum()
        w = int(rng.choice(field, p=p))
        date = f"2026-{(r // 28) % 12 + 1:02d}-{(r % 28) + 1:02d}"
        for i in range(field):
            rows.append(dict(date=date, venue="ST", race_no=r, runner=i + 1,
                             f0=float(X[i, 0]), f1=float(X[i, 1]), won=int(i == w)))
    return pl.DataFrame(rows)


def test_predict_proba_sums_to_one_per_race():
    df = _synth(60)
    X, g, y = _design(df, FCOLS, RKEYS, "won")
    m = ConditionalLogit().fit(X, g, y)
    p = m.predict_proba(X, g)
    sums = np.bincount(_codes(g), weights=p)
    assert np.allclose(sums, 1.0, atol=1e-9)


def test_recovers_coefficient_signs_and_beats_uniform():
    df = _synth(800)
    X, g, y = _design(df, FCOLS, RKEYS, "won")
    m = ConditionalLogit(l2=1e-6).fit(X, g, y)
    assert m.success_
    # standardized coefficients carry the true signs (+, -)
    assert m.coef_[0] > 0 and m.coef_[1] < 0
    p = m.predict_proba(X, g)
    assert cal.log_loss(p, y) < cal.log_loss(np.full_like(p, 1 / 8), y)


def test_determinism():
    df = _synth(120)
    X, g, y = _design(df, FCOLS, RKEYS, "won")
    a = ConditionalLogit().fit(X, g, y).coef_
    b = ConditionalLogit().fit(X, g, y).coef_
    assert np.allclose(a, b)


def test_fit_predict_through_harness_beats_uniform():
    # The proper score for win prediction is the multiclass race_win_log_loss;
    # binary one-vs-rest log-loss can favour the flatter uniform out-of-sample, so
    # we judge on the metric that actually measures putting mass on the winner.
    df = _synth(600)
    folds = wf.walk_forward_folds(df["date"].unique().to_list(), n_folds=3, embargo=1, min_train=40)

    model = harness.aggregate(harness.walk_forward_evaluate(
        df, folds, conditional_logit_fit_predict(FCOLS), race_col="race_no"))
    base = harness.aggregate(harness.walk_forward_evaluate(
        df, folds, uniform_fit_predict(), race_col="race_no"))

    assert model["race_win_log_loss"] < base["race_win_log_loss"]


def test_strengths_positive_for_plackett_luce():
    df = _synth(50)
    X, g, y = _design(df, FCOLS, RKEYS, "won")
    m = ConditionalLogit().fit(X, g, y)
    s = m.strengths(X)
    assert (s > 0).all() and np.isfinite(s).all()


def test_market_baseline_reproduces_market():
    df = pl.DataFrame({
        "date": ["d"] * 3, "venue": ["ST"] * 3, "race_no": [1] * 3, "runner": [1, 2, 3],
        "market_win_prob": [0.5, 0.3, 0.2],
    })
    out = market_baseline_fit_predict()(df, df)
    assert abs(out["p"].sum() - 1.0) < 1e-9
    assert np.allclose(out["p"].to_numpy(), [0.5, 0.3, 0.2])
