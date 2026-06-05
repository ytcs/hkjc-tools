"""Tests for calibration metrics."""
import numpy as np
import polars as pl

from hkjc.backtesting import scoring as cal


def test_perfect_predictions_score_zero():
    y = np.array([0, 1, 0, 1, 1])
    p = y.astype(float)
    assert cal.log_loss(p, y) < 1e-6
    assert cal.brier_score(p, y) < 1e-12
    assert cal.expected_calibration_error(p, y, bins=10) < 1e-12


def test_brier_constant_half():
    y = np.array([0, 1, 0, 1])
    assert abs(cal.brier_score(np.full(4, 0.5), y) - 0.25) < 1e-12


def test_reliability_table_counts_sum():
    rng = np.random.default_rng(0)
    p = rng.random(1000)
    y = (rng.random(1000) < p).astype(int)
    tab = cal.reliability_table(p, y, bins=10)
    assert tab["count"].sum() == 1000
    # well-calibrated by construction: pred ~ obs within each populated bin
    populated = tab.filter(pl.col("count") > 30)
    diff = (populated["mean_pred"] - populated["mean_obs"]).abs().max()
    assert diff < 0.12


def test_rps_order_sensitivity():
    # mass on the correct ordinal category -> 0; far-off mass -> larger penalty
    assert cal.ranked_probability_score([0, 0, 1], outcome_idx=2) < 1e-12
    near = cal.ranked_probability_score([0, 1, 0], outcome_idx=2)
    far = cal.ranked_probability_score([1, 0, 0], outcome_idx=2)
    assert far > near > 0


def test_race_win_log_loss():
    # two races; winner given 0.5 and 0.25 -> -mean(log[.5,.25])
    df = pl.DataFrame({
        "race_id": [1, 1, 2, 2],
        "p": [0.5, 0.5, 0.25, 0.75],
        "won": [1, 0, 1, 0],
    })
    expected = -np.mean(np.log([0.5, 0.25]))
    assert abs(cal.race_win_log_loss(df, race_col="race_id") - expected) < 1e-9
