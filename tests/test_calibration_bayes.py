"""Hierarchical-Bayes (NumPyro) energetics fit — synthetic recovery.

Runs only when the optional bayes stack is installed: ``uv run --group bayes pytest``.
"""
import numpy as np
import polars as pl
import pytest

pytest.importorskip("numpyro")

from hkjc.model.calibration_bayes import fit_energetics_bayes, horse_table_bayes  # noqa: E402


def _synth(n_horses=200, slope=0.03, tau=0.3, sigma=0.6, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for h in range(n_horses):
        rtg = float(rng.integers(20, 110))
        theta = 14.0 + slope * rtg + rng.normal(0, tau)
        for _ in range(rng.integers(1, 6)):
            rows.append(dict(horse_id=f"H{h}", distance=1200, rtg=rtg,
                             v_c=theta + rng.normal(0, sigma), w_prime=50.0,
                             true_vc=theta))
    return pl.DataFrame(rows)


def test_numpyro_recovers_slope_and_variance_components():
    df = _synth(slope=0.03, tau=0.3, sigma=0.6)
    fits = fit_energetics_bayes(df, rating_col="rtg", num_warmup=300, num_samples=300)
    vc = fits["v_c"]
    assert abs(vc.rating_slope - 0.03) < 0.012          # recovers the class slope
    assert abs(vc.sigma - 0.6) < 0.2                    # within-horse sd
    assert vc.tau < 0.6                                 # between-horse residual sd (small after rating)
    # per-horse posterior means track the truth
    tab = horse_table_bayes(df, fits, rating_col="rtg")
    by_horse = tab.group_by("horse_id").agg(
        pl.col("v_c_bayes").first().alias("est"), pl.col("true_vc").first().alias("truth"))
    corr = np.corrcoef(by_horse["est"].to_numpy(), by_horse["truth"].to_numpy())[0, 1]
    assert corr > 0.9
