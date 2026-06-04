"""Tests for the empirical/hierarchical-Bayes calibration of the race model."""
import numpy as np
import polars as pl

from hkjc.calibration import (RaceConfig, bend_posterior, distance_band,
                              empirical_draw_curve, fit_energetics_population,
                              fit_random_effects, horse_energetics,
                              horse_energetics_trailing)


def _synth_runs(n_horses=400, sigma=0.6, tau=0.8, seed=1):
    """Per-run v_c/w_prime from a known normal-normal model (1-6 runs per horse)."""
    rng = np.random.default_rng(seed)
    rows = []
    for h in range(n_horses):
        theta_vc = rng.normal(16.0, tau)
        theta_w = rng.normal(50.0, 6.0)
        for _ in range(rng.integers(1, 7)):
            dist = int(rng.choice([1200, 1650, 2000]))
            rows.append(dict(horse_id=f"H{h}", distance=dist,
                             v_c=theta_vc + rng.normal(0, sigma),
                             w_prime=theta_w + rng.normal(0, 5.0),
                             true_vc=theta_vc))
    return pl.DataFrame(rows)


def test_distance_band_monotone():
    assert distance_band(1000) == "sprint" and distance_band(1200) == "sprint"
    assert distance_band(1600) == "mile" and distance_band(2400) == "staying"


def test_random_effects_recovers_variance_components():
    df = _synth_runs(sigma=0.6, tau=0.8)
    re = fit_random_effects(df.with_columns(
        pl.col("distance").map_elements(distance_band, return_dtype=pl.Utf8).alias("_band")),
        "v_c", "horse_id", "_band")
    assert abs(re.sigma2 - 0.36) < 0.12      # sigma=0.6 -> 0.36
    assert abs(re.tau2 - 0.64) < 0.25        # tau=0.8 -> 0.64


def test_shrinkage_beats_raw_per_horse_means():
    # The empirical-Bayes estimate should track the TRUE horse effect better than
    # the noisy per-run mean — the whole point of partial pooling.
    df = _synth_runs(sigma=0.9, tau=0.6)
    pop = fit_energetics_population(df)
    eb = horse_energetics(df, pop, loo=False)
    by_horse = eb.group_by("horse_id").agg(
        pl.col("v_c").mean().alias("raw"),
        pl.col("v_c_eb").first().alias("eb"),
        pl.col("true_vc").first().alias("truth"))
    # band fixed effect ~16 for all bands here, so compare on the residual scale
    raw_err = ((by_horse["raw"] - by_horse["truth"]) ** 2).mean()
    eb_err = ((by_horse["eb"] - by_horse["truth"]) ** 2).mean()
    assert eb_err < raw_err


def test_loo_excludes_own_run():
    df = pl.DataFrame({
        "horse_id": ["A", "A", "B"], "distance": [1200, 1200, 1200],
        "v_c": [18.0, 14.0, 16.0], "w_prime": [50.0, 50.0, 50.0]})
    pop = fit_energetics_population(df)
    eb = horse_energetics(df, pop, loo=True)
    # A's two runs are symmetric around the pool; LOO -> each shrinks the OTHER run's
    # signal, so the high run gets a lower estimate than the low run (anti-correlated).
    a = eb.filter(pl.col("horse_id") == "A").sort("v_c")
    assert a["v_c_eb"][0] > a["v_c_eb"][1]   # low raw run -> higher LOO estimate (from the high run)


def _synth_runs_rated(n_horses=500, seed=3):
    """v_c truly increases with rating: class explains much of the between-horse var."""
    rng = np.random.default_rng(seed)
    rows = []
    for h in range(n_horses):
        rtg = float(rng.integers(20, 110))
        theta = 14.0 + 0.03 * rtg + rng.normal(0, 0.25)      # rating slope 0.03, small residual
        for _ in range(rng.integers(1, 6)):
            rows.append(dict(horse_id=f"H{h}", distance=1200, rtg=rtg,
                             v_c=theta + rng.normal(0, 0.6), w_prime=50.0, true_vc=theta))
    return pl.DataFrame(rows)


def test_rating_segmentation_recovers_class_slope_and_helps():
    df = _synth_runs_rated()
    mono = fit_energetics_population(df)                       # no rating
    rated = fit_energetics_population(df, rating_col="rtg")    # segmented by rating
    assert abs(rated.v_c.rating_slope - 0.03) < 0.015         # recovers the class slope
    # rating-conditioned prior tracks the true horse effect better than mono-population,
    # most visibly for one-run horses (which shrink fully to the population prior).
    eb_mono = horse_energetics(df, mono, loo=False)
    eb_rated = horse_energetics(df, rated, rating_col="rtg", loo=False)
    one_run = df.group_by("horse_id").len().filter(pl.col("len") == 1)["horse_id"].to_list()
    for eb, key in ((eb_mono, "mono"), (eb_rated, "rated")):
        sub = eb.filter(pl.col("horse_id").is_in(one_run))
        err = ((sub["v_c_eb"] - sub["true_vc"]) ** 2).mean()
        if key == "mono":
            mono_err = err
        else:
            rated_err = err
    assert rated_err < mono_err


def test_trailing_uses_only_prior_runs():
    # one horse, three dated runs; the trailing estimate must use only earlier runs.
    df = pl.DataFrame({
        "horse_id": ["A", "A", "A", "B", "B"],
        "date": ["2026-01-01", "2026-02-01", "2026-03-01", "2026-01-01", "2026-02-01"],
        "distance": [1200, 1200, 1200, 1200, 1200],
        "v_c": [18.0, 18.0, 18.0, 15.0, 15.0], "w_prime": [50.0, 50.0, 50.0, 50.0, 50.0]})
    pop = fit_energetics_population(df)
    tr = horse_energetics_trailing(df, pop).sort(["horse_id", "date"])
    a = tr.filter(pl.col("horse_id") == "A")
    assert list(a["v_c_prior_n"]) == [0, 1, 2]           # expanding prior count
    # run 1 has no history -> population mean (no horse signal); later runs lift toward 18
    assert a["v_c_eb"][0] < a["v_c_eb"][1] < a["v_c_eb"][2]
    # the leak-free guarantee: run-1 estimate ignores the (high) future runs
    assert abs(a["v_c_eb"][0] - pop.v_c.fixed("sprint")) < 1e-9


def test_empirical_draw_curve_shapes():
    df = pl.DataFrame({
        "draw": [1, 2, 1, 2], "won": [1, 0, 1, 0],
        "finish_pos": [1, 4, 1, 3], "field_size": [6, 6, 6, 6]})
    c = empirical_draw_curve(df)
    assert set(c["draw"].to_list()) == {1, 2}
    assert c.filter(pl.col("draw") == 1)["p_win"][0] == 1.0


def _draw_race(winner_draw, n=10, seed=0):
    draws = np.arange(1, n + 1)
    return RaceConfig(venue="HV", distance=1200, surface="turf", course="A",
                      draws=draws, v_c=np.full(n, 16.0), w_prime=np.full(n, 50.0),
                      winner_idx=winner_draw - 1)


def test_bend_posterior_responds_to_draw_signal():
    grid = np.linspace(0.001, 0.03, 6)
    # inner draws always win -> posterior should favour a LARGER bend_curvature than
    # a field where the winner is random.
    inner = [_draw_race(winner_draw=1, seed=s) for s in range(10)]
    rand = [_draw_race(winner_draw=int(1 + (s % 10)), seed=100 + s) for s in range(10)]
    p_inner = bend_posterior(inner, grid, n_samples=600)
    p_rand = bend_posterior(rand, grid, n_samples=600)
    assert p_inner.mean > p_rand.mean
    lo, hi = p_inner.interval(0.9)
    assert lo <= p_inner.mean <= hi
