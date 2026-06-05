"""Walk-forward backtest of the calibrated traffic model at Sha Tin — vs the market.

Honest, leak-free: inside every fold the energetics **population is fit on train
only**, and each test horse's ability is the empirical-Bayes estimate from its
**strictly-prior** (train) runs — adapted to the test race's distance band and the
horse's rating. A debut / no-sectional horse falls back to the rating-conditioned
population mean. The model never sees odds (it is structural); ``bend_curvature`` is
the one full-sample structural constant (the calibrated 0.0037 — geometry doesn't
drift). We then score the model's win probabilities against the de-vigged market and
a uniform baseline. The point is not to beat the market but to measure the gap and
check out-of-sample calibration.

Run: uv run --group bayes python experiments/backtest_traffic_st.py
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import polars as pl

from hkjc.calibration import (CALIBRATED_BEND_CURVATURE, EnergeticsPopulation,
                              distance_band, fit_energetics_population)
from hkjc.eval import harness
from hkjc.eval.walkforward import walk_forward_folds
from hkjc.model.traffic_jax import win_probs
from hkjc.track import track_for

CACHE = Path("data_cache")
# Full-history panels (3248 races / 5 seasons) are the DEFAULT — the 20x data over the old 3-month
# build (161 races) lifted the ST ABM from ~38% to 43.7% of market with zero modeling change.
# Rebuild them with build_energetics_full.py + build_wpq_full.py. Env vars still allow overriding
# back to the 3-month panels (energetics_dataset.parquet / wpq_dataset.parquet) for comparison.
ENERGETICS = os.environ.get("ENERGETICS_FILE", "energetics_dataset_full.parquet")
WPQ = os.environ.get("WPQ_FILE", "wpq_dataset_full.parquet")
N_SAMPLES = int(os.environ.get("N_SAMPLES", "6000"))
BEND = CALIBRATED_BEND_CURVATURE["ST"]
# Form is a far stronger ability measurement than sectional v_c (18% vs 5% of market
# skill) and survives the handicap. It maps to an effective critical velocity — a horse
# that consistently beats its field runs effectively faster than its sectional estimate.
# FORM_GAIN is m/s of v_c per unit form-deviation from the neutral 0.5.
# RECALIBRATED 2.0 -> 4.0 (market_residual_diagnosis showed the ABM under-weighted ability:
# market - model residual slope on form t=+32, i.e. the market concentrates more probability
# on strong horses than the model did). The ability_spread_sweep then confirmed an interior
# optimum at 4.0 (6.0 worse, 8.0 much worse) lifting ST 43.3% -> 48.2% of market at full scale,
# with ECE 0.0081 -> 0.0069 (now matching the market's own calibration). Note the sweep also
# showed LOWERING speed_noise hurts at every gain: the fix was amplifying the GOOD signal (form),
# not global sharpening — cutting noise sharpens the noisy sectional v_c too and overfits.
FORM_GAIN = float(os.environ.get("FORM_GAIN", "4.0"))
# Race-day unpredictability — must be CALIBRATED to the ability spread, else the sim is
# overconfident. With a real ability signal (form) the win is mostly unpredictable, so
# this is large; the old 0.02 only "worked" because heavy shrinkage made abilities equal.
# Sweep-confirmed: 0.12 is the optimum; 0.09/0.06 monotonically worse at the calibrated gain.
SPEED_NOISE = float(os.environ.get("SPEED_NOISE", "0.12"))
# Freshness (days-since-last-race) was explored as an engine-tune signal and DROPPED for
# parsimony: the only robust, form-orthogonal effect was the long-layoff penalty, but
# <2.4% of runners are long layoffs so it moved aggregate log-loss by only ~+0.001 (CI
# straddled 0). days-since is too crude a proxy for current condition anyway — the better
# sensor is the horse's most-recent BARRIER TRIAL (a direct fitness/health readout). See
# experiments/freshness_eda.py for the analysis. TODO: barrier-trial condition signal.


def _norm(col):
    return pl.col(col).cast(pl.Utf8).str.replace_all("/", "-")


# Recency note: an EXP-WEIGHTED (half-life ~2 race) form is +8pp of market over the career mean
# in a bare-form conditional logit (conditional_form.py: 28%->36%). But it does NOT transfer to
# the ABM: at each form's own best FORM_GAIN, EW is +0.0025 nats/race WORSE than the career mean
# (form_gain_recency_sweep.py, P(EW better)=26%). The recency the market prices is already carried
# in the ABM by the official rating, which enters NONLINEARLY via v_c.fixed(band, rtg) — a channel
# the bare logit lacks. So the career mean stays here; recency lives in the rating layer.
def current_form_table() -> pl.DataFrame:
    """Leak-free expanding mean of prior relative-finish per horse (all venues count).

    relative-finish = 1 − (finish_pos−1)/(field_size−1) ∈ [0,1], 1 = won. The expanding
    mean over a horse's STRICTLY PRIOR runs is a far stronger ability measurement than
    sectional v_c (corr with winning ~0.15 vs ~0.13), and it survives the handicap (a
    horse beating its mark keeps finishing well). Returns (date, venue, race_no, runner,
    form) keyed for joining; form is null on debut. (Recency weighting tested, see note above.)
    """
    e = pl.read_parquet(CACHE / ENERGETICS).with_columns(_norm("date").alias("date"))
    w = pl.read_parquet(CACHE / WPQ).with_columns(_norm("date").alias("date"))
    h = (e.select("date", "venue", "race_no", "runner", "horse_id")
         .join(w.select("date", "venue", "race_no", "runner", "finish_pos", "field_size"),
               on=["date", "venue", "race_no", "runner"])
         .with_columns(pl.col("date").str.strptime(pl.Date, "%Y-%m-%d", strict=False).alias("d"),
                       (1.0 - (pl.col("finish_pos") - 1) / (pl.col("field_size") - 1)).alias("rf"))
         .sort(["horse_id", "d"]))
    h = h.with_columns(
        ((pl.col("rf").cum_sum().over("horse_id") - pl.col("rf"))
         / (pl.col("rf").cum_count().over("horse_id") - 1)).alias("form"))
    return h.select("date", "venue", "race_no", "runner",
                    pl.when(pl.col("form").is_finite()).then(pl.col("form")).alias("form"))


# Running style → the sim's pace policy (front-runners take position early, closers wait).
# The MECHANISM is real and the sim reproduces it (front-runner +0.6pp, closer −0.9pp), but
# as a PREDICTOR off the trailing-rel_early proxy it is FALSIFIED on ST: every PACE_GAIN>0
# hurts OOS monotonically (0.03→−0.010, 0.15→−0.131 nats/race). Reason: the early-speed edge
# is already absorbed by FORM (front-runners that overperform have elevated form) and priced
# by the market, so coupling it double-counts. Kept OFF (gain 0 = identity). The pace policy
# survives in traffic_jax as dormant infrastructure for (a) trajectory-MEASURED pace_bias,
# (b) the venue/distance validation litmus, (c) the best-response policy layer — not as a
# live term. Revisit at HV (tight geometry may make position pay where ST routes don't).
PACE_GAIN = float(os.environ.get("PACE_GAIN", "0.0"))


# Jockey couples like form — a measurement of the effective critical velocity the horse
# realises (part better-trip, part the private info that a fancied horse gets a top rider).
# CONFIRMED in the ABM: paired Δ +0.030 nats/race at gain 3 (P=99.2%), interior optimum
# ~3-6 (gain 15 over-couples and hurts), lifting the model ~29%→~38% of market skill — the
# +7pp the at-scale regression predicted (signal_eval_rich). Baked at the plateau centre.
# JOCKEY_GAIN = m/s of v_c per unit of jockey strike-rate above the field-neutral mean.
JOCKEY_GAIN = float(os.environ.get("JOCKEY_GAIN", "5.0"))
JKY_MEAN = 0.0  # set in load_st to the data mean, for centering


def jockey_table() -> pl.DataFrame:
    """Leak-free trailing strike rate of the jockey riding each horse, keyed (date, horse_id)."""
    rr = (pl.read_parquet(CACHE / "results_rich.parquet")
          .with_columns(pl.col("date").cast(pl.Utf8).alias("date"),
                        (pl.col("Pla").cast(pl.Int64, strict=False) == 1).cast(pl.Int8).alias("w"),
                        pl.col("date").str.strptime(pl.Date, "%Y-%m-%d", strict=False).alias("d"))
          .sort(["Jockey", "d"]))
    rr = rr.with_columns(((pl.col("w").cum_sum().over("Jockey") - pl.col("w"))
                          / (pl.col("w").cum_count().over("Jockey") - 1)).alias("jky_sr"))
    return rr.select("date", pl.col("HorseID").alias("horse_id"),
                     pl.when(pl.col("jky_sr").is_finite()).then(pl.col("jky_sr")).alias("jky_sr"))


def running_style_table() -> pl.DataFrame:
    """Leak-free trailing mean rel_early per horse (front-runner ↑, closer ↓)."""
    s = (pl.read_parquet(CACHE / "sectional_dataset.parquet")
         .with_columns(pl.col("date").cast(pl.Utf8).str.strptime(pl.Date, "%d/%m/%Y", strict=False).alias("d")))
    s = (s.select("d", "venue", "race_no", "runner", "horse_id", "rel_early")
         .sort(["horse_id", "d"])
         .with_columns(((pl.col("rel_early").cum_sum().over("horse_id") - pl.col("rel_early"))
                        / (pl.col("rel_early").cum_count().over("horse_id") - 1)).alias("style"))
         .with_columns(pl.col("d").dt.strftime("%Y-%m-%d").alias("date")))
    return s.select("date", "venue", "race_no", "runner",
                    pl.when(pl.col("style").is_finite()).then(pl.col("style")).alias("style"))


def load_st() -> pl.DataFrame:
    e = pl.read_parquet(CACHE / ENERGETICS).with_columns(_norm("date").alias("date"))
    w = pl.read_parquet(CACHE / WPQ).with_columns(_norm("date").alias("date"))
    r = (pl.read_parquet(CACHE / "rating_dataset.parquet")
         .with_columns(_norm("date").str.strptime(pl.Date, "%Y-%m-%d", strict=False).alias("d"),
                       pl.col("rtg").cast(pl.Float64))
         .drop_nulls(["d", "rtg"]).sort(["horse_id", "d"]))
    keys = ["date", "venue", "race_no", "runner"]
    df = (e.select(keys + ["horse_id", "v_c", "w_prime", "distance"])
          .join(w.select(keys + ["draw", "won", "placed", "finish_pos", "field_size", "market_win_prob"]), on=keys)
          .filter(pl.col("venue") == "ST")
          .with_columns(pl.col("date").str.strptime(pl.Date, "%Y-%m-%d", strict=False).alias("d"))
          .sort(["horse_id", "d"]))
    # rating as-of *backward* (most recent at or before the race) — no look-ahead
    df = df.join_asof(r.select("horse_id", "d", "rtg"), on="d", by="horse_id", strategy="backward")
    gmed = float(r["rtg"].median())
    df = df.with_columns(pl.col("rtg").fill_null(gmed))
    # real per-race rail position (Course) — the rail directly changes the geometry, so
    # never assume "A". st_history carries it; fall back to A only if absent.
    hist = CACHE / "st_history.parquet"
    if hist.exists():
        ch = (pl.read_parquet(hist).with_columns(
                  pl.col("Course").cast(pl.Utf8).str.extract(r'"([A-C][+0-9]*)"', 1).alias("course"),
                  pl.col("date").cast(pl.Utf8).alias("date"),
                  pl.col("RaceNumber").cast(pl.Int64, strict=False).alias("race_no"))
              .select("date", "race_no", "course").unique(["date", "race_no"]))
        df = df.join(ch, on=["date", "race_no"], how="left")
    else:
        df = df.with_columns(pl.lit(None, dtype=pl.Utf8).alias("course"))
    df = df.with_columns(pl.col("course").fill_null("A"))
    # current-form ability state (leak-free); debut horses get the population-neutral 0.5
    df = df.join(current_form_table(), on=["date", "venue", "race_no", "runner"], how="left")
    df = df.with_columns(pl.col("form").fill_null(0.5))
    # running-style state (leak-free trailing early speed); unknown → field-neutral
    df = df.join(running_style_table(), on=["date", "venue", "race_no", "runner"], how="left")
    df = df.with_columns(pl.col("style").fill_null(pl.col("style").mean()))
    # jockey trip/ability measurement (leak-free trailing strike rate); unknown → field-neutral
    global JKY_MEAN
    df = df.join(jockey_table(), on=["date", "horse_id"], how="left")
    JKY_MEAN = float(df["jky_sr"].mean())
    df = df.with_columns(pl.col("jky_sr").fill_null(JKY_MEAN))
    return df.with_columns((pl.col("date") + "_" + pl.col("race_no").cast(pl.Utf8)).alias("race_id"))


def _surface(distance: int) -> str:
    return "awt" if distance == 1650 else "turf"


def _horse_residuals(train: pl.DataFrame, pop: EnergeticsPopulation) -> dict:
    """Per-horse mean EB residual (v_c, w') over the horse's train runs + count."""
    d = train.with_columns(pl.col("distance").map_elements(distance_band, return_dtype=pl.Utf8).alias("_b"))
    res = {}
    rtg = d["rtg"].to_numpy().astype(float)
    bands = d["_b"].to_list()
    horses = d["horse_id"].to_list()
    u_vc = d["v_c"].to_numpy() - np.array([pop.v_c.fixed(b, rtg[i]) for i, b in enumerate(bands)])
    u_w = d["w_prime"].to_numpy() - np.array([pop.w_prime.fixed(b, rtg[i]) for i, b in enumerate(bands)])
    agg: dict = {}
    for h, a, b in zip(horses, u_vc, u_w):
        s = agg.setdefault(h, [0, 0.0, 0.0])
        s[0] += 1
        s[1] += a
        s[2] += b
    for h, (n, sa, sb) in agg.items():
        res[h] = (n, sa / n, sb / n)
    return res


def fit_predict_traffic(train: pl.DataFrame, test: pl.DataFrame) -> pl.DataFrame:
    pop = fit_energetics_population(train, rating_col="rtg")
    res = _horse_residuals(train, pop)
    lam_vc = lambda n: pop.v_c.tau2 / (pop.v_c.tau2 + pop.v_c.sigma2 / n) if n > 0 else 0.0   # noqa: E731
    lam_w = lambda n: pop.w_prime.tau2 / (pop.w_prime.tau2 + pop.w_prime.sigma2 / n) if n > 0 else 0.0  # noqa: E731
    preds = []
    for race_id, g in test.group_by("race_id"):
        g = g.sort("runner")
        dist = int(g["distance"][0])
        rtg = g["rtg"].to_numpy().astype(float)
        form = g["form"].to_numpy().astype(float)
        jky = g["jky_sr"].to_numpy().astype(float)
        band = distance_band(dist)
        vc, wp = [], []
        for i, h in enumerate(g["horse_id"].to_list()):
            n, u_vc, u_w = res.get(h, (0, 0.0, 0.0))
            # effective critical velocity = sectional EB estimate + form + jockey measurements
            vc.append(pop.v_c.fixed(band, rtg[i]) + lam_vc(n) * u_vc + FORM_GAIN * (form[i] - 0.5)
                      + JOCKEY_GAIN * (jky[i] - JKY_MEAN))
            wp.append(pop.w_prime.fixed(band, rtg[i]) + lam_w(n) * u_w)
        geom = track_for("ST", dist, _surface(dist))
        course = g["course"][0] if "course" in g.columns else "A"
        # within-race style z-score → pace policy: front-runners take position, closers wait
        style = g["style"].to_numpy().astype(float)
        sd = style.std()
        pace_bias = (style - style.mean()) / sd if sd > 1e-9 else np.zeros_like(style)
        p = win_probs(np.array(vc), np.array(wp), g["draw"].to_numpy().astype(float), dist,
                      track=geom, course=course, size=N_SAMPLES, bend_curvature=BEND,
                      speed_noise=SPEED_NOISE, pace_bias=pace_bias, pace_gain=PACE_GAIN,
                      seed=abs(hash(race_id[0])) % (2**31))
        preds.append(g.with_columns(pl.Series("p", p), pl.Series("v_c_eb", vc),
                                    pl.Series("w_prime_eb", wp)))
    return pl.concat(preds)


def fit_predict_market(train, test):
    return test.with_columns(
        (pl.col("market_win_prob") / pl.col("market_win_prob").sum().over("race_id")).alias("p"))


def fit_predict_uniform(train, test):
    return test.with_columns((1.0 / pl.col("field_size")).alias("p"))


def main():
    df = load_st()
    dates = df["date"].unique().to_list()
    folds = walk_forward_folds(dates, n_folds=5, embargo=1, min_train=1)
    print(f"Sha Tin: {df['race_id'].n_unique()} races, {df.height} runners, "
          f"{len(dates)} meeting-days, {len(folds)} folds  (bend={BEND}, N={N_SAMPLES})")

    models = {"traffic": fit_predict_traffic, "market": fit_predict_market,
              "uniform": fit_predict_uniform}
    summary = {}
    for name, fp in models.items():
        fm = harness.walk_forward_evaluate(df, folds, fp, date_col="date",
                                           prob_col="p", outcome_col="won", race_col="race_id")
        summary[name] = harness.aggregate(fm)

    print("\n  model     race_win_log_loss   brier     ece      n_races")
    for name, s in summary.items():
        print(f"  {name:8s}  {s.get('race_win_log_loss', float('nan')):.4f}"
              f"            {s.get('brier', float('nan')):.4f}   {s.get('ece', float('nan')):.4f}"
              f"   {s.get('n_test_total', 0)}")
    base = summary["uniform"]["race_win_log_loss"]
    print(f"\n  skill vs uniform (lower=better):  traffic {base - summary['traffic']['race_win_log_loss']:+.4f}"
          f"   market {base - summary['market']['race_win_log_loss']:+.4f}  nats/race")
    print(f"  gap to market:  traffic − market = "
          f"{summary['traffic']['race_win_log_loss'] - summary['market']['race_win_log_loss']:+.4f} nats/race")


if __name__ == "__main__":
    main()
