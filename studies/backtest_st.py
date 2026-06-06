"""Walk-forward backtest of the ABM vs the market — rebuilt on the refactored library.

This is the first study rebuilt after the refactor: it proves the new seams end-to-end.
The model itself (leak-free features, EB ability fit, the win-prob sim, the baked
constants) now lives in the library — this file is only *study glue*: which parquet
panels to read, how to assemble the modelling frame, and the walk-forward reporting.

    loader (here)  ->  hkjc.model.features / hkjc.model.fit  ->  hkjc.backtesting.harness

Run: uv run --group bayes python studies/backtest_st.py            # ST
     VENUE=HV uv run --group bayes python studies/backtest_st.py   # HV
"""
from __future__ import annotations

import os
from pathlib import Path

import polars as pl

from hkjc.backtesting import harness, walk_forward_folds
from hkjc.model.features import (attach_rating_trajectory, barrier_trial_table,
                                 collateral_form_table, early_position_table, form_table,
                                 jockey_table, prior_starts_table, recency_form_table,
                                 speed_figure_table, style_table, trainer_hot_table,
                                 trainer_table, trajectory_trait_table)
from hkjc.model.constants import FADE_GAIN, FORM_HALF_LIFE
from hkjc.model.fit import fit_predict_traffic

CACHE = Path("data_cache")
VENUE = os.environ.get("VENUE", "ST")
ENERGETICS = CACHE / os.environ.get("ENERGETICS_FILE", "energetics_dataset_full.parquet")
WPQ = CACHE / os.environ.get("WPQ_FILE", "wpq_dataset_full.parquet")


def _norm(col: str) -> pl.Expr:
    return pl.col(col).cast(pl.Utf8).str.replace_all("/", "-")


def load(venue: str = VENUE) -> tuple[pl.DataFrame, float]:
    """Assemble the leak-free modelling frame for one venue. Returns (frame, jky_mean)."""
    e = pl.read_parquet(ENERGETICS).with_columns(_norm("date").alias("date"))
    w = pl.read_parquet(WPQ).with_columns(_norm("date").alias("date"))
    r = (pl.read_parquet(CACHE / "rating_dataset.parquet")
         .with_columns(_norm("date").str.strptime(pl.Date, "%Y-%m-%d", strict=False).alias("d"),
                       pl.col("rtg").cast(pl.Float64))
         .drop_nulls(["d", "rtg"]).sort(["horse_id", "d"]))
    keys = ["date", "venue", "race_no", "runner"]
    df = (e.select(keys + ["horse_id", "v_c", "w_prime", "distance"])
          .join(w.select(keys + ["draw", "won", "placed", "finish_pos", "field_size", "market_win_prob"]),
                on=keys)
          .filter(pl.col("venue") == venue)
          .with_columns(pl.col("date").str.strptime(pl.Date, "%Y-%m-%d", strict=False).alias("d"))
          .sort(["horse_id", "d"]))
    df = df.join_asof(r.select("horse_id", "d", "rtg"), on="d", by="horse_id", strategy="backward")
    df = df.with_columns(pl.col("rtg").fill_null(float(r["rtg"].median())))
    # real per-race rail (Course) from st_history — geometry depends on it
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
    # leak-free ability features from the library
    # production form = RECENCY (exp-weighted, half-life ~2 races): 48% -> 56% of market over the
    # career mean. Career form kept as `form_career` for diagnosis/comparison only.
    df = df.join(form_table(e, w).rename({"form": "form_career"}), on=keys, how="left")
    df = df.with_columns(pl.col("form_career").fill_null(0.5))
    # strictly-prior career start count (0 = debut) — the flag for the barrier-trial debut signal
    df = df.join(prior_starts_table(e, w), on=keys, how="left")
    df = df.with_columns(pl.col("prior_starts").fill_null(0))
    df = df.join(recency_form_table(e, w, half_life=FORM_HALF_LIFE).rename({"form_rec": "form"}),
                 on=keys, how="left").with_columns(pl.col("form").fill_null(0.5))
    sect = pl.read_parquet(CACHE / "sectional_dataset.parquet")
    df = df.join(style_table(sect), on=keys, how="left")
    df = df.with_columns(pl.col("style").fill_null(pl.col("style").mean()))
    # leak-free habitual early POSITION (front-runner tendency) for the positional-fade term
    rpq = CACHE / "run_policy_dataset.parquet"
    if rpq.exists():
        df = df.join(early_position_table(pl.read_parquet(rpq)), on=keys, how="left")
        df = df.with_columns(pl.col("early_pos").fill_null(pl.col("early_pos").mean()))
    else:
        df = df.with_columns(pl.lit(0.764).alias("early_pos"))
    rr = pl.read_parquet(CACHE / "results_rich.parquet")
    df = df.join(jockey_table(rr), on=["date", "horse_id"], how="left")
    jky_mean = float(df["jky_sr"].mean())
    df = df.with_columns(pl.col("jky_sr").fill_null(jky_mean))
    df = df.join(trainer_table(rr), on=["date", "horse_id"], how="left")
    df = df.with_columns(pl.col("trn_sr").fill_null(float(df["trn_sr"].mean())))
    # trainer RECENCY hot-streak (intent / stable-confidence overlay); leak-free strictly-prior
    df = df.join(trainer_hot_table(rr), on=["date", "horse_id"], how="left")
    df = df.with_columns(pl.col("trn_hot").fill_null(float(df["trn_hot"].mean())))
    # declared horse weight (published pre-race -> leak-free), keyed by runner
    bw = (rr.with_columns(pl.col("date").cast(pl.Utf8).str.replace_all("/", "-"),
                          pl.col("HorseNo").cast(pl.Int64, strict=False).alias("runner"),
                          pl.col("RaceNumber").cast(pl.Int64, strict=False).alias("race_no"),
                          pl.col("Venue").cast(pl.Utf8).alias("venue"),
                          pl.col("DeclarHorseWt").cast(pl.Float64, strict=False).alias("bodywt"))
          .select("date", "venue", "race_no", "runner", "bodywt"))
    df = df.join(bw, on=["date", "venue", "race_no", "runner"], how="left")
    # velocity-based speed figure (leak-free trailing); neutral 0 when unknown
    df = df.join(speed_figure_table(rr), on=["date", "venue", "race_no", "runner"], how="left")
    df = df.with_columns(pl.col("speedfig").fill_null(0.0))
    # collateral-form ability rating (opponent-adjusted velocity, leak-free strictly-prior); 0 = no prior
    df = df.join(collateral_form_table(rr), on=["date", "venue", "race_no", "runner"], how="left")
    df = df.with_columns(pl.col("collat_form").fill_null(0.0))
    # trajectory-shape traits (sustained-speed, fade-resistance) from per-section velocity path; 0 = unknown
    trajq = CACHE / "trajectory_dataset_full.parquet"
    if trajq.exists():
        df = df.join(trajectory_trait_table(pl.read_parquet(trajq)),
                     on=["date", "venue", "race_no", "runner"], how="left")
        df = df.with_columns(pl.col("traj_sustain").fill_null(0.0), pl.col("traj_fade").fill_null(0.0))
    else:
        df = df.with_columns(pl.lit(0.0).alias("traj_sustain"), pl.lit(0.0).alias("traj_fade"))
    # barrier-trial speed figure (the only prior on-track form for DEBUTANTS); leak-free as-of join
    # on horse brand. trial date strictly precedes the race (mornings vs race days). 0 = no prior trial.
    btq = CACHE / "barrier_trials.parquet"
    if btq.exists():
        tt = barrier_trial_table(pl.read_parquet(btq)).sort("d")
        df = df.sort("d").join_asof(tt, on="d", by="horse_id", strategy="backward")
        df = df.with_columns(pl.col("trial_fig").fill_null(0.0))
    else:
        df = df.with_columns(pl.lit(0.0).alias("trial_fig"))
    # rating-trajectory PROCESS feature (own-peak deficit); strict backward as-of (no same-day leak)
    df = attach_rating_trajectory(df, r)
    df = df.with_columns((pl.col("date") + "_" + pl.col("race_no").cast(pl.Utf8)).alias("race_id"))
    return df, jky_mean


def _market(train, test):
    return test.with_columns(
        (pl.col("market_win_prob") / pl.col("market_win_prob").sum().over("race_id")).alias("p"))


def _uniform(train, test):
    return test.with_columns((1.0 / pl.col("field_size")).alias("p"))


def main():
    df, jky_mean = load()
    dates = df["date"].unique().to_list()
    folds = walk_forward_folds(dates, n_folds=5, embargo=1, min_train=1)
    print(f"{VENUE}: {df['race_id'].n_unique()} races, {df.height} runners, {len(folds)} folds")

    wpg = float(os.environ.get("W_PRIME_GAIN", "2.0"))
    sim = os.environ.get("SIM", "wald")
    _ab = os.environ.get("ALLOMETRIC_B")
    ab = float(_ab) if _ab is not None else None  # None -> per-venue ALLOMETRIC_B_BY_VENUE
    _dfs = os.environ.get("DRAW_FS_GAIN")  # crowding-amplified draw; None -> per-venue DRAW_FS_GAIN_BY_VENUE
    dfs = float(_dfs) if _dfs is not None else None
    _tg = os.environ.get("TRIAL_GAIN")  # barrier-trial debut signal; None -> per-venue TRIAL_GAIN_BY_VENUE
    tg = float(_tg) if _tg is not None else None
    _cg = os.environ.get("COLLAT_GAIN")  # opponent-adjusted collateral-form; None -> per-venue baked
    cg = float(_cg) if _cg is not None else None
    _fg = os.environ.get("FADE_GAIN")  # trajectory fade-resistance; None -> baked default
    fg2 = float(_fg) if _fg is not None else FADE_GAIN
    _pgg = os.environ.get("PEAKGAP_GAIN")  # rating-trajectory own-peak deficit; None -> per-venue baked
    pgg = float(_pgg) if _pgg is not None else None
    _thg = os.environ.get("TRN_HOT_GAIN")  # trainer recency hot-streak (intent); None -> per-venue baked
    thg = float(_thg) if _thg is not None else None
    _dvg = os.environ.get("DRAW_VAR_GAIN")  # draw -> diffusion D (trip variance); None -> per-venue baked
    dvg = float(_dvg) if _dvg is not None else None

    def _model(train, test):
        return fit_predict_traffic(train, test, jky_mean=jky_mean, w_prime_gain=wpg, sim=sim,
                                   allometric_b=ab, draw_fs_gain=dfs, trial_gain=tg, collat_gain=cg,
                                   fade_gain=fg2, rtg_peakgap_gain=pgg, trn_hot_gain=thg, draw_var_gain=dvg)

    summary = {}
    for name, fp in {"traffic": _model, "market": _market, "uniform": _uniform}.items():
        fm = harness.walk_forward_evaluate(df, folds, fp, date_col="date",
                                           prob_col="p", outcome_col="won", race_col="race_id")
        summary[name] = harness.aggregate(fm)

    print("\n  model     race_win_log_loss   ece      n_races")
    for name, s in summary.items():
        print(f"  {name:8s}  {s.get('race_win_log_loss', float('nan')):.4f}"
              f"            {s.get('ece', float('nan')):.4f}   {s.get('n_test_total', 0)}")
    uni, mkt, mdl = (summary[k]["race_win_log_loss"] for k in ("uniform", "market", "traffic"))
    pct = (uni - mdl) / (uni - mkt) if uni > mkt else float("nan")
    print(f"\n  % of market (goodness-of-fit) = {pct:.1%}   "
          f"(gap to market {mdl - mkt:+.4f} nats/race)")


if __name__ == "__main__":
    main()
