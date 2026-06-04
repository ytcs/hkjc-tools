"""THE architecture test: does stacking the ABM with per-horse signals beat both alone?

The ABM (~38% of market) and a per-horse logit (EW form + jockey, 41%) capture ability
DIFFERENTLY: the ABM has field interaction (traffic, pace pressure, draw x geometry — a horse's
chance depends on the OTHERS) that a per-horse model cannot represent; the logit has recency form
the ABM's rating channel washes out. If the two signals are partly orthogonal, a stack should
exceed both.

We generate the ABM's leak-free OOS win prob via its own walk-forward, then use log(ABM_p) as a
feature in a SECOND-level conditional logit alongside z_EW_form and z_jockey (both keyed date+
horse_id from results_rich, trailing/leak-free). Coefficient 1 on log(ABM_p), 0 else == the ABM
exactly, so any improvement is real added skill. All models compared on the SAME race set + market.

Run: uv run python experiments/hybrid_stack.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from hkjc.eval import harness
from hkjc.eval.walkforward import walk_forward_folds
from hkjc.model.conditional_logit import conditional_logit_fit_predict

import backtest_traffic_st as bt

CACHE = Path("data_cache")
bt.N_SAMPLES = 4000


def ew_form_table() -> pl.DataFrame:
    """Leak-free exp-weighted (half-life 2) relative-finish per horse, keyed (date, horse_id)."""
    rr = (pl.read_parquet(CACHE / "results_rich.parquet")
          .with_columns(pl.col("date").cast(pl.Utf8).alias("date"),
                        pl.col("Pla").cast(pl.Int64, strict=False),
                        pl.col("date").str.strptime(pl.Date, "%Y-%m-%d", strict=False).alias("d"))
          .with_columns(pl.len().over(["date", "Venue", "RaceNumber"]).alias("fs"))
          .with_columns((1.0 - (pl.col("Pla") - 1) / (pl.col("fs") - 1)).alias("rf"))
          .sort(["HorseID", "d"]))
    rr = rr.with_columns(
        pl.col("rf").shift(1).ewm_mean(half_life=2.0, ignore_nulls=True).over("HorseID").alias("ewf"))
    return rr.select("date", pl.col("HorseID").alias("horse_id"),
                     pl.when(pl.col("ewf").is_finite()).then(pl.col("ewf")).alias("ewf"))


def abm_oos_preds() -> pl.DataFrame:
    """Run the ST ABM walk-forward; collect per-runner OOS win prob with keys for stacking."""
    df = bt.load_st()
    folds = walk_forward_folds(df["date"].unique().to_list(), n_folds=5, embargo=1, min_train=1)
    rows = []
    for fold in folds:
        train = df.filter(pl.col("date").is_in(fold.train))
        test = df.filter(pl.col("date").is_in(fold.test))
        if test.is_empty():
            continue
        pred = bt.fit_predict_traffic(train, test)
        rows.append(pred.select("date", "race_id", "horse_id", "won", "market_win_prob",
                                pl.col("p").alias("abm_p")))
    return pl.concat(rows)


def main():
    preds = abm_oos_preds()
    ew = ew_form_table()
    jk = bt.jockey_table()
    d = (preds.with_columns(pl.col("date").cast(pl.Utf8))
         .join(ew, on=["date", "horse_id"], how="left")
         .join(jk, on=["date", "horse_id"], how="left"))
    d = d.with_columns(
        pl.col("ewf").fill_null(0.5), pl.col("jky_sr").fill_null(float(jk["jky_sr"].mean())),
        pl.col("abm_p").clip(1e-6, 1.0).log().alias("lp_abm"),
        pl.col("date").str.strptime(pl.Date, "%Y-%m-%d", strict=False).alias("dt"))
    for c in ["lp_abm", "ewf", "jky_sr"]:
        s = d[c].std()
        d = d.with_columns(((pl.col(c) - d[c].mean()) / (s if s else 1.0)).alias(f"z_{c}"))

    dates = sorted(d["dt"].unique().to_list())
    folds = walk_forward_folds(dates, n_folds=5, embargo=1, min_train=1)
    print(f"{d.height} runners, {d['race_id'].n_unique()} races (ABM subset)\n")

    def skill(feats):
        fp = conditional_logit_fit_predict(feats, race_keys=["race_id"], prob_col="q")
        fm = harness.walk_forward_evaluate(d, folds, fp, date_col="dt", prob_col="q",
                                           outcome_col="won", race_col="race_id")
        return harness.aggregate(fm).get("race_win_log_loss", float("nan"))

    floor = harness.aggregate(harness.walk_forward_evaluate(
        d.with_columns(pl.col("won").len().over("race_id").alias("fs")).with_columns((1.0/pl.col("fs")).alias("uni")),
        folds, lambda tr, te: te.with_columns(pl.col("uni").alias("q")),
        date_col="dt", prob_col="q", outcome_col="won", race_col="race_id")).get("race_win_log_loss")
    market = harness.aggregate(harness.walk_forward_evaluate(
        d, folds, lambda tr, te: te.with_columns((pl.col("market_win_prob") /
            pl.col("market_win_prob").sum().over("race_id")).alias("q")),
        date_col="dt", prob_col="q", outcome_col="won", race_col="race_id")).get("race_win_log_loss")
    # raw ABM (no stacker) on this same set
    abm_raw = harness.aggregate(harness.walk_forward_evaluate(
        d, folds, lambda tr, te: te.with_columns((pl.col("abm_p") /
            pl.col("abm_p").sum().over("race_id")).alias("q")),
        date_col="dt", prob_col="q", outcome_col="won", race_col="race_id")).get("race_win_log_loss")

    def line(name, ll):
        sk = floor - ll
        print(f"  {name:32s} {ll:>9.4f} {sk:>+9.4f} {sk/(floor-market):>8.0%}")

    print(f"  {'model':32s} {'LL':>9s} {'skill':>9s} {'%mkt':>8s}")
    line("uniform (floor)", floor)
    line("ABM raw (no stacker)", abm_raw)
    line("stack: ABM only", skill(["z_lp_abm"]))
    line("stack: EW+jockey only", skill(["z_ewf", "z_jky_sr"]))
    line("stack: ABM + EW", skill(["z_lp_abm", "z_ewf"]))
    line("stack: ABM + EW + jockey", skill(["z_lp_abm", "z_ewf", "z_jky_sr"]))
    line("MARKET (ceiling)", market)

    # paired-bootstrap the two pivotal contrasts on per-race loss (positive = first is better)
    def race_losses(fp):
        out = {}
        for fold in folds:
            tr = d.filter(pl.col("dt").is_in(fold.train)); te = d.filter(pl.col("dt").is_in(fold.test))
            if te.is_empty():
                continue
            pred = fp(tr, te)
            for rid, g in pred.group_by("race_id"):
                q = np.clip(g["q"].to_numpy().astype(float), 1e-12, None); q = q / q.sum()
                won = g["won"].to_numpy().astype(bool)
                if won.any():
                    out[rid[0]] = -np.log(q[won][0])
        return out

    def logit_losses(feats):
        return race_losses(conditional_logit_fit_predict(feats, race_keys=["race_id"], prob_col="q"))
    abm_losses = race_losses(lambda tr, te: te.with_columns(
        (pl.col("abm_p") / pl.col("abm_p").sum().over("race_id")).alias("q")))
    ewj = logit_losses(["z_ewf", "z_jky_sr"])
    full = logit_losses(["z_lp_abm", "z_ewf", "z_jky_sr"])
    rng = np.random.default_rng(0)

    def boot(a, b, label):  # mean(a - b); positive = a better (lower loss)
        ks = sorted(set(a) & set(b))
        diff = np.array([a[k] - b[k] for k in ks])
        bs = np.array([rng.choice(diff, diff.size, replace=True).mean() for _ in range(5000)])
        lo, hi = np.percentile(bs, [2.5, 97.5])
        print(f"  {label:42s} Δ {diff.mean():+.4f} [{lo:+.4f},{hi:+.4f}] P={ (bs>0).mean():.1%}")

    print("\n  paired bootstrap (Δ = second − first per-race loss; positive ⇒ FIRST model better):")
    boot(abm_losses, ewj, "ABM raw  vs  EW+jockey logit")
    boot(full, ewj, "ABM+EW+jockey  vs  EW+jockey  (does ABM add?)")


if __name__ == "__main__":
    main()
