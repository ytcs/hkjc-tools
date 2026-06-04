"""WHERE does the ABM's field-interaction edge actually live? (aggregate-zero may hide local signal)

At scale the ABM adds nothing to an EW-form+jockey logit IN AGGREGATE (Δ=0.0002). But the ABM's one
non-substitutable idea is FIELD INTERACTION: a horse's chance depends on the OTHERS (pace pressure,
lone speed, draw x geometry) — which a per-horse logit structurally cannot see. If that edge is real
but localized, it nets to ~zero only because most races are interaction-neutral. We compute, per race,
the ABM's advantage over the logit (logit_loss − abm_loss; +ve ⇒ ABM wins that race) and slice it by
race-shape covariates where interaction should bite hardest:
  - lone speed: one clear front-runner in a field of closers (gap in early-position tendency)
  - pace pressure: many front-runners (a speed duel that should cook the leaders)
  - field size, distance, venue, draw spread.
A systematic ABM advantage in any slice = real structural value worth keeping; flat everywhere = the
ABM is genuinely redundant.

Run: uv run --group bayes python experiments/field_interaction_probe.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from hkjc.eval.walkforward import walk_forward_folds
from hkjc.model.conditional_logit import conditional_logit_fit_predict

import backtest_traffic_st as bt
from hybrid_stack import abm_oos_preds, ew_form_table

CACHE = Path("data_cache")
bt.N_SAMPLES = 4000


def style_table() -> pl.DataFrame:
    """Trailing early-position tendency per horse (1=led, 0=last), keyed (date, horse_id). Leak-free."""
    rr = (pl.read_parquet(CACHE / "results_rich.parquet")
          .with_columns(pl.col("date").cast(pl.Utf8),
                        pl.col("date").str.strptime(pl.Date, "%Y-%m-%d", strict=False).alias("d"),
                        pl.len().over(["date", "Venue", "RaceNumber"]).alias("fs"),
                        pl.col("RunningPosition").str.split(" ").list.first()
                        .cast(pl.Float64, strict=False).alias("early")))
    rr = rr.with_columns(((pl.col("early") - 1) / (pl.col("fs") - 1)).alias("early_norm"))  # 0=led
    rr = rr.sort(["HorseID", "d"]).with_columns(
        ((pl.col("early_norm").cum_sum().over("HorseID") - pl.col("early_norm"))
         / (pl.col("early_norm").cum_count().over("HorseID") - 1)).alias("style_raw"))
    # convert so HIGH = front-runner: 1 - trailing early_norm
    return rr.select("date", pl.col("HorseID").alias("horse_id"),
                     pl.when(pl.col("style_raw").is_finite()).then(1.0 - pl.col("style_raw")).alias("frontiness"))


def logit_oos(d, folds, feats):
    """Per-runner OOS logit prob q across folds."""
    parts = []
    fp = conditional_logit_fit_predict(feats, race_keys=["race_id"], prob_col="q")
    for fold in folds:
        tr = d.filter(pl.col("dt").is_in(fold.train)); te = d.filter(pl.col("dt").is_in(fold.test))
        if not te.is_empty():
            parts.append(fp(tr, te))
    return pl.concat(parts)


def main():
    preds = abm_oos_preds()
    ew = ew_form_table(); jk = bt.jockey_table(); st = style_table()
    dist_map = (pl.read_parquet(CACHE / bt.ENERGETICS)
                .with_columns(bt._norm("date").alias("date"))
                .select("date", "horse_id", "distance").unique())
    d = (preds.with_columns(pl.col("date").cast(pl.Utf8))
         .join(dist_map, on=["date", "horse_id"], how="left")
         .join(ew, on=["date", "horse_id"], how="left")
         .join(jk, on=["date", "horse_id"], how="left")
         .join(st, on=["date", "horse_id"], how="left"))
    d = d.with_columns(pl.col("ewf").fill_null(0.5), pl.col("jky_sr").fill_null(float(jk["jky_sr"].mean())),
                       pl.col("frontiness").fill_null(0.5),
                       pl.col("date").str.strptime(pl.Date, "%Y-%m-%d", strict=False).alias("dt"))
    for c in ["ewf", "jky_sr"]:
        s = d[c].std(); d = d.with_columns(((pl.col(c) - d[c].mean()) / (s if s else 1.0)).alias(f"z_{c}"))

    folds = walk_forward_folds(sorted(d["dt"].unique().to_list()), n_folds=5, embargo=1, min_train=1)
    lg = logit_oos(d, folds, ["z_ewf", "z_jky_sr"]).select("race_id", "horse_id", "q")
    d = d.join(lg, on=["race_id", "horse_id"], how="inner")

    # per-race losses + shape covariates
    rows = []
    for rid, g in d.group_by("race_id"):
        won = g["won"].to_numpy().astype(bool)
        if not won.any():
            continue
        ap = g["abm_p"].to_numpy().astype(float); ap = np.clip(ap / ap.sum(), 1e-12, 1)
        lp = g["q"].to_numpy().astype(float); lp = np.clip(lp / lp.sum(), 1e-12, 1)
        fr = np.sort(g["frontiness"].to_numpy().astype(float))[::-1]
        lone = float(fr[0] - fr[1]) if len(fr) > 1 else 0.0          # gap leader->2nd: lone speed
        n_front = int((g["frontiness"].to_numpy() > 0.66).sum())     # speed-duel count
        rows.append({"race_id": rid[0],
                     "abm_loss": -np.log(ap[won][0]), "lg_loss": -np.log(lp[won][0]),
                     "field_size": g.height, "distance": int(g["distance"][0]),
                     "lone_speed": lone, "n_front": n_front})
    r = pl.DataFrame(rows).with_columns((pl.col("lg_loss") - pl.col("abm_loss")).alias("abm_adv"))
    print(f"{r.height} races.  mean ABM advantage (logit_loss − abm_loss) = {r['abm_adv'].mean():+.4f} "
          f"(+ve ⇒ ABM better)\n")

    def slice_by(col, qs=(0.0, 0.33, 0.66, 1.0)):
        edges = [r[col].quantile(q) for q in qs]
        print(f"  by {col}:")
        for i in range(len(edges) - 1):
            lo, hi = edges[i], edges[i + 1]
            sub = r.filter((pl.col(col) >= lo) & (pl.col(col) <= hi)) if i == len(edges) - 2 \
                else r.filter((pl.col(col) >= lo) & (pl.col(col) < hi))
            if sub.height:
                print(f"    [{lo:7.2f},{hi:7.2f}]  n={sub.height:4d}  ABM adv {sub['abm_adv'].mean():+.4f}")

    for col in ["lone_speed", "n_front", "field_size", "distance"]:
        slice_by(col)
    # correlations
    print("\n  Pearson corr(ABM advantage, covariate):")
    for col in ["lone_speed", "n_front", "field_size", "distance"]:
        c = np.corrcoef(r["abm_adv"].to_numpy(), r[col].to_numpy().astype(float))[0, 1]
        print(f"    {col:12s} {c:+.3f}")


if __name__ == "__main__":
    main()
