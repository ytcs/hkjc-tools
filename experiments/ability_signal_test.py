"""Locate the 10x gap: the win-skill each ability signal carries (walk-forward).

Our structural model recovers ~12% of the market's win skill. This decomposes the gap:
walk-forward conditional-logit race-win log-loss for each signal alone and combined,
vs uniform (floor) and market (ceiling). Tells us whether the shortfall is (a) weak
ability signals or (b) our dynamics WASTING signals it already has (e.g. rating) — so
we know which mechanism to strengthen next.

Run: uv run python experiments/ability_signal_test.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from hkjc.eval import harness
from hkjc.eval.walkforward import walk_forward_folds
from hkjc.model.conditional_logit import conditional_logit_fit_predict

CACHE = Path("data_cache")


def main():
    panel = pl.read_parquet(CACHE / "oos_panel.parquet").with_columns(pl.col("date").cast(pl.Utf8).alias("dd"))
    form = pl.read_parquet(CACHE / "form_dataset.parquet").with_columns(
        pl.col("HorseNo").alias("horse_id"), pl.col("Date").alias("d"))
    p = panel.join(form.select("horse_id", "d", "win_rate_prior", "avg_finish_prior3"),
                   on=["horse_id", "d"], how="left").with_columns(
        pl.col("win_rate_prior").fill_null(0.05),
        pl.col("avg_finish_prior3").fill_null(7.0),
        pl.col("rtg").fill_null(pl.col("rtg").median()),
        pl.col("v_c_eb").alias("vc"),
        pl.col("p_traffic").log().alias("log_traffic"),
        pl.col("p_market").log().alias("log_market"))

    folds = walk_forward_folds(p["dd"].unique().to_list(), n_folds=5, embargo=1, min_train=1)

    def skill(feats):
        fp = conditional_logit_fit_predict(feats, race_keys=["race_id"], prob_col="q")
        fm = harness.walk_forward_evaluate(p, folds, fp, date_col="dd", prob_col="q",
                                           outcome_col="won", race_col="race_id")
        return harness.aggregate(fm).get("race_win_log_loss", float("nan"))

    rows = [
        ("v_c (our energetics)", ["vc"]),
        ("rating only", ["rtg"]),
        ("win_rate_prior only", ["win_rate_prior"]),
        ("avg_finish_prior3 only", ["avg_finish_prior3"]),
        ("rating + form", ["rtg", "win_rate_prior", "avg_finish_prior3"]),
        ("rating + form + v_c", ["rtg", "win_rate_prior", "avg_finish_prior3", "vc"]),
        ("our traffic model p", ["log_traffic"]),
        ("MARKET (ceiling)", ["log_market"]),
    ]
    lls = {name: skill(feats) for name, feats in rows}
    floor = lls["v_c (our energetics)"]
    mk = lls["MARKET (ceiling)"]
    print(f"\n{'signal':28s} {'race-win log-loss':>17s}   {'% of market skill':>17s}")
    for name in lls:
        share = (floor - lls[name]) / (floor - mk) if floor > mk else float("nan")
        print(f"{name:28s} {lls[name]:>17.4f}   {share:>16.0%}")
    print("\n(% of market skill uses our v_c as the 0% floor and market as 100%.)")


if __name__ == "__main__":
    main()
