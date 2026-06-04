"""Does the agent-based model earn its keep vs a 1-factor regression on form?

The honest ablation: take the EXACT same leak-free form signal the ABM couples in,
and feed it to a plain walk-forward conditional logit (1 free parameter). Score it on
the IDENTICAL races/folds. If the ABM (traffic sim + energetics + geometry) doesn't
beat form-logit, the physics is decoration. We also test form+draw and form+v_c to see
whether the structural pieces (draw via traffic, v_c via energetics) add anything a
flat regression can't.

ABM/market/uniform numbers are reproduced from the confirmed N=5000 run on this exact
pipeline (load_st + 5 walk-forward folds): traffic 2.4054 / market 2.0922 / uniform 2.5338.

Run: uv run python experiments/abm_vs_form_logit.py
"""
from __future__ import annotations

import numpy as np
import polars as pl

from hkjc.eval import harness
from hkjc.eval.walkforward import walk_forward_folds
from hkjc.model.conditional_logit import conditional_logit_fit_predict

from backtest_traffic_st import load_st

# from the confirmed ABM run on this identical pipeline (N=5000, same folds)
ABM = {"traffic": 2.4054, "market": 2.0922, "uniform": 2.5338}


def main():
    df = load_st()
    # standardise features so the logit coefficient is well-scaled
    df = df.with_columns(
        ((pl.col("form") - pl.col("form").mean()) / pl.col("form").std()).alias("z_form"),
        ((pl.col("v_c") - pl.col("v_c").mean()) / pl.col("v_c").std()).alias("z_vc"),
        ((pl.col("draw") - pl.col("draw").mean()) / pl.col("draw").std()).alias("z_draw"),
    )
    dates = df["date"].unique().to_list()
    folds = walk_forward_folds(dates, n_folds=5, embargo=1, min_train=1)

    def skill(feats):
        fp = conditional_logit_fit_predict(feats, race_keys=["race_id"], prob_col="q")
        fm = harness.walk_forward_evaluate(df, folds, fp, date_col="date", prob_col="q",
                                           outcome_col="won", race_col="race_id")
        return harness.aggregate(fm).get("race_win_log_loss", float("nan"))

    rows = [
        ("form-logit (1 factor)", ["z_form"]),
        ("form + draw", ["z_form", "z_draw"]),
        ("form + v_c", ["z_form", "z_vc"]),
        ("form + v_c + draw", ["z_form", "z_vc", "z_draw"]),
    ]
    print(f"{df['race_id'].n_unique()} races, {df.height} runners, {len(folds)} folds\n")
    print(f"{'model':28s} {'race_win_log_loss':>17s} {'skill vs uniform':>17s} {'% of market':>12s}")
    base = ABM["uniform"]
    mk = ABM["market"]
    def line(name, ll):
        share = (base - ll) / (base - mk) if base > mk else float("nan")
        print(f"{name:28s} {ll:>17.4f} {base - ll:>+17.4f} {share:>11.0%}")
    for name, feats in rows:
        line(name, skill(feats))
    print("  " + "-" * 70)
    line("ABM (traffic sim)", ABM["traffic"])
    line("MARKET (ceiling)", ABM["market"])
    line("uniform (floor)", ABM["uniform"])


if __name__ == "__main__":
    main()
