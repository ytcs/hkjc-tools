"""Real-data validation: can a results-trained model beat the market?

Builds a Win/Place/Quinella dataset from HKJC result pages (closing win odds +
draw + weight + finishing position), then compares — on identical walk-forward
folds — uniform, the market baseline, and conditional-logit models that add the
market as one feature. The Benter test: does refitting the market to results
(favourite–longshot correction) plus fundamentals beat the raw market?

Run:  uv run python experiments/validate_market_model.py
The fetched dataset is cached to data_cache/ (gitignored) so re-runs are instant.
"""
from __future__ import annotations

import datetime as dt
import time
from pathlib import Path

import numpy as np
import polars as pl

from hkjc.data.feature_store import results_to_features, results_to_outcomes
from hkjc.data.historical import get_race_data
from hkjc.eval import harness
from hkjc.eval import walkforward as wf
from hkjc.model.baselines import market_baseline_fit_predict, uniform_fit_predict
from hkjc.model.conditional_logit import conditional_logit_fit_predict

CACHE = Path("data_cache/wpq_dataset.parquet")
KEYS = ["date", "venue", "race_no", "runner"]


def build_dataset(start: dt.date, end: dt.date, venues=("ST", "HV"),
                  max_races: int = 12, delay: float = 0.12) -> pl.DataFrame:
    frames, d = [], start
    while d <= end:
        ds = d.strftime("%Y/%m/%d")
        for v in venues:
            try:
                r1 = get_race_data(ds, v, 1)
            except Exception:
                continue
            races = [r1]
            for rn in range(2, max_races + 1):
                try:
                    races.append(get_race_data(ds, v, rn))
                    time.sleep(delay)
                except Exception:
                    break
            for rdf in races:
                try:
                    feats = results_to_features(rdf)
                    outs = results_to_outcomes(rdf).select(
                        KEYS + ["finish_pos", "won", "placed", "field_size"])
                    frames.append(feats.join(outs, on=KEYS, how="inner"))
                except Exception:
                    continue
            print(f"  {ds} {v}: {len(races)} races")
        d += dt.timedelta(days=1)
    return pl.concat(frames, how="diagonal_relaxed") if frames else pl.DataFrame()


def collect_oos(df, folds, fit_predict, prob_col="p") -> pl.DataFrame:
    """Concatenate out-of-sample test predictions across all folds."""
    parts = []
    for f in folds:
        train = df.filter(pl.col("date").is_in(f.train))
        test = df.filter(pl.col("date").is_in(f.test))
        if train.height and test.height:
            parts.append(fit_predict(train, test))
    return pl.concat(parts, how="diagonal_relaxed") if parts else pl.DataFrame()


def flat_stake_roi(preds: pl.DataFrame, edge_threshold: float = 0.0) -> dict:
    """Flat 1-unit bet on every runner whose model edge clears the threshold.

    Edge = p_model * closing_decimal_odds - 1. Payout uses the closing win odds as
    a proxy for the dividend. A positive ROI here means the model's disagreement
    with the market is, on average, correct — the real signal we're after.
    """
    d = preds.filter(pl.col("market_win_odds").is_not_null() & pl.col("p").is_not_null())
    d = d.with_columns((pl.col("p") * pl.col("market_win_odds") - 1.0).alias("edge"))
    bets = d.filter(pl.col("edge") > edge_threshold)
    if bets.height == 0:
        return {"n_bets": 0}
    ret = bets.select(
        (pl.col("won") * pl.col("market_win_odds") - 1.0).alias("ret")
    )["ret"]
    return {
        "n_bets": bets.height,
        "roi": float(ret.mean()),
        "total_return": float(ret.sum()),
        "hit_rate": float(bets["won"].mean()),
    }


def main():
    if CACHE.exists():
        df = pl.read_parquet(CACHE)
        print(f"Loaded cached dataset: {CACHE}")
    else:
        print("Building dataset from HKJC result pages (this hits the network)...")
        df = build_dataset(dt.date(2026, 3, 1), dt.date(2026, 5, 31))
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(CACHE)
        print(f"Saved {CACHE}")

    if df.height == 0:
        print("No data fetched.")
        return

    df = df.with_columns(
        pl.concat_str([pl.col(c).cast(pl.Utf8) for c in ["date", "venue", "race_no"]],
                      separator="|").alias("race_id")
    )
    dates = sorted(df["date"].unique().to_list())
    n_races = df["race_id"].n_unique()
    print(f"\nDataset: {df.height} runner-rows, {n_races} races, {len(dates)} meeting-days "
          f"({dates[0]} … {dates[-1]})")

    folds = wf.walk_forward_folds(dates, n_folds=5, embargo=1, min_train=max(6, len(dates) // 3))
    print(f"Walk-forward: {len(folds)} folds\n")

    models = {
        "uniform": uniform_fit_predict(),
        "market": market_baseline_fit_predict(),
        "clogit[mkt]": conditional_logit_fit_predict(["log_market_prob"]),
        "clogit[mkt,draw,wt]": conditional_logit_fit_predict(["log_market_prob", "draw", "act_wt"]),
        "clogit[mkt,draw,wt,declwt]":
            conditional_logit_fit_predict(["log_market_prob", "draw", "act_wt", "decl_wt"]),
    }

    # compute all aggregates first, then print (market is the baseline)
    results = {
        name: harness.aggregate(harness.walk_forward_evaluate(df, folds, fp, race_col="race_id"))
        for name, fp in models.items()
    }
    base = results["market"]["race_win_log_loss"]
    print(f"{'model':<28}{'race_logloss':>14}{'brier':>10}{'vs_market':>11}")
    print("-" * 63)
    for name, agg in results.items():
        rl = agg.get("race_win_log_loss", float("nan"))
        tag = "baseline" if name == "market" else f"{100*(1-rl/base):+.1f}%"
        print(f"{name:<28}{rl:>14.4f}{agg.get('brier', float('nan')):>10.4f}{tag:>11}")

    # profitability proxy for the richest model
    best = "clogit[mkt,draw,wt,declwt]"
    oos = collect_oos(df, folds, models[best])
    print(f"\nFlat-stake ROI (proxy, payout at closing odds) — {best}:")
    for thr in (0.0, 0.05, 0.10):
        roi = flat_stake_roi(oos, edge_threshold=thr)
        if roi["n_bets"]:
            print(f"  edge>{thr:>4.0%}: {roi['n_bets']:>5} bets  ROI {roi['roi']:+.3f}  "
                  f"hit {roi['hit_rate']:.3f}")
        else:
            print(f"  edge>{thr:>4.0%}: no bets")


if __name__ == "__main__":
    main()
