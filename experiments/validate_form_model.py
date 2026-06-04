"""Richer-features test: do rating + recent form beat the efficient win market?

Journal 0011 showed crude features (market+draw+weight) don't beat the closing win
line. This adds **official rating** and **leakage-controlled recent form** (from
horse form pages via generate_historical_data) and re-runs the walk-forward
comparison vs the market baseline. The honest question: is the win market efficient
w.r.t. fundamentals too, or is there a model edge once rating/form are included?

Run: uv run python experiments/validate_form_model.py   (dataset cached)
"""
from __future__ import annotations

from pathlib import Path

import polars as pl

from hkjc.data.feature_store import trailing_form_features
from hkjc.data.processing import generate_historical_data
from hkjc.eval import harness
from hkjc.eval import walkforward as wf
from hkjc.model.baselines import market_baseline_fit_predict, uniform_fit_predict
from hkjc.model.conditional_logit import conditional_logit_fit_predict

CACHE = Path("data_cache/form_dataset.parquet")
RKEYS = ["RaceId"]


def build_dataset(start: str, end: str) -> pl.DataFrame:
    df = generate_historical_data(start, end)            # cleaned per-horse runs
    df = df.with_columns((1.0 / pl.col("WinOdds")).alias("_inv"))
    df = df.with_columns(
        (pl.col("_inv") / pl.col("_inv").sum().over("RaceId")).alias("market_win_prob")
    ).drop("_inv")
    df = df.with_columns([
        pl.col("market_win_prob").log().alias("log_market_prob"),
        (pl.col("Pla") == 1).cast(pl.Int64).alias("won"),
    ])
    df = trailing_form_features(df, group_col="HorseNo", date_col="Date",
                                finish_col="Pla", won_col="won")
    return df


def main():
    if CACHE.exists():
        df = pl.read_parquet(CACHE)
        print(f"Loaded {CACHE}")
    else:
        print("Building form dataset from HKJC horse pages (network-heavy)...")
        df = build_dataset("2026-04-15", "2026-05-31")
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(CACHE)
        print(f"Saved {CACHE}")

    df = df.filter(pl.col("market_win_prob").is_not_null() & pl.col("won").is_not_null())
    dates = sorted(df["Date"].unique().to_list())
    print(f"\nDataset: {df.height} runner-rows, {df['RaceId'].n_unique()} races, "
          f"{len(dates)} meeting-days ({dates[0]} … {dates[-1]})")

    folds = wf.walk_forward_folds(dates, n_folds=4, embargo=1, min_train=max(4, len(dates) // 3))

    models = {
        "uniform": uniform_fit_predict(race_keys=RKEYS),
        "market": market_baseline_fit_predict(race_keys=RKEYS),
        "clogit[mkt]": conditional_logit_fit_predict(["log_market_prob"], race_keys=RKEYS),
        "clogit[mkt,rtg]": conditional_logit_fit_predict(["log_market_prob", "Rtg"], race_keys=RKEYS),
        "clogit[mkt,rtg,form]": conditional_logit_fit_predict(
            ["log_market_prob", "Rtg", "avg_finish_prior3", "win_rate_prior", "days_since_last"],
            race_keys=RKEYS),
    }
    results = {n: harness.aggregate(harness.walk_forward_evaluate(
        df, folds, fp, date_col="Date", race_col="RaceId", outcome_col="won")) for n, fp in models.items()}

    base = results["market"]["race_win_log_loss"]
    print(f"\n{'model':<26}{'race_logloss':>14}{'brier':>10}{'vs_market':>11}")
    print("-" * 61)
    for name, agg in results.items():
        rl = agg.get("race_win_log_loss", float("nan"))
        tag = "baseline" if name == "market" else f"{100*(1-rl/base):+.1f}%"
        print(f"{name:<26}{rl:>14.4f}{agg.get('brier', float('nan')):>10.4f}{tag:>11}")


if __name__ == "__main__":
    main()
