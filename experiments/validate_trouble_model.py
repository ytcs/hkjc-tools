"""Trip-trouble model vs the market — the cheap gate for the video question.

Tests the lateral-dimension thesis with HKJC's own stewards' running comments: a
horse blocked / denied a clear run last time has a result that *understates* its
ability, so the market (anchored on results) may under-rate it next time. We flag
prior-run trouble (leakage-controlled) and test whether it beats the closing win
line. If the experts' trouble notes don't beat the market, a CV pipeline that
re-derives the same signal won't either — so this gates the (expensive) video build.

Run: uv run python experiments/validate_trouble_model.py   (dataset cached)
"""
from __future__ import annotations

import datetime as dt
import time
from pathlib import Path

import polars as pl

from hkjc.data.corunning import get_running_comments
from hkjc.data.historical import get_race_data
from hkjc.eval import harness
from hkjc.eval import walkforward as wf
from hkjc.model.baselines import market_baseline_fit_predict, uniform_fit_predict
from hkjc.model.conditional_logit import conditional_logit_fit_predict

CACHE = Path("data_cache/trouble_dataset.parquet")
RKEYS = ["race_id"]


def build(start: str, end: str, delay: float = 0.1) -> pl.DataFrame:
    rows = []
    d = dt.datetime.strptime(start, "%Y-%m-%d")
    end_d = dt.datetime.strptime(end, "%Y-%m-%d")
    while d <= end_d:
        ds_dash, ds_slash = d.strftime("%Y-%m-%d"), d.strftime("%Y/%m/%d")
        venue = None
        for v in ("ST", "HV"):
            try:
                if get_running_comments(ds_dash, v, 1).height:
                    venue = v
                    break
            except Exception:
                pass
        if venue is None:
            d += dt.timedelta(days=1)
            continue
        print(f"  {ds_dash} {venue}")
        for race in range(1, 15):
            cr = get_running_comments(ds_dash, venue, race)
            if cr.height == 0:
                break
            try:
                res = get_race_data(ds_slash, venue, race)
            except Exception:
                continue
            mkt = res.select(
                pl.col("HorseNo").cast(pl.Int64, strict=False).alias("runner"),
                pl.col("WinOdds").cast(pl.Float64, strict=False).alias("win_odds"),
                (pl.col("Pla").cast(pl.Int64, strict=False) == 1).cast(pl.Int64).alias("won"))
            run = cr.select(["date", "venue", "race_no", "runner", "horse_id", "trouble"]).join(
                mkt, on="runner", how="inner")
            rows.append(run)
            time.sleep(delay)
        d += dt.timedelta(days=1)
    return pl.concat(rows, how="diagonal_relaxed") if rows else pl.DataFrame()


def _prepare(panel: pl.DataFrame) -> pl.DataFrame:
    panel = panel.with_columns(
        pl.concat_str([pl.col("date"), pl.col("venue"), pl.col("race_no").cast(pl.Utf8)],
                      separator="|").alias("race_id"),
        pl.col("date").str.strptime(pl.Date, "%Y/%m/%d", strict=False).alias("date_d"),
    )
    panel = panel.with_columns((1.0 / pl.col("win_odds")).alias("_inv"))
    panel = panel.with_columns(
        (pl.col("_inv") / pl.col("_inv").sum().over("race_id")).alias("market_win_prob")
    ).drop("_inv").with_columns(pl.col("market_win_prob").log().alias("log_market_prob"))
    # leakage-controlled prior-run trouble
    panel = panel.sort(["horse_id", "date_d", "race_no"])
    panel = panel.with_columns([
        pl.col("trouble").shift(1).over("horse_id").alias("trouble_last"),
        pl.col("trouble").shift(1).cum_sum().over("horse_id").alias("_cum"),
        (pl.col("date_d").cum_count().over("horse_id") - 1).alias("prior_runs"),
    ])
    return panel.with_columns(
        pl.when(pl.col("prior_runs") > 0).then(pl.col("_cum") / pl.col("prior_runs"))
        .otherwise(None).alias("trouble_rate_prior")
    ).drop("_cum")


def main():
    if CACHE.exists():
        panel = pl.read_parquet(CACHE)
        print(f"Loaded {CACHE}")
    else:
        print("Building trouble dataset from HKJC running comments (network-heavy)...")
        panel = build("2026-03-01", "2026-05-31")
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        panel.write_parquet(CACHE)
        print(f"Saved {CACHE}")

    panel = _prepare(panel)
    panel = panel.filter(pl.col("market_win_prob").is_not_null() & pl.col("won").is_not_null())
    dates = sorted(panel["date_d"].unique().to_list())
    n_trouble_rows = int(panel.filter(pl.col("trouble_last") == 1).height)
    print(f"\nDataset: {panel.height} runner-rows, {panel['race_id'].n_unique()} races, "
          f"{len(dates)} meeting-days; {n_trouble_rows} rows had trouble last start")

    folds = wf.walk_forward_folds(dates, n_folds=4, embargo=1, min_train=max(4, len(dates) // 3))
    models = {
        "uniform": uniform_fit_predict(race_keys=RKEYS),
        "market": market_baseline_fit_predict(race_keys=RKEYS),
        "clogit[mkt]": conditional_logit_fit_predict(["log_market_prob"], race_keys=RKEYS),
        "clogit[mkt,trouble_last]": conditional_logit_fit_predict(
            ["log_market_prob", "trouble_last"], race_keys=RKEYS),
        "clogit[mkt,trouble_last,rate]": conditional_logit_fit_predict(
            ["log_market_prob", "trouble_last", "trouble_rate_prior"], race_keys=RKEYS),
    }
    results = {n: harness.aggregate(harness.walk_forward_evaluate(
        panel, folds, fp, date_col="date_d", race_col="race_id", outcome_col="won")) for n, fp in models.items()}

    base = results["market"]["race_win_log_loss"]
    print(f"\n{'model':<32}{'race_logloss':>14}{'brier':>10}{'vs_market':>11}")
    print("-" * 67)
    for name, agg in results.items():
        rl = agg.get("race_win_log_loss", float("nan"))
        tag = "baseline" if name == "market" else f"{100*(1-rl/base):+.1f}%"
        print(f"{name:<32}{rl:>14.4f}{agg.get('brier', float('nan')):>10.4f}{tag:>11}")


if __name__ == "__main__":
    main()
