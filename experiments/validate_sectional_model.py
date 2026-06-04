"""Physics-inspired sectional model vs the market (the trajectory-line kill/confirm).

Builds a per-horse run panel from HKJC sectional pages (velocity profiles), derives
field-relative energetics features (finishing speed, early speed, fade = last/best
section), and tests whether **trailing** (leakage-controlled, prior-run) versions of
them beat the closing win market — the one feature class the market plausibly can't
fully price. If this shows nothing, the next lever is video (finer trajectory).

Run: uv run python experiments/validate_sectional_model.py   (dataset cached)
"""
from __future__ import annotations

import datetime as dt
import time
from pathlib import Path

import polars as pl

from hkjc.data.historical import get_race_data
from hkjc.data.sectional import get_sectional_times, sectional_run_features
from hkjc.eval import harness
from hkjc.eval import walkforward as wf
from hkjc.model.baselines import market_baseline_fit_predict, uniform_fit_predict
from hkjc.model.conditional_logit import conditional_logit_fit_predict

CACHE = Path("data_cache/sectional_dataset.parquet")
RKEYS = ["race_id"]
FEATS = ["rel_finish", "rel_early", "fade", "best_section"]


def build(start: str, end: str, delay: float = 0.1) -> pl.DataFrame:
    rows = []
    d = dt.datetime.strptime(start, "%Y-%m-%d")
    end_d = dt.datetime.strptime(end, "%Y-%m-%d")
    while d <= end_d:
        ds_dash, ds_slash = d.strftime("%Y-%m-%d"), d.strftime("%Y/%m/%d")
        if get_sectional_times(ds_dash, 1).height == 0:
            d += dt.timedelta(days=1)
            continue
        venue = None
        for v in ("ST", "HV"):
            try:
                if get_race_data(ds_slash, v, 1).height:
                    venue = v
                    break
            except Exception:
                pass
        if venue is None:
            d += dt.timedelta(days=1)
            continue
        print(f"  {ds_dash} {venue}")
        for race in range(1, 15):
            sec = get_sectional_times(ds_dash, race)
            if sec.height == 0:
                break
            try:
                res = get_race_data(ds_slash, venue, race)
            except Exception:
                continue
            feats = sectional_run_features(sec)
            if feats.height == 0:
                continue
            mkt = res.select(
                pl.col("HorseNo").cast(pl.Int64, strict=False).alias("runner"),
                pl.col("WinOdds").cast(pl.Float64, strict=False).alias("win_odds"))
            run = feats.join(mkt, on="runner", how="left").with_columns([
                pl.lit(venue).alias("venue"),
                (pl.col("finish_order") == 1).cast(pl.Int64).alias("won"),
            ])
            rows.append(run)
            time.sleep(delay)
        d += dt.timedelta(days=1)
    return pl.concat(rows, how="diagonal_relaxed") if rows else pl.DataFrame()


def _prepare(panel: pl.DataFrame) -> pl.DataFrame:
    panel = panel.with_columns(
        pl.concat_str([pl.col("date"), pl.col("venue"), pl.col("race_no").cast(pl.Utf8)],
                      separator="|").alias("race_id"),
        pl.col("date").str.strptime(pl.Date, "%d/%m/%Y", strict=False).alias("date_d"),
    )
    panel = panel.with_columns((1.0 / pl.col("win_odds")).alias("_inv"))
    panel = panel.with_columns(
        (pl.col("_inv") / pl.col("_inv").sum().over("race_id")).alias("market_win_prob")
    ).drop("_inv").with_columns(pl.col("market_win_prob").log().alias("log_market_prob"))
    # trailing (prior-run) means of the physics features, per horse
    panel = panel.sort(["horse_id", "date_d", "race_no"])
    panel = panel.with_columns([
        pl.col(f).shift(1).rolling_mean(window_size=6, min_samples=1).over("horse_id").alias("prior_" + f)
        for f in FEATS
    ])
    return panel


def main():
    if CACHE.exists():
        panel = pl.read_parquet(CACHE)
        print(f"Loaded {CACHE}")
    else:
        print("Building sectional dataset from HKJC (network-heavy)...")
        panel = build("2026-03-01", "2026-05-31")
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        panel.write_parquet(CACHE)
        print(f"Saved {CACHE}")

    panel = _prepare(panel)
    panel = panel.filter(pl.col("market_win_prob").is_not_null() & pl.col("won").is_not_null())
    dates = sorted(panel["date_d"].unique().to_list())
    print(f"\nDataset: {panel.height} runner-rows, {panel['race_id'].n_unique()} races, "
          f"{len(dates)} meeting-days")

    folds = wf.walk_forward_folds(dates, n_folds=4, embargo=1, min_train=max(4, len(dates) // 3))
    models = {
        "uniform": uniform_fit_predict(race_keys=RKEYS),
        "market": market_baseline_fit_predict(race_keys=RKEYS),
        "clogit[mkt]": conditional_logit_fit_predict(["log_market_prob"], race_keys=RKEYS),
        "clogit[mkt,finish,fade]": conditional_logit_fit_predict(
            ["log_market_prob", "prior_rel_finish", "prior_fade"], race_keys=RKEYS),
        "clogit[mkt,all-sectional]": conditional_logit_fit_predict(
            ["log_market_prob"] + ["prior_" + f for f in FEATS], race_keys=RKEYS),
    }
    results = {n: harness.aggregate(harness.walk_forward_evaluate(
        panel, folds, fp, date_col="date_d", race_col="race_id", outcome_col="won")) for n, fp in models.items()}

    base = results["market"]["race_win_log_loss"]
    print(f"\n{'model':<28}{'race_logloss':>14}{'brier':>10}{'vs_market':>11}")
    print("-" * 63)
    for name, agg in results.items():
        rl = agg.get("race_win_log_loss", float("nan"))
        tag = "baseline" if name == "market" else f"{100*(1-rl/base):+.1f}%"
        print(f"{name:<28}{rl:>14.4f}{agg.get('brier', float('nan')):>10.4f}{tag:>11}")


if __name__ == "__main__":
    main()
