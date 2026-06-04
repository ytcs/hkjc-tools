"""First-principles energetics model vs the market (dynamics, not statistics).

Recovers each horse's critical-velocity / anaerobic reserve (v_c, W') from its
sectional velocity profiles, takes the **trailing** (prior-run, leakage-controlled)
pace-neutral parameters, and for each race *simulates* finishing times via the
generative EnergeticsRaceModel -> physics win probabilities. Tests (a) the physics
probs alone and (b) whether they ADD to the closing market (clogit[market+physics])
through the harness. The question: does mechanistic physiology see anything the
crowd doesn't?

Run: uv run python experiments/validate_energetics_model.py   (dataset cached)
"""
from __future__ import annotations

import datetime as dt
import time
from pathlib import Path

import numpy as np
import polars as pl

from hkjc.data.historical import get_race_data
from hkjc.data.sectional import get_sectional_times
from hkjc.eval import harness
from hkjc.eval import walkforward as wf
from hkjc.model.baselines import market_baseline_fit_predict, uniform_fit_predict
from hkjc.model.conditional_logit import conditional_logit_fit_predict
from hkjc.model.energetics import EnergeticsRaceModel, estimate_run_energetics, section_distances
from hkjc.simulation import RaceSimulator

CACHE = Path("data_cache/energetics_dataset.parquet")
RKEYS = ["race_id"]


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
                dist = int(res["RaceDistance"][0])
            except Exception:
                continue
            erows = []
            for hid, g in sec.group_by("horse_id"):
                g = g.sort("section")
                if g.height < 2 or hid[0] is None:
                    continue
                e = estimate_run_energetics(g["section_time"].to_list(), section_distances(dist, g.height))
                erows.append({"runner": int(g["runner"][0]), "horse_id": hid[0],
                              "v_c": e.v_c, "w_prime": e.w_prime,
                              "finish_order": int(g["finish_order"][0])})
            if not erows:
                continue
            mkt = res.select(
                pl.col("HorseNo").cast(pl.Int64, strict=False).alias("runner"),
                pl.col("WinOdds").cast(pl.Float64, strict=False).alias("win_odds"),
                (pl.col("Pla").cast(pl.Int64, strict=False) == 1).cast(pl.Int64).alias("won"))
            run = pl.DataFrame(erows).join(mkt, on="runner", how="inner").with_columns([
                pl.lit(ds_dash).alias("date"), pl.lit(venue).alias("venue"),
                pl.lit(int(race)).alias("race_no"), pl.lit(dist).alias("distance")])
            rows.append(run)
            time.sleep(delay)
        d += dt.timedelta(days=1)
    return pl.concat(rows, how="diagonal_relaxed") if rows else pl.DataFrame()


def _prepare(panel: pl.DataFrame) -> pl.DataFrame:
    panel = panel.with_columns(
        pl.concat_str([pl.col("date"), pl.col("venue"), pl.col("race_no").cast(pl.Utf8)],
                      separator="|").alias("race_id"),
        pl.col("date").str.strptime(pl.Date, "%Y-%m-%d", strict=False).alias("date_d"),
    )
    panel = panel.with_columns((1.0 / pl.col("win_odds")).alias("_inv"))
    panel = panel.with_columns(
        (pl.col("_inv") / pl.col("_inv").sum().over("race_id")).alias("market_win_prob")
    ).drop("_inv").with_columns(pl.col("market_win_prob").log().alias("log_market_prob"))
    # trailing (prior-run) pace-neutral energetics, per horse
    panel = panel.sort(["horse_id", "date_d", "race_no"])
    return panel.with_columns([
        pl.col("v_c").shift(1).rolling_mean(window_size=6, min_samples=1).over("horse_id").alias("trail_v_c"),
        pl.col("w_prime").shift(1).rolling_mean(window_size=6, min_samples=1).over("horse_id").alias("trail_w_prime"),
    ])


def _add_physics(panel: pl.DataFrame, time_noise: float = 2.5, n_samples: int = 20000) -> pl.DataFrame:
    parts = []
    for _, g in panel.group_by("race_id", maintain_order=True):
        vc = g["trail_v_c"].to_numpy().astype(float)
        wp = g["trail_w_prime"].to_numpy().astype(float)
        dist = float(g["distance"][0])
        med_vc = np.nanmedian(vc) if np.isfinite(vc).any() else 16.0
        med_wp = np.nanmedian(wp) if np.isfinite(wp).any() else 50.0
        vc = np.where(np.isfinite(vc), vc, med_vc)
        wp = np.where(np.isfinite(wp), wp, med_wp)
        sim = RaceSimulator(EnergeticsRaceModel(vc, wp, dist, time_noise), n_samples=n_samples)
        p = np.array([sim.prob(sim.wins(i)) for i in range(len(vc))])
        parts.append(g.with_columns(pl.Series("physics_win_prob", p)))
    out = pl.concat(parts)
    return out.with_columns(pl.col("physics_win_prob").log().alias("log_physics_prob"))


def _precomputed(col: str):
    return lambda train, test: test.with_columns(pl.col(col).alias("p"))


def main():
    if CACHE.exists():
        panel = pl.read_parquet(CACHE)
        print(f"Loaded {CACHE}")
    else:
        print("Building energetics dataset from HKJC sectionals (network-heavy)...")
        panel = build("2026-03-01", "2026-05-31")
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        panel.write_parquet(CACHE)
        print(f"Saved {CACHE}")

    panel = _prepare(panel)
    panel = panel.filter(pl.col("market_win_prob").is_not_null() & pl.col("won").is_not_null())
    panel = _add_physics(panel)
    dates = sorted(panel["date_d"].unique().to_list())
    print(f"\nDataset: {panel.height} runner-rows, {panel['race_id'].n_unique()} races, {len(dates)} meeting-days")

    folds = wf.walk_forward_folds(dates, n_folds=4, embargo=1, min_train=max(4, len(dates) // 3))
    models = {
        "uniform": uniform_fit_predict(race_keys=RKEYS),
        "market": market_baseline_fit_predict(race_keys=RKEYS),
        "physics-only": _precomputed("physics_win_prob"),
        "clogit[mkt]": conditional_logit_fit_predict(["log_market_prob"], race_keys=RKEYS),
        "clogit[mkt,physics]": conditional_logit_fit_predict(
            ["log_market_prob", "log_physics_prob"], race_keys=RKEYS),
    }
    results = {n: harness.aggregate(harness.walk_forward_evaluate(
        panel, folds, fp, date_col="date_d", race_col="race_id", outcome_col="won")) for n, fp in models.items()}

    base = results["market"]["race_win_log_loss"]
    print(f"\n{'model':<26}{'race_logloss':>14}{'brier':>10}{'vs_market':>11}")
    print("-" * 61)
    for name, agg in results.items():
        rl = agg.get("race_win_log_loss", float("nan"))
        tag = "baseline" if name == "market" else f"{100*(1-rl/base):+.1f}%"
        print(f"{name:<26}{rl:>14.4f}{agg.get('brier', float('nan')):>10.4f}{tag:>11}")


if __name__ == "__main__":
    main()
