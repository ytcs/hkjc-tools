"""Extend the energetics dataset from 3 months (161 races) to the FULL 5-season history.

The ABM's statistical power is throttled 20x: energetics_dataset covers only 2026-03..05 (161 ST
races) while results_rich has 3248 races over 5 seasons. The limiter was a hardcoded 3-month build
range, NOT data availability — HKJC serves sectionals across the whole span (probed 2022-09..2026-05).
This rebuilds the panel over every meeting in results_rich. We reuse results_rich for the race-level
fields (distance, odds, result, venue), so only ONE sectional fetch per race is needed, parallelised
across races. Output -> energetics_dataset_full.parquet (kept separate until backtests validate it).

Run: uv run python experiments/build_energetics_full.py
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import polars as pl

from hkjc.data.sectional import get_sectional_times
from hkjc.model.energetics import estimate_run_energetics, section_distances

CACHE = Path("data_cache")
OUT = CACHE / "energetics_dataset_full.parquet"


def race_energetics(date_iso: str, venue: str, race_no: int, dist: int,
                    mkt: pl.DataFrame) -> pl.DataFrame | None:
    """Fetch sectionals for one race, estimate per-horse (v_c, W'), join market rows."""
    try:
        sec = get_sectional_times(date_iso, race_no)
    except Exception:
        return None
    if sec.height == 0:
        return None
    erows = []
    for hid, g in sec.group_by("horse_id"):
        g = g.sort("section")
        if g.height < 2 or hid[0] is None:
            continue
        e = estimate_run_energetics(g["section_time"].to_list(), section_distances(dist, g.height))
        erows.append({"runner": int(g["runner"][0]), "horse_id": hid[0],
                      "v_c": e.v_c, "w_prime": e.w_prime, "finish_order": int(g["finish_order"][0])})
    if not erows:
        return None
    run = (pl.DataFrame(erows).join(mkt, on="runner", how="inner")
           .with_columns(pl.lit(date_iso).alias("date"), pl.lit(venue).alias("venue"),
                         pl.lit(int(race_no)).alias("race_no"), pl.lit(int(dist)).alias("distance")))
    return run


def main():
    rr = (pl.read_parquet(CACHE / "results_rich.parquet")
          .with_columns(pl.col("date").cast(pl.Utf8),
                        pl.col("HorseNo").cast(pl.Int64, strict=False).alias("runner"),
                        pl.col("WinOdds").cast(pl.Float64, strict=False).alias("win_odds"),
                        (pl.col("Pla").cast(pl.Int64, strict=False) == 1).cast(pl.Int64).alias("won"),
                        pl.col("RaceDistance").cast(pl.Int64, strict=False).alias("dist")))
    races = (rr.select("date", "Venue", "RaceNumber", "dist").unique()
             .drop_nulls(["dist"]).sort(["date", "RaceNumber"]))
    print(f"{races.height} races across {races['date'].n_unique()} meetings to rebuild")

    def task(row):
        date_iso, venue, rno, dist = row["date"], row["Venue"], int(row["RaceNumber"]), int(row["dist"])
        mkt = (rr.filter((pl.col("date") == date_iso) & (pl.col("RaceNumber") == rno) &
                         (pl.col("Venue") == venue))
               .select("runner", "win_odds", "won"))
        return race_energetics(date_iso, venue, rno, dist, mkt)

    rows, done = [], 0
    with ThreadPoolExecutor(max_workers=12) as ex:
        futs = [ex.submit(task, r) for r in races.iter_rows(named=True)]
        for fut in as_completed(futs):
            r = fut.result()
            if r is not None:
                rows.append(r)
            done += 1
            if done % 400 == 0:
                print(f"  {done}/{races.height} races, {len(rows)} with sectionals")
    out = pl.concat(rows, how="diagonal_relaxed") if rows else pl.DataFrame()
    out.write_parquet(OUT)
    nr = out.select(["date", "venue", "race_no"]).unique().height if out.height else 0
    print(f"wrote {OUT}: {out.height} runner-rows, {nr} races, {out['horse_id'].n_unique() if out.height else 0} horses")


if __name__ == "__main__":
    main()
