"""Build the FULL per-section trajectory dataset for every race (roadmap step 1).

`build_energetics_full` fetches each race's sectionals but immediately COLLAPSES the
velocity profile to two scalars (v_c, W') and throws the trajectory away. The policy /
game-theory layer needs the trajectory itself: per-section **velocity** (how effort was
spent — early/sustained/kick) and per-section **running position** (front/stalk/close,
and the position-drop that co-signs a W'-depletion fade). This build persists that raw,
design-agnostic long frame once, so downstream policy/energy work is offline (no re-fetch).

Source of truth per section:
- `section_time` (-> velocity = section_dist / section_time) from the sectional page.
- `position` from the sectional page (running order at each split).
Section distances follow the verified HKJC convention (first = D-400*(n-1), rest 400m).

Output -> data_cache/trajectory_dataset_full.parquet, one row per (race, runner, section):
  date, venue, race_no, distance, runner, horse_id, finish_order, won, win_odds,
  n_sections, section, section_dist, section_time, velocity, position.

Run: uv run python experiments/build_trajectory_full.py
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import polars as pl

from hkjc.data.sectional import get_sectional_times
from hkjc.model.energetics import section_distances

CACHE = Path("data_cache")
OUT = CACHE / "trajectory_dataset_full.parquet"


def race_trajectory(date_iso: str, venue: str, race_no: int, dist: int,
                    mkt: pl.DataFrame) -> pl.DataFrame | None:
    """Fetch one race's sectionals and return the tidy per-section trajectory frame."""
    try:
        sec = get_sectional_times(date_iso, race_no)
    except Exception:
        return None
    if sec.height == 0:
        return None
    sec = sec.filter(pl.col("section_time").is_not_null() & pl.col("runner").is_not_null())
    if sec.height == 0:
        return None

    # Per-section distance + velocity, keyed off each runner's own section count.
    n_by_runner = sec.group_by("runner").agg(pl.col("section").max().alias("n_sections"))
    sec = sec.join(n_by_runner, on="runner", how="left")

    rows = []
    for (runner,), g in sec.group_by("runner"):
        g = g.sort("section")
        n = int(g["n_sections"][0])
        dists = section_distances(dist, n)
        for r in g.iter_rows(named=True):
            i = int(r["section"])
            if i < 1 or i > n:
                continue
            sd = float(dists[i - 1])
            st = r["section_time"]
            rows.append({
                "runner": int(runner), "horse_id": r["horse_id"],
                "finish_order": int(r["finish_order"]),
                "n_sections": n, "section": i, "section_dist": sd,
                "section_time": float(st), "velocity": sd / st if st else None,
                "position": r["position"],
            })
    if not rows:
        return None

    traj = (pl.DataFrame(rows).join(mkt, on="runner", how="inner")
            .with_columns(pl.lit(date_iso).alias("date"), pl.lit(venue).alias("venue"),
                          pl.lit(int(race_no)).alias("race_no"), pl.lit(int(dist)).alias("distance")))
    return traj


def main():
    rr = (pl.read_parquet(CACHE / "results_rich.parquet")
          .with_columns(pl.col("date").cast(pl.Utf8),
                        pl.col("HorseNo").cast(pl.Int64, strict=False).alias("runner"),
                        pl.col("WinOdds").cast(pl.Float64, strict=False).alias("win_odds"),
                        (pl.col("Pla").cast(pl.Int64, strict=False) == 1).cast(pl.Int64).alias("won"),
                        pl.col("RaceDistance").cast(pl.Int64, strict=False).alias("dist")))
    races = (rr.select("date", "Venue", "RaceNumber", "dist").unique()
             .drop_nulls(["dist"]).sort(["date", "RaceNumber"]))
    print(f"{races.height} races across {races['date'].n_unique()} meetings to build")

    def task(row):
        date_iso, venue, rno, dist = row["date"], row["Venue"], int(row["RaceNumber"]), int(row["dist"])
        mkt = (rr.filter((pl.col("date") == date_iso) & (pl.col("RaceNumber") == rno) &
                         (pl.col("Venue") == venue))
               .select("runner", "win_odds", "won"))
        return race_trajectory(date_iso, venue, rno, dist, mkt)

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
    print(f"wrote {OUT}: {out.height} section-rows, {nr} races, "
          f"{out['horse_id'].n_unique() if out.height else 0} horses")


if __name__ == "__main__":
    main()
