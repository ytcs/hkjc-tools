"""Calibrate the interacting traffic model — empirical & hierarchical Bayes.

Two layers (see :mod:`hkjc.calibration`):
1. Fit the horse-level energetics hierarchy and compute leave-one-out empirical-Bayes
   ``(v_c, W')`` for every run (partial pooling — stabilises the noisy per-run values).
2. Hold those abilities fixed and fit the per-venue ``bend_curvature`` to *which draw
   actually won*, with a venue-level empirical-Bayes prior pooling HV and Sha Tin.

Then a posterior-predictive check: does the calibrated model reproduce the empirical
draw→finish curve? Writes ``data_cache/calibration.json`` and prints the report.

Run: uv run python experiments/calibrate_traffic_model.py   (datasets cached)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl

from hkjc.calibration import (RaceConfig, empirical_draw_curve,
                              fit_energetics_population, hierarchical_bend,
                              horse_energetics)
from hkjc.track import track_for

CACHE = Path("data_cache")
BEND_GRID = np.linspace(0.001, 0.05, 14)
N_SAMPLES = 1500


def _norm_date(col: str) -> pl.Expr:
    return pl.col(col).str.replace_all("/", "-")


def _infer_surface(venue: str, distance: int) -> str:
    # Approximation: the cached datasets carry no surface label. 1650 m at Sha Tin is
    # the all-weather mile (turf races 1600). Everything else treated as turf.
    return "awt" if (venue == "ST" and distance == 1650) else "turf"


def attach_rating(e: pl.DataFrame) -> pl.DataFrame:
    """Join official rating per run: exact (horse,date) → as-of nearest → fallbacks."""
    path = CACHE / "rating_dataset.parquet"
    if not path.exists():
        return e.with_columns(pl.lit(None, dtype=pl.Float64).alias("rtg"))
    r = (pl.read_parquet(path)
         .with_columns(pl.col("date").cast(pl.Utf8).str.replace_all("/", "-")
                       .str.strptime(pl.Date, "%Y-%m-%d", strict=False).alias("d"),
                       pl.col("rtg").cast(pl.Float64))
         .drop_nulls(["d", "rtg"]).sort(["horse_id", "d"]))
    e = e.with_columns(pl.col("date").str.strptime(pl.Date, "%Y-%m-%d", strict=False).alias("d")).sort(["horse_id", "d"])
    e = e.join_asof(r.select("horse_id", "d", "rtg"), on="d", by="horse_id", strategy="nearest")
    # fallbacks: per-horse median rating, then global median
    horse_med = r.group_by("horse_id").agg(pl.col("rtg").median().alias("_hm"))
    gmed = float(r["rtg"].median())
    return (e.join(horse_med, on="horse_id", how="left")
            .with_columns(pl.col("rtg").fill_null(pl.col("_hm")).fill_null(gmed))
            .drop(["d", "_hm"]))


def load_configs() -> dict:
    """Build per-venue RaceConfig lists with LOO-EB abilities + realised winners."""
    e = pl.read_parquet(CACHE / "energetics_dataset.parquet").with_columns(_norm_date("date"))
    w = pl.read_parquet(CACHE / "wpq_dataset.parquet").with_columns(_norm_date("date"))
    e = attach_rating(e)

    pop = fit_energetics_population(e, rating_col="rtg")
    eb = horse_energetics(e, pop, rating_col="rtg", loo=True)

    keys = ["date", "venue", "race_no", "runner"]
    joined = (eb.select(keys + ["v_c_eb", "w_prime_eb", "distance"])
              .join(w.select(keys + ["draw", "finish_pos", "won", "field_size"]), on=keys)
              .drop_nulls(["draw", "v_c_eb", "w_prime_eb", "finish_pos"]))

    venues: dict[str, list[RaceConfig]] = {"HV": [], "ST": []}
    for (date, venue, race_no), g in joined.group_by(["date", "venue", "race_no"]):
        if venue not in venues or g.height < 4:
            continue
        g = g.sort("runner")
        won = g["won"].to_numpy()
        if won.sum() != 1:                       # need exactly one realised winner
            continue
        dist = int(g["distance"][0])
        venues[venue].append(RaceConfig(
            venue=venue, distance=dist, surface=_infer_surface(venue, dist), course="A",
            draws=g["draw"].to_numpy().astype(float),
            v_c=g["v_c_eb"].to_numpy(), w_prime=g["w_prime_eb"].to_numpy(),
            winner_idx=int(np.argmax(won))))
    return venues, pop, w


def model_draw_slope(configs, bend, size=1500, seed=11):
    """Mean-finish-vs-draw slope the model produces at ``bend`` (real fields/abilities)."""
    from hkjc.model.traffic_jax import finish_times
    rows = []
    for c in configs:
        geom = track_for(c.venue, c.distance, c.surface)
        ft = finish_times(c.v_c, c.w_prime, c.draws, c.distance, track=geom,
                          course=c.course, size=size, bend_curvature=bend, seed=seed)
        mean_finish = ft.argsort(1).argsort(1).mean(0) + 1.0
        rows.extend({"draw": int(d), "mf": float(m)} for d, m in zip(c.draws, mean_finish))
    a = pl.DataFrame(rows).group_by("draw").agg(pl.col("mf").mean()).sort("draw")
    return float(np.polyfit(a["draw"].to_numpy(), a["mf"].to_numpy(), 1)[0])


def moment_match_bend(configs, emp_slope, grid):
    """Bend whose model draw-slope matches the empirical slope (simulated moments)."""
    slopes = np.array([model_draw_slope(configs, b) for b in grid])
    order = np.argsort(slopes)                                   # ensure increasing for interp
    return float(np.interp(emp_slope, slopes[order], grid[order])), slopes


def main():
    venues, pop, w = load_configs()
    print("=== Layer 1: energetics hierarchy (empirical Bayes, rating-segmented) ===")
    for name, re in (("v_c", pop.v_c), ("w_prime", pop.w_prime)):
        print(f"  {name:8s}  band means={ {k: round(v,2) for k,v in re.band_mean.items()} }")
        print(f"            rating_slope={re.rating_slope:+.4f} per pt (mean rtg={re.rating_mean:.0f})"
              f"  sigma_within={np.sqrt(re.sigma2):.3f}  tau_between={np.sqrt(re.tau2):.3f}"
              f"  shrinkage@n=3 -> {re.tau2/(re.tau2+re.sigma2/3):.2f}")
    print(f"  races usable: HV={len(venues['HV'])}  ST={len(venues['ST'])}")

    print("\n=== Layer 2: per-venue bend_curvature (hierarchical Bayes, JAX backend) ===")
    posts = hierarchical_bend(venues, BEND_GRID, n_samples=N_SAMPLES, backend="jax")
    out = {"bend_grid": BEND_GRID.tolist(), "venues": {}}
    for v, p in posts.items():
        lo, hi = p.interval(0.9)
        print(f"  {v}:  mean={p.mean:.4f}  MAP={p.map:.4f}  90%=[{lo:.4f},{hi:.4f}]"
              f"  (1/{1/p.mean:.0f})  n_races={len(venues[v])}")
        out["venues"][v] = {"mean": p.mean, "map": p.map, "ci90": [lo, hi],
                            "n_races": len(venues[v]), "post": p.post.tolist()}

    print("\n=== Layer 2b: draw-slope moment match (directly targets the draw curve) ===")
    for v in posts:
        emp = empirical_draw_curve(w.filter(pl.col("venue") == v))
        emp_slope = float(np.polyfit(emp["draw"].to_numpy(), emp["mean_finish"].to_numpy(), 1)[0])
        bend_mm, slopes = moment_match_bend(venues[v], emp_slope, BEND_GRID)
        print(f"  {v}: empirical draw-slope={emp_slope:+.3f}  -> moment-match bend={bend_mm:.4f}"
              f"  (1/{1/max(bend_mm,1e-9):.0f})  [winner-likelihood mean={posts[v].mean:.4f}]")
        out["venues"][v]["empirical_draw_slope"] = emp_slope
        out["venues"][v]["bend_moment_match"] = bend_mm
        out["venues"][v]["model_slopes_on_grid"] = slopes.tolist()

    (CACHE / "calibration.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {CACHE / 'calibration.json'}")


if __name__ == "__main__":
    main()
