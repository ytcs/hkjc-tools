"""S7 — the FIELD/DEPLOYMENT causal dimension, step 1: the MEAN-FIELD EQUILIBRIUM PACE.

GOAL (philosophy-pure, causal — NOT a regressed pace feature). 0049 conditioned the EOM on the
OBSERVED field pace (a contemporaneous boundary condition, fine for path RECONSTRUCTION but NOT
predictive pre-race). Here we MODEL the field pace as a self-consistent FIXED POINT of the field's
deployment interaction (THEORY §III.4a, the mean-field game of pace; 0060 the FIELD/DEPLOYMENT
cause), so a race's equilibrium pace is available from PRIOR-race per-horse constants alone.

Each runner has, strictly-prior (leak-free — earlier runs only):
  * ``base_i`` = its own opening-pace tendency [m/s] — the expanding mean of its prior opening-section
    velocity (the velocity it runs to its OWN preferred deployment, uncontested);
  * ``style_i`` ∈ [0,1] = its forwardness — ``1 − trail_early_pos_norm`` (the orientation fix of
    journal 0063/0065: ``trail_early_pos_norm`` is the trailing position RANK, high = closer, so
    forwardness = 1 − it; 1 = front-runner, low = closer).

The realised opening/early field pace EMERGES from the interaction of those preferences: a field of
front-runners bids the pace UP (they contest the lead), a field of closers settles it slow — the
same horse, a different equilibrium. We solve ``field.equilibrium_pace`` for the self-consistent V*.

VALIDATION (leak-free): does V* (from PRIOR styles/paces only) predict the race's REALISED opening
field pace (section-1 field-mean velocity from the trajectory data)? We report correlation + RMSE
vs the NAIVE baseline (the plain field-mean of the prior individual opening paces ``base``), both
venues. The mean-field equilibrium earns the dimension only if it BEATS that baseline.

NO market / win / pool. NO coefficient fitted to any outcome. Run synchronously:
    PYTHONHASHSEED=0 uv run --group bayes python studies/equilibrium_pace.py
"""
from __future__ import annotations

import numpy as np
import polars as pl

from hkjc.model.field import equilibrium_pace

DATA = "data_cache"
MIN_FIELD = 5              # need a real field to talk about an equilibrium pace
MIN_PRIOR_FRAC = 0.5       # need ≥ half the field to have a prior opening pace (leak-free coverage)


def _race_id(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        (pl.col("date") + "_" + pl.col("venue") + "_" + pl.col("race_no").cast(str)).alias("race_id")
    )


def load_panel() -> pl.DataFrame:
    """Per-runner panel with the strictly-prior opening-pace tendency + forwardness style.

    ``base_i`` is the horse's prior expanding-mean OPENING velocity (section-1 velocity of its
    earlier runs only; null on debut). ``style_i`` is ``trail_early_pos_norm`` (the prior
    expanding-mean early-position rank, already strictly-prior in the dataset). ``v_open_i`` is the
    CURRENT race's realised opening velocity (the target the equilibrium must predict in aggregate).
    """
    tj = _race_id(pl.read_parquet(f"{DATA}/trajectory_dataset_full.parquet"))
    op = (
        tj.filter(pl.col("section") == 1)
        .select(["race_id", "horse_id", "date", "venue", "distance", "velocity"])
        .rename({"velocity": "v_open_i"})
        .sort(["horse_id", "date"])
    )
    # strictly-prior expanding mean of the horse's own opening velocity (shift-by-one via cum_count-1)
    op = op.with_columns(
        [
            (pl.col("v_open_i").cum_sum().over("horse_id") - pl.col("v_open_i")).alias("_cs"),
            (pl.col("v_open_i").cum_count().over("horse_id") - 1).alias("_cc"),
        ]
    ).with_columns(
        (pl.col("_cs") / pl.when(pl.col("_cc") > 0).then(pl.col("_cc")).otherwise(None)).alias("base_i")
    )

    rp = _race_id(pl.read_parquet(f"{DATA}/run_policy_dataset.parquet")).select(
        ["race_id", "horse_id", "trail_early_pos_norm"]
    )
    # ORIENTATION FIX (journal 0063/0065): ``trail_early_pos_norm`` is the strictly-prior trailing
    # early-position RANK and is POSITIVELY correlated with the section-1 position (1 = leader), so
    # HIGH = a CLOSER, LOW = a front-runner — the INVERSE of the original docstring. The mean-field
    # game needs FORWARDNESS (1 = front-runner contesting the lead), so invert at the feed.
    panel = op.join(rp, on=["race_id", "horse_id"], how="left").with_columns(
        (1.0 - pl.col("trail_early_pos_norm")).clip(0.0, 1.0).alias("style_i")
    ).drop("trail_early_pos_norm")
    return panel


def per_race(panel: pl.DataFrame) -> pl.DataFrame:
    """Collapse to one row per race with the equilibrium pace, the naive baseline, and the target."""
    rows = []
    for (race_id,), g in panel.group_by(["race_id"], maintain_order=True):
        n = g.height
        base = g["base_i"].to_numpy()
        style = g["style_i"].to_numpy()
        v_open = g["v_open_i"].to_numpy()
        prior_ok = np.isfinite(base)
        if n < MIN_FIELD or prior_ok.sum() < MIN_PRIOR_FRAC * n:
            continue
        # leak-free per-horse inputs: where a horse has no prior opening pace, fall back to the
        # field's prior-mean base (a neutral pre-race estimate); style null (debut) → field-neutral.
        bm = float(np.nanmean(base[prior_ok])) if prior_ok.any() else float("nan")
        if not np.isfinite(bm):
            continue
        b = np.where(prior_ok, base, bm)
        sm = float(np.nanmean(style)) if np.isfinite(style).any() else 0.5
        s = np.clip(np.where(np.isfinite(style), style, sm), 0.0, 1.0)
        # NAIVE baseline: the plain field-mean of the prior individual opening paces.
        naive = float(b.mean())
        # MEAN-FIELD equilibrium pace (front-group best-response self-consistent pace).
        v_eq = equilibrium_pace(b, s)
        realized = float(np.mean(v_open))
        rows.append(
            {
                "race_id": race_id,
                "venue": g["venue"][0],
                "distance": int(g["distance"][0]),
                "field_size": n,
                "naive": naive,
                "v_eq": v_eq,
                "realized": realized,
            }
        )
    return pl.DataFrame(rows)


def _metrics(df: pl.DataFrame, col: str) -> tuple[float, float, float]:
    pred = df[col].to_numpy()
    y = df["realized"].to_numpy()
    corr = float(np.corrcoef(pred, y)[0, 1])
    rmse = float(np.sqrt(np.mean((pred - y) ** 2)))
    bias = float(np.mean(pred - y))
    return corr, rmse, bias


def main() -> None:
    panel = load_panel()
    races = per_race(panel)
    print("=" * 92)
    print("S7 — MEAN-FIELD EQUILIBRIUM PACE  (predict realised opening field pace, leak-free)")
    print("=" * 92)
    print(f"min_field={MIN_FIELD}  min_prior_frac={MIN_PRIOR_FRAC}")
    print(f"total races with leak-free coverage: {races.height}\n")

    for venue in ["ST", "HV"]:
        rv = races.filter(pl.col("venue") == venue)
        if rv.height == 0:
            continue
        nc, nr, nb = _metrics(rv, "naive")
        ec, er, eb = _metrics(rv, "v_eq")
        print(f"--- {venue}  (n={rv.height}) ---  target = realised section-1 field-mean velocity")
        print(f"  NAIVE  (field-mean of prior base) : corr {nc:+.3f}   RMSE {nr:.4f}   bias {nb:+.4f}")
        print(f"  EQUILIBRIUM  (mean-field game)    : corr {ec:+.3f}   RMSE {er:.4f}   bias {eb:+.4f}")
        d_corr = ec - nc
        d_rmse = nr - er
        verdict = "BEATS" if (d_rmse > 0 and d_corr > 0) else (
            "MIXED" if (d_rmse > 0 or d_corr > 0) else "does NOT beat")
        print(f"  Δcorr {d_corr:+.3f}   ΔRMSE {d_rmse:+.4f} (positive = equilibrium better)  → {verdict}\n")


if __name__ == "__main__":
    main()
