"""Measure each run's POLICY + observed W'-depletion from the trajectory dataset (roadmap step 2).

The ABM currently grounds each run on two noisy scalars (v_c, W') and an OPEN-LOOP policy
(pace_gain=0). The full per-section trajectory lets us read the policy the horse ACTUALLY ran
and the energy it ACTUALLY spent — the empirical analogues of the EOM control knobs
(model/dynamics.py): `kick_frac` (when W' is deployed), running style (where it sits), and the
fade (W' running out). These per-run reads, averaged leak-free over a horse's PRIOR runs, become
stable per-horse policy traits to drive the generative sim — far sharper than the 2-scalar fit.

Per-run measures (one row per race x runner), all field-relative or self-normalised so they
compare across distance/going/field size:

  EFFORT / KICK TIMING
  - v_c_run        settling critical velocity (slower of the last two sections; matches energetics)
  - effort_dtg     distance-to-go CENTROID of velocity-excess-over-v_c, in [0,1]. This is where the
                   reserve was spent: ~1 => spent early (gun/front-runner), ~0 => spent late
                   (sit-and-pounce). The direct empirical reading of the EOM `kick_frac`.
  - kick_frac_obs  largest distance-to-go fraction at which the horse is still above v_c and stays
                   above to the line (the EOM kick_frac literal reading); NaN if never sustained.

  RUNNING STYLE / POSITION
  - early_pos_norm running position at the first call / field size, in (0,1]; low = up front.
  - late_pos_norm  running position at the final call / field size.
  - pos_gain       early_pos_norm - late_pos_norm (>0 = closed ground from behind).

  OBSERVED W' DEPLETION (the fade)
  - fade           v_peak/v_end - 1 (>=0; velocity collapse off the peak).
  - pos_drop_last  position(final) - position(penultimate) (>0 = lost places in the last section).
  - depleted       fade > FADE_THR AND pos_drop_last > 0 — the co-signed "tank empty" event.

Then leak-free TRAILING per-horse means (prior runs only, shifted by date) of the policy traits.

Run: uv run python experiments/measure_policy.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

CACHE = Path("data_cache")
TRAJ = CACHE / "trajectory_dataset_full.parquet"
OUT = CACHE / "run_policy_dataset.parquet"
FADE_THR = 0.04   # >4% velocity collapse off peak counts as a fade


def measure_run(g: pl.DataFrame) -> dict | None:
    """Per-run policy + depletion measures from one runner's per-section frame (sorted by section)."""
    n = g.height
    if n < 2:
        return None
    v = g["velocity"].to_numpy().astype(float)
    sd = g["section_dist"].to_numpy().astype(float)
    pos = g["position"].to_numpy()
    dist_total = float(sd.sum())

    # settling critical velocity (slower of last two sections) — pace-robust, matches energetics.
    v_c = float(min(v[-2:]))

    # distance-to-go at each section MIDPOINT, as a fraction of the race (1 at start, 0 at line).
    cum_end = np.cumsum(sd)              # distance covered by the END of each section
    cum_mid = cum_end - sd / 2.0
    dtg = (dist_total - cum_mid) / dist_total              # in (0,1)

    # effort = velocity excess over v_c, distance-weighted (the W' actually spent each section).
    # Undefined (-> null, skipped downstream) for an even-paced run that never exceeds v_c: there
    # is genuinely no anaerobic-deployment LOCATION to report, not a value to fudge.
    excess = np.clip(v - v_c, 0.0, None) * sd
    tot = float(excess.sum())
    effort_dtg = float((excess * dtg).sum() / tot) if tot > 1e-9 else None

    # literal EOM kick_frac: the earliest (largest dtg) section from which the horse stays > v_c
    # all the way to the line (a sustained deployment to the finish).
    above = v > v_c
    kick_frac_obs = None
    for i in range(n):
        if above[i] and above[i:].all():
            kick_frac_obs = float(dtg[i] + sd[i] / 2.0 / dist_total)   # dtg at the START of that section
            break

    field = float(np.nanmax(pos)) if pos is not None else float("nan")
    early_pos_norm = float(pos[0] / field) if field and field > 0 else float("nan")
    late_pos_norm = float(pos[-1] / field) if field and field > 0 else float("nan")
    pos_gain = early_pos_norm - late_pos_norm

    v_peak, v_end = float(v.max()), float(v[-1])
    fade = v_peak / v_end - 1.0 if v_end > 0 else float("nan")
    pos_drop_last = float(pos[-1] - pos[-2])
    depleted = int(fade > FADE_THR and pos_drop_last > 0)

    return {
        "runner": int(g["runner"][0]), "horse_id": g["horse_id"][0],
        "finish_order": int(g["finish_order"][0]), "won": int(g["won"][0]),
        "win_odds": float(g["win_odds"][0]) if g["win_odds"][0] is not None else None,
        "distance": int(g["distance"][0]), "venue": g["venue"][0],
        "v_c_run": v_c, "effort_dtg": effort_dtg, "kick_frac_obs": kick_frac_obs,
        "early_pos_norm": early_pos_norm, "late_pos_norm": late_pos_norm, "pos_gain": pos_gain,
        "fade": fade, "pos_drop_last": pos_drop_last, "depleted": depleted,
    }


def main():
    traj = pl.read_parquet(TRAJ)
    print(f"trajectory rows={traj.height}, races={traj.select(['date','venue','race_no']).unique().height}")

    rows = []
    for (date, venue, rno, runner), g in traj.group_by(["date", "venue", "race_no", "runner"]):
        rec = measure_run(g.sort("section"))
        if rec is not None:
            rec |= {"date": date, "race_no": int(rno)}
            rows.append(rec)
    runs = pl.DataFrame(rows)
    print(f"per-run measures: {runs.height} runs, {runs['horse_id'].n_unique()} horses")

    # leak-free trailing per-horse policy traits: mean over STRICTLY PRIOR runs, null-safe
    # (a null run contributes to neither the sum nor the count, so the mean isn't biased).
    runs = runs.sort(["horse_id", "date", "race_no"])
    for col in ("effort_dtg", "early_pos_norm", "kick_frac_obs", "fade", "depleted"):
        c = pl.col(col).cast(pl.Float64)
        cnt = c.is_not_null().cast(pl.Int64)
        prior_sum = c.fill_null(0).cum_sum().over("horse_id") - c.fill_null(0)
        prior_cnt = cnt.cum_sum().over("horse_id") - cnt
        runs = runs.with_columns(
            pl.when(prior_cnt > 0).then(prior_sum / prior_cnt).alias(f"trail_{col}")
        )
    runs.write_parquet(OUT)
    print(f"wrote {OUT}: {runs.height} rows, {runs.width} cols\n")

    # --- validation: do the measures behave as the mechanism predicts? ---
    won = runs.filter(pl.col("won") == 1)
    lost = runs.filter(pl.col("won") == 0)
    print("WINNERS vs LOSERS (per-run, mean):")
    for c in ("effort_dtg", "pos_gain", "fade", "depleted", "late_pos_norm"):
        wv, lv = won[c].mean(), lost[c].mean()
        print(f"  {c:<14} winners {wv:+.3f}   losers {lv:+.3f}")

    # depletion should be MUCH rarer among winners than faders (the tank-empty signature).
    print(f"\ndepleted rate: winners {won['depleted'].mean():.3f}  losers {lost['depleted'].mean():.3f}")

    # front-runners (low early_pos_norm) should spend effort EARLY (high effort_dtg): mechanism sanity.
    style_corr = runs.select(pl.corr("early_pos_norm", "effort_dtg")).item()
    print(f"corr(early_pos_norm, effort_dtg) = {style_corr:+.3f}  "
          f"(expect <0: front-runners spend early)")

    # trait stability: a horse's run-to-run effort_dtg vs its own trailing mean (is policy a trait?).
    st = runs.filter(pl.col("trail_effort_dtg").is_not_null())
    trait_corr = st.select(pl.corr("effort_dtg", "trail_effort_dtg")).item()
    print(f"corr(effort_dtg, trailing effort_dtg) = {trait_corr:+.3f}  "
          f"(expect >0: running style/pace is a stable trait)")


if __name__ == "__main__":
    main()
