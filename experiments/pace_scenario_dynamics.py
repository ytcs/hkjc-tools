"""Policy layer / game theory II: the pace scenario on the REAL W′-depletion EOM.

pace_scenario.py used the kinematic traffic sim (no anaerobic exhaustion) and found NO
real pace game — front-running was just robustly good, closers never benefited. Here we
re-run on DynamicsRaceModel, whose EOM actually depletes a finite W′ reserve and forces a
fade when it empties. Hypothesis: with exhaustion in the loop, a crowd of front-runners
that over-spend early should FADE, and the conserved closer should INHERIT the race — i.e.
closer win% should now RISE with the number of front-runners. That is the emergent,
un-coded pace game the policy layer is supposed to produce.

Run: uv run python experiments/pace_scenario_dynamics.py
"""
from __future__ import annotations

import numpy as np

from hkjc.model.dynamics import DynamicsRaceModel
from hkjc.track import track_for

SIZE = 1500
FIELD = 12
VC, WP, MASS = 16.5, 18.0, 500.0
PACE_GAIN = 0.10
BIAS = 1.0
N_PERM = 6


def scenario(dist, k, rng):
    roles = np.array([+BIAS] * k + [-BIAS] + [0.0] * (FIELD - k - 1))
    geom = track_for("ST", dist, "turf")
    draws = np.arange(1, FIELD + 1).astype(float)
    front, closer, neutral = [], [], []
    for _ in range(N_PERM):
        pb = roles[rng.permutation(FIELD)]
        model = DynamicsRaceModel.from_kinematic(
            np.full(FIELD, VC), np.full(FIELD, WP), np.full(FIELD, MASS), draws, dist,
            track=geom, course="A", bend_curvature=0.0037, speed_noise=0.05,
            pace_bias=pb, pace_gain=PACE_GAIN)
        ft = model.finish_times(SIZE, rng)
        win = (np.argmin(ft, axis=1)[:, None] == np.arange(FIELD)[None, :]).mean(axis=0) * 100
        front.append(win[pb > 0].mean() if k > 0 else np.nan)
        closer.append(win[pb < 0][0])
        neutral.append(win[pb == 0].mean() if (FIELD - k - 1) > 0 else np.nan)
    return np.nanmean(front), np.nanmean(closer), np.nanmean(neutral)


def main():
    rng = np.random.default_rng(0)
    fair = 100.0 / FIELD
    for dist in (1200, 1600):
        print(f"\nST {dist}m  (fair {fair:.1f}%; W′-DEPLETION EOM, pace_gain {PACE_GAIN}, bias ±{BIAS})")
        print(f"{'#front':>7s} {'front win%':>11s} {'closer win%':>12s} {'neutral win%':>13s}")
        cl_by_k = {}
        for k in (1, 2, 4, 6, 8):
            fr, cl, ne = scenario(dist, k, rng)
            cl_by_k[k] = cl
            print(f"{k:>7d} {fr:>11.2f} {cl:>12.2f} {ne:>13.2f}")
        lo, hi = cl_by_k[1], cl_by_k[8]
        print(f"  closer win% vs speed pressure: {lo:.1f} (k=1) → {hi:.1f} (k=8)  Δ={hi-lo:+.1f}pp  "
              f"({'closer inherits the duel ✓' if hi > lo else 'still no pace game ✗'})")


if __name__ == "__main__":
    main()
