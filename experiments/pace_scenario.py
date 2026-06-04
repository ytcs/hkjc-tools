"""Policy layer / game theory I: does an emergent PACE SCENARIO arise from the field?

Racing's central tactical truth: speed is only an advantage if it's UNCONTESTED. One lone
front-runner dictates a soft pace and is hard to catch; pack several together and they
duel, burn each other's W' early, and hand the race to a closer. None of this is coded —
it must EMERGE from the field-coupled sim (each horse's pace_bias deploys W' early/late;
the car-following dynamics make them interact). We hold ability identical and sweep the
NUMBER of front-runners k, tracking the win-prob of (a) the front-runners, (b) a lone
designated closer, (c) the neutral pack. Draws are randomised across roles to decorrelate.

Run: uv run --group bayes python experiments/pace_scenario.py
"""
from __future__ import annotations

import numpy as np

from hkjc.model.traffic_jax import win_probs
from hkjc.track import track_for

N = 8000
FIELD = 12
VC, WP = 16.5, 18.0
PACE_GAIN = 0.08
BIAS = 1.3
N_PERM = 12


def scenario(geom, dist, k, rng):
    """k front-runners (+BIAS), 1 closer (−BIAS), rest neutral. Returns mean win% by role."""
    roles = np.array([+BIAS] * k + [-BIAS] + [0.0] * (FIELD - k - 1))
    front, closer, neutral = [], [], []
    draws = np.arange(1, FIELD + 1).astype(float)
    for p in range(N_PERM):
        perm = rng.permutation(FIELD)
        pb = roles[perm]                                  # role→random draw slot
        wp = win_probs(np.full(FIELD, VC), np.full(FIELD, WP), draws, dist,
                       track=geom, course="A", size=N, bend_curvature=0.003,
                       speed_noise=0.12, pace_bias=pb, pace_gain=PACE_GAIN, seed=300 + p) * 100
        front.append(wp[pb > 0].mean() if k > 0 else np.nan)
        closer.append(wp[pb < 0][0])
        neutral.append(wp[pb == 0].mean() if (FIELD - k - 1) > 0 else np.nan)
    return np.nanmean(front), np.nanmean(closer), np.nanmean(neutral)


def main():
    rng = np.random.default_rng(0)
    fair = 100.0 / FIELD
    for dist in (1200, 1600):
        geom = track_for("ST", dist, "turf")
        print(f"\nST {dist}m  (fair share = {fair:.1f}%; field {FIELD}, pace_gain {PACE_GAIN}, bias ±{BIAS})")
        print(f"{'#front':>7s} {'front win%':>11s} {'closer win%':>12s} {'neutral win%':>13s}")
        front_by_k = {}
        for k in (1, 2, 3, 4, 6, 8):
            fr, cl, ne = scenario(geom, dist, k, rng)
            front_by_k[k] = fr
            print(f"{k:>7d} {fr:>11.2f} {cl:>12.2f} {ne:>13.2f}")
        lone = front_by_k[1]; crowd = front_by_k[8]
        print(f"  lone-speed premium: front-runner win% {lone:.1f} (k=1) → {crowd:.1f} (k=8)  "
              f"Δ={lone-crowd:+.1f}pp  ({'cannibalise ✓' if lone > crowd else 'no duel ✗'})")


if __name__ == "__main__":
    main()
