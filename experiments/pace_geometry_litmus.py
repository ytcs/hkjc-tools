"""Validation litmus: does the front-runner advantage EMERGE from geometry alone?

No data, no fitting — pure simulation. We hold ability identical across a field and use
the sim's pace policy only as a PROBE: spread running-styles across the field (pace_bias
from closer −1.5 to front-runner +1.5) and measure how strongly style maps to winning,
per venue × distance. The slope d(win%)/d(pace_bias) is the "front-running advantage".

Racing wisdom the physics must reproduce WITHOUT being told:
  (1) shorter distance ⇒ bigger front-runner edge (closers need room to convert W').
  (2) HV ⇒ bigger front-runner edge than ST (tight bends + short run-in punish being
      back/wide; ST's long straight lets closers wind up).
We decorrelate style from draw by averaging the (pace_bias, win%) cloud over many random
style↔draw permutations. We also report the pure draw slope (pace_gain=0) as a sanity
check that inside draws emerge stronger at HV too.

Run: uv run --group bayes python experiments/pace_geometry_litmus.py
"""
from __future__ import annotations

import numpy as np

from hkjc.model.traffic_jax import win_probs
from hkjc.track import track_for

N = 6000
PACE_GAIN = 0.06
N_PERM = 10
FIELD = 12
VC, WP = 16.5, 18.0

VENUE_DIST = {
    "HV": [1000, 1200, 1650, 1800],
    "ST": [1000, 1200, 1400, 1600, 2000, 2400],
}


def _slope(x, y):
    return float(np.polyfit(x, y, 1)[0])


def front_runner_slope(geom, dist, rng):
    """win% per unit pace_bias, marginalised over draw via style↔draw permutations."""
    draws = np.arange(1, FIELD + 1).astype(float)
    style = np.linspace(-1.5, 1.5, FIELD)
    xs, ys = [], []
    for k in range(N_PERM):
        pb = style[rng.permutation(FIELD)]                     # decorrelate style from draw
        p = win_probs(np.full(FIELD, VC), np.full(FIELD, WP), draws, dist,
                      track=geom, course="A", size=N, bend_curvature=0.003,
                      speed_noise=0.12, pace_bias=pb, pace_gain=PACE_GAIN, seed=1000 + k)
        xs.append(pb); ys.append(p * 100.0)
    return _slope(np.concatenate(xs), np.concatenate(ys))


def draw_slope(geom, dist):
    """win% per draw step at pace_gain=0 — inside-draw advantage from geometry alone."""
    draws = np.arange(1, FIELD + 1).astype(float)
    p = win_probs(np.full(FIELD, VC), np.full(FIELD, WP), draws, dist,
                  track=geom, course="A", size=N, bend_curvature=0.003,
                  speed_noise=0.12, pace_gain=0.0, seed=7)
    return _slope(draws, p * 100.0)


def main():
    rng = np.random.default_rng(0)
    print(f"front-runner advantage = d(win%)/d(pace_bias); field={FIELD}, N={N}, "
          f"pace_gain={PACE_GAIN}\n")
    print(f"{'venue':>6s} {'dist':>5s} {'bends?':>7s} {'front-runner slope':>19s} {'draw slope':>11s}")
    rows = []
    for venue, dists in VENUE_DIST.items():
        for dist in dists:
            geom = track_for(venue, dist, "turf")
            has_bends = bool(geom.bend_segments)
            fr = front_runner_slope(geom, dist, rng)
            dr = draw_slope(geom, dist)
            rows.append((venue, dist, fr))
            print(f"{venue:>6s} {dist:>5d} {str(has_bends):>7s} {fr:>+19.3f} {dr:>+11.3f}")

    print("\nChecks the physics should pass (no covariate, no fit — pure geometry):")
    # (1) monotone-ish decline with distance, within venue
    for venue in VENUE_DIST:
        sub = [(d, fr) for v, d, fr in rows if v == venue]
        ds = [d for d, _ in sub]; frs = [fr for _, fr in sub]
        c = float(np.corrcoef(ds, frs)[0, 1])
        print(f"  [{venue}] corr(distance, front-runner slope) = {c:+.2f}  "
              f"(expect NEGATIVE: sprints favour speed)")
    # (2) HV vs ST at comparable distances
    for d in (1000, 1200):
        hv = next((fr for v, dd, fr in rows if v == "HV" and dd == d), None)
        st = next((fr for v, dd, fr in rows if v == "ST" and dd == d), None)
        if hv is not None and st is not None:
            print(f"  [{d}m] HV slope {hv:+.3f} vs ST slope {st:+.3f}  "
                  f"({'HV>ST ✓' if hv > st else 'HV≤ST ✗'} expect HV>ST)")


if __name__ == "__main__":
    main()
