"""Policy layer / game theory III: BEST RESPONSE — is closing the optimal counter to a hot pace?

pace_scenario_dynamics showed W′-depletion produces front-runner cannibalisation, but a
naive all-the-way closer fell too far back to profit. The real game-theoretic question:
given a field with m front-runners, what pace policy MAXIMISES one focal horse's win prob,
and how does that optimum move as the field gets hotter?

We hold ability identical, fix (FIELD−1) background horses with m of them front-running,
and sweep the FOCAL horse's pace_bias from strong-closer (−2) to strong-leader (+2) on the
real W′-depletion EOM. b*(m) = argmax focal win%. Game-theoretic signature we're testing:
as m rises (hotter pace), b* should shift NEGATIVE (closing becomes the best response) and
its win% should rise — i.e. a suicidal pace structurally rewards the horse that sits off it.

Run: uv run python experiments/best_response.py
"""
from __future__ import annotations

import numpy as np

from hkjc.model.dynamics import DynamicsRaceModel
from hkjc.track import track_for

SIZE = 1200
FIELD = 12
VC, WP, MASS = 16.5, 18.0, 500.0
PACE_GAIN = 0.10
FRONT = 1.0                                   # background front-runner bias
BVALS = np.array([-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0])
N_PERM = 4


def focal_winprob(geom, dist, m, b, rng):
    others = [FRONT] * m + [0.0] * (FIELD - 1 - m)
    wins = []
    for _ in range(N_PERM):
        draws = (rng.permutation(FIELD) + 1).astype(float)
        pb = np.array([b] + others)                       # focal = index 0
        model = DynamicsRaceModel.from_kinematic(
            np.full(FIELD, VC), np.full(FIELD, WP), np.full(FIELD, MASS), draws, dist,
            track=geom, course="A", bend_curvature=0.0037, speed_noise=0.05,
            pace_bias=pb, pace_gain=PACE_GAIN)
        ft = model.finish_times(SIZE, rng)
        wins.append((np.argmin(ft, axis=1) == 0).mean() * 100)
    return float(np.mean(wins))


def main():
    rng = np.random.default_rng(0)
    fair = 100.0 / FIELD
    for dist in (1200, 1800, 2400):
        geom = track_for("ST", dist, "turf")
        print(f"\nST {dist}m  W′-depletion EOM; focal win% vs its own pace_bias, by field hotness m"
              f"  (fair = {fair:.1f}%)")
        header = "  m\\b " + "".join(f"{b:>7.1f}" for b in BVALS) + "   | best b*  win@b*"
        print(header)
        for m in (0, 3, 6, 9):
            row = [focal_winprob(geom, dist, m, b, rng) for b in BVALS]
            i = int(np.argmax(row))
            star = "".join(f"{v:>7.1f}" for v in row)
            print(f"{m:>4d} {star}   |  {BVALS[i]:>+5.1f}  {row[i]:>6.1f}")
    print("\nRead: peak of each row = best response to m front-runners. Game-theoretic signature =")
    print("the peak sliding LEFT (toward closing) as m grows AND as distance grows (closers get room).")


if __name__ == "__main__":
    main()
