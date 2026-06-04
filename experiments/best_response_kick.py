"""Policy layer / game theory IV: best response under the KICK feedback policy.

The open-loop tilt couldn't make closing pay because a negative-bias horse DAWDLES (runs
below v_c) and falls out of it. The kick policy fixes that: every horse HOLDS at sustainable
v_c (staying in touch — all v_c equal — while banking W′) and then DUMPS the reserve inside
kick_frac·D of home. kick_frac→1 = spend from the gun (front-runner); small kick_frac =
sit and pounce (closer). Now a closer stays connected, so the question is finally fair:

  against a field of m gun-spenders, what kick_frac maximises the focal horse's win prob,
  and does the optimum move to a LATE kick (small kick_frac) on routes / hot fields?

If on a route behind many front-runners the best response is to kick late AND it wins, the
"closer inherits a burned-out speed duel" tactic has EMERGED from the W′ physics — no tactic
coded, just deployment timing on the real reserve dynamics.

Run: uv run python experiments/best_response_kick.py
"""
from __future__ import annotations

import numpy as np

from hkjc.model.dynamics import DynamicsRaceModel
from hkjc.track import track_for

SIZE = 1500
FIELD = 12
VC, WP, MASS = 16.5, 18.0, 500.0
FRONT_KF, PACK_KF = 1.0, 0.5                  # background: gun-spenders + a moderate pack
KICKS = np.array([0.15, 0.25, 0.40, 0.55, 0.70, 0.85, 1.00])
N_PERM = 5


def focal_winprob(geom, dist, m, kf, rng):
    others = [FRONT_KF] * m + [PACK_KF] * (FIELD - 1 - m)
    wins = []
    for _ in range(N_PERM):
        draws = (rng.permutation(FIELD) + 1).astype(float)
        kick = np.array([kf] + others)                    # focal = index 0
        model = DynamicsRaceModel.from_kinematic(
            np.full(FIELD, VC), np.full(FIELD, WP), np.full(FIELD, MASS), draws, dist,
            track=geom, course="A", bend_curvature=0.0037, speed_noise=0.05,
            kick_frac=kick, kick_overspeed=0.35)
        ft = model.finish_times(SIZE, rng)
        wins.append((np.argmin(ft, axis=1) == 0).mean() * 100)
    return float(np.mean(wins))


def main():
    rng = np.random.default_rng(0)
    fair = 100.0 / FIELD
    for dist in (1200, 2400):
        geom = track_for("ST", dist, "turf")
        print(f"\nST {dist}m  KICK policy; focal win% vs its kick_frac (1.0=lead from gun, "
              f"0.15=late pounce), by #gun-spenders m  (fair {fair:.1f}%)")
        print("  m\\kf " + "".join(f"{k:>7.2f}" for k in KICKS) + "   | best kf*  win")
        for m in (0, 4, 8):
            row = [focal_winprob(geom, dist, m, k, rng) for k in KICKS]
            i = int(np.argmax(row))
            print(f"{m:>4d}  " + "".join(f"{v:>7.1f}" for v in row) +
                  f"   |   {KICKS[i]:>4.2f}  {row[i]:>5.1f}")
    print("\nSignature: on the ROUTE behind many gun-spenders, the peak should move to a SMALL")
    print("kick_frac (late pounce) and beat the fading leaders — closing emergent from W′ physics.")


if __name__ == "__main__":
    main()
