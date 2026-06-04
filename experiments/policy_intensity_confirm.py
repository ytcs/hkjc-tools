"""Firm up the kick-closing headline + fill the INTENSITY knob (kick_overspeed).

Scenario fixed at the place the closing tactic lives: ST 2400m route, field of 6 gun-spenders
(kick_frac=1.0) + 5 moderate pack (0.5) + 1 focal. Tighter sampling than the scan, with a
per-config standard error (across draw permutations) so we can separate signal from noise.

  Part 1 — TIMING (confirm): sweep the focal horse's kick_frac; expect a mid-late kick to beat
           leading from the gun on the route behind a hot pace.
  Part 2 — INTENSITY: fix the focal at its best kick_frac and sweep kick_overspeed (how hard the
           reserve is dumped); expect an interior optimum (too soft = no separation, too hard =
           burns out before the line).

Run: uv run python experiments/policy_intensity_confirm.py
"""
from __future__ import annotations

import numpy as np

from hkjc.model.dynamics import DynamicsRaceModel
from hkjc.track import track_for

SIZE = 3000
FIELD = 12
VC, WP, MASS = 16.5, 18.0, 500.0
DIST = 2400
M_GUN = 6
N_PERM = 8


def focal_win(kf, overspeed, rng):
    others = [1.0] * M_GUN + [0.5] * (FIELD - 1 - M_GUN)
    geom = track_for("ST", DIST, "turf")
    wins = []
    for _ in range(N_PERM):
        draws = (rng.permutation(FIELD) + 1).astype(float)
        kick = np.array([kf] + others)
        model = DynamicsRaceModel.from_kinematic(
            np.full(FIELD, VC), np.full(FIELD, WP), np.full(FIELD, MASS), draws, DIST,
            track=geom, course="A", bend_curvature=0.0037, speed_noise=0.05,
            kick_frac=kick, kick_overspeed=overspeed)
        ft = model.finish_times(SIZE, rng)
        wins.append((np.argmin(ft, axis=1) == 0).mean() * 100)
    a = np.array(wins)
    return a.mean(), a.std(ddof=1) / np.sqrt(len(a))


def main():
    rng = np.random.default_rng(0)
    fair = 100.0 / FIELD
    print(f"ST {DIST}m, {M_GUN} gun-spenders + pack; focal win% ± SE  (fair {fair:.1f}%, "
          f"N={SIZE}, perms={N_PERM})\n")

    print("Part 1 — TIMING (kick_overspeed=0.35):")
    print(f"  {'kick_frac':>10s} {'win% ± SE':>14s}")
    best_kf, best = 0.4, -1
    for kf in (0.30, 0.40, 0.55, 0.70, 1.00):
        mu, se = focal_win(kf, 0.35, rng)
        flag = " ← gun-from-gun" if kf == 1.0 else (" ← late pounce" if kf <= 0.4 else "")
        print(f"  {kf:>10.2f} {mu:>8.1f} ± {se:>3.1f}{flag}")
        if mu > best:
            best, best_kf = mu, kf

    print(f"\nPart 2 — INTENSITY (kick_frac={best_kf:.2f}, the Part-1 best):")
    print(f"  {'overspeed':>10s} {'win% ± SE':>14s}")
    for ov in (0.15, 0.25, 0.35, 0.50, 0.70):
        mu, se = focal_win(best_kf, ov, rng)
        print(f"  {ov:>10.2f} {mu:>8.1f} ± {se:>3.1f}")
    print("\nInterpretation: Part 1 peak below gun = closing beats the duel; Part 2 interior peak")
    print("= an optimal kick HARDNESS (soft = no separation, hard = empties W′ before the wire).")


if __name__ == "__main__":
    main()
