"""TRIAGE (selection only; visual is the gate): find candidate racing/fixed rail IDENTITY SWAPS.

A swap shows as an ABRUPT JUMP of the GREEN (running) rail between consecutive frames — the rails are
static world structures seen by a smoothly-moving camera, so the true running curve moves smoothly;
a jump of ~the rail separation in one step is an identity event, not motion. Prints the jump stretches
to inspect on the exp_railfit overlays.

    RACE=R08 PYTHONHASHSEED=0 uv run --group bayes python studies/cv_rail_swap_triage.py
"""
from __future__ import annotations

import os

import numpy as np

from hkjc.vision.feeds.aerial_rail import _sample_curve, _signed_curvature, track_sequence

RACE = os.environ.get("RACE", "R08")
CLEAN = f"data_cache/_railclean_{RACE}.npz"


def main():
    zc = np.load(CLEAN, allow_pickle=True)
    keys, clean = list(zc["keys"]), zc["clean"]
    seq = track_sequence(keys, clean)
    fks = sorted(seq)
    W = max(float(np.asarray(c, np.float32)[:, 0].max()) for c in clean if len(np.asarray(c))) + 1

    fp, rad_flag = {}, {}
    for fi in fks:
        r = [c for c in seq[fi] if c.get("running")]
        o = [c for c in seq[fi] if not c.get("running")]
        if r:
            fp[fi] = _sample_curve(r[0]["pts"])
        if r and o:
            kr, _ = _signed_curvature(r[0]["pts"])
            ko, _ = _signed_curvature(o[0]["pts"])
            if np.isfinite(kr) and np.isfinite(ko) and kr != 0 and ko != 0:
                rad_flag[fi] = (1.0 / abs(kr)) < (1.0 / abs(ko))   # running NARROWER than outer

    jumps = []
    prev = None
    for fi in fks:
        if fi in fp and fp[fi] is not None:
            if prev is not None and fks.index(fi) - fks.index(prev) == 1 and fp[prev] is not None:
                d = float(np.median(np.hypot(*(fp[fi] - fp[prev]).T)))
                if d > 0.03 * W:                                    # ~28 px @949 — far above smooth motion
                    jumps.append((prev, fi, d))
            prev = fi

    print(f"{RACE}: {len(fks)} frames, W={W:.0f}")
    print(f"running-rail JUMPS (> 3% W between consecutive kept frames): {len(jumps)}")
    for a, b, d in jumps:
        print(f"  f{a:05d} -> f{b:05d}: median fingerprint jump {d:.0f} px")
    flagged = [fi for fi, v in rad_flag.items() if v]
    print(f"two-rail frames where running is the NARROWER rail: {len(flagged)}/{len(rad_flag)}")
    # contiguous stretches of narrow-flag (sustained = candidate swap zone)
    if flagged:
        runs, s, p = [], flagged[0], flagged[0]
        for fi in flagged[1:]:
            if fks.index(fi) - fks.index(p) <= 2:
                p = fi
            else:
                runs.append((s, p))
                s = p = fi
        runs.append((s, p))
        long_runs = [(a, b) for a, b in runs if fks.index(b) - fks.index(a) >= 5]
        print(f"sustained narrow stretches (>=5 kept frames): {len(long_runs)}")
        for a, b in long_runs:
            print(f"  f{a:05d} .. f{b:05d}  ({fks.index(b) - fks.index(a) + 1} frames)")


if __name__ == "__main__":
    main()
