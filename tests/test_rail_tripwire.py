"""R08 pixel-level rail-reconstruction regression TRIPWIRE (owner request, journal 0122).

Re-runs the promoted aerial-rail reconstruction (``track_sequence`` + ``thread_rails`` identity) on
the cached R08 clean pixels and diffs it against the frozen, visually-audited gold reconstruction
(``studies/cv_rail_freeze_r08.py``). R08 is the gold race (config C, 1227 frames, all phases incl.
the rotating bend); any behavioural drift here — a moved curve, a lost rail, a racing/fixed identity
flip — trips this test instead of silently degrading downstream recon.

Local-only by construction: both inputs live in the git-ignored ``data_cache/`` (the 35 MB pixel
cache cannot ship with the repo), so the test SKIPS where the caches are absent (CI/fresh clones).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

CLEAN = Path("data_cache/_railclean_R08.npz")
FROZEN = Path("data_cache/rail_tripwire_R08.npz")

TOL_MED = 2.0      # px: median curve-to-curve distance — fit noise, not a moved rail
TOL_P95 = 15.0     # px: 95th-pct directed distance — catches an end-trim/extent regression
                   # that a median (robust to <50% loss) would absorb


def _directed(a, b):
    """Nearest-distance from each vertex of polyline a to polyline b."""
    return np.sqrt(((a[:, None, 0] - b[None, :, 0]) ** 2
                    + (a[:, None, 1] - b[None, :, 1]) ** 2).min(1))


@pytest.mark.skipif(not (CLEAN.exists() and FROZEN.exists()),
                    reason="R08 rail caches absent (local data_cache only)")
def test_r08_rail_reconstruction_matches_frozen_gold():
    from hkjc.vision.feeds.aerial_rail import track_sequence

    zf = np.load(FROZEN, allow_pickle=True)
    zc = np.load(CLEAN, allow_pickle=True)
    seq = track_sequence(list(zc["keys"]), zc["clean"])

    fkeys = [int(k) for k in zf["keys"]]
    assert sorted(seq) == fkeys, "kept-frame set changed"

    bad = []
    for j, fi in enumerate(fkeys):
        got = seq[fi]
        want_curves, want_run = zf["curves"][j], zf["running"][j]
        if len(got) != len(want_curves):
            bad.append(f"f{fi:05d}: {len(want_curves)} rails -> {len(got)}")
            continue
        for i, c in enumerate(got):
            if bool(c["running"]) != bool(want_run[i]):
                bad.append(f"f{fi:05d}.{i}: running {bool(want_run[i])} -> {bool(c['running'])}")
            new = np.asarray(c["pts"], float)[::4]
            old = np.asarray(want_curves[i], float)[::4]
            d_no, d_on = _directed(new, old), _directed(old, new)
            med = max(float(np.median(d_no)), float(np.median(d_on)))
            p95 = max(float(np.percentile(d_no, 95)), float(np.percentile(d_on, 95)))
            if med > TOL_MED or p95 > TOL_P95:
                bad.append(f"f{fi:05d}.{i}: med {med:.1f}px / p95 {p95:.1f}px off frozen")
    assert not bad, "rail recon drifted from frozen R08 gold:\n" + "\n".join(bad[:40]) + (
        f"\n... +{len(bad) - 40} more" if len(bad) > 40 else "")
