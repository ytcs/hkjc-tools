"""Freeze the owner-confirmed R08 rail reconstruction as the pixel-level regression TRIPWIRE.

Runs ``track_sequence`` (the promoted aerial-rail reconstruction incl. the ``thread_rails`` global
identity) on the cached R08 clean pixels and freezes the per-frame rail curves + running labels to
``data_cache/rail_tripwire_R08.npz``. ``tests/test_rail_tripwire.py`` diffs every future run against
this freeze (median curve distance + extent + exact label match), so ANY behavioural drift in the
reconstruction trips a test instead of silently degrading the gold race.

Re-run this script ONLY after a re-verified visual audit (CV_RECON §0/§0b) — the freeze IS the
ground truth; refreshing it without looking re-bases the tripwire onto unaudited output.

    PYTHONHASHSEED=0 uv run --group bayes python studies/cv_rail_freeze_r08.py
"""
from __future__ import annotations

import numpy as np

from hkjc.vision.feeds.aerial_rail import track_sequence

CLEAN = "data_cache/_railclean_R08.npz"
OUT = "data_cache/rail_tripwire_R08.npz"


def main():
    zc = np.load(CLEAN, allow_pickle=True)
    keys, clean = list(zc["keys"]), zc["clean"]
    seq = track_sequence(keys, clean)
    fks = sorted(seq)
    curves = np.empty(len(fks), object)
    running = np.empty(len(fks), object)
    for j, fi in enumerate(fks):
        curves[j] = [np.asarray(c["pts"], np.float32) for c in seq[fi]]
        running[j] = np.array([bool(c["running"]) for c in seq[fi]])
    np.savez_compressed(OUT, keys=np.array(fks), curves=curves, running=running)
    n2 = sum(1 for j in range(len(fks)) if len(curves[j]) == 2)
    print(f"froze {len(fks)} frames ({n2} two-rail) -> {OUT}")


if __name__ == "__main__":
    main()
