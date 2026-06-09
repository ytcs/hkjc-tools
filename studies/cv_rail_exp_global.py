"""GLOBAL track geometry DRIVER (Stage 1): visualise the running rail's IMAGE-curvature profile
kappa(s) vs odometer arclength, to confirm it is clean and captures both bends.

The algorithm lives in the library (``hkjc.vision.feeds.aerial_rail``: ``track_sequence``,
``_signed_curvature``, ``build_kappa``). This script measures + plots kappa(s) and caches it.
    RACE=R08 PYTHONHASHSEED=0 uv run --group bayes python studies/cv_rail_exp_global.py
"""
from __future__ import annotations

import glob
import os
import sys

import numpy as np

sys.path.insert(0, "studies")

RACE = os.environ.get("RACE", "R08")
CLEAN = f"data_cache/_railclean_{RACE}.npz"
_pp = glob.glob(f"data_cache/_railprep_{RACE}_*.npz")
PREP = max(_pp, key=lambda p: int(p.split("_")[-1].split(".")[0]) - int(p.split("_")[-2])) if _pp else None
OUT = "data_cache/diag/exp_global" + ("" if RACE == "R08" else f"_{RACE}")


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from hkjc.vision.feeds.aerial_rail import _signed_curvature, build_kappa, track_sequence
    os.makedirs(OUT, exist_ok=True)

    z = np.load(CLEAN, allow_pickle=True)
    keys, clean = list(z["keys"]), z["clean"]
    zp = np.load(PREP, allow_pickle=True)
    odo_of = {int(k): float(o) for k, o in zip(zp["keys"], zp["odo"])}   # odometer only (NO field)
    seq = track_sequence(keys, clean)

    rows = []
    for fi in keys:
        fi = int(fi)
        v = seq.get(fi, [])
        s = odo_of.get(fi, np.nan)
        run = [c for c in v if c["running"]]
        if not run or not np.isfinite(s):
            continue
        k, span = _signed_curvature(run[0]["pts"])
        if np.isfinite(k):
            rows.append((fi, s, k, span))
    rows = np.array(rows, float)
    _fi_a, s_a, k_a, span_a = rows.T

    s_grid, k_smooth = build_kappa(seq, odo_of, running=True)
    well = span_a > np.percentile(span_a, 40)
    np.savez(f"data_cache/_railkappa_{RACE}.npz", s_grid=s_grid, k_smooth=k_smooth)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
    ax1.scatter(s_a[well], k_a[well], s=8, c="C2", label="well-seen (pins kappa)")
    ax1.scatter(s_a[~well], k_a[~well], s=8, c="C3", alpha=0.5, label="short (to be re-fit)")
    ax1.plot(s_grid, k_smooth, "k-", lw=2, label="kappa(s) global")
    ax1.set_ylabel("signed image curvature 1/px")
    ax1.legend()
    ax1.set_title(f"{RACE} global track geometry kappa(s)")
    ax2.scatter(s_a, span_a, s=8, c="C0")
    ax2.axhline(np.percentile(span_a, 40), color="grey", ls="--")
    ax2.set_ylabel("running-rail span (px)")
    ax2.set_xlabel("odometer arclength s (m)")
    fig.savefig(f"{OUT}/_kappa_profile.png", dpi=110, bbox_inches="tight")
    print(f"{RACE}: {len(rows)} frames; kappa(s) -> data_cache/_railkappa_{RACE}.npz + {OUT}/_kappa_profile.png")
    print(f"well-seen {int(well.sum())}/{len(rows)}; |kappa| peak {np.nanmax(np.abs(k_smooth)):.5f}/px "
          f"(min R ~{1/np.nanmax(np.abs(k_smooth)):.0f}px)")


if __name__ == "__main__":
    main()
