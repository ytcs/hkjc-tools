"""Artifact-level equivalence check (speed work must not change results): compare every array in
the per-stage .npz artifacts of two run directories; print max abs diff / mismatch counts."""

import sys
from pathlib import Path

import numpy as np

FILES = [
    "frames_idx.npz",
    "feats.npz",
    "pose.npz",
    "dets.npz",
    "hud.npz",
    "legend.npz",
    "rail.npz",
    "metric.npz",
    "tracks.npz",
    "finish.npz",
    "labels.npz",
    "det_labels.npz",
]


def cmp(a, b):
    out = []
    for f in FILES:
        pa, pb = Path(a) / f, Path(b) / f
        if not (pa.exists() and pb.exists()):
            continue
        za, zb = np.load(pa, allow_pickle=True), np.load(pb, allow_pickle=True)
        for k in sorted(set(za.files) | set(zb.files)):
            if k not in za.files or k not in zb.files:
                out.append(f"{f}:{k} MISSING in one")
                continue
            x, y = za[k], zb[k]
            if x.dtype == object or y.dtype == object:
                same = len(x) == len(y) and all(np.array_equal(u, v) for u, v in zip(x, y))
                out.append(f"{f}:{k} object {'EQUAL' if same else 'DIFF'}")
                continue
            if x.shape != y.shape:
                out.append(f"{f}:{k} shape {x.shape} vs {y.shape}")
                continue
            if np.issubdtype(x.dtype, np.number):
                d = np.nanmax(np.abs(x.astype(float) - y.astype(float))) if x.size else 0.0
                nan_mis = int((np.isnan(x.astype(float)) != np.isnan(y.astype(float))).sum())
                out.append(
                    f"{f}:{k} maxdiff {d:.3g}" + (f" nan-mismatch {nan_mis}" if nan_mis else "")
                )
            else:
                out.append(f"{f}:{k} {'EQUAL' if np.array_equal(x, y) else 'DIFF'}")
    return out


if __name__ == "__main__":
    print("\n".join(cmp(sys.argv[1], sys.argv[2])))
