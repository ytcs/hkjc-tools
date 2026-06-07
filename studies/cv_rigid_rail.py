"""Rigid-body rail backbone — STEP 1: derive the rail centreline from the fixed track shape.

Per the diagnosis (walkthrough video): the rail must be treated as a RIGID BODY whose planar
shape is KNOWN from track geometry, not a free per-frame polynomial. This module derives the
inner-rail centreline ``(X(d), Y(d))`` (metres, track plane) parameterised by arc-length-from-
finish ``d``, plus the LOCAL signed curvature/radius at each ``d``. Downstream steps fit a
per-frame homography track-plane->image by pinning this rigid curve to observed rail pixels +
distance-marker poles (known-arclen landmarks) + the odometer (absolute arclen anchor).

HV "C": circumference 1450 m; bends (m-from-finish) (300,770)=470 m arc & (920,1450)=530 m arc;
straights home 0-300 (300 m) + back 770-920 (150 m). The two bends together must turn 2*pi to
close the loop; with antiparallel straights each turns pi, giving local radii R=arc/pi.

Pure geometry + matplotlib QC. No video here — this is the rigid template the fits snap to.

    PYTHONHASHSEED=0 uv run --group bayes python studies/cv_rigid_rail.py
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hkjc.core.track import TRACKS


@dataclass(frozen=True)
class RailGeometry:
    """The rigid inner-rail centreline of one track configuration, in a fixed track plane.

    ``d`` is arc-length from the finish (m), increasing BACKWARDS along the running direction
    (the HKJC distance-to-go convention is the negative; we keep from-finish to match the
    marker poles). ``X,Y`` are metres in an arbitrary but fixed planar frame (finish at origin,
    running direction +X at the line). ``kappa`` is signed curvature (1/m); ``radius`` its
    reciprocal (inf on straights).
    """

    circ: float
    seg_bounds: tuple            # ((d0,d1,kind,R_signed), ...) kind in {"straight","bend"}
    # dense sampled centreline (for projection + QC)
    d: np.ndarray
    X: np.ndarray
    Y: np.ndarray
    heading: np.ndarray          # tangent angle (rad) at each sample
    kappa: np.ndarray


def _hv_segments(circ: float, bends: tuple) -> list[tuple]:
    """Ordered (d0, d1, kind) segments over one lap from the bend list + circumference."""
    segs, prev = [], 0.0
    for (a, b) in bends:
        if a > prev:
            segs.append((prev, a, "straight"))
        segs.append((a, b, "bend"))
        prev = b
    if prev < circ:
        segs.append((prev, circ, "straight"))
    return segs


def build_rail_geometry(track_key: str = "HV", course: str = "C", *, step: float = 1.0,
                        total_turn: float = 2 * np.pi) -> RailGeometry:
    """Integrate heading along the known segments to get the rigid centreline.

    Each bend turns by ``arc_len / sum(arc_len) * total_turn`` (so the two bends share the full
    2*pi closure in proportion to their lengths — exact closure for a clean oval, near-closure
    for HV's slightly irregular one). Straights keep heading constant. Returns a RailGeometry.
    """
    tg = TRACKS[track_key]
    circ = tg.circumference
    segs = _hv_segments(circ, tg.bend_segments)
    bend_total = sum(b - a for (a, b, k) in segs if k == "bend")
    # signed radius per bend so total turn = total_turn (left turn => +)
    seg_bounds = []
    for (a, b, kind) in segs:
        if kind == "bend":
            turn = (b - a) / bend_total * total_turn
            R = (b - a) / turn
            seg_bounds.append((a, b, kind, R))
        else:
            seg_bounds.append((a, b, kind, np.inf))

    ds = np.arange(0.0, circ + step, step)
    X = np.zeros_like(ds)
    Y = np.zeros_like(ds)
    head = np.zeros_like(ds)
    kappa = np.zeros_like(ds)
    x = y = h = 0.0
    seg_i = 0
    for i, d in enumerate(ds):
        # advance segment pointer
        while seg_i < len(seg_bounds) - 1 and d >= seg_bounds[seg_i][1]:
            seg_i += 1
        a, b, kind, R = seg_bounds[seg_i]
        k = 0.0 if kind == "straight" else 1.0 / R
        X[i], Y[i], head[i], kappa[i] = x, y, h, k
        # step forward
        x += step * np.cos(h)
        y += step * np.sin(h)
        h += step * k
    return RailGeometry(circ, tuple(seg_bounds), ds, X, Y, head, kappa)


def local_radius(geom: RailGeometry, d_from_finish: float) -> float:
    """Signed local radius (m, inf on straight) at arc-length ``d`` (mod circumference)."""
    dm = d_from_finish % geom.circ
    for (a, b, kind, R) in geom.seg_bounds:
        if a <= dm < b:
            return R
    return geom.seg_bounds[-1][3]


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    geom = build_rail_geometry("HV", "C")
    print("=" * 78)
    print("RIGID RAIL GEOMETRY — HV 'C'  (circumference %.0f m)" % geom.circ)
    print("=" * 78)
    for (a, b, kind, R) in geom.seg_bounds:
        rtxt = "straight" if not np.isfinite(R) else f"bend  R={R:6.1f} m  turn={np.degrees((b-a)/R):5.1f} deg"
        print(f"  d {a:6.0f}..{b:6.0f}  ({b-a:5.0f} m)  {rtxt}")
    # closure error
    dx, dy = geom.X[-1] - geom.X[0], geom.Y[-1] - geom.Y[0]
    print(f"\n  loop closure gap: {np.hypot(dx, dy):.1f} m  "
          f"(final heading {np.degrees(geom.heading[-1]):.1f} deg; ideal 360)")
    print(f"  bbox: X [{geom.X.min():.0f},{geom.X.max():.0f}]  Y [{geom.Y.min():.0f},{geom.Y.max():.0f}] m")
    for dd in (50, 200, 400, 600, 850, 1100, 1400):
        print(f"  local radius @ d={dd:5d} m-to-finish: "
              f"{local_radius(geom, dd):8.1f} m" if np.isfinite(local_radius(geom, dd))
              else f"  local radius @ d={dd:5d} m-to-finish:  straight")

    fig, ax = plt.subplots(figsize=(7, 7))
    sc = ax.scatter(geom.X, geom.Y, c=geom.kappa, cmap="coolwarm", s=6)
    ax.scatter([geom.X[0]], [geom.Y[0]], c="k", s=60, marker="s", label="finish (d=0)")
    for dd in (300, 770, 920):
        i = int(dd)
        ax.annotate(f"d={dd}", (geom.X[i], geom.Y[i]), fontsize=8)
    ax.set_aspect("equal")
    ax.legend()
    ax.set_title("HV 'C' rigid rail centreline (colour=curvature)")
    plt.colorbar(sc, label="kappa (1/m)")
    out = "data_cache/figs/cv_rigid_rail_geom.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    print(f"\n[fig] {out}")


if __name__ == "__main__":
    main()
