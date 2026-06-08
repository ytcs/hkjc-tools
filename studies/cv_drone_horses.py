"""DRONE horse detection (step 1-2): LANE-BOUNDED, FIELD-COUNT-consolidated per-frame detection.

Leverages what the rail work built:
  - LANE BOUND: restrict detection to the running surface = a track-width strip on the TRACK side of
    the INNER rail (95% available). Track side = away from the band centroid (infield is the large
    interior). Kills the stands/structure/infield false positives the naive probe showed.
  - FIELD COUNT: k-means the in-lane dark-on-green foreground to N = known runners (R08=11) to
    consolidate fragments (body/shadow/silk split) and split merges -> one point per horse.
Identity is out of scope (anonymous positions; identity comes from finish-cam/cross-feed). Output =
per-horse image positions, later projected to (arclen, lateral) via the rails.
"""
from __future__ import annotations

import importlib.util

import av
import cv2
import numpy as np

_t = importlib.util.spec_from_file_location("tr", "studies/cv_drone_rail_track.py")
tr = importlib.util.module_from_spec(_t)
_t.loader.exec_module(tr)
cdr = tr.cdr

VIDEO = "data_cache/diag/drone_scan/drone_R08_half.mp4"
NFIELD = 11
WIDTH_DEFAULT = 166.0   # median white-rail separation px (half-res); per-frame when both rails seen


RAILS = "data_cache/diag/drone_scan/_rec_R08_s1_rails.npz"


def load_rails(path=RAILS):
    """Filled per-frame rails from the rail tracker (pass2 export, full coverage)."""
    z = np.load(path)
    return {int(i): (z["inner"][k], z["outer"][k]) for k, i in enumerate(z["idx"])}


def lane_mask(shape, inner_pts, outer_pts, pad=6.0):
    """The track ring = polygon between the inner and outer rails (small inward pad to drop the rail
    lines themselves)."""
    outer = outer_pts
    if tr.cdist(inner_pts, outer[::-1]) < tr.cdist(inner_pts, outer):
        outer = outer[::-1]                # align traversal so the polygon doesn't self-cross
    m = np.zeros(shape[:2], np.uint8)
    poly = np.vstack([inner_pts, outer[::-1]]).astype(np.int32)
    cv2.fillPoly(m, [poly], 255)
    return cv2.erode(m, np.ones((int(pad), int(pad)), np.uint8))


def horse_fg(bgr, lane, gh):
    """Dark / non-green / non-white foreground within the lane = horses (+ shadows)."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[..., 0].astype(int), hsv[..., 1], hsv[..., 2]
    green = (np.abs(H - gh) <= 18) & (S > 30) & (V > 35)
    white = (V > 175) & (S < 60)
    fg = ((lane > 0) & ~green & ~white & (V < 170)).astype(np.uint8) * 255
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))


def detect_horses(bgr, inner, outer, n=NFIELD):
    if not np.isfinite(inner).all() or not np.isfinite(outer).all():
        return None, None, None
    en = cdr.enhance(bgr)
    _, gh = cdr.turf_mask(en)
    lane = lane_mask(bgr.shape, inner, outer)
    fg = horse_fg(en, lane, gh)
    ys, xs = np.where(fg > 0)
    if len(xs) < n:
        return None, lane, fg
    pts = np.c_[xs, ys].astype(np.float32)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, lbl, cen = cv2.kmeans(pts, n, None, crit, 4, cv2.KMEANS_PP_CENTERS)
    sizes = np.bincount(lbl.ravel(), minlength=n)
    return cen, lane, fg, sizes


if __name__ == "__main__":
    import subprocess
    railmap = load_rails()
    times = [10, 30, 54, 78, 96, 110]
    cont = av.open(VIDEO)
    st = cont.streams.video[0]
    want = {int(round(t * 25)): t for t in times}
    got = {}
    i = -1
    for fr in cont.decode(st):
        i += 1
        if i in want:
            got[want[i]] = fr.to_ndarray(format="bgr24")
        if i > max(want):
            break
    tiles = []
    for t in times:
        bgr = got[t]
        fi = int(round(t * 25))
        if fi not in railmap:
            continue
        inner, outer = railmap[fi]
        res = detect_horses(bgr, inner, outer)
        ov = bgr.copy()
        cv2.polylines(ov, [inner.astype(np.int32)], False, (255, 255, 0), 1)
        cv2.polylines(ov, [outer.astype(np.int32)], False, (255, 0, 255), 1)
        if res[0] is not None:
            cen, lane, fg, sizes = res
            ov[lane > 0] = (0.6 * ov[lane > 0] + np.array([40, 0, 0])).astype(np.uint8)
            ov[fg > 0] = (0, 140, 255)
            for (x, y), s in zip(cen, sizes):
                cv2.circle(ov, (int(x), int(y)), 7, (0, 0, 255), 2)
            cv2.putText(ov, f"t={t}s N={NFIELD} clusters (lane-bounded)", (8, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            print(f"t={t}: fg_px={int((fg>0).sum())} cluster sizes={sorted(sizes.tolist())}")
        else:
            cv2.putText(ov, f"t={t}s no-inner-rail/insufficient", (8, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            print(f"t={t}: FAIL")
        p = f"data_cache/diag/drone_scan/_horse2_t{t}.png"
        cv2.imwrite(p, cv2.resize(ov, (900, 506)))
        tiles.append(p)
    subprocess.run(["montage", *tiles, "-tile", "2x", "-geometry", "+2+2",
                    "-background", "black", "data_cache/diag/drone_scan/_horse_lane.png"])
