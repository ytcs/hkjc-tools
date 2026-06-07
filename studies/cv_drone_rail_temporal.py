"""DRONE rail TEMPORAL tracker (scratch) — frame-to-frame coherence for 100% 2-rail coverage.

Per-frame detection (cv_drone_rail) loses the inner rail on the dark, occluded far turn (the field
runs ON it). But the rails barely move frame-to-frame (slow pan), and the two rails are concentric.
So a stateful forward pass at FINE stride:
  (1) SELECTION: assign each elongated candidate to the previous frame's inner/outer rail by curve
      distance -> rejects distractors (the right pair, not just any parallel pair). [user steer]
  (2) SYNTHESIS: when only one rail survives, predict the missing one by offsetting it by the
      temporally-tracked track WIDTH (rails concentric -> well-posed).
  (3) HOLD: when nothing is found, carry the previous rails.
Seeded from the first frame that pairs cleanly. Validate VISUALLY (CV_RECON s0); count is trivial
to inflate with hold/synth, so the strip is the truth.
"""
from __future__ import annotations

import importlib.util

import av
import cv2
import numpy as np

_spec = importlib.util.spec_from_file_location("cdr", "studies/cv_drone_rail.py")
cdr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cdr)

VIDEO = "data_cache/diag/drone_scan/drone_R08_half.mp4"   # 960x540 race segment (4x faster)
STRIDE = 5          # 0.2 s — rail moves slowly frame-to-frame
END = 2833
GATE = 90.0         # max curve dist (px @1080) for a candidate to match a tracked rail; *SCALE'd


def curve_dist(ma, mb, n=40):
    """Median nearest-point distance between two fitted rail curves (px)."""
    a = ma["sample"](n)
    b = mb["sample"](n)
    d = np.sqrt(((a[:, None, :] - b[None, :, :]) ** 2).sum(-1))
    return float(np.median(d.min(axis=1)))


def synth_partner(model, width, prev_partner):
    """Build the missing rail by offsetting `model` by `width`; pick the side nearest prev_partner."""
    if model["type"] == "arc":
        center, R = model["center"], model["R"]
        ang0, ang1 = model.get("_ang", (None, None))
        base = model["sample"](40)
        a = np.arctan2(base[:, 1] - center[1], base[:, 0] - center[0])
        cands = []
        for Rp in (R - width, R + width):
            pts = np.c_[center[0] + Rp * np.cos(a), center[1] + Rp * np.sin(a)].astype(np.float32)
            cands.append((Rp, pts))
        best = min(cands, key=lambda rp: float(np.median(np.sqrt(
            ((rp[1][:, None] - prev_partner["sample"](40)[None]) ** 2).sum(-1)).min(1))))
        Rp, pts = best
        return {"type": "arc", "center": center, "R": Rp, "synth": True,
                "sample": lambda nn=40, _p=pts: _p}
    base = model["sample"](40)
    d = base[-1] - base[0]
    d = d / (np.linalg.norm(d) + 1e-6)
    perp = cdr._rot90(d)
    cands = [base + perp * width, base - perp * width]
    pp = prev_partner["sample"](40)
    best = min(cands, key=lambda pts: float(np.median(np.sqrt(
        ((pts[:, None] - pp[None]) ** 2).sum(-1)).min(1))))
    return {"type": "line", "synth": True, "sample": lambda nn=40, _p=best: _p}


def track(out_strip="data_cache/diag/drone_scan/temporal_track.png", strip_range=None,
          pstart=0, pend=END):
    cont = av.open(VIDEO)
    st = cont.streams.video[0]
    state = None        # {"inner":model,"outer":model,"width":float}
    i = -1
    n_ok = n_real2 = n_synth = n_hold = tot = 0
    tiles = []
    for fr in cont.decode(st):
        i += 1
        if i > min(END, pend):
            break
        if i % STRIDE or i < pstart:
            continue
        bgr = fr.to_ndarray(format="bgr24")
        t = i / 25.0
        # per-frame detection is PRIMARY (re-anchors every frame it succeeds, so a single miss
        # cannot cascade into a stale-hold lock). analyze() = white -> tophat fallback + pairing.
        cands, band, res = cdr.analyze(bgr)
        for c in cands:
            if "model" not in c:
                c["model"] = cdr.fit_model(c["xs"], c["ys"])
        kind = None
        gate = GATE * cdr.SCALE
        if res is not None:
            inner, outer = res["inner"]["model"], res["outer"]["model"]
            w = res["sep"] if state is None else 0.8 * state["width"] + 0.2 * res["sep"]
            state = {"inner": inner, "outer": outer, "width": w}
            kind = "real2"
        elif state is not None:
            # per-frame pairing failed (bend/occlusion). FIRST try the GEOMETRY-CONSTRAINED fill:
            # both rails are concentric arcs separated by the tracked width (user steer).
            width_px = state["width"]
            expected_R_px = (159.0 / 22.5) * width_px   # HV "C" bend ~159 m, width 22.5 m
            cf = cdr.concentric_fill(cands, band, width_px, expected_R_px)
            if cf is not None:
                inner, outer = cf["inner"]["model"], cf["outer"]["model"]
                kind = "synth"
            else:
                # match the surviving candidate(s) to the tracked rails (perp/line fallback).
                din = [(curve_dist(c["model"], state["inner"]), c) for c in cands]
                dout = [(curve_dist(c["model"], state["outer"]), c) for c in cands]
                bi = min(din, key=lambda z: z[0]) if din else (1e9, None)
                bo = min(dout, key=lambda z: z[0]) if dout else (1e9, None)
                inner = bi[1]["model"] if bi[0] < gate else None
                outer = bo[1]["model"] if bo[0] < gate else None
                if outer is not None and inner is None:
                    inner = synth_partner(outer, state["width"], state["inner"])
                    kind = "synth"
                elif inner is not None and outer is None:
                    outer = synth_partner(inner, state["width"], state["outer"])
                    kind = "synth"
                elif inner is not None and outer is not None:
                    kind = "real2"
                else:
                    inner, outer = state["inner"], state["outer"]
                    kind = "hold"
            state = {"inner": inner, "outer": outer, "width": state["width"]}
        tot += 1
        if kind is not None:
            n_ok += 1
            n_real2 += kind in ("real2", "seed")
            n_synth += kind == "synth"
            n_hold += kind == "hold"
        if (strip_range and strip_range[0] <= t <= strip_range[1] and state is not None
                and i % 25 == 0):   # 1 tile/sec
            ov = (bgr // 2).copy()
            for c in cands:
                ov[c["ys"], c["xs"]] = (0, 120, 180)
            ip = state["inner"]["sample"](80).astype(np.int32)
            op = state["outer"]["sample"](80).astype(np.int32)
            cv2.polylines(ov, [op], False, (255, 0, 255), 3)
            cv2.polylines(ov, [ip], False, (255, 255, 0), 3)
            cv2.putText(ov, f"t={t:.1f} {kind} w={state['width']:.0f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            tiles.append(cv2.resize(ov, (640, 360)))
    if tiles:
        cols = 4
        rows = [np.hstack(tiles[r:r + cols] + [np.zeros_like(tiles[0])] * (cols - len(tiles[r:r + cols])))
                for r in range(0, len(tiles), cols)]
        cv2.imwrite(out_strip, np.vstack(rows))
    print(f"frames {tot}  OK {n_ok} ({100 * n_ok / tot:.0f}%)  "
          f"real-2 {n_real2}  synth {n_synth}  hold {n_hold}")


if __name__ == "__main__":
    import sys
    if "--bend" in sys.argv:      # fast: seed before the bend, validate through it
        track(strip_range=(48, 82), pstart=int(40 * 25), pend=int(84 * 25))
    else:
        track(strip_range=(56, 80))
