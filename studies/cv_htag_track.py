"""R&D study — HORSE TAGGING: per-frame detections threaded into stable per-horse tags.

Layered like the rail identity rework (journal 0122/0124 principles):
  LOOP 1  per-frame class-segmentation detection (cv_htag_detect.detect_core) — observations
          only, count-aware never count-forced;
  LOOP 2  TRACKLETS by continuity: constant-velocity prediction + Hungarian assignment with
          a motion gate and an appearance VETO. No evidence -> the tracklet ENDS (occlusion /
          out-of-frame); coasting only re-acquires, it never emits a tag;
  LOOP 3  GLOBAL identity: tracklets pooled by silk signature (the pooling axis that
          separates — per-det colour is noisy, the tracklet median is the evidence unit),
          linked across gaps by appearance + motion consistency under the CO-OCCURRENCE
          exclusion (two tracklets overlapping in time are two different horses).

Output: stable integer tags per frame + a full-race overlay render for the visual gate.

Run:  RACE=R08 PYTHONHASHSEED=0 uv run --group bayes python studies/cv_htag_track.py
Out:  data_cache/diag/htag_track_{RACE}/fNNNNN.png + data_cache/_htag_tracks_{RACE}.npz
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, "src")
sys.path.insert(0, "studies")

from cv_htag_detect import detect_core, sig_dist  # noqa: E402

RACE = os.environ.get("RACE", "R08")
VID = {"R08": "aerial_20260603_08", "R03": "aerial_20260603_03",
       "R02A": "aerial_20260520_02"}[RACE]
FIELD = {"R08": 11, "R03": 12, "R02A": 12}[RACE]
OUTDIR = f"data_cache/diag/htag_track_{RACE}"
RAILS = f"data_cache/_htag_rails_{RACE}.npz"

# ---- tracking constants (render kinematics, not per-race tunables)
GATE_PX = 30.0        # max screen motion between sampled frames (camera pans with the field,
#                       so on-screen per-step motion is small; zoom jumps stay under this)
SIG_VETO = 60.0       # LAB ΔE above this = definitely a different silk (veto only — silk
#                       families overlap, so appearance never CONFIRMS a match by itself)
COAST_MAX = 4         # steps a track may coast (predict, no emit) before it ENDS
LINK_GAP_SHORT = 40   # frames: short-gap links are POSITION-gated (camera barely moved)
LINK_GAP_MAX = 400    # frames: long-gap links are SIGNATURE-only (re-entry after exit;
#                       camera moved too far for any position extrapolation to mean much)
LINK_SIG_SHORT = 45.0  # pooled ΔE cap for a position-confirmed short link
LINK_SIG_LONG = 18.0   # pooled ΔE cap for a signature-only long link
LINK_DIST_FAC = 14.0  # px/step: how far the field can drift per step across a short gap
DUP_OVERLAP = 0.8     # tracklets co-temporal >=this (of the shorter) and near = SAME horse
DUP_DIST = 16.0       # px: mean distance for the duplicate-track merge
MERGE_SIG = 30.0      # pooled ΔE for the count-aware final chain merge
STATIC_SPREAD = 14.0  # px: position spread below this over STATIC_SPAN+ frames = HUD
STATIC_SPAN = 300     # frames: no horse holds a 14 px screen box for 12 s (formation,
#                       zoom and reframing drift); the disclaimer text / odometer box does
MIN_TRACKLET = 3      # members below this = unsupported flicker, dropped from LOOP 3
FURN_RESID_MIN = 6.0  # px/step FIELD-relative speed above which a tracklet is FURNITURE.
#                       The camera follows the FIELD: horses ride near the field's median
#                       screen motion (residual ~1 px/step, measured R08), while
#                       ground-fixed furniture (crowd dots, painted glyphs, rail dashes)
#                       sweeps across at the pan rate — the pooled-motion separator for
#                       everything per-frame geometry cannot exclude (R02A f400 crowd).
#                       Field shift is the MEDIAN MATCHED-DET displacement per step
#                       (phase correlation was tried and MEASURED to lock onto the field
#                       itself — the avatars are the highest-contrast content — so it
#                       cannot serve as a ground reference; the det median is direct).


def _hungarian(cost: np.ndarray):
    try:
        from scipy.optimize import linear_sum_assignment

        return linear_sum_assignment(cost)
    except Exception:                          # greedy fallback
        ri, ci = [], []
        c = cost.copy()
        while np.isfinite(c).any() and c.min() < np.inf:
            i, j = np.unravel_index(np.argmin(c), c.shape)
            ri.append(i)
            ci.append(j)
            c[i, :] = np.inf
            c[:, j] = np.inf
        return np.array(ri, int), np.array(ci, int)


class Track:
    """Tracks in the GROUND frame: screen positions sweep with camera pans/rotations
    (tens of px/step at the bends — a screen-coord gate fragments there), so velocity is
    learned net of the measured camera shift and predictions add the accumulated shift."""

    _next = 0

    def __init__(self, frame, det):
        self.id = Track._next
        Track._next += 1
        self.frames = [frame]
        self.dets = [det]
        self.coast = 0
        self.gx = 0.0          # ground velocity (px/step), camera-compensated
        self.gy = 0.0
        self.pcx = 0.0         # camera shift accumulated since the last matched frame
        self.pcy = 0.0
        self.gsteps: list[float] = []   # per-step ground speeds (furniture metric)
        self.misses: list[tuple] = []   # diag: (frame, n_in_gate, n_vetoed) per missed step

    @property
    def x(self):
        return self.dets[-1]["cx"]

    @property
    def y(self):
        return self.dets[-1]["cy"]

    def predict(self, steps=1):
        return (self.x + self.gx * steps + self.pcx,
                self.y + self.gy * steps + self.pcy)

    def add(self, frame, det):
        steps = max(1, (frame - self.frames[-1]) // 2)
        dgx = (det["cx"] - self.x - self.pcx) / steps
        dgy = (det["cy"] - self.y - self.pcy) / steps
        a = 0.5
        self.gx = (1 - a) * self.gx + a * dgx
        self.gy = (1 - a) * self.gy + a * dgy
        self.gsteps.append(float(np.hypot(dgx, dgy)))
        self.frames.append(frame)
        self.dets.append(det)
        self.coast = 0
        self.pcx = 0.0
        self.pcy = 0.0

    def ground_speed(self):
        return float(np.median(self.gsteps)) if self.gsteps else 0.0

    def pooled_sig(self):
        sigs = [d["sig"] for d in self.dets if d["sig"] is not None and len(d["sig"])]
        if not sigs:
            return np.zeros((0, 3), np.float32)
        k = max(len(s) for s in sigs)
        prim = np.median([s[0] for s in sigs], axis=0)
        if k == 1:
            return np.asarray([prim], np.float32)
        sec = np.median([s[1] for s in sigs if len(s) > 1], axis=0)
        return np.asarray([prim, sec], np.float32)


def build_tracklets(frames_dets):
    """LOOP 2: thread per-frame detections into tracklets by FIELD-frame continuity.

    The field's screen shift per step = the median displacement of the dets matched at
    that step (most dets ARE the field); each track predicts as pos + field-relative
    velocity + accumulated field shift. Returns (tracklets, field_shift_by_frame).
    """
    active: list[Track] = []
    done: list[Track] = []
    field_shift = {}
    ema = (0.0, 0.0)
    for frame, dets in frames_dets:
        for tr in active:
            tr.pcx += ema[0]
            tr.pcy += ema[1]
        if not active:
            active = [Track(frame, d) for d in dets]
            field_shift[frame] = ema
            continue
        cost = np.full((len(active), len(dets)), np.inf)
        near = np.zeros(len(active), int)      # diag: candidates inside the motion gate
        veto = np.zeros(len(active), int)      # diag: of those, killed by the sig veto
        for i, tr in enumerate(active):
            px, py = tr.predict(1 + tr.coast)
            for j, d in enumerate(dets):
                dist = float(np.hypot(d["cx"] - px, d["cy"] - py))
                if dist > GATE_PX * (1 + 0.5 * tr.coast):
                    continue
                near[i] += 1
                if sig_dist(tr.dets[-1]["sig"], d["sig"]) > SIG_VETO:
                    veto[i] += 1
                    continue                   # frame-to-frame veto; pooled sig is LOOP 3's
                cost[i, j] = dist
        ri, ci = _hungarian(cost)
        matched_t, matched_d = set(), set()
        disps = []
        for i, j in zip(ri, ci):
            if np.isfinite(cost[i, j]):
                tr = active[i]
                if tr.coast == 0:              # 1-step pair: a clean field-shift sample
                    disps.append((dets[j]["cx"] - tr.x, dets[j]["cy"] - tr.y))
                tr.add(frame, dets[j])
                matched_t.add(i)
                matched_d.add(j)
        if len(disps) >= 4:
            real = (float(np.median([d[0] for d in disps])),
                    float(np.median([d[1] for d in disps])))
            ema = (0.6 * ema[0] + 0.4 * real[0], 0.6 * ema[1] + 0.4 * real[1])
            field_shift[frame] = real
        else:
            field_shift[frame] = ema
        nxt = []
        for i, tr in enumerate(active):
            if i in matched_t:
                nxt.append(tr)
            else:
                tr.coast += 1
                tr.misses.append((frame, int(near[i]), int(veto[i])))
                if tr.coast > COAST_MAX:
                    done.append(tr)
                else:
                    nxt.append(tr)
        for j, d in enumerate(dets):
            if j not in matched_d:
                nxt.append(Track(frame, d))
        active = nxt
    done.extend(active)
    return [t for t in done if len(t.frames) >= MIN_TRACKLET], field_shift


def _is_static(frames, xs, ys):
    """Screen-static = HUD furniture: tiny position spread held over a long span."""
    if not frames or frames[-1] - frames[0] < STATIC_SPAN:
        return False
    sx = np.percentile(xs, 95) - np.percentile(xs, 5)
    sy = np.percentile(ys, 95) - np.percentile(ys, 5)
    return float(np.hypot(sx, sy)) < STATIC_SPREAD


def merge_duplicates(tracklets):
    """Two tracklets co-temporal and co-located are the SAME horse seen as split dets
    (helmet/torso fragments threading in parallel) — fold the shorter into the longer."""
    tracklets = sorted(tracklets, key=lambda t: -len(t.frames))
    kept: list[Track] = []
    for t in tracklets:
        tset = dict(zip(t.frames, t.dets))
        dup = None
        for k in kept:
            common = [f for f in k.frames if f in tset]
            if len(common) < DUP_OVERLAP * min(len(k.frames), len(t.frames)):
                continue
            kk = dict(zip(k.frames, k.dets))
            d = float(np.mean([np.hypot(kk[f]["cx"] - tset[f]["cx"],
                                        kk[f]["cy"] - tset[f]["cy"]) for f in common]))
            if d < DUP_DIST:
                dup = k
                break
        if dup is None:
            kept.append(t)
        # else: drop t — its dets are dup's split fragments
    return kept


def _chain_span(ch):
    return ch[0].frames[0], ch[-1].frames[-1]


def _chain_sig(ch, sigs):
    ws = [(len(t.frames), sigs[t.id]) for t in ch if len(sigs[t.id])]
    if not ws:
        return np.zeros((0, 3), np.float32)
    return max(ws, key=lambda w: w[0])[1]      # the longest member's pooled sig


def link_global(tracklets, cum_cam):
    """LOOP 3: link tracklets into per-horse identities. Two-tier links under the
    co-occurrence exclusion: short gaps are POSITION-gated in the ground frame
    (``cum_cam`` = cumulative camera shift); long gaps (out-of-frame re-entry, FINISH-paint
    dropouts) are SIGNATURE-only — pooled silk evidence, the tracklet-median axis."""
    tracklets = sorted(tracklets, key=lambda t: t.frames[0])
    sigs = {t.id: t.pooled_sig() for t in tracklets}
    chains: list[list] = []        # each chain = one horse (list of tracklets, time-ordered)
    for t in tracklets:
        best, bcost = None, np.inf
        for ch in chains:
            last = ch[-1]
            gap = t.frames[0] - last.frames[-1]
            if gap <= 0 or gap > LINK_GAP_MAX:
                continue                       # co-occurrence exclusion / too far apart
            ds = sig_dist(sigs[t.id], sigs[last.id])
            if gap <= LINK_GAP_SHORT:
                steps = gap / 2.0
                c0 = cum_cam.get(last.frames[-1], (0.0, 0.0))
                c1 = cum_cam.get(t.frames[0], (0.0, 0.0))
                px = last.x + last.gx * steps + (c1[0] - c0[0])
                py = last.y + last.gy * steps + (c1[1] - c0[1])
                dist = float(np.hypot(t.dets[0]["cx"] - px, t.dets[0]["cy"] - py))
                if dist > LINK_DIST_FAC * steps + 40.0 or ds > LINK_SIG_SHORT:
                    continue
                c = ds + 0.3 * dist / max(1.0, steps)
            else:
                if ds > LINK_SIG_LONG:
                    continue                   # long gap: silk evidence only, strict
                c = 100.0 + ds                 # short links always preferred
            if c < bcost:
                best, bcost = ch, c
        if best is None:
            chains.append([t])
        else:
            best.append(t)
    # count-aware final merge: while over the field size, fold the best non-overlapping
    # silk-compatible pair (count CAP, never count-forcing — incompatible chains survive)
    field = FIELD
    while len(chains) > field:
        best, bcost = None, MERGE_SIG
        for a in range(len(chains)):
            for b in range(a + 1, len(chains)):
                s0, e0 = _chain_span(chains[a])
                s1, e1 = _chain_span(chains[b])
                if min(e0, e1) - max(s0, s1) > 4:
                    continue                   # co-temporal: two different horses
                ds = sig_dist(_chain_sig(chains[a], sigs), _chain_sig(chains[b], sigs))
                if ds < bcost:
                    best, bcost = (a, b), ds
        if best is None:
            break
        a, b = best
        chains[a] = sorted(chains[a] + chains[b], key=lambda t: t.frames[0])
        del chains[b]
    chains.sort(key=lambda ch: -sum(len(t.frames) for t in ch))
    return {t.id: tag for tag, ch in enumerate(chains) for t in ch}, chains


def main():
    import av
    import cv2

    from hkjc.vision.aerial_layout import legend_top, render_active

    os.makedirs(OUTDIR, exist_ok=True)
    for p in os.listdir(OUTDIR):
        os.remove(os.path.join(OUTDIR, p))
    z = np.load(RAILS)
    rails = {}
    for k in z.files:
        f, name = k.rsplit("_", 1)
        rails.setdefault(int(f), {})[name] = z[k]

    # ---- pass 1: detect per frame; cached so threading experiments iterate without
    # the decode+detect cost
    dcache = f"data_cache/_htag_dets_{RACE}.npz"
    if os.path.exists(dcache):
        z2 = np.load(dcache, allow_pickle=True)
        frames_dets = list(z2["frames_dets"])
        ty0_by_frame = {int(k): int(v) for k, v in z2["ty0"].item().items()}
        print(f"loaded {dcache}")
    else:
        frames_dets = []
        ty0_by_frame = {}
        for i, fr in enumerate(av.open(f"data_cache/video/{VID}.mp4").decode(video=0)):
            if i % 2:
                continue
            panel = fr.to_ndarray(format="bgr24")[:, fr.width // 2:]
            if not render_active(panel) or i not in rails:
                continue
            ty0, ty1 = 0, legend_top(panel) - 2
            off = np.array([0.0, ty0], np.float32)
            rd = {k2: (v - off if len(v) else v) for k2, v in rails[i].items()}
            dets, _dbg = detect_core(panel[ty0:ty1], rd)
            for d in dets:
                d["cy"] += ty0
            frames_dets.append((i, dets))
            ty0_by_frame[i] = ty0
        np.savez_compressed(dcache,
                            frames_dets=np.array(frames_dets, dtype=object),
                            ty0=np.array(ty0_by_frame, dtype=object))
    print(f"detected on {len(frames_dets)} frames; "
          f"mean dets/frame={np.mean([len(d) for _f, d in frames_dets]):.1f} field={FIELD}")

    # ---- LOOP 2 + 3
    tracklets, field_shift = build_tracklets(frames_dets)
    # cumulative field shift per sampled frame (field-frame extrapolation in LOOP 3)
    cum_cam = {}
    cx = cy = 0.0
    for f, _d in frames_dets:
        s = field_shift.get(f, (0.0, 0.0))
        cx += s[0]
        cy += s[1]
        cum_cam[f] = (cx, cy)
    furn = [t for t in tracklets
            if (len(t.frames) >= 5 and t.ground_speed() > FURN_RESID_MIN)
            or _is_static(t.frames, [d["cx"] for d in t.dets], [d["cy"] for d in t.dets])]
    print(f"furniture tracklets dropped: {len(furn)} "
          f"(resid {[round(t.ground_speed(), 1) for t in furn[:12]]}...)")
    tracklets = [t for t in tracklets if t not in furn]
    ndup = len(tracklets)
    tracklets = merge_duplicates(tracklets)
    print(f"duplicate tracklets folded: {ndup - len(tracklets)}")
    # ---- death diagnostics: why did horse tracklets end?
    deaths = {"no_candidate": 0, "sig_veto": 0, "competition": 0}
    for t in tracklets:
        if not t.misses:
            continue
        _f, n_in, n_veto = t.misses[-1]
        if n_in == 0:
            deaths["no_candidate"] += 1
        elif n_veto == n_in:
            deaths["sig_veto"] += 1
        else:
            deaths["competition"] += 1
    print("tracklet-death causes (last miss):", deaths)
    tag_of, chains = link_global(tracklets, cum_cam)
    # chain-level HUD test: intermittent screen-static fragments only pool to "static"
    # once LINKED (each fragment alone is too short to fail the tracklet test)
    static_chains = []
    for ch in chains:
        fs = [f for t in ch for f in t.frames]
        xs = [d["cx"] for t in ch for d in t.dets]
        ys = [d["cy"] for t in ch for d in t.dets]
        if _is_static(sorted(fs), xs, ys):
            static_chains.append(ch)
    if static_chains:
        print(f"static (HUD) chains dropped: {len(static_chains)}")
        drop = {t.id for ch in static_chains for t in ch}
        tracklets = [t for t in tracklets if t.id not in drop]
        tag_of, chains = link_global(tracklets, cum_cam)
    print(f"tracklets={len(tracklets)} horses(chains)={len(chains)}")
    lens = sorted((sum(len(t.frames) for t in ch) for ch in chains), reverse=True)
    print("chain lengths:", lens[:20], "..." if len(lens) > 20 else "")

    # ---- emit per-frame tags + overlay render
    bytag: dict[int, list] = {}
    for t in tracklets:
        for f, d in zip(t.frames, t.dets):
            bytag.setdefault(f, []).append((tag_of[t.id], d))
    rng = np.random.RandomState(7)
    colors = {tag: tuple(int(c) for c in rng.randint(60, 255, 3)) for tag in set(tag_of.values())}
    rows = []
    for i, fr in enumerate(av.open(f"data_cache/video/{VID}.mp4").decode(video=0)):
        if i % 2 or i not in ty0_by_frame:
            continue
        panel = fr.to_ndarray(format="bgr24")[:, fr.width // 2:]
        vis = panel.copy()
        for tag, d in bytag.get(i, []):
            x, y = int(d["cx"]), int(d["cy"])
            cv2.circle(vis, (x, y), 12, colors[tag], 2)
            cv2.putText(vis, str(tag), (x + 10, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, colors[tag], 2)
            rows.append((i, tag, d["cx"], d["cy"], d["area"]))
        cv2.putText(vis, f"f{i:05d} tags={len(bytag.get(i, []))} field={FIELD}",
                    (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        cv2.imwrite(f"{OUTDIR}/f{i:05d}.png", vis)
    arr = np.array(rows, np.float32)
    np.savez_compressed(f"data_cache/_htag_tracks_{RACE}.npz", rows=arr)
    print(f"render -> {OUTDIR}/  tracks -> data_cache/_htag_tracks_{RACE}.npz")


if __name__ == "__main__":
    main()
