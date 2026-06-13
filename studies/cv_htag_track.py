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
COAST_MAX = 4         # steps a tracklet may coast (predict, no emit) before it ENDS
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


# ---- global second pass: N-target propagation (no-teleport, known field count) -------------
GP_GATE = 34.0        # px: max screen jump a horse can make between sampled frames (no teleport)
GP_SIG = 55.0         # LAB ΔE appearance veto inside the gate
GP_COAST = 40         # steps a track may dead-reckon through occlusion before it stops claiming
GP_VEL_DECAY = 0.7    # coasting velocity relaxes toward the field shift


def _apply(M, x, y):
    return (M[0, 0] * x + M[0, 1] * y + M[0, 2],
            M[1, 0] * x + M[1, 1] * y + M[1, 2])


class GTrack:
    """One of the N persistent horse tracks in the global pass. The camera common mode
    (pan + ZOOM + rotation) is a per-step SIMILARITY transform M applied to every track's
    position; each horse carries a small field-relative RESIDUAL velocity on top. A pure
    translation cannot model the start/finish zoom (horses move radially), which is why a
    translation-only field shift lost the bunched start."""

    def __init__(self, tag, frame, det):
        self.tag = tag
        self.x, self.y = det["cx"], det["cy"]
        self.rx = self.ry = 0.0            # own residual displacement / step (propagation order)
        self.sig = det["sig"]
        self.sigs = [det["sig"]] if det["sig"] is not None and len(det["sig"]) else []
        self.coast = 0
        self.pos = {frame: (det["cx"], det["cy"], det["area"])}

    def pooled_sig(self):
        if not self.sigs:
            return np.zeros((0, 3), np.float32)
        k = max(len(s) for s in self.sigs)
        prim = np.median([s[0] for s in self.sigs], axis=0)
        if k == 1:
            return np.asarray([prim], np.float32)
        sec = np.median([s[1] for s in self.sigs if len(s) > 1], axis=0)
        return np.asarray([prim, sec], np.float32)

    def reset(self, frame):
        self.x, self.y, _a = self.pos[frame]
        self.rx = self.ry = 0.0
        self.coast = 0

    def predict(self, M, step):
        bx, by = _apply(M, self.x, self.y)
        return bx + step * self.rx, by + step * self.ry

    def update(self, frame, det, M, step):
        bx, by = _apply(M, self.x, self.y)
        self.rx = 0.5 * self.rx + 0.5 * (det["cx"] - bx) / step
        self.ry = 0.5 * self.ry + 0.5 * (det["cy"] - by) / step
        self.x, self.y = det["cx"], det["cy"]
        if det["sig"] is not None and len(det["sig"]):
            self.sig = det["sig"]
            self.sigs.append(det["sig"])
        self.coast = 0
        self.pos[frame] = (det["cx"], det["cy"], det["area"])

    def coast_step(self, M, step, freeze=False):
        self.coast += 1
        self.x, self.y = _apply(M, self.x, self.y)
        self.x += step * self.rx
        self.y += step * self.ry
        if not freeze:                          # a detection collapse must not decay the
            self.rx *= GP_VEL_DECAY             # residual to zero — keep the last good
            self.ry *= GP_VEL_DECAY             # velocity so the horse is re-found on exit


def _anchor_frame(frames_dets, N):
    """Pick a seed frame in the stable middle: exactly N dets, maximally separated."""
    lo, hi = int(0.30 * len(frames_dets)), int(0.70 * len(frames_dets))
    best, bscore = None, -1.0
    for k in range(lo, hi):
        frame, dets = frames_dets[k]
        if len(dets) != N:
            continue
        p = np.array([[d["cx"], d["cy"]] for d in dets])
        dm = np.sqrt(((p[:, None] - p[None]) ** 2).sum(-1))
        np.fill_diagonal(dm, 1e9)
        score = float(dm.min())                # widest-separated config = least ambiguous
        if score > bscore:
            best, bscore = k, score
    if best is None:                           # no exact-N frame: take closest-to-N
        best = min(range(lo, hi), key=lambda k: abs(len(frames_dets[k][1]) - N))
    return best


GP_M_MIN = 6          # min matched pairs to TRUST a fresh camera-transform estimate
GP_COLLAPSE = 0.5     # matched < this*N = a detection COLLAPSE: ride the carried transform


def _similarity(src, dst):
    """Per-step camera transform (pan+zoom+rotation) from matched old->new positions."""
    import cv2

    if len(src) < 3:
        return None
    M, _inl = cv2.estimateAffinePartial2D(np.asarray(src, np.float32),
                                          np.asarray(dst, np.float32),
                                          method=cv2.RANSAC, ransacReprojThreshold=6.0)
    return M


def _propagate(tracks, seq):
    """March the N tracks through ``seq`` (ordered frame,dets) in propagation order.

    The camera common mode is a per-step SIMILARITY transform M (pan+zoom+rotation). It is
    only UPDATED when >= GP_M_MIN tracks match (a sparse estimate is noise); otherwise the
    last good M is carried. Through a detection COLLAPSE (the zoomed-out bend) tracks ride the
    carried M without corrupting their residual velocity, so they re-acquire on the far side
    instead of drifting out — the no-teleport prior crossing a bad-pixels gap."""
    n = len(tracks)
    prev_f = seq[0][0]
    M = np.array([[1.0, 0, 0], [0, 1.0, 0]])    # last GOOD camera transform, prev->cur
    for frame, dets in seq[1:]:
        step = max(1, abs(frame - prev_f) // 2)
        prev_f = frame
        cost = np.full((len(tracks), len(dets)), np.inf)
        for i, t in enumerate(tracks):
            px, py = t.predict(M, step)
            gate = GP_GATE * step * (1 + 0.5 * min(t.coast, 12))
            for j, d in enumerate(dets):
                dist = float(np.hypot(d["cx"] - px, d["cy"] - py))
                if dist > gate or sig_dist(t.sig, d["sig"]) > GP_SIG:
                    continue
                cost[i, j] = dist
        ri, ci = _hungarian(cost)
        pairs = [(i, j) for i, j in zip(ri, ci) if np.isfinite(cost[i, j])]
        if len(pairs) >= GP_M_MIN:              # trust + carry only a well-supported estimate
            est = _similarity([(tracks[i].x, tracks[i].y) for i, _j in pairs],
                              [(dets[j]["cx"], dets[j]["cy"]) for _i, j in pairs])
            if est is not None:
                M = est
        collapse = len(pairs) < GP_COLLAPSE * n
        matched_t = {i for i, _j in pairs}
        for i, j in pairs:
            tracks[i].update(frame, dets[j], M, step)
        for i, t in enumerate(tracks):
            if i not in matched_t:
                t.coast_step(M, step, freeze=collapse)


def _stable_segments(frames_dets, N, tol=1, min_run=15):
    """Maximal index runs where detection count is within ``tol`` of N for >= min_run frames.
    These are the trustworthy zones; the gaps between them (e.g. the zoomed-out bend, the
    finish-line paint) are where per-frame detection is unreliable and must be BRIDGED, not
    coasted across. Returns list of (lo, hi) index ranges."""
    ok = [abs(len(d) - N) <= tol for _f, d in frames_dets]
    segs, i, n = [], 0, len(ok)
    while i < n:
        if not ok[i]:
            i += 1
            continue
        j = i
        gap = 0
        while j < n and gap <= 4:              # tolerate <=4 bad frames inside a segment
            if ok[j]:
                gap = 0
            else:
                gap += 1
            j += 1
        hi = j - gap
        if hi - i >= min_run:
            segs.append((i, hi))
        i = j
    return segs


def _seg_tracks(frames_dets, seg, N):
    """N tracks for one stable segment: seed at its cleanest frame, propagate to both seg
    ends plus a margin into the adjacent bad zone (so the stitch has overlap to match on)."""
    lo, hi = seg
    sub = frames_dets[lo:hi]
    a_local = _anchor_frame(sub, N)
    frame0, dets0 = sub[a_local]
    seed = sorted(dets0, key=lambda d: -d["area"])[:N]
    tracks = [GTrack(tag, frame0, d) for tag, d in enumerate(seed)]
    margin = 60                                # frames of overhang into the gap for stitching
    fwd_hi = min(len(frames_dets), hi + margin)
    bwd_lo = max(0, lo - margin)
    _propagate(tracks, frames_dets[lo + a_local:fwd_hi])
    for t in tracks:
        t.reset(frame0)
    _propagate(tracks, list(reversed(frames_dets[bwd_lo:lo + a_local + 1])))
    return tracks


def _stitch(left, right):
    """Match right-segment tracks onto left-segment tracks (N<->N) by pooled silk signature,
    refined by position where their coverage overlaps. Returns {right_track: left_track}."""
    cost = np.zeros((len(left), len(right)))
    for i, lt in enumerate(left):
        ls = lt.pooled_sig()
        for j, rt in enumerate(right):
            ds = sig_dist(ls, rt.pooled_sig())
            common = set(lt.pos) & set(rt.pos)
            if common:                         # overlap: position agreement is decisive
                pd = float(np.median([np.hypot(lt.pos[f][0] - rt.pos[f][0],
                                               lt.pos[f][1] - rt.pos[f][1]) for f in common]))
                cost[i, j] = pd + 0.2 * ds
            else:
                cost[i, j] = ds + 200.0        # appearance-only (across a gap): weaker
    ri, ci = _hungarian(cost)
    return {right[j].tag: left[i].tag for i, j in zip(ri, ci)}


def propagate_ntracks(frames_dets, N):
    """Global second pass: per stable segment, exactly N no-teleport tracks; segments STITCHED
    left-to-right by pooled appearance (+ position overlap) so identity survives the unreliable
    bend/finish zones WITHOUT coasting across them. Returns N merged GTracks."""
    segs = _stable_segments(frames_dets, N)
    if not segs:
        segs = [(0, len(frames_dets))]
    seg_tracks = [_seg_tracks(frames_dets, s, N) for s in segs]
    base = seg_tracks[0]                        # tags 0..N-1 anchor on the first segment
    by_tag = {t.tag: t for t in base}
    for nxt in seg_tracks[1:]:
        mp = _stitch(list(by_tag.values()), nxt)
        for rt in nxt:
            gtag = mp.get(rt.tag)
            if gtag is None:
                continue
            gt = by_tag[gtag]
            for f, p in rt.pos.items():         # later segment wins on overlap (fresher local fit)
                gt.pos[f] = p
            gt.sigs.extend(rt.sigs)
    return list(by_tag.values())



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

    # ---- pass 1: detect per frame; the cache is built by cv_htag_dets_cache.py (with the
    # temporally-smoothed avatar scale). The inline fallback below is unsmoothed (dev only).
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

    # ---- FIRST PASS: forward tracklets, used ONLY to identify furniture (a moving-vs-field
    # residual + a screen-static test); the per-horse tags come from the global second pass.
    tracklets, _field_shift = build_tracklets(frames_dets)
    furn = [t for t in tracklets
            if (len(t.frames) >= 5 and t.ground_speed() > FURN_RESID_MIN)
            or _is_static(t.frames, [d["cx"] for d in t.dets], [d["cy"] for d in t.dets])]
    print(f"furniture tracklets dropped: {len(furn)} "
          f"(resid {[round(t.ground_speed(), 1) for t in furn[:12]]}...)")
    # ---- furniture removal from the DETECTION set (the tracklets are only used to find it)
    furn_pos = set()
    for t in furn:
        for f, d in zip(t.frames, t.dets):
            furn_pos.add((f, round(d["cx"]), round(d["cy"])))
    clean_fd = []
    for f, dets in frames_dets:
        clean_fd.append((f, [d for d in dets
                             if (f, round(d["cx"]), round(d["cy"])) not in furn_pos]))

    # ---- GLOBAL SECOND PASS: exactly N tracks, no teleport, seeded at the cleanest frame
    tracks = propagate_ntracks(clean_fd, FIELD)
    # reliability filter: where per-frame detection is far from N the matches are unreliable
    # (the zoomed-out bend, finish paint) — drop those positions rather than emit a wrong grab.
    # Identity is still carried THROUGH them (no-teleport state), only the OUTPUT is suppressed.
    reliable = {f for f, d in clean_fd if FIELD - 2 <= len(d) <= FIELD + 2}
    for t in tracks:
        t.pos = {f: p for f, p in t.pos.items() if f in reliable}
    cover = [len(t.pos) for t in tracks]
    print(f"global tracks={len(tracks)} field={FIELD} "
          f"coverage(min/med/max)={min(cover)}/{int(np.median(cover))}/{max(cover)} "
          f"of {len(clean_fd)} frames")

    # ---- emit per-frame tags + overlay render
    bytag: dict[int, list] = {}
    for t in tracks:
        for f, (x, y, area) in t.pos.items():
            bytag.setdefault(f, []).append((t.tag, x, y, area))
    rng = np.random.RandomState(7)
    colors = {t.tag: tuple(int(c) for c in rng.randint(60, 255, 3)) for t in tracks}
    rows = []
    for i, fr in enumerate(av.open(f"data_cache/video/{VID}.mp4").decode(video=0)):
        if i % 2 or i not in ty0_by_frame:
            continue
        panel = fr.to_ndarray(format="bgr24")[:, fr.width // 2:]
        vis = panel.copy()
        for tag, x, y, area in bytag.get(i, []):
            cv2.circle(vis, (int(x), int(y)), 12, colors[tag], 2)
            cv2.putText(vis, str(tag), (int(x) + 10, int(y) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, colors[tag], 2)
            rows.append((i, tag, x, y, area))
        cv2.putText(vis, f"f{i:05d} tags={len(bytag.get(i, []))} field={FIELD}",
                    (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        cv2.imwrite(f"{OUTDIR}/f{i:05d}.png", vis)
    arr = np.array(rows, np.float32)
    np.savez_compressed(f"data_cache/_htag_tracks_{RACE}.npz", rows=arr)
    print(f"render -> {OUTDIR}/  tracks -> data_cache/_htag_tracks_{RACE}.npz")


if __name__ == "__main__":
    main()
