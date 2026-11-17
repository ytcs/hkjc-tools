"""Finish-line camera (RF/drone shared finish replay): per-horse finish ORDER and time GAPS.

After the race the RF (and drone — same edit) replay shows a FIXED overhead camera on the finish
line (white vertical line, the judge's post at the top), in slow motion with a variable playback
rate. Each UNIQUE frame is one source-camera frame (evenly spaced in real time); repeated frames are
the slow-motion padding. Horses carry saddlecloth numbers (yellow digits on a dark cloth) that are
readable here, so identity comes for free.

Per horse: the cloth's x over unique-frame index u is ~linear across the line (constant speed over
~1 s), so a robust line fit gives the index u* at which it crosses the finish line. The cloth sits a
near-constant distance behind the nose for every horse, so cloth-crossing gaps = nose-crossing gaps
(speed differences of ~1 m/s shift this by < 0.01 s). Real time per unique frame (k) and the absolute
offset are not observable here; they are fitted against the other feeds' arrival times (``calibrate``).
"""

from __future__ import annotations

import re

import numpy as np

from . import ocr

LINE_ROWS = (300, 1000)


def line_x(img):
    """x of the finish line (a white vertical column spanning the turf), or None."""
    import cv2

    hsv = cv2.cvtColor(img[LINE_ROWS[0] : LINE_ROWS[1]], cv2.COLOR_BGR2HSV)
    white = ((hsv[..., 1] < 70) & (hsv[..., 2] > 160)).mean(0)
    x = int(np.argmax(white))
    return x if white[x] > 0.6 else None


RING_MIN = 0.40  # white fraction of the ring around a digit (0.55 lost small far-side cloths: ST7's
#                  winner #8 read on 2 frames only; the OCR confidence filters the extra candidates)


def _white_cloths(img, y_min):
    """Candidate boxes of DARK DIGITS ON WHITE saddlecloths (Sha Tin side-on finish camera; HV's
    overhead camera shows yellow digits on dark cloth): dark components of digit size whose
    surrounding ring is mostly white, grouped side by side for two-digit numbers. Segmenting the
    whole white cloth was REJECTED: it merges with jockeys' white breeches and grey horses."""
    import cv2

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    s, v = hsv[..., 1].astype(int), hsv[..., 2].astype(int)
    white = (s < 70) & (v > 175)
    # digits = dark strokes INSIDE the closed white cloth (closing fills the digit holes). A fixed
    # v < 100 cut fragments motion-blurred strokes; a black-hat also lights up whole dark horse
    # bodies on green turf (ST7 wide shot) and merges them with the digits — both REJECTED.
    wc = cv2.morphologyEx(white.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    dark = (wc.astype(bool) & ~white & (v < 150)).astype(np.uint8)
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    dark[: max(y_min, 200)] = 0  # the disclaimer overlay band
    n, _lab, st, _cen = cv2.connectedComponentsWithStats(dark)
    H, W = v.shape
    digs = []
    for x, y, w, hh, a in st[1:]:
        if not (9 <= hh <= 70 and 3 <= w <= 0.9 * hh and a >= 0.15 * w * hh):
            continue
        m = max(4, hh // 4)
        x0, y0, x1, y1 = max(0, x - m), max(0, y - m), min(W, x + w + m), min(H, y + hh + m)
        ring = white[y0:y1, x0:x1].copy()
        ring[m : m + hh, m : m + w] = False
        nring = (y1 - y0) * (x1 - x0) - w * hh
        if nring > 0 and ring.sum() / nring > RING_MIN:
            digs.append((x, y, w, hh))
    digs.sort()
    out, used = [], set()
    for i, (x, y, w, hh) in enumerate(digs):  # pair digits of two-digit numbers
        if i in used:
            continue
        box = [x, y, x + w, y + hh]
        for j in range(i + 1, len(digs)):
            x2, y2, w2, h2 = digs[j]
            if j not in used and 0 <= x2 - box[2] < 0.6 * hh and abs(y2 - y) < 0.3 * hh:
                box = [box[0], min(box[1], y2), x2 + w2, max(box[3], y2 + h2)]
                used.add(j)
                break
        out.append((box[0], box[1], box[2] - box[0], box[3] - box[1]))
    return out


def cloth_reads(img, *, y_min=150, n_field=14, x_center=None, x_half=700, white=True):
    """Saddlecloth number reads: [(x, y, number, conf)] — yellow digit clusters on dark cloth (HV
    overhead finish camera) and dark digits on white cloth (Sha Tin side-on finish camera)."""
    import cv2

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = [hsv[..., i].astype(int) for i in range(3)]
    yel = ((h > 18) & (h < 40) & (s > 110) & (v > 140)).astype(np.uint8)
    yel &= cv2.dilate((v < 90).astype(np.uint8), np.ones((15, 15), np.uint8))
    yel[:y_min] = 0
    yel = cv2.dilate(yel, np.ones((5, 9), np.uint8))  # join the digits of "11"
    n, _lab, st, _cen = cv2.connectedComponentsWithStats(yel)
    boxes = [
        (x, y, w, hh) for x, y, w, hh, a in st[1:] if 12 <= hh <= 70 and 8 <= w <= 90 and a > 60
    ]
    if white:  # Sha Tin side-on camera; HV's overhead camera has yellow-on-dark cloths only
        boxes += _white_cloths(img, y_min)
    if (
        x_center is not None
    ):  # only cloths near the line matter for crossings/gaps (OCR is the cost)
        boxes = [b for b in boxes if abs(b[0] + b[2] / 2 - x_center) < x_half]
    crops = [img[max(0, y - 8) : y + hh + 8, max(0, x - 10) : x + w + 10] for x, y, w, hh in boxes]
    out = []
    for (x, y, w, hh), (txt, conf) in zip(boxes, ocr.rec_batch(crops)):
        dg = re.sub(r"\D", "", txt)
        if dg and 1 <= int(dg) <= n_field and conf >= 0.6:
            out.append((x + w / 2, y + hh / 2, int(dg), conf))
    return out


HORSE_LEN_M = 2.6  # nose-to-tail of a galloping thoroughbred at full stretch (+-8%): the ruler
COCO_WEIGHTS = "models/coco_yolov8x.pt"


def horse_scale(frames_imgs, line_x):
    """px per metre ALONG the running direction near the finish line, from isolated horse boxes
    (COCO 'horse') of known nose-to-tail length. Median over frames; None if too few."""
    import os

    from ultralytics import YOLO

    if not os.path.exists(COCO_WEIGHTS):
        return None
    dev = os.environ.get("HKJC_DEVICE", "cpu")
    if dev.startswith("intel:") and frames_imgs and frames_imgs[0].shape[:2] == (540, 960):
        # Intel iGPU via OpenVINO (fixed 544x960 input for the half-res 960x540 frames): YOLOv8x
        # at ~2.4 s/frame on the CPU was 75 s of the R08 finish scan
        ov_dir = COCO_WEIGHTS.replace(".pt", "_544x960_openvino_model")
        if not os.path.exists(ov_dir):
            p = YOLO(COCO_WEIGHTS).export(format="openvino", imgsz=[544, 960], verbose=False)
            os.rename(p, ov_dir)
        m, kw = YOLO(ov_dir, task="detect"), dict(imgsz=(544, 960), device=dev)
    else:
        m, kw = YOLO(COCO_WEIGHTS), dict(imgsz=960)
    w = []
    for img in frames_imgs:
        r = m.predict(img, conf=0.4, classes=[17], verbose=False, **kw)[0]
        b = r.boxes.xyxy.cpu().numpy()
        for i, (x0, y0, x1, y1) in enumerate(b):
            others = np.delete(b, i, axis=0)
            ov = (
                ((np.minimum(x1, others[:, 2]) - np.maximum(x0, others[:, 0])) > 0)
                & ((np.minimum(y1, others[:, 3]) - np.maximum(y0, others[:, 1])) > 0)
                if len(others)
                else []
            )
            if (
                (len(others) == 0 or not np.any(ov))
                and abs((x0 + x1) / 2 - line_x) < 500
                and x0 > 5
            ):
                w.append(x1 - x0)
    return (float(np.median(w)) / HORSE_LEN_M, len(w)) if len(w) >= 5 else None


def scan(path, *, start_s=100.0, stop_s=160.0, fps=25.0, n_field=14, min_run=40, white=True):
    """Find the finish-camera segment (finish line present at a stable x over many unique frames;
    of several candidate runs keep the one with the most cloth reads) and read the cloths on its
    unique frames. Returns dict(line_x, frames, reads=[(u, frame, x, y, num, conf)])."""
    import av

    runs, cur = [], None
    prev = None
    with av.open(str(path)) as c:
        c.streams.video[0].thread_type = "AUTO"  # frame-threaded decode: same pixels, ~3x faster
        for i, fr in enumerate(c.decode(video=0)):
            if i < start_s * fps:
                continue
            if i > stop_s * fps:
                break
            img = fr.to_ndarray(format="bgr24")
            lx = line_x(img)
            if lx is None or (cur is not None and abs(lx - cur["x0"]) > 6):
                if cur is not None and len(cur["frames"]) >= min_run:
                    runs.append(cur)
                cur, prev = None, None
                if lx is None:
                    continue
            if cur is None:
                cur = dict(x0=lx, frames=[], xs=[], reads=[])
            small = img[::4, ::4]
            if prev is not None and float(np.abs(small.astype(np.int16) - prev).mean()) < 1.0:
                continue  # slow-motion repeat
            prev = small
            u = len(cur["frames"])
            cur["frames"].append(i)
            cur["xs"].append(lx)
            if u % 6 == 0:  # a few half-res frames for the horse-length ruler (horse_scale)
                cur.setdefault("imgs", []).append(img[::2, ::2].copy())
            for x, y, num, conf in cloth_reads(img, n_field=n_field, x_center=lx, white=white):
                cur["reads"].append((u, i, x, y, num, conf))
    if cur is not None and len(cur["frames"]) >= min_run:
        runs.append(cur)
    if not runs:
        return dict(line_x=np.nan, frames=np.zeros(0), reads=np.zeros((0, 6)))
    best = max(runs, key=lambda r: len(r["reads"]))
    lx = float(np.median(best["xs"]))
    hs = horse_scale(best.get("imgs", []), lx / 2)  # half-res frames -> x2 for full-res px/m
    return dict(
        line_x=lx,
        frames=np.array(best["frames"]),
        reads=np.array(best["reads"], float).reshape(-1, 6),
        ppm=2 * hs[0] if hs else np.nan,
    )


_TRACKS_CACHE: dict = {}


def _reads_key(sc):
    import hashlib

    R = np.ascontiguousarray(np.asarray(sc["reads"]))
    return (R.shape, str(R.dtype), hashlib.sha1(R.tobytes()).hexdigest())


def tracks(sc, *, min_reads=4, y_tol=40.0):
    """Memoised ``_tracks`` (a pure function of the reads; called ~12x per fuse on the same reads).
    Exact: returns deep copies."""
    import copy

    key = (_reads_key(sc), float(min_reads), float(y_tol))
    if key not in _TRACKS_CACHE:
        if len(_TRACKS_CACHE) > 64:
            _TRACKS_CACHE.clear()
        _TRACKS_CACHE[key] = _tracks(sc, min_reads=min_reads, y_tol=y_tol)
    return copy.deepcopy(_TRACKS_CACHE[key])


def _tracks(sc, *, min_reads=4, y_tol=40.0):
    """Per number, its reads restricted to the dominant lateral (y) cluster (a misread of another
    horse's cloth sits at a different y), sorted by unique-frame index u. {num: (u, x, y) arrays}.

    ONE CLOTH = ONE MOVING TRACK (DEFAULT since 2026-11-16; HKJC_FC_MOVE=0 / HKJC_FC_CONT=0 restore the
    old rules for diagnosis):
    - the dominant y-cluster is the one read at the most DISTINCT PLACES (45 px bins, the same-cloth
      tolerance), then the most reads. MEASURED (pixels): ST8's "#1" was a static RAIL POST read ~60x at
      x 1566 (it passed the ptp "must move" test only through 2 stray reads the running median later
      removed) and outvoted the real #1 cloth (35 reads, x 1532 -> 293, conf 0.99); ST9's "#7" was a
      tie (8 v 8 reads, first index won) between "7" read on #5's checked jockey silk (static) and the
      real #7 cloth. REJECTED: the ptp test after the running median -- it drops genuine SHORT tracks
      (ST4's winner #11 spans 95 px, ST9 #8 94 px over 4-7 unique frames).
    - the coincidence rule (two numbers on one cloth) also counts reads on the other track CONTINUED
      past its ends (linear, +-10 unique frames -- the digit-drop window). MEASURED (pixels): HV 04-15 R1's
      "#1" was the "7" cloth read as "1" on the 4 unique frames BEFORE the "7" track starts (x on #7's
      line extrapolated back to 2 px); the shared-frame test saw 1 shared frame and kept a phantom #1
      9th (the real #1 finished ~4th, render silks) -> 2 finish-order inversions on every seed.
    26 races: finish tracks change on 4 (all pixel-verified), 23 races' paths byte-identical; 6-seed
    ensemble: every gate / idscore / hand-truth yardstick identical (paired Delta 0); finish-order audit
    s0 8 -> 4 (interpolated crossings)."""
    R = np.asarray(sc["reads"])
    out = {}

    import os

    move = os.environ.get("HKJC_FC_MOVE", "1") != "0"

    def support(r, inc):
        if not move:
            return inc.sum()
        # distinct PLACES read (45 px = the same-cloth tolerance below), then reads: a static
        # scene feature read at one spot in 60 frames must not outvote a moving cloth
        return (len(np.unique(np.round(r[inc, 2] / 45.0))), inc.sum())

    def build(r):
        if len(r) < min_reads:
            return None
        ys = r[:, 3]
        best = max(range(len(ys)), key=lambda i: support(r, np.abs(ys - ys[i]) < y_tol))
        r = r[np.abs(ys - ys[best]) < y_tol]
        if len(r) < min_reads or np.ptp(r[:, 2]) < 100:  # must actually move across the frame
            return None
        o = np.argsort(r[:, 0], kind="stable")
        uu, xx, yy = r[o, 0], r[o, 2], r[o, 3]
        # drop isolated misreads (e.g. a half-occluded "11" read as "1"): far from a running median.
        # Known cost: SPARSE reads of a moving cloth look isolated (ST 0419 R4's winner #11, read at u
        # 25/27/56 then 69-86 in the slow-motion post camera, loses the reads near the line -> no gap).
        # NOT ADOPTED (flips untraced): a robust line x(u) instead (+ a motion-rate test for short
        # slow-motion tracks, which admits ST4's runner-up #10): restores #11 and #10 (ST4 rho 1.00,
        # paths complete) but 26 races: timing gates 126 -> 123 -- ST4 sectional 0.173 -> 0.288 (untraced),
        # ST 0509 R1 rho 0.87 (its #8/#9 reads are GENUINE, pixels; their gaps place them wrongly), ST1's
        # winner flip is a near dead heat (#13 / unread #9 official 0.05 s apart; noise). Motion-rate
        # alone: 126 -> 124.
        med = np.array([np.median(xx[max(0, i - 3) : i + 4]) for i in range(len(xx))])
        k = np.abs(xx - med) < 80
        return (uu[k], xx[k], yy[k]) if k.sum() >= min_reads else None

    raw = {int(n): R[R[:, 4] == n] for n in np.unique(R[:, 4]).astype(int)}
    for num, r in raw.items():
        tr = build(r)
        if tr is not None:
            out[num] = tr
    # DIGIT-DROP pre-pass: reads of a SHORTER number lying on a longer number's track ("1" on the "11"
    # cloth) are that longer number's, minus a digit -- remove them and rebuild the shorter track from
    # what is left. MEASURED (pixels, ST 0419 R4): "1" pooled the winner #11's cloth ("11" read as "1")
    # with the real #1; its dominant y-cluster became #11's cloth, the coincidence rule below kept
    # "1" (more reads) and DROPPED the winner #11 from the finish order.
    for lng in sorted(list(out), key=lambda n: -len(str(n))):
        if lng not in out:
            continue
        ul, xl, yl = out[lng]
        for sh in list(raw):
            if sh == lng or str(sh) not in str(lng) or len(str(sh)) >= len(str(lng)):
                continue
            r = raw[sh]
            # the long number's RAW reads (its y-cluster), robust LINE x(u): the running-median filter
            # trims sparse early reads (ST4 "11": u 25-86 read, track u 69-74), and the misread "1"s
            # sat exactly in the trimmed part
            rl = raw[lng]
            rl = rl[np.abs(rl[:, 3] - np.median(yl)) < 30]
            if len(rl) < 3:
                continue
            A = np.c_[rl[:, 0], np.ones(len(rl))]
            coef = np.linalg.lstsq(A, rl[:, 2], rcond=None)[0]
            res = np.abs(A @ coef - rl[:, 2])
            keep = res < max(60.0, 2.5 * np.median(res))
            coef = np.linalg.lstsq(A[keep], rl[keep, 2], rcond=None)[0]
            inside = (r[:, 0] >= rl[:, 0].min() - 10) & (r[:, 0] <= rl[:, 0].max() + 10)
            on = (
                inside
                & (np.abs(r[:, 0] * coef[0] + coef[1] - r[:, 2]) < 60)
                & (np.abs(r[:, 3] - np.median(yl)) < 30)
            )
            if on.sum() >= 2:
                raw[sh] = r[~on]
                tr = build(raw[sh])
                if tr is None:
                    out.pop(sh, None)
                else:
                    out[sh] = tr
    # a MISREAD is the same physical cloth read under a second number (Sha Tin side-on camera:
    # 1/4/14 confusions put "#14" 1.4 s ahead of where it finished): two numbers whose tracks
    # coincide in (x, y) on shared frames (or on one track continued past its ends: see the
    # docstring) -> keep the one with more reads
    cont = os.environ.get("HKJC_FC_CONT", "1") != "0"
    nums = sorted(out, key=lambda n: -len(out[n][0]))
    drop = set()
    for i, a in enumerate(nums):
        if a in drop:
            continue
        ua, xa, ya = out[a]
        for b in nums[i + 1 :]:
            if b in drop:
                continue
            ub, xb, yb = out[b]
            same = 0
            for u, x, y in zip(ub, xb, yb):
                j = np.where(np.abs(ua - u) <= 1)[0]
                if len(j) and np.abs(xa[j] - x).min() < 45 and np.abs(ya[j] - y).min() < 30:
                    same += 1
                elif (
                    cont
                    and not len(j)
                    and ua[0] - 10 <= u <= ua[-1] + 10
                    and abs(_x_at(out[a], u) - x) < 45
                    and abs((ya[:4] if u < ua[0] else ya[-4:]).mean() - y) < 30
                ):
                    same += 1  # on a's track CONTINUED past its ends (see the docstring)
            if same >= max(2, 0.5 * len(ub)):
                # a dropped digit is the common misread: prefer the LONGER number ("11" over "1")
                if str(a) in str(b) and len(str(a)) < len(str(b)):
                    drop.add(a)
                    break
                drop.add(b)
    return {n: v for n, v in out.items() if n not in drop}


def _x_at(tr, u):
    """Cloth x at (fractional) unique frame u, linear between reads, linear extrapolation outside."""
    uu, xx = tr[0], tr[1]
    if u <= uu[0] or u >= uu[-1]:
        k = slice(0, min(4, len(uu))) if u <= uu[0] else slice(max(0, len(uu) - 4), len(uu))
        if np.ptp(uu[k]) == 0:
            return np.nan
        b, a = np.polyfit(uu[k], xx[k], 1)
        return a + b * u
    return float(np.interp(u, uu, xx))


def crossing_index(tr, lx):
    """Fractional unique-frame index where the cloth passes the line (moving leftward)."""
    uu, xx = tr[0], tr[1]
    k = np.where((xx[:-1] >= lx) & (xx[1:] < lx))[0]
    if len(k) == 0:
        return np.nan
    i = k[0]
    return uu[i] + (xx[i] - lx) / max(xx[i] - xx[i + 1], 1e-9) * (uu[i + 1] - uu[i])


_GAPS_CACHE: dict = {}


def finish_gaps(sc, *, max_dx=1500.0, with_sd=False):
    """Memoised ``_finish_gaps`` (a pure function of line_x and the reads; the fuse stage calls it
    ~17x per race on the same cached reads, 1.6 s of a 13.5 s race). Exact: returns deep copies."""
    import copy

    key = (float(sc["line_x"]), _reads_key(sc), float(max_dx), bool(with_sd))
    if key not in _GAPS_CACHE:
        if len(_GAPS_CACHE) > 64:
            _GAPS_CACHE.clear()
        _GAPS_CACHE[key] = _finish_gaps(sc, max_dx=max_dx, with_sd=with_sd)
    return copy.deepcopy(_GAPS_CACHE[key])


def _finish_gaps(sc, *, max_dx=1500.0, with_sd=False):
    """Order + SPATIAL gaps at the finish (playback-rate independent).

    1. Horses whose cloth is seen crossing the line get a crossing index; consecutive crossers are
       chained: at the instant B crosses, A's cloth is gap_px ahead.
    2. EVERY horse with reads then gets gap estimates from shared instants with referenced horses:
       gap_h = gap_r + (x_h(u) - x_r(u)) (speeds near the line are ~equal, so the spacing is ~stable);
       the median of all estimates is used. This recovers horses occluded exactly at the line.
    Returns (order list, {num: gap_px behind the first finisher}, {num: u*}); with_sd=True appends
    {num: gap sd px} (robust spread of the per-instant estimates / sqrt(n), floor 15 px)."""
    lx = sc["line_x"]
    T = tracks(sc)
    ustar = {n: crossing_index(t, lx) for n, t in T.items()}
    crossers = sorted([n for n in ustar if np.isfinite(ustar[n])], key=lambda n: ustar[n])
    if not crossers:
        return ([], {}, ustar, {}) if with_sd else ([], {}, ustar)
    def _chain(cr):
        ch = {cr[0]: 0.0}
        for i, b in enumerate(cr[1:], start=1):
            for a in reversed(cr[:i]):
                if a not in ch:
                    continue
                xa = _x_at(T[a], ustar[b])
                if np.isfinite(xa) and 0 <= lx - xa < max_dx:
                    ch[b] = ch[a] + (lx - xa)
                    break
        return ch

    # Start the chain at the crosser that links the MOST crossers (ties: the earliest). A first
    # crosser whose cloth has left the frame before the next horse crosses (a wide winning margin
    # in the slow-motion post camera) links nobody, and starting there wiped every other gap.
    # MEASURED (ST 0419 R4 with the winner #11's crossing injected): 1 gap instead of 8. On all 26
    # turf races the first crosser already links the most, so this is a no-op there.
    chains = [_chain(crossers[s:]) for s in range(len(crossers))]
    chain = max(chains, key=len)
    est = {n: ([chain[n]] if n in chain else []) for n in T}
    for h, (uh, xh, _yh) in T.items():
        for r, gr in chain.items():
            if r == h:
                continue
            ur, xr = T[r][0], T[r][1]
            for u, x in zip(uh, xh):
                k = np.where(np.abs(ur - u) <= 1)[0]
                if len(k):
                    dx = x - xr[k[0]]
                    if abs(dx) < max_dx:
                        est[h].append(gr + dx)
    # horses seen CROSSING the line (chain members) are direct measurements: keep them with a single
    # estimate. Requiring >= 2 dropped the first crosser (its own gap is defined, not estimated) and
    # re-zeroed every other gap on the next horse (ST8: winner #12 seen crossing, order came out
    # [2, 7, 1]). Non-crossers still need >= 2 shared-frame estimates.
    gaps = {n: float(np.median(v)) for n, v in est.items() if len(v) >= 2 or (n in chain and v)}
    # DIGIT-DROP misreads: "14" read as "4" when the "1" is missed (ST7: "#4" at 0.17 s next to #14 at
    # 0.19 s; real #4 finished 2.12 s back). Drop a number whose digits are contained in another read
    # number when its reads COINCIDE physically with that number's (same cloth: within 45/30 px on
    # >= 2 shared frames, and >= half of the shared frames) and it has fewer reads. Gap proximity alone
    # REJECTED: it dropped ST8's real winner #2 next to #12 (a close finish; their reads ~240 px apart).
    R = np.asarray(sc["reads"])
    nreads = {n: len(T[n][0]) for n in T}
    for n in list(gaps):
        for m in list(gaps):
            if _digit_drop(n, m, R, T, nreads):
                gaps.pop(n)
                break
    if not gaps:
        return ([], {}, ustar, {}) if with_sd else ([], {}, ustar)
    g0 = min(gaps.values())
    gaps = {n: g - g0 for n, g in gaps.items()}
    order = sorted(gaps, key=lambda n: gaps[n])
    if not with_sd:
        return order, gaps, ustar
    sd = {}
    for n in gaps:
        v = np.asarray(est[n])
        sd[n] = max(15.0, 1.4826 * float(np.median(np.abs(v - np.median(v)))) / np.sqrt(len(v)))
    return order, gaps, ustar, sd


def _digit_drop(n, m, R, T, nreads) -> bool:
    """True when number ``n``'s reads are a DIGIT-DROP misread of ``m``'s cloth (see finish_gaps)."""
    if m == n or str(n) not in str(m) or nreads.get(n, 0) >= nreads.get(m, 0):
        return False
    rn = R[R[:, 4] == n]
    # compare against m's TRACK interpolated to each n-read's frame: one cloth yields ONE read
    # per frame ("3" OR "13"), so same-frame coincidence never fired, and the horse moves
    # ~60 px per unique frame. MEASURED (pixels, ST 0419 R4): 9 of 10 "3" reads sit on the
    # "13" cloth (one at conf 0.99); kept, "#3" (tailed off, 7.5 s back) took #13's crossing
    # -> placed 6.2 s early (behind-winner gate failure).
    rm = R[R[:, 4] == m]
    sh0 = sa0 = 0  # same-frame coincidence (the original test: ST7's "4" next to "14")
    for r in rn:
        k = np.where(np.abs(rm[:, 0] - r[0]) <= 1)[0]
        if len(k):
            sh0 += 1
            sa0 += bool(
                (np.abs(rm[k, 2] - r[2]).min() < 45) and (np.abs(rm[k, 3] - r[3]).min() < 30)
            )
    # ... and only when MOST of n's reads sit on m's cloth: the tests count shared frames
    # only, so 3 dropped-digit "2" reads on the "12" cloth deleted the REAL #2 (37 reads,
    # crossing after #12 had left; pixels: a different silk) on HV 0603 R3 and HV 0415 R1.
    if sa0 >= 2 and sa0 >= 0.5 * sh0 and sa0 >= 0.5 * len(rn):
        return True
    if m not in T:
        return False
    um, xm, ym = T[m]
    shared = same = 0
    for r in rn:
        if not (um[0] - 3 <= r[0] <= um[-1] + 3):
            continue
        shared += 1
        same += bool(
            abs(np.interp(r[0], um, xm) - r[2]) < 45 and abs(np.interp(r[0], um, ym) - r[3]) < 30
        )
    return same >= 2 and same >= 0.5 * shared and same >= 0.5 * len(rn)


def read_numbers(sc) -> set:
    """Numbers the finish camera READ (a cloth track, minus digit-drop misreads of another read
    number) -- whether or not finish_gaps could give them a gap. A gap needs a crossing chained to
    an earlier crosser or a frame shared with a referenced horse; a back-marker crossing after the
    last referenced cloth has left the frame gets neither. Example (ST 0509 R1): #8/#9 (16/22 reads,
    1.45/1.51 s back) had no gap, so the E-step's negative evidence treated them as unread and
    "blind" (CLUTTER on every detection in the last 50 m). Used only with HKJC_FC_READS=reads
    (opt-in, NOT assessed on ensembles): it did NOT fix that race's short #4 path (finish-n 7 -> 7);
    the cause there was the straight-course rail side (rails.inner_side). Gapless read numbers on the 26 turf
    races: real runners 0.87-2.4 s back (ST3 #14, ST9 #8, ST 0531 #6), the ST4 winner #11, and ST7
    "#4" (a digit-drop of #14, removed here).
    REJECTED for giving such horses a gap: integrating the cloth speed (px per unique frame) over
    frames between crossings -- the post camera pans with the leaders (their cloths sit still in
    frame), so the integral saturates (ST 0509 R1 #1/#6/#7, ST7 eight horses at one value)."""
    T = tracks(sc)
    R = np.asarray(sc["reads"])
    nreads = {n: len(T[n][0]) for n in T}
    return {int(n) for n in T if not any(_digit_drop(n, m, R, T, nreads) for m in T)}


def calibrate(gaps, ref_behind, *, c_bounds=(200.0, 6000.0)):
    """Fit C (px per second = speed x scale) so that gap_px / C matches reference time-behind-winner
    {num: s} from another feed (robust, through the origin). Returns (C, {num: seconds behind})."""
    nums = [
        n
        for n in gaps
        if n in ref_behind
        and np.isfinite(gaps[n])
        and np.isfinite(ref_behind[n])
        and ref_behind[n] > 0.02
    ]
    if len(nums) < 3:
        return None
    ratios = np.array([gaps[n] / ref_behind[n] for n in nums])
    C = float(np.clip(np.median(ratios), *c_bounds))
    return C, {n: (g / C if np.isfinite(g) else np.nan) for n, g in gaps.items()}


def anchor(gaps, t_cross, *, c_bounds=(800.0, 4000.0), c_prior=None, c_prior_rel=0.30, t_sd=0.10):
    """Jointly fit the first finisher's clock time t_w and C (px/s) so that
    t_cross[h] ~ t_w + gap_h / C over the horses another feed has fused (robust LS, residuals in
    units of t_sd = the render crossing-time noise), with the PHYSICAL prior C ~ c_prior (relative
    sd c_prior_rel) as one extra residual. The camera fixes ORDER and GAPS; the other feeds only set
    the clock origin and the px->s scale — so horses without any other-feed path (back-markers)
    still get a finish time. Returns (t_w, C, n, cov) or None; cov = 2x2 covariance of (t_w, 1/C)
    from the Gauss-Newton normal matrix, residual scale estimated robustly (never below t_sd).
    (Before 2026-11-13 c_prior was accepted but NOT used: C drifted to 2395 vs prior 1942 on R08.)
    c_prior_rel = 0.30: the COCO-box horse ruler is MEASURED 15-25 % high on HV; at 10 % the prior
    dragged C up and compressed the tail-of-field gaps (R02A #9 -1.35 s, #11 -0.68 s)."""
    from scipy.optimize import least_squares

    nums = [n for n in gaps if np.isfinite(gaps[n]) and np.isfinite(t_cross.get(n, np.nan))]
    if len(nums) < 3:
        return None
    g = np.array([gaps[n] for n in nums])
    t = np.array([t_cross[n] for n in nums])
    use_prior = c_prior is not None and np.isfinite(c_prior) and c_prior > 0

    def res(p):
        r = (p[0] + g * p[1] - t) / t_sd
        if use_prior:
            r = np.r_[r, (p[1] * c_prior - 1.0) / c_prior_rel]
        return r

    x0 = [t.min(), 1 / (c_prior if use_prior else 1700.0)]
    lo, hi = 1 / c_bounds[1], 1 / c_bounds[0]
    x0[1] = float(np.clip(x0[1], lo * 1.001, hi * 0.999))
    r = least_squares(res, x0, loss="soft_l1", f_scale=1.5, bounds=([-np.inf, lo], [np.inf, hi]))
    rt = (r.x[0] + g * r.x[1] - t) / t_sd
    scale = max(1.0, 1.4826 * float(np.median(np.abs(rt))))  # robust residual scale (>= t_sd)
    J = np.c_[np.ones_like(g), g] / (t_sd * scale)
    H = J.T @ J
    if use_prior:
        H[1, 1] += (c_prior / c_prior_rel) ** 2
    cov = np.linalg.inv(H + np.eye(2) * 1e-12)
    return float(r.x[0]), float(1.0 / r.x[1]), len(nums), cov


def identity_match(
    t_cross, gaps, *, c_prior=None, c_rel=0.25, tol=0.08, margin=2.0, sep=0.15, t_step=0.01
):
    """IDENTITY from the finish camera: 1D registration of anonymous line-crossing times
    ``t_cross`` {key: race-clock s} (render tracklets) to the camera's gap profile
    ``gaps`` {num: px behind the first finisher}. Scans the px/s scale C (c_prior +- c_rel, or
    1200..2600 px/s) and the clock t_w, scores the Hungarian assignment with residuals capped at
    0.3 s, then accepts a pair only if its residual < ``tol`` and every other horse's predicted time
    is ``margin`` x farther AND >= ``sep`` s away (well-separated finishers only: on a packed finish
    — R08, neighbours 0.01-0.1 s apart — the relative form alone renamed legend-#2 as #10). The camera fixes ORDER and GAPS exactly at the line; it
    names the tracklets that reach it (R02A: the 13 m clear winner was never numbered by the legend
    votes). -> ({key: num}, fit) with fit = label-free clock/scale {t_w, C, edge, n_close, n} (MEASURED
    on 4 HV races: C within ~15 %, t_w 0.10-0.24 s EARLY — the render leads near the line) or None."""
    from scipy.optimize import linear_sum_assignment

    keys = [k for k in t_cross if np.isfinite(t_cross[k])]
    nums = [n for n in gaps if np.isfinite(gaps[n])]
    if len(keys) < 3 or len(nums) < 3:
        return {}, None
    t = np.array([t_cross[k] for k in keys])
    g = np.array([gaps[n] for n in nums])
    if c_prior is not None and np.isfinite(c_prior):
        Cs = c_prior * np.linspace(1 - c_rel, 1 + c_rel, 41)
    else:
        Cs = np.linspace(1200.0, 2600.0, 57)
    best = None
    for C in Cs:
        for tw in np.arange(t.min() - 1.5, t.min() + 0.5, t_step):
            P = tw + g / C
            R = np.minimum(np.abs(t[:, None] - P[None, :]), 0.3)
            r, c = linear_sum_assignment(R)
            cost = R[r, c].sum() + 0.3 * (len(keys) - len(r))
            if best is None or cost < best[0]:
                best = (cost, C, tw)
    _cost, C, tw = best
    P = tw + g / C
    R = np.abs(t[:, None] - P[None, :])
    r, c = linear_sum_assignment(np.minimum(R, 0.3))
    out = {}
    for i, j in zip(r, c):
        if R[i, j] >= tol:
            continue
        others = np.delete(R[i], j)
        if len(others) and others.min() < max(margin * R[i, j], sep):
            continue
        out[keys[i]] = int(nums[j])
    fit = dict(
        t_w=float(tw),
        C=float(C),
        edge=bool(C <= Cs[0] or C >= Cs[-1]),
        n_close=int((R[r, c] < tol).sum()),
        n=len(keys),
    )
    return out, fit


def eliminate(fit, labelled_keys, used_nums, *, tol=0.12, sep=0.15):
    """ELIMINATION at the line: tracklets that reach the line WITHOUT a label are matched only to
    numbers that have NO line-reaching tracklet (``used_nums``), under the race-wide label-free
    registration (``fit`` from identity_match: t_w, C, tc, gaps). Back-markers never appear in the
    top-8 legend, so the legend cannot name them (R08: three whole-race tracklets unnamed, #1/#5
    path-less). A pair is accepted if its residual < ``tol`` and forbidding it raises the optimal
    assignment cost by >= ``sep`` s (assignment-level ambiguity; a pairwise neighbour test REJECTED:
    #1/#5 predicted 0.16 s apart blocked an unambiguous joint match). -> {key: num}."""
    from scipy.optimize import linear_sum_assignment

    keys = [k for k in fit["tc"] if k not in labelled_keys]
    nums = [n for n, g in fit["gaps"].items() if n not in used_nums and np.isfinite(g)]
    if not keys or not nums:
        return {}
    t = np.array([fit["tc"][k] for k in keys])
    P = fit["t_w"] + np.array([fit["gaps"][n] for n in nums]) / fit["C"]
    R = np.minimum(np.abs(t[:, None] - P[None, :]), 1.0)
    pad = max(len(keys), len(nums))
    Rp = np.full((pad, pad), 1.0)  # dummy rows/cols = "unmatched" at the cap cost
    Rp[: len(keys), : len(nums)] = R

    def cost(M):
        r, c = linear_sum_assignment(M)
        return M[r, c].sum(), r, c

    c0, r, c = cost(Rp)
    out = {}
    for i, j in zip(r, c):
        if i >= len(keys) or j >= len(nums) or R[i, j] >= tol:
            continue
        M = Rp.copy()
        M[i, j] = 10.0
        if cost(M)[0] - c0 >= sep:
            out[keys[i]] = int(nums[j])
    return out
