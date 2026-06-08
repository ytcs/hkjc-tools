"""DRONE rail extraction experiment (scratch) — GREEN-BAND borders, not white lines.

Premise: the running surface is one large green band; its two long edges ARE the rails.
Robust to white distractors (gate/road/stands/floodlights sit OUTSIDE the band). Detecting BOTH
borders gives per-frame track width (px) -> scale, and a normalized lateral coordinate.
Validate VISUALLY across the whole race segment (frames 0-2833), per CV_RECON s0.
"""
from __future__ import annotations
import cv2
import numpy as np

VIDEO="data_cache/video/drone_20260603_08.mp4"

# All pixel-unit constants below are defined for 1080p and scaled by SCALE = H/1080, so the
# detector is RESOLUTION-INDEPENDENT (lets us run at half res for speed; identical behaviour).
SCALE = 1.0

def _ksz(v_1080):
    k = int(round(v_1080 * SCALE))
    k = max(3, k)
    return k + 1 - (k % 2)  # force odd

def turf_mask(bgr):
    global SCALE
    SCALE = bgr.shape[0] / 1080.0
    hsv=cv2.cvtColor(bgr,cv2.COLOR_BGR2HSV)
    H,S,V=hsv[...,0].astype(int),hsv[...,1].astype(int),hsv[...,2].astype(int)
    # modal hue among reasonably saturated, non-dark pixels
    sel=(S>40)&(V>40)&(V<250)
    hist=np.bincount(hsv[...,0][sel].ravel(),minlength=180)
    # green band ~ 30-90 in OpenCV hue; restrict modal search there to avoid red track/brown
    gh=np.argmax(hist[30:95])+30
    m=((np.abs(H-gh)<=18)&(S>30)&(V>35)).astype(np.uint8)*255
    # include slightly-worn (lower sat) turf near the band by closing
    m=cv2.morphologyEx(m,cv2.MORPH_CLOSE,np.ones((_ksz(25),_ksz(25)),np.uint8))
    m=cv2.morphologyEx(m,cv2.MORPH_OPEN,np.ones((_ksz(7),_ksz(7)),np.uint8))
    return m, gh

def largest_band(mask):
    n,lab,stats,_=cv2.connectedComponentsWithStats((mask>0).astype(np.uint8),8)
    if n<=1:
        return None
    i=1+int(np.argmax(stats[1:,cv2.CC_STAT_AREA]))
    return (lab==i).astype(np.uint8)*255

def _rot90(v):  # +90 deg
    return np.array([-v[1], v[0]], np.float32)

def trace_ribbon(band, step=14.0, slab=16.0, min_slice=12):
    """Walk the band's medial axis with perpendicular slices whose orientation tracks the LOCAL
    tangent -> borders follow the arc on bends (fixes the global-PCA chord failure)."""
    ys, xs = np.where(band > 0)
    if len(xs) < 500:
        return None
    pts = np.c_[xs, ys].astype(np.float32)
    mean = pts.mean(0)
    _, _, vt = np.linalg.svd(pts - mean, full_matrices=False)
    axis = vt[0].astype(np.float32)
    t = (pts - mean) @ axis
    start = mean + axis * np.percentile(t, 3)
    tan = axis.copy()
    if ((pts - start) @ tan).mean() < 0:  # orient into the band
        tan = -tan
    centers, left, right, widths = [], [], [], []
    c = start.copy()
    H, W = band.shape
    for _ in range(400):
        perp = _rot90(tan)
        d = pts - c
        sel = np.abs(d @ tan) < slab / 2
        if sel.sum() < min_slice:
            c2 = c + tan * step  # hop a gap (horses / mask hole)
            if (np.abs((pts - c2) @ tan) < slab / 2).sum() < min_slice:
                break
            c = c2
            continue
        wcoord = d[sel] @ perp
        wl, wr = np.percentile(wcoord, 1), np.percentile(wcoord, 99)
        mid = c + perp * (wl + wr) / 2
        centers.append(mid)
        left.append(c + perp * wl)
        right.append(c + perp * wr)
        widths.append(wr - wl)
        c = mid + tan * step
        if len(centers) >= 4:  # update tangent from recent centers (smoothed)
            tn = np.array(centers[-1]) - np.array(centers[-4])
            n = np.linalg.norm(tn)
            if n > 1e-3:
                tan = 0.5 * tan + 0.5 * tn / n
                tan /= np.linalg.norm(tan)
        if not (0 <= c[0] < W and 0 <= c[1] < H):
            break
    if len(centers) < 6:
        return None
    return (np.array(left), np.array(right), np.array(centers),
            np.array(widths), axis, _rot90(axis), mean)

def band_borders(band):  # back-compat for the overlay driver
    r = trace_ribbon(band)
    if r is None:
        return None
    left, right, centers, widths, axis, perp, mean = r
    return left, right, axis, perp, mean

def _pca(ys, xs):
    p = np.c_[xs, ys].astype(np.float32)
    mean = p.mean(0)
    _, s, vt = np.linalg.svd(p - mean, full_matrices=False)
    direc = vt[0]
    proj = (p - mean) @ direc
    extent = float(proj.max() - proj.min())          # true major-axis length (px)
    width = len(xs) / max(extent, 1.0)               # mean thickness (px) — a rail is THIN
    return mean, direc, s[0] / max(s[1], 1e-3), extent, width

def _rail_pixels(bgr, band, tophat):
    """Interior rail pixels. white = bright thin lines (V>175,S<60); tophat = thin BRIGHT
    structures regardless of absolute brightness (recovers low-contrast/shadowed rail, user steer)
    — only added on demand because it also adds distractor edges that confuse pairing."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    px = (hsv[..., 2] > 175) & (hsv[..., 1] < 60)
    green_in = cv2.erode((band > 0).astype(np.uint8), np.ones((_ksz(9), _ksz(9)), np.uint8)) > 0
    if tophat:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        th = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT,
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_ksz(15), _ksz(15))))
        if green_in.any():
            px = px | (th > max(12.0, th[green_in].mean() + 2 * th[green_in].std()))
    wi = (px & green_in).astype(np.uint8) * 255
    return cv2.morphologyEx(wi, cv2.MORPH_CLOSE, np.ones((_ksz(5), _ksz(5)), np.uint8))

def enhance(bgr):
    """CLAHE on L (luminance) — recovers the low-contrast rails on the dark, under-lit far turn
    (validated: t=70 raw 0 candidates -> both rails; t=66/72 second rail recovered). Hue/sat
    preserved, so the green-segmentation + white thresholds are unaffected on bright frames."""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    lo, a, b = cv2.split(lab)
    lo = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(lo)
    return cv2.cvtColor(cv2.merge([lo, a, b]), cv2.COLOR_LAB2BGR)

def rail_candidates(bgr, band=None, tophat=False, min_ratio=4.0, clahe=True):
    """Elongated interior rail curves (inner rail green-on-both-sides + outer rail), longest first.
    Length/area/width thresholds scale with SCALE (resolution-independent); CLAHE recovers dark rails."""
    gh = None
    if clahe:
        bgr = enhance(bgr)
    if band is None:
        mask, gh = turf_mask(bgr)
        band = largest_band(mask)
    if band is None:
        return [], None, gh
    min_pts = 120 * SCALE * SCALE
    min_len = 80.0 * SCALE
    max_width = 14.0 * SCALE
    wi = _rail_pixels(bgr, band, tophat)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(wi, 8)
    cands = []
    for k in range(1, n):
        ys, xs = np.where(lab == k)
        if len(xs) < min_pts:
            continue
        c, d, ratio, extent, width = _pca(ys, xs)
        if ratio > min_ratio and extent > min_len and width < max_width:
            cands.append({"k": k, "xs": xs, "ys": ys, "centroid": c, "dir": d,
                          "ratio": ratio, "len": extent, "width": width})
    cands.sort(key=lambda z: -z["len"])
    return cands, band, gh

def fit_model(xs, ys):
    """Fit a rail curve to a straight LINE or a circular ARC (track is straight or known-R bend).
    Returns dict with type, params, residual, and a sample() that returns ordered curve points."""
    x = xs.astype(np.float64)
    y = ys.astype(np.float64)
    c = np.array([x.mean(), y.mean()])
    _, _, vt = np.linalg.svd(np.c_[x - c[0], y - c[1]], full_matrices=False)
    d = vt[0]
    nrm = _rot90(d)
    proj = np.c_[x - c[0], y - c[1]] @ d
    line_res = float(np.sqrt(np.mean((np.c_[x - c[0], y - c[1]] @ nrm) ** 2)))
    # circle (Kasa)
    A = np.c_[x, y, np.ones_like(x)]
    bb = x * x + y * y
    sol, *_ = np.linalg.lstsq(A, bb, rcond=None)
    cx, cy = sol[0] / 2, sol[1] / 2
    R = float(np.sqrt(max(sol[2] + cx * cx + cy * cy, 1e-6)))
    rad = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    circ_res = float(np.sqrt(np.mean((rad - R) ** 2)))
    use_arc = (R < 4000 * SCALE) and (circ_res < 0.7 * line_res)
    if use_arc:
        ang = np.arctan2(y - cy, x - cx)
        a0, a1 = ang.min(), ang.max()
        if a1 - a0 > np.pi:  # wrap
            ang2 = np.where(ang < 0, ang + 2 * np.pi, ang)
            a0, a1 = ang2.min(), ang2.max()
        def sample(npix=60, _a0=a0, _a1=a1):
            a = np.linspace(_a0, _a1, npix)
            return np.c_[cx + R * np.cos(a), cy + R * np.sin(a)].astype(np.float32)
        return {"type": "arc", "center": np.array([cx, cy]), "R": R, "res": circ_res,
                "sample": sample, "extent": float(R * (a1 - a0))}
    lo, hi = proj.min(), proj.max()
    def sample(npix=60, _lo=lo, _hi=hi, _c=c, _d=d):
        t = np.linspace(_lo, _hi, npix)
        return (_c + np.outer(t, _d)).astype(np.float32)
    return {"type": "line", "centroid": c, "dir": d, "res": line_res,
            "sample": sample, "extent": float(hi - lo)}

def pair_rails(cands, band, gh):
    """Pick the two curves that form a track: parallel LINES or concentric ARCS with a plausible
    track-width offset. Returns (inner, outer, info) with smooth fitted models; labels via
    'what lies beyond the rail' (green=infield=inner, clutter=outer) cross-checked by arc radius."""
    if len(cands) < 1:
        return None
    for c in cands:
        c["model"] = fit_model(c["xs"], c["ys"])
    top = cands[:6]
    best = None
    for i in range(len(top)):
        for j in range(i + 1, len(top)):
            a, b = top[i]["model"], top[j]["model"]
            if a["type"] == "arc" and b["type"] == "arc":
                csep = np.linalg.norm(a["center"] - b["center"])
                off = abs(a["R"] - b["R"])
                ok = csep < 0.25 * min(a["R"], b["R"]) and 40 * SCALE < off < 700 * SCALE
                geom = "arc"
                sep = off
            elif a["type"] == "line" and b["type"] == "line":
                cosang = abs(float(a["dir"] @ b["dir"]))
                sep = abs(float((b["centroid"] - a["centroid"]) @ _rot90(a["dir"])))
                ok = cosang > 0.92 and 40 * SCALE < sep < 700 * SCALE
                geom = "line"
            else:
                ok = False
            if ok:
                score = top[i]["len"] + top[j]["len"]
                if best is None or score > best[0]:
                    best = (score, top[i], top[j], sep, geom)
    if best is None:
        return None
    _, ca, cb, sep, geom = best
    # inner = the rail with infield (green) on its far side; outer = clutter on far side
    def beyond_is_green(c, toward):
        m = c["model"]
        p = m["sample"](12).mean(0)
        if m["type"] == "arc":
            out = p - m["center"]
            out = out / (np.linalg.norm(out) + 1e-6)
        else:
            out = _rot90(m["dir"])
        probe = (p + toward * out * 30 * SCALE).astype(int)
        H, W = band.shape
        if not (0 <= probe[1] < H and 0 <= probe[0] < W):
            return None
        return band[probe[1], probe[0]] > 0
    # decide orientation per rail: which perpendicular side is green
    def classify(c):
        for s in (+1, -1):
            g = beyond_is_green(c, s)
            if g is True:
                return "inner"  # green beyond -> infield side -> inner rail
        return "outer"
    la, lb = classify(ca), classify(cb)
    if geom == "arc":  # arc radius is the reliable tiebreak (inner = smaller R = toward centre)
        inner, outer = (ca, cb) if ca["model"]["R"] < cb["model"]["R"] else (cb, ca)
    elif la != lb:
        inner, outer = (ca, cb) if la == "inner" else (cb, ca)
    else:  # fallback: keep order
        inner, outer = ca, cb
    return {"inner": inner, "outer": outer, "sep": sep, "geom": geom}


def _force_arc(c):
    """Circle (Kasa) fit to a candidate's pixels regardless of its line/arc label; returns an arc
    model + fit quality. Used on bends where a short fragment may have been labelled a line."""
    x = c["xs"].astype(np.float64)
    y = c["ys"].astype(np.float64)
    A = np.c_[x, y, np.ones_like(x)]
    sol, *_ = np.linalg.lstsq(A, x * x + y * y, rcond=None)
    cx, cy = sol[0] / 2, sol[1] / 2
    R = float(np.sqrt(max(sol[2] + cx * cx + cy * cy, 1e-6)))
    center = np.array([cx, cy])
    ang = np.arctan2(y - cy, x - cx)
    a0, a1 = ang.min(), ang.max()
    if a1 - a0 > np.pi:
        ang2 = np.where(ang < 0, ang + 2 * np.pi, ang)
        a0, a1 = ang2.min(), ang2.max()
    rad = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    res = float(np.sqrt(np.mean((rad - R) ** 2)))
    return {"center": center, "R": R, "res": res, "span": float(a1 - a0),
            "a0": a0, "a1": a1, "len": c["len"]}

def concentric_fill(cands, band, width_px, expected_R_px):
    """GEOMETRY-CONSTRAINED bend fill (user steer): the two rails are CONCENTRIC arcs separated by
    the track width. Fit the cleanest surviving rail as an arc, then build its partner as a
    concentric arc offset by `width_px`. inner/outer by a radial green probe (centre of curvature is
    on the infield side). Rejects stub fits (poor arc / implausible R) so it can't fabricate garbage.
    Returns {'inner','outer','sep'} or None (-> caller should hold)."""
    arcs = []
    for c in cands:
        a = _force_arc(c)
        plausible = 0.45 * expected_R_px < a["R"] < 2.2 * expected_R_px
        good = a["res"] < 6.0 * SCALE and a["span"] > 0.12 and a["len"] > 90 * SCALE
        if plausible and good:
            arcs.append(a)
    if not arcs:
        return None
    base = max(arcs, key=lambda a: a["span"] * a["len"])   # the most-supported arc
    center = base["center"]
    ang = np.linspace(base["a0"], base["a1"], 80)

    def arc_model(Rr):
        pts = np.c_[center[0] + Rr * np.cos(ang), center[1] + Rr * np.sin(ang)].astype(np.float32)
        return {"type": "arc", "center": center, "R": Rr, "synth": True,
                "sample": lambda nn=80, _p=pts: _p}

    # is `base` the inner or outer rail? probe radially OUTWARD (away from centre of curvature):
    # green beyond -> track is outside it -> base is INNER; non-green (clutter) -> base is OUTER.
    mid = np.array([center[0] + base["R"] * np.cos(ang[len(ang) // 2]),
                    center[1] + base["R"] * np.sin(ang[len(ang) // 2])])
    out = (mid - center)
    out = out / (np.linalg.norm(out) + 1e-6)
    probe = (mid + out * 25 * SCALE).astype(int)
    H, W = band.shape
    base_is_inner = (0 <= probe[1] < H and 0 <= probe[0] < W and band[probe[1], probe[0]] > 0)
    if base_is_inner:
        inner, outer = arc_model(base["R"]), arc_model(base["R"] + width_px)
    else:
        inner, outer = arc_model(base["R"] - width_px), arc_model(base["R"])
    return {"inner": {"model": inner}, "outer": {"model": outer}, "sep": width_px, "geom": "arc"}

def analyze(bgr):
    cands, band, gh = rail_candidates(bgr)              # white-only first (clean)
    res = pair_rails(cands, band, gh) if band is not None else None
    used_tophat = False
    if band is not None and res is None:               # fallback: recover low-contrast rail
        c2, _, _ = rail_candidates(bgr, band=band, tophat=True)
        r2 = pair_rails(c2, band, gh)
        if r2 is not None:
            cands, res, used_tophat = c2, r2, True
    if res is not None:
        res["tophat"] = used_tophat
    return cands, band, res

def _curve(ov, model, color, w=3):
    pts = model["sample"](80)
    cv2.polylines(ov, [pts.astype(np.int32)], False, color, w)

def _draw(bgr, cands, res, t, big=False):
    ov = (bgr // 2).copy()
    for c in cands:                       # raw rail pixels = faint orange
        ov[c["ys"], c["xs"]] = (0, 120, 180)
    if res is not None:
        im, om = res["inner"]["model"], res["outer"]["model"]
        ip, op = im["sample"](80), om["sample"](80)
        # shade the track region between the two rails (light yellow)
        poly = np.vstack([ip, op[::-1]]).astype(np.int32)
        sh = ov.copy()
        cv2.fillPoly(sh, [poly], (40, 90, 90))
        ov = cv2.addWeighted(ov, 0.8, sh, 0.2, 0)
        cv2.polylines(ov, [op.astype(np.int32)], False, (255, 0, 255), 3)   # OUTER magenta
        cv2.polylines(ov, [ip.astype(np.int32)], False, (255, 255, 0), 3)   # INNER cyan
        cv2.putText(ov, "IN", tuple(ip[len(ip) // 2].astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 3)
        cv2.putText(ov, "OUT", tuple(op[len(op) // 2].astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 255), 3)
        def tag(m):
            return f"arc R={m['R']:.0f}" if m["type"] == "arc" else "line"
        cv2.putText(ov, f"t={t:.0f}s [{res['geom']}] IN={tag(im)} OUT={tag(om)} sep={res['sep']:.0f}px",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(ov, "INNER=cyan(IN)  OUTER=magenta(OUT)  track=shaded", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    else:
        cv2.putText(ov, f"t={t:.0f}s PAIR-FAIL nc={len(cands)}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    return ov if big else cv2.resize(ov, (480, 270))

if __name__ == "__main__":
    import av
    cont = av.open(VIDEO)
    st = cont.streams.video[0]
    # (1) dense stats over the race segment (every 2s)
    STRIDE, END = 50, 2833
    # (2) a hand-picked high-res panel spanning straights + the bend for visual co-review
    panel_t = [4, 18, 30, 42, 54, 66, 72, 78, 90, 96, 104, 110]
    panel_f = {int(t * 25): t for t in panel_t}
    i = -1
    tiles = []
    ok = 0
    tot = 0
    seps = []
    fails = []
    panels = {}
    for fr in cont.decode(st):
        i += 1
        if i > END:
            break
        do_dense = (i % STRIDE == 0)
        do_panel = i in panel_f
        if not (do_dense or do_panel):
            continue
        bgr = fr.to_ndarray(format='bgr24')
        cands, band, res = analyze(bgr)
        t = i / 25.0
        if do_dense:
            tot += 1
            if res is not None:
                ok += 1
                seps.append(res["sep"])
            else:
                fails.append(round(t, 1))
            tiles.append(_draw(bgr, cands, res, t))
        if do_panel:
            panels[panel_f[i]] = _draw(bgr, cands, res, t, big=True)
    cols = 6
    rows = [np.hstack(tiles[r:r + cols] + [np.zeros_like(tiles[0])] * (cols - len(tiles[r:r + cols])))
            for r in range(0, len(tiles), cols)]
    cv2.imwrite("data_cache/diag/drone_scan/dense_rails.png", np.vstack(rows))
    # high-res review panel
    pt = [cv2.resize(panels[t], (620, 349)) for t in panel_t if t in panels]
    prows = [np.hstack(pt[r:r + 3] + [np.zeros_like(pt[0])] * (3 - len(pt[r:r + 3])))
             for r in range(0, len(pt), 3)]
    cv2.imwrite("data_cache/diag/drone_scan/rail_review.png", np.vstack(prows))
    print(f"frames {tot}  2-rail OK {ok} ({100*ok/tot:.0f}%)")
    print(f"sep px: med {np.median(seps):.0f}  iqr {np.percentile(seps,75)-np.percentile(seps,25):.0f}" if seps else "no picks")
    print(f"FAIL times (s): {fails}")
