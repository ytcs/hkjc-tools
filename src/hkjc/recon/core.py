"""Stage A core: render detections + race facts -> labelled paths, one objective (RECON 4, identify / fuse).

The unit of identity is the SEGMENT: a piece of a tracker tracklet between cuts (stitched gaps, lateral jumps,
encounters with another tracklet's end, render camera transitions). Tracker continuity across a cut is a LINK whose
reward enters the objective; per-frame exclusivity holds between segments (one segment per horse per frame).

Latent: the label z_k of every segment (a runner or clutter), every runner's along-track position s_h(t) and lateral
lat_h(t) on the 0.1 s grid (seen or not), and the fault states of the running-order list (a list-level frozen state,
a per-horse omission state). One objective, J = -2 log p(labels, observations, list) with the paths integrated out
(Laplace at the joint mode):

  J(z) = J_s + J_lat + J_list + sum over detections (horse prior | clutter cost) + broken links + reads + silk

  * J_s: every runner's s in ONE joint banded Gaussian (node-major, lower bandwidth 2H) under the physics prior, the
    observations of its segments (chunk means) and the list's pairwise order factors; mode by Newton with line
    search, the factor weights from the fault chains (EM); value at the mode + log det of the Laplace Hessian.
  * J_lat: per runner, the lateral smoother (box-bounded) at its mode + log det.
  * J_list: -2 log of the list's fault-state chains (forward algorithm) at the mode.

The assignment step is one MILP over segments x runners (per-frame exclusivity, links) whose unary costs are J's
increments linearised at the current state: the block predictive of the segment's chunks against each runner's
leave-segment-out posterior, the list's rank-count vote (identity-free counts of the horses ahead, weighted by each
detection's P(horse) and each undetected runner's posterior), the per-detection horse/clutter prior, disc reads and
the silk term. A proposal is kept only when the state step finds J lower; the loop ends when J stops falling.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.linalg import cho_solve_banded, cholesky_banded
from scipy.special import ndtr

DT = 0.1
# detections
SD_S, SD_L = 0.8, 0.35  # m, render units: per-detection white noise
CHUNK = 0.5  # s: a segment's detections are summarised by their mean per CHUNK
SD_S_COMMON, SD_L_COMMON = 0.25, 0.15  # correlated part of a chunk mean's error (not averaged away)
CLUTTER = 16.0  # -2 log: a detection of P(horse) = P_HORSE0 left as clutter, against a perfect fit
P_HORSE0 = 0.97
P_GHOST = 0.01  # P(horse) of a short-lived low-confidence box in an overlay zone (legend band, corner, text, edge)
GHOST_TRACK, GHOST_CONF, GHOST_TEXT_SIZE = 50, 0.75, 0.70
GHOST_EDGE_X, GHOST_EDGE_LAT, GHOST_CORNER_X = 40.0, 6.0, 60.0
LEGEND_BAND = 62  # px below the legend top occupied by number boxes and silk icons
# known occluders (the list band at the panel bottom, the disclaimer text at the top): a box overlapping one is a
# horse partly hidden or an icon / glyph; tracker continuity across an occluder's edge is weak
P_EDGE = 0.5  # P(horse) of a box overlapping an occluder band (not otherwise a ghost)
EDGE_PX = 10.0  # px: a box within this of an occluder band is at its edge
# segments
CUT_GAP = 1.0  # s: a stitched gap inside a tracklet
CUT_JUMP_T, CUT_JUMP_LAT = 0.2, 0.6  # s, units: a lateral jump between consecutive detections
ENC_T, ENC_S, ENC_LAT = 0.2, 3.0, 1.5  # another tracklet starts or ends here (s, m, units)
CAM_RESID = 1.0  # px: a sequential pose step with a larger residual (or none) is a render camera transition
CAM_SD_S, CAM_SD_L = 1.0, 0.5  # extra chunk sd within CAM_WIN of a camera transition
CAM_WIN = 0.3  # s
LINK_TRACK, LINK_WEAK, LINK_CROSS = 4.0, 1.0, 2.0  # -2 log link rewards: tracker continuity / jump or camera / cross
CROSS_GAP, CROSS_S, CROSS_LAT = 1.0, 3.0, 1.0
# physics prior (as fusion.Prior)
A_SD_S, A_SD_START, T_START, A_SD_L, VL_SD, L_MIN = 1.2, 6.0, 6.0, 0.25, 2.0, 0.3
S0_SD, V0, V0_SD = 100.0, 16.0, 8.0  # proper prior at the grid's first node (m, m/s, m/s)
# running-order list
LAG = 0.3  # s: the list shows the order 0.3 s before its render tick
SIG_AB = 1.0  # m: order noise of two horses abreast
EPS_LIST = 0.02  # iid misread / wrong-slot probability of one statement
P_FROZEN_IN, P_FROZEN_STAY = 1e-3, 0.995  # per list row
P_OMIT_IN, P_OMIT_STAY = 1e-4, 0.999  # per list row, per horse
P_AHEAD_UNSEEN = 0.25  # start solve: P(an undetected horse is ahead of a detection) before any path exists
MAX_RANK = 8
# detection: frame-level probability that an in-view horse is boxed, from the frame's own count
P_DET_MIN = 0.05
# ceiling of a frame's detection probability: the detector's measured recall of visible horses (0330): silk 0.95,
# 0.86 on bends (small vertical / diagonal avatars), disc 0.985
P_DET_MAX = {("silk", False): 0.95, ("silk", True): 0.86, ("disc", False): 0.985, ("disc", True): 0.985}
VIEW_PX = 5.0  # px: projection uncertainty floor of the in-view test
# disc reads (circle render)
READ_ERR, READ_ERR_TOUCH, READ_ERR_DROP, TOUCH_PX = 0.01, 0.10, 0.25, 50.0
INF = 1e6


# ------------------------------------------------------------------------------------------------------------------
# evidence (label-free)


@dataclass
class Evidence:
    horses: np.ndarray  # runner numbers (columns)
    fr: np.ndarray  # per detection: render frame
    t: np.ndarray
    s: np.ndarray  # metres run
    lat: np.ndarray
    p_horse: np.ndarray
    seg: np.ndarray  # per detection: segment index
    n_seg: int
    links: np.ndarray  # (L, 3): segment a, segment b, reward
    seg_lo: np.ndarray  # first / last frame of each segment
    seg_hi: np.ndarray
    chunk: np.ndarray  # per detection: chunk index
    ch_seg: np.ndarray  # per chunk
    ch_t: np.ndarray
    ch_s: np.ndarray
    ch_l: np.ndarray
    ch_sds: np.ndarray
    ch_sdl: np.ndarray
    ch_scat: np.ndarray  # within-chunk chi^2 (s and lat) about the chunk mean
    cam: np.ndarray  # per detection: within CAM_WIN of a camera transition
    read_cost: np.ndarray | None = None  # (S, H) disc reads, -2 log likelihood ratio vs a uniform read
    silk_cost: np.ndarray | None = None  # (S, H) -2 x silk LLR
    forbid: np.ndarray | None = None  # (S, H) bool: hard exclusions
    extra: dict = field(default_factory=dict)
    view: dict = field(default_factory=dict)  # per render frame: local (s, lat) -> panel px map, panel, occluders

    @property
    def horse_prior(self):
        return -2.0 * np.log(self.p_horse / P_HORSE0)

    @property
    def clutter(self):
        return CLUTTER - 2.0 * np.log((1.0 - self.p_horse) / (1.0 - P_HORSE0))


def legend_top(cy, panel_h: float) -> float:
    """The silk legend's top row from the detections alone: the start of the largest box-free row interval >= 50 px in
    the panel's lower part (the detector drops boxes in the icon band); 0.8 x the panel height when none."""
    lo, hi = 0.6 * panel_h, 0.97 * panel_h
    cy = np.asarray(cy, float)
    ys = np.sort(np.r_[cy[(cy >= lo) & (cy <= hi)], lo, hi])
    gap = np.diff(ys)
    i = int(np.argmax(gap))
    return float(ys[i]) if gap[i] >= 50 else 0.8 * panel_h


def occluder_edge(rows, panel_h: float, silk: bool = True):
    """Per detection: (box overlaps an occluder band, box within EDGE_PX of one). Occluders: the disclaimer text band
    at the top and, on the silk render, the list band (from ``legend_top``) at the bottom."""
    from .video import DISCLAIMER_ROWS

    rows = np.asarray(rows, float)
    top, bot = rows[:, 2] - rows[:, 4] / 2, rows[:, 2] + rows[:, 4] / 2
    over = top < DISCLAIMER_ROWS
    near = top < DISCLAIMER_ROWS + EDGE_PX
    if silk and len(rows):
        lt = legend_top(rows[:, 2], panel_h)
        over |= bot > lt
        near |= bot > lt - EDGE_PX
    return over, near


def p_horse(rows, lat, trk_len, panel_h: float, silk: bool = True) -> np.ndarray:
    """Per detection (``rows``: [frame, cx, cy, w, h, conf] in panel px): P(the box is a horse). P_GHOST for a
    short-lived (tracklet < GHOST_TRACK) box in an overlay zone: legend icon band or the dark corner beside it below
    GHOST_CONF (silk render), the disclaimer band at glyph size, or the left edge far outside its frame's field;
    P_EDGE for any other box overlapping an occluder band (``occluder_edge``); else P_HORSE0."""
    from .video import DISCLAIMER_ROWS

    rows = np.asarray(rows, float)
    lat = np.asarray(lat, float)
    zone = np.zeros(len(rows), bool)
    if len(rows):
        cx, cy, cf = rows[:, 1], rows[:, 2], rows[:, 5]
        size = np.maximum(rows[:, 3], rows[:, 4])
        _u, inv = np.unique(rows[:, 0].astype(int), return_inverse=True)
        med = np.bincount(inv, lat) / np.bincount(inv)
        zone |= (cx < GHOST_EDGE_X) & (lat - med[inv] > GHOST_EDGE_LAT)
        zone |= (cy < DISCLAIMER_ROWS) & (size < GHOST_TEXT_SIZE * np.median(size))
        if silk:
            lt = legend_top(cy, panel_h)
            band = cy <= lt + LEGEND_BAND - 4
            zone |= (cf < GHOST_CONF) & (((cy >= lt + 4) & band) | ((cx < GHOST_CORNER_X) & ~band & (cy > lt)))
    zone &= np.asarray(trk_len) < GHOST_TRACK
    over, _near = occluder_edge(rows, panel_h, silk) if len(rows) else (zone, zone)
    return np.where(zone, P_GHOST, np.where(over, P_EDGE, P_HORSE0))


def camera_transitions(pose, frames) -> np.ndarray:
    """Render frames (sorted ``frames``) that start a camera transition: the sequential pose step into them is missing
    or has a residual > CAM_RESID px."""
    idx = np.asarray(pose["idx"])
    e, kinds, res = pose["edges"], pose["kinds"], pose["resid"]
    ok = {}
    for (a, b, _w), k, r in zip(e, kinds, res):
        if k == "seq":
            ok[int(idx[int(b)])] = r <= CAM_RESID
    frames = np.asarray(frames)
    return np.array([f for f in frames[1:] if not ok.get(int(f), False)], int)


def segments(t, s, lat, fr, members, cam_frames, edge=None):
    """Cut tracker tracklets into segments at stitched gaps, lateral jumps, encounters, render camera transitions and
    where a tracklet enters or leaves an occluder's edge (``edge`` per detection). -> (seg per detection, links)."""
    seg = np.full(len(t), -1)
    mem = [np.asarray(m, int)[np.argsort(t[np.asarray(m, int)])] for m in members if len(m)]
    ends = [(t[m[e]], s[m[e]], lat[m[e]], i) for i, m in enumerate(mem) for e in (0, -1)]
    cam = set(int(f) for f in cam_frames)
    pieces, links = [], []
    for i, m in enumerate(mem):
        cut = {}  # position in m where a new segment starts -> link reward (None: no link)
        dt, dl = np.diff(t[m]), np.abs(np.diff(lat[m]))
        for j in np.where(dt > CUT_GAP)[0]:
            cut[j + 1] = LINK_TRACK
        for j in np.where((dt <= CUT_JUMP_T) & (dl > CUT_JUMP_LAT))[0]:
            cut[j + 1] = LINK_WEAK
        for j in range(1, len(m)):
            if int(fr[m[j]]) in cam or (edge is not None and edge[m[j]] != edge[m[j - 1]]):
                cut[j] = min(cut.get(j, LINK_WEAK), LINK_WEAK)
        for te, se, le, o in ends:
            if o == i or te <= t[m[0]] or te >= t[m[-1]]:
                continue
            j = int(np.searchsorted(t[m], te))
            jj = j if abs(t[m[j]] - te) < abs(t[m[j - 1]] - te) else j - 1
            if abs(t[m[jj]] - te) <= ENC_T and abs(s[m[jj]] - se) < ENC_S and abs(lat[m[jj]] - le) < ENC_LAT:
                cut.setdefault(j, LINK_TRACK)
        bnd = sorted(cut)
        first = len(pieces)
        pieces.extend(np.split(m, bnd))
        links.extend((first + q, first + q + 1, cut[b]) for q, b in enumerate(bnd))
    for k, p in enumerate(pieces):
        seg[p] = k
    # cross links: the end of one segment continues as the start of another (<= CROSS_GAP s, kinematic match)
    have = {(a, b) for a, b, _ in links}
    st = np.array([[t[p[0]], s[p[0]], lat[p[0]]] for p in pieces])
    for a, p in enumerate(pieces):
        ta, sa, la = t[p[-1]], s[p[-1]], lat[p[-1]]
        k = p[-min(10, len(p)):]
        v = np.polyfit(t[k], s[k], 1)[0] if len(k) >= 3 and np.ptp(t[k]) > 0 else 16.0
        g = st[:, 0] - ta
        cand = np.where((g >= 0) & (g <= CROSS_GAP) & (np.abs(st[:, 1] - (sa + v * g)) < CROSS_S)
                        & (np.abs(st[:, 2] - la) < CROSS_LAT))[0]
        for b in cand:
            if b != a and (a, b) not in have:
                links.append((a, int(b), LINK_CROSS))
    return seg, np.array(links, float).reshape(-1, 3)


def chunks(t, s, lat, seg, cam):
    """Per segment, its detections' means per CHUNK s. -> chunk id per detection and the chunk table."""
    key = np.c_[seg, np.floor((t - t.min()) / CHUNK).astype(int)]
    _u, cid = np.unique(key, axis=0, return_inverse=True)
    cid = cid.ravel()
    nc = int(cid.max()) + 1 if len(cid) else 0
    n = np.bincount(cid, minlength=nc).astype(float)
    mt, ms, ml = (np.bincount(cid, x, nc) / n for x in (t, s, lat))
    scat = np.bincount(cid, (s - ms[cid]) ** 2 / SD_S**2 + (lat - ml[cid]) ** 2 / SD_L**2, nc)
    camc = np.bincount(cid, cam.astype(float), nc) > 0
    sds = np.sqrt(SD_S**2 / n + SD_S_COMMON**2 + np.where(camc, CAM_SD_S**2, 0.0))
    sdl = np.sqrt(SD_L**2 / n + SD_L_COMMON**2 + np.where(camc, CAM_SD_L**2, 0.0))
    chs = np.zeros(nc, int)
    chs[cid] = seg
    return cid, dict(ch_seg=chs, ch_t=mt, ch_s=ms, ch_l=ml, ch_sds=sds, ch_sdl=sdl, ch_scat=scat)


def read_costs(seg, n_seg, horses, nums, touch):
    """(segments x runners) disc-read term: per confident read r on a detection, -2 log P(r | horse) + 2 log(1/N),
    error READ_ERR (READ_ERR_TOUCH where another disc is within TOUCH_PX), READ_ERR_DROP between a number and its
    dropped-digit partner (1 <-> 11)."""
    H = len(horses)
    out = np.zeros((n_seg, H))
    j = np.flatnonzero(nums > 0)
    if not len(j):
        return out
    err = np.where(touch[j], READ_ERR_TOUCH, READ_ERR)
    for c, h in enumerate(horses):
        r = nums[j]
        drop = np.array([str(h) != str(x) and (str(h) in str(x) or str(x) in str(h)) for x in r])
        p = np.where(r == h, 1.0 - err, np.where(drop, READ_ERR_DROP, err / max(H - 1, 1)))
        np.add.at(out[:, c], seg[j], -2.0 * np.log(p * H))
    return out


def silk_costs(seg, n_seg, horses, feat, proto_no, proto):
    """(segments x runners) -2 x the silk log-likelihood ratio of each segment's mean render signature against each
    runner's mapped official prototype (``silkmap``), segments with >= 10 silk features."""
    from .silkmap import SILK_LLR_A, SILK_LLR_B, SILK_LLR_CLIP, SILK_MAP

    def collapse(a):
        nh = (a.shape[-1] - 2) // 2
        return np.concatenate([a[..., :nh] + a[..., nh:2 * nh], a[..., 2 * nh:]], -1)

    Bm = np.asarray(SILK_MAP)
    q = {}
    for h, p in zip(proto_no, proto):
        v = np.clip(np.r_[collapse(np.asarray(p, float)), 1.0] @ Bm, 1e-9, None)
        q[int(h)] = v / v.sum()
    out = np.zeros((n_seg, len(horses)))
    F = np.asarray(feat, float)
    ok = F.sum(1) > 0
    for k in range(n_seg):
        f = F[(seg == k) & ok]
        if len(f) < 10:
            continue
        sig = collapse((f / f.sum(1, keepdims=True)).mean(0))
        for c, h in enumerate(horses):
            if int(h) in q:
                tv = 0.5 * np.abs(q[int(h)] - sig).sum()
                out[k, c] = -2.0 * np.clip(SILK_LLR_A + SILK_LLR_B * tv, -SILK_LLR_CLIP, SILK_LLR_CLIP)
    return out


def view_maps(rows, t, s, lat, ph, panel_h, panel_w, silk_style, bend=None):
    """Per render frame with detections: a local affine map (s, lat) -> panel (cx, cy) fitted to the frame's horse
    boxes (P(horse) >= 0.5; >= 4 of them, else the nearest frame's map), the frame's box count, and the visible
    panel rows (below the disclaimer text, above the list band on the silk render)."""
    from .video import DISCLAIMER_ROWS

    fr = rows[:, 0].astype(int)
    F, inv = np.unique(fr, return_inverse=True)
    A = np.full((len(F), 3, 2), np.nan)
    good = ph >= 0.5
    for i in range(len(F)):
        j = np.flatnonzero((inv == i) & good)
        if len(j) < 4:
            continue
        M = np.c_[s[j] - s[j].mean(), lat[j] - lat[j].mean(), np.ones(len(j))]
        if np.linalg.matrix_rank(M) < 3 or np.ptp(s[j]) < 2.0 or np.ptp(lat[j]) < 0.5:
            continue
        sol, *_ = np.linalg.lstsq(M, rows[j, 1:3], rcond=None)
        A[i] = sol
        A[i, 2] -= s[j].mean() * sol[0] + lat[j].mean() * sol[1]  # absolute origin
    ok = np.flatnonzero(np.isfinite(A[:, 0, 0]))
    if len(ok):
        near = ok[np.clip(np.searchsorted(ok, np.arange(len(F))), 0, len(ok) - 1)]
        prev = ok[np.clip(np.searchsorted(ok, np.arange(len(F))) - 1, 0, len(ok) - 1)]
        pick = np.where(np.abs(prev - np.arange(len(F))) < np.abs(near - np.arange(len(F))), prev, near)
        A = A[pick]
    tf = np.bincount(inv, t) / np.bincount(inv)
    y_lo = float(DISCLAIMER_ROWS)
    y_hi = legend_top(rows[:, 2], panel_h) if silk_style else float(panel_h)
    on_bend = np.zeros(len(F), bool)
    if bend is not None:
        on_bend = np.bincount(inv, np.asarray(bend, float), len(F)) / np.bincount(inv, minlength=len(F)) > 0.5
    style = "silk" if silk_style else "disc"
    cap = np.where(on_bend, P_DET_MAX[(style, True)], P_DET_MAX[(style, False)])
    return dict(frames=F, t=tf, A=A if len(ok) else None, n=np.bincount(inv, good.astype(float), len(F)),
                x=(0.0, float(panel_w)), y=(y_lo, y_hi), cap=cap)


def in_view(ev: Evidence, S, L, sdS, sdL, tg):
    """(H, F) P(horse h's position projects into the visible panel at frame f), from its posterior mean and sd."""
    V = ev.view
    if not V or V["A"] is None:
        return np.zeros((S.shape[0], 0))
    k, w = _interp(tg, V["t"])

    def at(X):
        return X[:, k] * (1 - w) + X[:, k + 1] * w

    s_, l_, ss, sl = at(S), at(L), at(sdS), at(sdL)
    A = V["A"]
    cx = s_ * A[None, :, 0, 0] + l_ * A[None, :, 1, 0] + A[None, :, 2, 0]
    cy = s_ * A[None, :, 0, 1] + l_ * A[None, :, 1, 1] + A[None, :, 2, 1]
    ux = np.hypot(ss * A[None, :, 0, 0], sl * A[None, :, 1, 0]) + VIEW_PX
    uy = np.hypot(ss * A[None, :, 0, 1], sl * A[None, :, 1, 1]) + VIEW_PX
    (x0, x1), (y0, y1) = V["x"], V["y"]
    return ndtr((cx - x0) / ux) * ndtr((x1 - cx) / ux) * ndtr((cy - y0) / uy) * ndtr((y1 - cy) / uy)


def miss_terms(ev: Evidence, pin, seg_lab):
    """Detection terms per (horse, frame): with p_f the frame's detection probability (its horse-box count over the
    horses projected in view, smoothed, in [P_DET_MIN, the render style's recall ceiling, lower on bends]), a horse
    labelled in the frame costs -2 log p_f and
    an unlabelled one -2 log(1 - p_f) times P(in view). -> (J term, p_f, per (H, F) cost if unlabelled, if labelled)."""
    V = ev.view
    if not pin.shape[1]:
        return 0.0, None, None, None
    p = np.clip((V["n"] + 1.0) / (pin.sum(0) + 2.0), P_DET_MIN, V["cap"])
    c_un = pin * (-2.0 * np.log(1.0 - p))[None, :]
    c_lab = np.broadcast_to(-2.0 * np.log(p)[None, :], pin.shape)
    labd = np.zeros(pin.shape, bool)
    fi = np.searchsorted(V["frames"], ev.fr)
    lab = seg_lab[ev.seg]
    m = lab >= 0
    labd[lab[m], fi[m]] = True
    return float(np.where(labd, c_lab, c_un).sum()), p, c_un, c_lab


def build_evidence(horses, T, members, pose, panel_h, silk_style, nums=None, silk=None, panel_w=None, bend=None):
    """Label-free evidence from observe's tracks (``T``: rows, t, d->s supplied as T['s'], lat), the tracker's
    tracklets, the pose graph, disc reads and silk features."""
    t, s, lat = np.asarray(T["t"], float), np.asarray(T["s"], float), np.asarray(T["lat"], float)
    rows = np.asarray(T["rows"], float)
    fr = rows[:, 0].astype(int)
    trk_len = np.zeros(len(t), int)
    for m in members:
        trk_len[np.asarray(m, int)] = len(m)
    ph = p_horse(rows, lat, trk_len, panel_h, silk_style)
    camf = camera_transitions(pose, np.unique(fr))
    _over, near = occluder_edge(rows, panel_h, silk_style)
    seg, links = segments(t, s, lat, fr, members, camf, near)
    lone = seg < 0  # detections in no tracklet: one segment each
    seg[lone] = seg.max() + 1 + np.arange(lone.sum())
    n_seg = int(seg.max()) + 1
    tcam = np.sort(np.unique(t[np.isin(fr, camf)]))
    if len(tcam):
        i = np.clip(np.searchsorted(tcam, t), 1, len(tcam)) - 1
        near = np.minimum(np.abs(t - tcam[i]), np.abs(t - tcam[np.minimum(i + 1, len(tcam) - 1)]))
        cam = near <= CAM_WIN
    else:
        cam = np.zeros(len(t), bool)
    cid, ch = chunks(t, s, lat, seg, cam)
    lo = np.full(n_seg, np.iinfo(np.int64).max)
    hi = np.full(n_seg, -1)
    np.minimum.at(lo, seg, fr)
    np.maximum.at(hi, seg, fr)
    ev = Evidence(np.asarray(horses, int), fr, t, s, lat, ph, seg, n_seg, links, lo, hi, cid, **ch, cam=cam)
    ev.view = view_maps(rows, t, s, lat, ph, panel_h, panel_w, silk_style, bend)
    H = len(horses)
    ev.read_cost = np.zeros((n_seg, H))
    if nums is not None:
        cx, cy = rows[:, 1], rows[:, 2]
        touch = np.zeros(len(t), bool)
        o = np.argsort(fr, kind="stable")
        b = np.r_[0, np.flatnonzero(np.diff(fr[o])) + 1, len(o)]
        for a, e in zip(b[:-1], b[1:]):
            j = o[a:e]
            if len(j) > 1:
                dd = np.hypot(cx[j][:, None] - cx[j][None, :], cy[j][:, None] - cy[j][None, :])
                np.fill_diagonal(dd, np.inf)
                touch[j] = dd.min(1) < TOUCH_PX
        ev.read_cost = read_costs(seg, n_seg, ev.horses, np.asarray(nums), touch)
    ev.silk_cost = np.zeros((n_seg, H)) if silk is None else silk_costs(seg, n_seg, ev.horses, *silk)
    ev.forbid = np.zeros((n_seg, H), bool)
    return ev


# ------------------------------------------------------------------------------------------------------------------
# the running-order list


@dataclass
class ListRows:
    tau: np.ndarray  # race time the row describes (render tick time - LAG)
    slots: np.ndarray  # (R, 8) runner column index, -1 unread / not a runner
    full: np.ndarray  # every slot the field fills read, distinct runners
    same: np.ndarray  # identical to the previous full row


def list_rows(leg, t_of_frame, horses) -> ListRows:
    """legend rows (frame, 8 slot numbers; masked reads are 0) -> ListRows on the race clock."""
    col = {int(h): c for c, h in enumerate(horses)}
    H = len(horses)
    need = min(MAX_RANK, H)
    if leg is None or not len(leg):
        z = np.zeros(0)
        return ListRows(z, np.zeros((0, MAX_RANK), int), z.astype(bool), z.astype(bool))
    leg = leg[np.argsort(leg[:, 0], kind="stable")]
    tau = t_of_frame(leg[:, 0]) - LAG
    sl = np.vectorize(lambda x: col.get(int(x), -1))(leg[:, 1:1 + MAX_RANK])
    sl = sl.reshape(len(leg), MAX_RANK)
    sl[:, need:] = -1
    rd = sl[:, :need] >= 0
    full = rd.all(1) & np.array([len(set(r[:need].tolist())) == need for r in sl])
    same = np.zeros(len(sl), bool)
    last = None
    for i in range(len(sl)):
        if full[i]:
            same[i] = last is not None and np.array_equal(sl[i], sl[last])
            last = i
    keep = rd.any(1)
    return ListRows(tau[keep], sl[keep], full[keep], same[keep])


def _hmm(log_e0, log_e1, p_in, p_stay):
    """Two-state chains (0 = ok, 1 = fault) starting in 0, one per row of ``log_e0`` / ``log_e1`` (C x n, or one chain
    as a 1-d array). -> (log Z per chain, P(fault) per chain and step)."""
    e0, e1 = np.atleast_2d(log_e0).astype(float), np.atleast_2d(log_e1).astype(float)
    C, n = e0.shape
    if not n:
        return (0.0, np.zeros(0)) if np.ndim(log_e0) == 1 else (np.zeros(C), np.zeros((C, 0)))
    l00, l01, l10, l11 = np.log(1 - p_in), np.log(p_in), np.log(1 - p_stay), np.log(p_stay)
    a0 = np.zeros((C, n))
    a1 = np.zeros((C, n))
    a0[:, 0] = e0[:, 0]
    a1[:, 0] = -690.0 + e1[:, 0]
    for i in range(1, n):
        a0[:, i] = np.logaddexp(a0[:, i - 1] + l00, a1[:, i - 1] + l10) + e0[:, i]
        a1[:, i] = np.logaddexp(a0[:, i - 1] + l01, a1[:, i - 1] + l11) + e1[:, i]
    b0 = np.zeros((C, n))
    b1 = np.zeros((C, n))
    for i in range(n - 2, -1, -1):
        f0, f1 = e0[:, i + 1] + b0[:, i + 1], e1[:, i + 1] + b1[:, i + 1]
        b0[:, i] = np.logaddexp(l00 + f0, l01 + f1)
        b1[:, i] = np.logaddexp(l10 + f0, l11 + f1)
    lz = np.logaddexp(a0[:, -1], a1[:, -1])
    post = np.clip(np.exp(a1 + b1 - lz[:, None]), 0.0, 1.0)
    return (float(lz[0]), post[0]) if np.ndim(log_e0) == 1 else (lz, post)


def _p_order(d):
    """P(stated order | the ahead-minus-behind gap d) with the misread mixture; -> (p, -2 log p, d/dd, d2/dd2)."""
    u = d / SIG_AB
    phi = np.exp(-0.5 * u * u) / np.sqrt(2 * np.pi)
    P = ndtr(u)
    p = (1 - EPS_LIST) * P + 0.5 * EPS_LIST
    f = -2.0 * np.log(p)
    g = -2.0 * (1 - EPS_LIST) * phi / (SIG_AB * p)
    h = 2.0 * (1 - EPS_LIST) * phi * u / (SIG_AB**2 * p) + 0.25 * g * g
    return p, f, g, np.maximum(h, 0.0)


@dataclass
class Statements:
    """Pairwise order statements of the list: horse a ahead of horse b at time tau (row r)."""

    row: np.ndarray
    tau: np.ndarray
    a: np.ndarray
    b: np.ndarray
    unl: np.ndarray  # bool: an unlisted horse (b) behind the last listed (a)


def statements(L: ListRows, H: int) -> Statements:
    rr, aa, bb, uu = [], [], [], []
    need = min(MAX_RANK, H)
    for i in range(len(L.tau)):
        sl = L.slots[i, :need]
        for q in range(need - 1):
            if sl[q] >= 0 and sl[q + 1] >= 0 and sl[q] != sl[q + 1]:
                rr.append(i)
                aa.append(sl[q])
                bb.append(sl[q + 1])
                uu.append(False)
        if L.full[i] and H > MAX_RANK:
            for u in sorted(set(range(H)) - set(sl.tolist())):
                rr.append(i)
                aa.append(sl[need - 1])
                bb.append(u)
                uu.append(True)
    rr = np.array(rr, int)
    return Statements(rr, L.tau[rr] if len(rr) else np.zeros(0), np.array(aa, int), np.array(bb, int),
                      np.array(uu, bool))


OMIT_MARGIN = 2.4  # m: an omitted horse explains an unlisted horse only ahead of the last listed one by > a length


def list_chains(L: ListRows, St: Statements, f_stmt, H, delta=None):
    """Fault chains at the statements' -2 log p ``f_stmt`` (``delta``: their ahead-minus-behind gaps), each fact's
    likelihood taken relative to a clearly kept order (q = p / p_max). The list-level frozen chain acts on the adjacent
    facts: a frozen row (only a row identical to the previous full row can be one) makes each fact a coin flip (1/2).
    Then one omission chain per horse over every list row on its unlisted facts: an omitted horse explains being ahead
    of the last listed horse by more than OMIT_MARGIN (likelihood max(q(d), 1 - q(d + OMIT_MARGIN))), so a horse level
    with the last listed one gains nothing from the omitted state and the list keeps ordering it; a listed horse is
    not omitted. -> (J_list, P(frozen) per row, P(omitted) per (horse, row))."""
    R = len(L.tau)
    pmax = 1.0 - 0.5 * EPS_LIST
    lp = -0.5 * np.asarray(f_stmt) - np.log(pmax)
    adj = ~St.unl
    e_ok = np.bincount(St.row[adj], lp[adj], R) if R else np.zeros(0)
    n_adj = np.bincount(St.row[adj], minlength=R) if R else np.zeros(0)
    e_fz = np.where(L.same, n_adj * np.log(0.5), -690.0)
    lz_f, pf = _hmm(e_ok, e_fz, P_FROZEN_IN, P_FROZEN_STAY)
    ok = 1.0 - pf
    om = np.zeros((H, R))
    if not R or not St.unl.any():
        return -2.0 * lz_f, pf, om
    k = np.flatnonzero(St.unl)
    d = np.zeros(len(St.a)) if delta is None else np.asarray(delta)
    q_far = 1.0 - _p_order(d[k] + OMIT_MARGIN)[0] / pmax
    l_om = np.maximum(lp[k], np.log(np.maximum(q_far, 1e-300)))
    e0 = np.zeros((H, R))
    e1 = np.zeros((H, R))
    np.add.at(e0, (St.b[k], St.row[k]), lp[k] * ok[St.row[k]])
    np.add.at(e1, (St.b[k], St.row[k]), l_om * ok[St.row[k]])
    for q in range(L.slots.shape[1]):
        sl = L.slots[:, q]
        m = sl >= 0
        e1[sl[m], np.flatnonzero(m)] = -690.0
    lz, om = _hmm(e0, e1, P_OMIT_IN, P_OMIT_STAY)
    return -2.0 * (lz_f + float(lz.sum())), pf, om


# ------------------------------------------------------------------------------------------------------------------
# the state step


def _interp(tg, t):
    x = (np.asarray(t, float) - tg[0]) / DT
    k = np.clip(np.floor(x).astype(int), 0, len(tg) - 2)
    return k, np.clip(x - k, 0.0, 1.0)


@dataclass
class Obs:
    """Point observations of one coordinate for the state step: horse column, time, value, sd."""

    h: np.ndarray
    t: np.ndarray
    y: np.ndarray
    sd: np.ndarray

    @staticmethod
    def cat(parts):
        parts = [p for p in parts if p is not None and len(p.h)]
        if not parts:
            z = np.zeros(0)
            return Obs(z.astype(int), z, z, z)
        return Obs(*(np.concatenate([getattr(p, a) for p in parts]) for a in ("h", "t", "y", "sd")))


class Band:
    """Symmetric banded matrix in scipy's lower form, with additive assembly."""

    def __init__(self, n, bw):
        self.ab = np.zeros((bw + 1, n))
        self.n = n

    def add(self, i, j, v):
        i, j, v = np.broadcast_arrays(np.asarray(i), np.asarray(j), np.asarray(v, float))
        lo, hi = np.minimum(i, j), np.maximum(i, j)
        w = np.where(i == j, v, v)  # symmetric: store the lower entry once per (i, j) pair given
        np.add.at(self.ab, (hi - lo, lo), w)


def _physics(n, a_sd, v_sd=None):
    """Per-horse precision of the physics prior on n nodes: 3 x n lower band (second differences, optional first)."""
    g2 = (1.0 / (np.asarray(a_sd, float) * DT * DT)) ** 2
    ab = np.zeros((3, n))
    i = np.arange(n - 2)
    np.add.at(ab[0], i, g2)
    np.add.at(ab[0], i + 1, 4 * g2)
    np.add.at(ab[0], i + 2, g2)
    np.add.at(ab[1], i, -2 * g2)
    np.add.at(ab[1], i + 1, -2 * g2)
    np.add.at(ab[2], i, g2)
    if v_sd:
        h2 = (1.0 / (v_sd * DT)) ** 2
        j = np.arange(n - 1)
        np.add.at(ab[0], j, h2)
        np.add.at(ab[0], j + 1, h2)
        np.add.at(ab[1], j, -h2)
    return ab


def _start_prior(n, m0, sd0, v0, sdv):
    """Proper prior at the first two nodes: x0 ~ N(m0, sd0^2), (x1 - x0)/DT ~ N(v0, sdv^2). -> (3 x n band, rhs)."""
    ab = np.zeros((3, n))
    r = np.zeros(n)
    ab[0, 0] += 1 / sd0**2
    r[0] += m0 / sd0**2
    c = 1.0 / (sdv * DT) ** 2
    ab[0, 0] += c
    ab[0, 1] += c
    ab[1, 0] -= c
    r[0] -= v0 * DT * c
    r[1] += v0 * DT * c
    return ab, r


def _quad_band(ab, x):
    """x' A x for a lower-banded symmetric A."""
    out = float((ab[0] * x * x).sum())
    for d in range(1, ab.shape[0]):
        out += 2.0 * float((ab[d, :-d] * x[:-d] * x[d:]).sum())
    return out


def _logdet(ab):
    c = cholesky_banded(ab, lower=True)
    return 2.0 * float(np.log(c[0]).sum()), c


@dataclass
class State:
    tg: np.ndarray
    S: np.ndarray  # (H, n) along-track mode
    L: np.ndarray  # (H, n) lateral mode
    As: list  # per horse 3 x n band: physics + prior + own obs + diagonal order-factor curvature (E-step marginal)
    Al: list
    pf: np.ndarray  # P(list frozen) per row
    om: np.ndarray  # P(omitted) per (horse, row)
    J: float
    parts: dict
    obs_s: Obs = None
    obs_l: Obs = None
    has_obs: np.ndarray = None  # per horse: any observation

    def sd(self, A, every=10):
        """Posterior sd of every horse on the grid (from the per-horse marginal bands)."""
        n = len(self.tg)
        idx = np.unique(np.r_[np.arange(0, n, every), n - 1])
        out = np.zeros((len(A), n))
        for h, ab in enumerate(A):
            c = cholesky_banded(ab, lower=True)
            e = np.zeros((n, len(idx)))
            e[idx, np.arange(len(idx))] = 1.0
            v = np.einsum("ij,ij->j", e, cho_solve_banded((c, True), e))
            out[h] = np.sqrt(np.interp(np.arange(n), idx, v))
        return out


class Model:
    """The objective's state step for one race: grid, priors, list statements."""

    def __init__(self, ev: Evidence, L: ListRows, tg, width, s_ref=None):
        self.ev, self.Lr, self.tg = ev, L, tg
        self.H, self.n = len(ev.horses), len(tg)
        self.width = width
        self.St = statements(L, self.H)
        a_s = np.where(tg[1:-1] < T_START, A_SD_START, A_SD_S)
        self.phys_s = _physics(self.n, a_s)
        self.phys_l = _physics(self.n, np.full(self.n - 2, A_SD_L), VL_SD)
        m0 = 0.0 if s_ref is None else float(s_ref)
        v0 = 0.0 if tg[0] < 0.5 else V0
        self.pri_s = _start_prior(self.n, m0 if tg[0] < 0.5 else V0 * tg[0], 1.0 if tg[0] < 0.5 else S0_SD, v0, V0_SD)
        self.pri_l = _start_prior(self.n, 0.5 * width, width, 0.0, VL_SD)
        self.ld_prior_s = _logdet(self.phys_s + self.pri_s[0])[0]
        self.ld_prior_l = _logdet(self.phys_l + self.pri_l[0])[0]
        ks, ws = _interp(tg, self.St.tau)
        self.st_k, self.st_w = ks, ws
        # the order factors' indices in the node-major joint system and their lower-band Hessian positions
        H, St = self.H, self.St
        self.st_II = np.stack([ks * H + St.a, (ks + 1) * H + St.a, ks * H + St.b, (ks + 1) * H + St.b], 1)
        self.st_vv = np.stack([1 - ws, ws, -(1 - ws), -ws], 1)
        nn = H * self.n
        idx, val = [], []
        for p in range(4):
            for q in range(p + 1):
                lo = np.minimum(self.st_II[:, p], self.st_II[:, q])
                hi = np.maximum(self.st_II[:, p], self.st_II[:, q])
                idx.append((hi - lo) * nn + lo)
                val.append(self.st_vv[:, p] * self.st_vv[:, q])
        self.st_hidx = np.array(idx).reshape(10, -1)
        self.st_hval = np.array(val).reshape(10, -1)

    # -- along-track: the joint system over all horses
    def _joint_base(self, obs: Obs):
        H, n = self.H, self.n
        B = Band(H * n, 2 * H)
        r = np.zeros(H * n)
        g = lambda k, h: k * H + h  # noqa: E731
        k = np.arange(n)
        for d in range(3):
            v = self.phys_s[d, :n - d] + self.pri_s[0][d, :n - d]
            for h in range(H):
                B.add(g(k[:n - d] + d, h), g(k[:n - d], h), v)
        for h in range(H):
            r[g(k, h)] += self.pri_s[1]
        if len(obs.h):
            kk, w = _interp(self.tg, obs.t)
            a, b = (1 - w) / obs.sd, w / obs.sd
            i0, i1 = g(kk, obs.h), g(kk + 1, obs.h)
            B.add(i0, i0, a * a)
            B.add(i1, i1, b * b)
            B.add(i1, i0, a * b)
            np.add.at(r, i0, a * obs.y / obs.sd)
            np.add.at(r, i1, b * obs.y / obs.sd)
        return B.ab, r

    def _stmt_delta(self, X):
        St, k, w = self.St, self.st_k, self.st_w
        sa = X[St.a, k] * (1 - w) + X[St.a, k + 1] * w
        sb = X[St.b, k] * (1 - w) + X[St.b, k + 1] * w
        return sa - sb

    def solve_s(self, obs: Obs, X0=None, rounds=4):
        """Joint mode of every horse's s under physics, observations and the weighted order factors (EM over the
        fault chains, Newton with line search). -> (X, J_s, J_list, pf, om, per-horse marginal bands)."""
        H, n, St = self.H, self.n, self.St
        A0, r0 = self._joint_base(obs)
        x = cho_solve_banded((cholesky_banded(A0, lower=True), True), r0) if X0 is None else X0.T.ravel().copy()
        k, w = self.st_k, self.st_w
        II, vv = self.st_II, self.st_vv
        ia0, ia1, ib0, ib1 = II.T
        m_st = len(St.a)

        def delta(x):
            return x[ia0] * (1 - w) + x[ia1] * w - x[ib0] * (1 - w) - x[ib1] * w

        def hess(cw):
            Hb = A0.copy()
            if m_st:
                Hb += np.bincount(self.st_hidx.ravel(), (cw[None, :] * self.st_hval).ravel(),
                                  Hb.size).reshape(Hb.shape)
            return Hb

        wt = np.ones(m_st)
        pf = np.zeros(len(self.Lr.tau))
        om = np.zeros((H, len(self.Lr.tau)))
        for _ in range(rounds if m_st else 1):
            if m_st:
                dl = delta(x)
                _J, pf, om = list_chains(self.Lr, St, _p_order(dl)[1], H, dl)
                wt_new = (1 - pf[St.row]) * np.where(St.unl, 1 - om[St.b, St.row], 1.0)
                if _ > 0 and np.abs(wt_new - wt).max() < 1e-3:
                    break
                wt = wt_new
            for _it in range(30):
                Ax = _band_matvec(A0, x)
                gq = 2.0 * (Ax - r0)
                q0 = float(x @ Ax) - 2.0 * float(r0 @ x)
                grad = gq.copy()
                obj = q0
                cw = np.zeros(m_st)
                if m_st:
                    _p, f, gd, hd = _p_order(delta(x))
                    obj += float((wt * f).sum())
                    grad += np.bincount(II.ravel(), ((wt * gd)[:, None] * vv).ravel(), len(x))
                    cw = 0.5 * wt * hd
                step = cho_solve_banded((cholesky_banded(hess(cw), lower=True), True), -0.5 * grad)
                dec = float(-grad @ step)
                if dec < 1e-6:
                    break
                g1 = float(gq @ step)
                g2 = float(step @ _band_matvec(A0, step))
                ds = delta(step) if m_st else None
                d0 = delta(x) if m_st else None
                lam, on = 1.0, np.inf
                while lam > 1e-4:
                    on = q0 + lam * g1 + lam * lam * g2
                    if m_st:
                        on += float((wt * _p_order(d0 + lam * ds)[1]).sum())
                    if on <= obj:
                        break
                    lam *= 0.5
                if on > obj:
                    break
                x = x + lam * step
                if obj - on < 1e-4:
                    break
        Xs = x.reshape(n, H).T.copy()
        # J_s: the objective at the mode with the exact chains, + Laplace log det
        J_list = 0.0
        hd = np.zeros(m_st)
        if m_st:
            dl = delta(x)
            _p, f, _g, hd = _p_order(dl)
            J_list, pf, om = list_chains(self.Lr, St, f, H, dl)
            wt = (1 - pf[St.row]) * np.where(St.unl, 1 - om[St.b, St.row], 1.0)
        ld = _logdet(hess(0.5 * wt * hd))[0]
        J_s = float(x @ _band_matvec(A0, x)) - 2.0 * float(r0 @ x) + self._const_s(obs) + ld - H * self.ld_prior_s
        # per-horse marginal bands: own physics, prior, observations + the order factors' curvature on its nodes
        As = []
        for h in range(H):
            ab = (self.phys_s + self.pri_s[0]).copy()
            As.append(ab)
        if len(obs.h):
            kk, ww = _interp(self.tg, obs.t)
            a, b = (1 - ww) / obs.sd, ww / obs.sd
            for h in np.unique(obs.h):
                m = obs.h == h
                np.add.at(As[h][0], kk[m], a[m] ** 2)
                np.add.at(As[h][0], kk[m] + 1, b[m] ** 2)
                np.add.at(As[h][1], kk[m], a[m] * b[m])
        if len(St.a):
            cw = 0.5 * wt * hd
            for side, sgn in ((St.a, 1.0), (St.b, -1.0)):
                for h in np.unique(side):
                    m = side == h
                    np.add.at(As[h][0], k[m], cw[m] * (1 - w[m]) ** 2)
                    np.add.at(As[h][0], k[m] + 1, cw[m] * w[m] ** 2)
                    np.add.at(As[h][1], k[m], cw[m] * (1 - w[m]) * w[m])
        return Xs, J_s, J_list, pf, om, As

    def _const_s(self, obs):
        """The quadratic objective's constant: sum y^2/sd^2 of the observations and the prior's m0, v0 terms."""
        c = float((obs.y**2 / obs.sd**2).sum()) if len(obs.h) else 0.0
        _ab, r = self.pri_s
        m0 = r[0] * 0  # the start prior's constant, per horse
        sd0 = 1.0 if self.tg[0] < 0.5 else S0_SD
        mm = 0.0 if self.tg[0] < 0.5 else V0 * self.tg[0]
        v0 = 0.0 if self.tg[0] < 0.5 else V0
        return c + self.H * ((mm / sd0) ** 2 + (v0 / V0_SD) ** 2) + m0

    # -- lateral: per horse, box-bounded
    def solve_l(self, obs: Obs):
        H, n = self.H, self.n
        X = np.zeros((H, n))
        A = []
        J = 0.0
        base = self.phys_l + self.pri_l[0]
        kk, ww = _interp(self.tg, obs.t) if len(obs.h) else (None, None)
        lb, ub = L_MIN, self.width - L_MIN
        for h in range(H):
            ab = base.copy()
            r = self.pri_l[1].copy()
            c = (0.5 * self.width / self.width) ** 2
            m = obs.h == h if len(obs.h) else np.zeros(0, bool)
            if m.any():
                a, b, y = (1 - ww[m]) / obs.sd[m], ww[m] / obs.sd[m], obs.y[m] / obs.sd[m]
                np.add.at(ab[0], kk[m], a * a)
                np.add.at(ab[0], kk[m] + 1, b * b)
                np.add.at(ab[1], kk[m], a * b)
                np.add.at(r, kk[m], a * y)
                np.add.at(r, kk[m] + 1, b * y)
                c += float((y * y).sum())
            pen = np.zeros(n)
            tgt = np.zeros(n)
            for _ in range(8):
                abp = ab.copy()
                abp[0] += pen
                x = cho_solve_banded((cholesky_banded(abp, lower=True), True), r + pen * tgt)
                lo_v, hi_v = (x < lb - 1e-6) & (pen == 0), (x > ub + 1e-6) & (pen == 0)
                if not (lo_v.any() or hi_v.any()):
                    break
                pen[lo_v | hi_v] = 1e4 * float(ab[0].max())
                tgt[lo_v] = lb
                tgt[hi_v] = ub
            X[h] = x
            A.append(ab)
            J += _quad_band(ab, x) - 2.0 * float(r @ x) + c + _logdet(ab)[0] - self.ld_prior_l
        return X, J, A


def _band_matvec(ab, x):
    y = ab[0] * x
    for d in range(1, ab.shape[0]):
        y[d:] += ab[d, :-d] * x[:-d]
        y[:-d] += ab[d, :-d] * x[d:]
    return y


# ------------------------------------------------------------------------------------------------------------------
# labels -> observations, the objective


def chunk_obs(ev: Evidence, seg_lab, s_shift=None):
    """Observations of the labelled segments' chunks: (Obs s, Obs lat)."""
    h = seg_lab[ev.ch_seg]
    m = h >= 0
    s = ev.ch_s[m] if s_shift is None else ev.ch_s[m] + s_shift(ev.ch_s[m])
    return (Obs(h[m], ev.ch_t[m], s, ev.ch_sds[m]), Obs(h[m], ev.ch_t[m], ev.ch_l[m], ev.ch_sdl[m]))


def label_terms(ev: Evidence, seg_lab):
    """J's label terms: per detection the horse prior or the clutter cost, the within-chunk scatter of labelled
    chunks, broken links, reads, silk."""
    lab_det = seg_lab[ev.seg]
    on = lab_det >= 0
    J = float(ev.horse_prior[on].sum() + ev.clutter[~on].sum())
    J += float(ev.ch_scat[seg_lab[ev.ch_seg] >= 0].sum())
    if len(ev.links):
        a, b, wl = ev.links[:, 0].astype(int), ev.links[:, 1].astype(int), ev.links[:, 2]
        J += float(wl[(seg_lab[a] != seg_lab[b]) | (seg_lab[a] < 0)].sum())
    k = np.flatnonzero(seg_lab >= 0)
    J += float(ev.read_cost[k, seg_lab[k]].sum() + ev.silk_cost[k, seg_lab[k]].sum())
    if ev.forbid[k, seg_lab[k]].any():
        J += INF
    return J


def evaluate(model: Model, seg_lab, extra_s=None, extra_l=None, X0=None, s_shift=None):
    """State step for the labelling ``seg_lab`` (-1 = clutter): the joint mode and J."""
    ev = model.ev
    os_, ol = chunk_obs(ev, seg_lab, s_shift)
    os_ = Obs.cat([os_, extra_s])
    ol = Obs.cat([ol, extra_l])
    Xs, J_s, J_list, pf, om, As = model.solve_s(os_, X0)
    Xl, J_l, Al = model.solve_l(ol)
    J_lab = label_terms(ev, seg_lab)
    has = np.zeros(model.H, bool)
    has[np.unique(os_.h)] = True
    st = State(model.tg, Xs, Xl, As, Al, pf, om, 0.0, {}, os_, ol, has)
    st.sdS, st.sdL = st.sd(As), st.sd(Al)
    st.pin = in_view(ev, Xs, Xl, st.sdS, st.sdL, model.tg)
    J_det, st.p_det, st.c_un, st.c_lab = miss_terms(ev, st.pin, seg_lab)
    st.J = J_s + J_l + J_list + J_lab + J_det
    st.parts = dict(J_s=J_s, J_lat=J_l, J_list=J_list, J_lab=J_lab, J_det=J_det)
    return st


# ------------------------------------------------------------------------------------------------------------------
# the assignment step


def _block_cov(ab, H_rows):
    """Posterior cov of linear functionals (rows of H: (m, n) dense) under the precision band ``ab``."""
    c = cholesky_banded(ab, lower=True)
    X = cho_solve_banded((c, True), H_rows.T)
    return H_rows @ X


WINDOWS = (10.0, 30.0)  # s: window proposals leave each horse's own observations this close to the segment out


def _band_add_obs(ab, r, tg, t, y, sd, sign=1.0):
    """Add (sign +1) or remove (-1) point observations to a per-horse 3 x n band and its right-hand side."""
    k, w = _interp(tg, t)
    a, b = (1 - w) / sd, w / sd
    np.add.at(ab[0], k, sign * a * a)
    np.add.at(ab[0], k + 1, sign * b * b)
    np.add.at(ab[1], k, sign * a * b)
    np.add.at(r, k, sign * a * y / sd)
    np.add.at(r, k + 1, sign * b * y / sd)


def window_kin_costs(model: Model, st: State, delta: float):
    """(segments x horses) block predictive of each segment's chunks against each horse's posterior with that horse's
    own observations within ``delta`` s of the segment left out (its evidence farther away, the extra observations
    and the order factors' curvature kept): the kinematic term of a window proposal, which can move a whole stretch
    of labels at once (a list omission shifts every horse behind the omitted one)."""
    ev, tg = model.ev, model.tg
    S, H = ev.n_seg, model.H
    out = np.zeros((S, H))
    lo = np.full(S, np.inf)
    hi = np.full(S, -np.inf)
    np.minimum.at(lo, ev.ch_seg, ev.ch_t)
    np.maximum.at(hi, ev.ch_seg, ev.ch_t)
    order = np.argsort(ev.ch_seg, kind="stable")
    bnd = np.r_[0, np.flatnonzero(np.diff(ev.ch_seg[order])) + 1, len(order)]
    segs = ev.ch_seg[order[bnd[:-1]]]
    for coord in ("s", "l"):
        A = st.As if coord == "s" else st.Al
        X = st.S if coord == "s" else st.L
        ob = st.obs_s if coord == "s" else st.obs_l
        y = (ev.ch_s if coord == "s" else ev.ch_l)
        R = (ev.ch_sds if coord == "s" else ev.ch_sdl) ** 2
        for h in range(H):
            mine = ob.h == h
            r_full = _band_matvec(A[h], X[h])
            # only the chunk observations are left out; extra observations (finish, official) carry no segment
            for q, k in enumerate(segs):
                c = order[bnd[q]:bnd[q + 1]]
                m = mine & (ob.t >= lo[k] - delta) & (ob.t <= hi[k] + delta) & (ob.sd < 1e3)
                ab, r = A[h].copy(), r_full.copy()
                if m.any():
                    _band_add_obs(ab, r, tg, ob.t[m], ob.y[m], ob.sd[m], -1.0)
                ab[0] = np.maximum(ab[0], 1e-12)
                try:
                    cb = cholesky_banded(ab, lower=True)
                except np.linalg.LinAlgError:
                    out[k, h] += INF
                    continue
                x = cho_solve_banded((cb, True), r)
                kk, ww = _interp(tg, ev.ch_t[c])
                Hm = np.zeros((len(c), len(tg)))
                Hm[np.arange(len(c)), kk] += 1 - ww
                Hm[np.arange(len(c)), kk + 1] += ww
                Z = cho_solve_banded((cb, True), Hm.T)
                C = Hm @ Z + np.diag(R[c])
                mu = Hm @ x
                try:
                    cp = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    out[k, h] += INF
                    continue
                z = np.linalg.solve(cp, y[c] - mu)
                out[k, h] += float(z @ z) + 2.0 * float(np.log(np.diag(cp)).sum()) - float(np.log(R[c]).sum())
    return out


def permutation_moves(ev: Evidence, lab):
    """Label permutations to try over a stretch of the race. Where tracker links are broken with both ends labelled,
    the labels before and after a cut time T define a permutation sigma (label before -> label after). Each move
    applies sigma to every labelled segment lying in [B, T) for an earlier cut time B (or the race start), or
    sigma^-1 to those in (T, B'] for a later one (B, B' where >= 3 labelled segments start together, or the race's
    ends): the same reassignment carried across a whole stretch at once.
    Horses sigma maps onto but does not map from lose their segments in the stretch. -> list of labellings."""
    lo = np.full(ev.n_seg, np.inf)
    hi = np.full(ev.n_seg, -np.inf)
    np.minimum.at(lo, ev.seg, ev.t)
    np.maximum.at(hi, ev.seg, ev.t)
    groups = {}
    for a, b, _w in ev.links:
        a, b = int(a), int(b)
        if lab[a] >= 0 and lab[b] >= 0 and lab[a] != lab[b]:
            groups.setdefault(round(float(lo[b]) * 5) / 5, []).append((int(lab[a]), int(lab[b])))
    st_t, cnt = np.unique(np.round(lo[lab >= 0] * 5) / 5, return_counts=True)
    cuts = st_t[cnt >= 3]  # stretch ends: times where >= 3 labelled segments start together (camera cuts)
    # at such a cut the tracker's continuity is often lost: pair the segments ending just before it with those
    # starting just after it by along-track rank (a render camera cut moves every box alike)
    s_lo = np.zeros(ev.n_seg)
    s_hi = np.zeros(ev.n_seg)
    first = np.full(ev.n_seg, -1)
    last = np.full(ev.n_seg, -1)
    o = np.argsort(ev.t, kind="stable")
    first[ev.seg[o[::-1]]] = o[::-1]
    last[ev.seg[o]] = o
    s_lo, s_hi = ev.s[first], ev.s[last]
    for T in cuts:
        a = np.flatnonzero((lab >= 0) & (hi < T - 0.01) & (hi >= T - 0.35))
        b = np.flatnonzero((lab >= 0) & (lo >= T - 0.11) & (lo <= T + 0.35))
        if len(a) >= 3 and len(a) == len(b):
            a, b = a[np.argsort(-s_hi[a])], b[np.argsort(-s_lo[b])]
            for x, y in zip(a, b):
                if lab[x] != lab[y]:
                    groups.setdefault(float(T), []).append((int(lab[x]), int(lab[y])))
    out, seen = [], set()
    for T, pairs in sorted(groups.items()):
        sig = {}
        for x, y in pairs:
            if sig.get(x, y) != y or y in [v for k, v in sig.items() if k != x]:
                continue
            sig[x] = y
        if len(sig) < 2:
            continue
        inv = {v: k for k, v in sig.items()}
        for side, f in ((-1, sig), (1, inv)):
            bounds = [b for b in cuts if (b < T - 0.5 if side < 0 else b > T + 0.5)] + [(-np.inf if side < 0 else np.inf)]
            for B in bounds:
                w0, w1 = (B - 0.15, T - 0.05) if side < 0 else (T + 0.05, B + 0.15)
                inside = (lo >= w0) & (hi <= w1) & (lab >= 0)
                if not inside.any():
                    continue
                new = lab.copy()
                for k in np.flatnonzero(inside):
                    if lab[k] in f:
                        new[k] = f[lab[k]]
                    elif lab[k] in f.values():
                        new[k] = -1
                key = new.tobytes()
                if not np.array_equal(new, lab) and key not in seen:
                    seen.add(key)
                    out.append((T, B, new))
    return out


INSERT_MIN_T, INSERT_JUMP = 10.0, 2  # s, ranks


def insertion_moves(model: Model, st: State, lab):
    """Insertions over a stretch between camera cuts (times where >= 3 labelled segments start together, or the race's
    ends): where a horse's box rank in the first labelled frame after the stretch (or the last before it) is at least
    INSERT_JUMP places ahead of its rank by mean position inside the stretch, the horse is put at that rank over the
    stretch and every horse from there to its old rank moves one avatar back (a horse the list omitted sits ahead of
    where the list's order put it). Stretches of >= INSERT_MIN_T s. -> list of (description, labelling)."""
    ev, tg = model.ev, model.tg
    lo = np.full(ev.n_seg, np.inf)
    hi = np.full(ev.n_seg, -np.inf)
    np.minimum.at(lo, ev.seg, ev.t)
    np.maximum.at(hi, ev.seg, ev.t)
    st_t, cnt = np.unique(np.round(lo[lab >= 0] * 5) / 5, return_counts=True)
    cuts = np.r_[tg[0], st_t[cnt >= 3], tg[-1] + 1.0]
    lab_det = lab[ev.seg]
    frames = np.unique(ev.fr[lab_det >= 0])
    ft = np.array([ev.t[ev.fr == f][0] for f in frames]) if len(frames) else np.zeros(0)

    def ranks_at(f):
        j = np.flatnonzero((ev.fr == f) & (lab_det >= 0))
        j = j[np.argsort(-ev.s[j])]
        return {int(lab_det[x]): r for r, x in enumerate(j)}

    out, seen = [], set()
    for i, b0 in enumerate(cuts):
        for b1 in cuts[i + 1:]:
            if b1 - b0 < INSERT_MIN_T:
                continue
            k, w = _interp(tg, np.clip(np.arange(b0, b1, 0.5), tg[0], tg[-1]))
            order = list(np.argsort(-(st.S[:, k] * (1 - w) + st.S[:, k + 1] * w).mean(1)))
            inside = np.flatnonzero((lo >= b0 - 0.15) & (hi < b1 - 0.05) & (lab >= 0))
            if not len(inside):
                continue
            edge = []
            after = frames[ft >= b1]
            before = frames[ft < b0]
            if len(after):
                edge.append(ranks_at(after[0]))
            if len(before):
                edge.append(ranks_at(before[-1]))
            for rk in edge:
                for u, r in rk.items():
                    q = order.index(u)
                    if q - r < INSERT_JUMP:
                        continue
                    mp = {order[r]: u, **{order[j]: order[j - 1] for j in range(r + 1, q + 1)}}
                    new = lab.copy()
                    for s_ in inside:
                        new[s_] = mp.get(int(lab[s_]), lab[s_])
                    key = new.tobytes()
                    if key not in seen and not np.array_equal(new, lab):
                        seen.add(key)
                        out.append((f"insert #{ev.horses[u]} at rank {r + 1} over {b0:.1f}-{b1:.1f} s", new))
    return out


def kin_costs(model: Model, st: State, seg_lab):
    """(segments x horses) block predictive -2 log-likelihood ratio of each segment's chunks against each horse's
    posterior (leave-segment-out for its current owner), along-track and lateral."""
    ev, tg = model.ev, model.tg
    S, H = ev.n_seg, model.H
    out = np.zeros((S, H))
    order = np.argsort(ev.ch_seg, kind="stable")
    bnd = np.r_[0, np.flatnonzero(np.diff(ev.ch_seg[order])) + 1, len(order)]
    segs = ev.ch_seg[order[bnd[:-1]]]
    kk, ww = _interp(tg, ev.ch_t)
    n = len(tg)
    allc = order
    # per horse: cov of every chunk functional (linear interpolation rows), by segment blocks
    Hm = np.zeros((len(allc), n))
    Hm[np.arange(len(allc)), kk[allc]] += 1 - ww[allc]
    Hm[np.arange(len(allc)), kk[allc] + 1] += ww[allc]
    import scipy.sparse as sp

    Hsp = sp.csr_matrix(Hm)
    k0, w0 = kk[allc], ww[allc]
    for coord in ("s", "l"):
        A = st.As if coord == "s" else st.Al
        X = st.S if coord == "s" else st.L
        y = (ev.ch_s if coord == "s" else ev.ch_l)[allc]
        R = ((ev.ch_sds if coord == "s" else ev.ch_sdl)[allc]) ** 2
        for h in range(H):
            c = cholesky_banded(A[h], lower=True)
            Z = cho_solve_banded((c, True), Hm.T)  # n x M
            mu = Hsp @ X[h]
            for q, k in enumerate(segs):
                a, b = bnd[q], bnd[q + 1]
                C = Z[k0[a:b], a:b] * (1 - w0[a:b])[:, None] + Z[k0[a:b] + 1, a:b] * w0[a:b][:, None]
                m = mu[a:b]
                yy, rr = y[a:b], R[a:b]
                if seg_lab[k] == h:  # leave this segment out
                    Ci = np.linalg.inv(C + 1e-12 * np.eye(len(C)))
                    M = Ci - np.diag(1.0 / rr)
                    M = 0.5 * (M + M.T)
                    lam, V = np.linalg.eigh(M)
                    lam = np.maximum(lam, 1e-8)
                    Cl = (V / lam) @ V.T
                    m = Cl @ (Ci @ m - yy / rr)
                    C = Cl
                P = C + np.diag(rr)
                try:
                    cp = np.linalg.cholesky(P)
                except np.linalg.LinAlgError:
                    out[k, h] += INF
                    continue
                z = np.linalg.solve(cp, yy - m)
                out[k, h] += float(z @ z) + 2.0 * float(np.log(np.diag(cp)).sum()) - float(np.log(rr).sum())
    return out


def list_votes(model: Model, st: State | None, seg_lab):
    """(segments x horses) the list's rank-count vote: per detection i and listed horse h at slot r,
    -2 log(N P(slot r = h | i is h)), with the count of horses ahead of i identity-free: every other detection
    weighted by its P(horse) (by 1 when labelled, 0 when clutter, once labels exist) and every runner without a
    detection in the frame by its posterior P(ahead); omitted runners count for nothing, a frozen row says nothing.
    Unlisted horses (full rows): -2 log(P(rank > 8) / ((N-8)/N))."""
    ev, L = model.ev, model.Lr
    H, S = model.H, ev.n_seg
    out = np.zeros((S, H))
    if not len(L.tau):
        return out
    fr, t, s = ev.fr, ev.t, ev.s
    o = np.argsort(fr, kind="stable")
    bnd = np.r_[0, np.flatnonzero(np.diff(fr[o])) + 1, len(o)]
    lab_det = seg_lab[ev.seg] if seg_lab is not None else None
    need = min(MAX_RANK, H)
    for a, b in zip(bnd[:-1], bnd[1:]):
        j = o[a:b]
        tf = t[j[0]]
        r = int(np.argmin(np.abs(L.tau - tf)))
        if abs(L.tau[r] - tf) > 0.15:
            continue
        ok = 1.0 - (st.pf[r] if st is not None else 0.0)
        omit = st.om[:, r] if st is not None else np.zeros(H)
        sl = L.slots[r]
        sj = s[j]
        if st is None or lab_det is None:
            w = ev.p_horse[j]
            n_un = max(0, H - int((w >= 0.5).sum()))
            und_p = np.full(n_un, P_AHEAD_UNSEEN)[None, :].repeat(len(j), 0)
        else:
            lj = lab_det[j]
            w = np.where(lj >= 0, 1.0 - omit[np.maximum(lj, 0)], 0.0)
            seen = set(lj[lj >= 0].tolist())
            un = np.array([h for h in range(H) if h not in seen and st.has_obs[h]], int)
            blind = np.array([h for h in range(H) if h not in seen and not st.has_obs[h]], int)
            k, ww = _interp(model.tg, np.array([tf]))
            mu = st.S[un, k[0]] * (1 - ww[0]) + st.S[un, k[0] + 1] * ww[0]
            sdv = st.sdS[un, k[0]]
            und_p = ndtr((mu[None, :] - sj[:, None]) / np.sqrt(sdv[None, :] ** 2 + SIG_AB**2))
            und_p = np.concatenate([und_p * (1 - omit[un])[None, :],
                                    np.full((len(j), len(blind)), P_AHEAD_UNSEEN) * (1 - omit[blind])[None, :]], 1)
        pa = ndtr((sj[None, :] - sj[:, None]) / SIG_AB) * w[None, :]
        np.fill_diagonal(pa, 0.0)
        probs = np.concatenate([pa, und_p], 1)  # D x (D + U)
        # Poisson-binomial count of horses ahead, per detection
        cnt = np.zeros((len(j), H + 1))
        cnt[:, 0] = 1.0
        for q in range(probs.shape[1]):
            p = probs[:, q:q + 1]
            cnt[:, 1:] = cnt[:, 1:] * (1 - p) + cnt[:, :-1] * p
            cnt[:, 0] *= 1 - p[:, 0]
        cnt /= np.maximum(cnt.sum(1, keepdims=True), 1e-300)
        segj = ev.seg[j]
        for q in range(need):
            h = sl[q]
            if h < 0:
                continue
            P = ok * ((1 - EPS_LIST) * cnt[:, q] + EPS_LIST / H) + (1 - ok) / H
            np.add.at(out[:, h], segj, -2.0 * np.log(P * H))
        if L.full[r] and H > MAX_RANK:
            base = (H - MAX_RANK) / H
            pb = cnt[:, MAX_RANK:].sum(1)
            for h in sorted(set(range(H)) - set(sl[:need].tolist())):
                P = ok * ((1 - omit[h]) * ((1 - EPS_LIST) * pb + EPS_LIST * base) + omit[h] * base) + (1 - ok) * base
                np.add.at(out[:, h], segj, -2.0 * np.log(P / base))
    return out


def segment_base(ev: Evidence):
    """(segments,) label-independent part of a labelled segment's cost relative to clutter: horse prior minus
    clutter over its detections, plus its chunks' within-chunk scatter."""
    b = np.bincount(ev.seg, ev.horse_prior - ev.clutter, ev.n_seg)
    return b + np.bincount(ev.ch_seg, ev.ch_scat, ev.n_seg)


def milp(U, ev: Evidence, allow=None):
    """min sum U x - sum link rewards z over x[segment, horse] binary (a segment has at most one horse; leaving it
    out is clutter, cost 0), one segment per horse per frame, z[link, horse] <= x at both ends."""
    from scipy.optimize import Bounds, LinearConstraint, milp as _milp
    from scipy.sparse import coo_matrix

    S, H = U.shape
    ok = np.isfinite(U) & (U < INF / 2) & (~ev.forbid)
    if allow is not None:
        ok &= allow
    links = ev.links
    nx, nz = S * H, len(links) * H
    c = np.r_[np.where(ok, U, 0.0).ravel(), -np.repeat(links[:, 2], H) if len(links) else np.zeros(0)]
    rows, cols, vals = [], [], []
    r = 0
    for s in range(S):
        rows += [r] * H
        cols += list(range(s * H, s * H + H))
        vals += [1] * H
        r += 1
    lo, hi = ev.seg_lo, ev.seg_hi
    for x in np.unique(lo):
        cl = np.flatnonzero((lo <= x) & (hi >= x))
        if len(cl) > 1:
            for h in range(H):
                rows += [r] * len(cl)
                cols += [int(k) * H + h for k in cl]
                vals += [1] * len(cl)
                r += 1
    n_ub1 = r
    for li, (a, b, _w) in enumerate(links):
        for h in range(H):
            zi = nx + li * H + h
            for s in (int(a), int(b)):
                rows += [r, r]
                cols += [zi, s * H + h]
                vals += [1, -1]
                r += 1
    A = coo_matrix((vals, (rows, cols)), shape=(r, nx + nz)).tocsr()
    ub = np.r_[ok.ravel().astype(float), np.ones(nz)]
    res = _milp(c, constraints=LinearConstraint(A, -np.inf, np.r_[np.ones(n_ub1), np.zeros(r - n_ub1)]),
                integrality=np.ones(nx + nz), bounds=Bounds(0, ub), options=dict(time_limit=300))
    if res.x is None:
        raise RuntimeError(f"segment MILP failed: {res.message}")
    x = np.round(res.x[:nx]).reshape(S, H)
    return np.where(x.max(1) > 0.5, x.argmax(1), -1)


def unaries(model: Model, st: State, seg_lab):
    """The assignment step's linearised increments of J per (segment, horse), relative to clutter."""
    ev = model.ev
    U = kin_costs(model, st, seg_lab) + list_votes(model, st, seg_lab)
    U += segment_base(ev)[:, None] + ev.read_cost + ev.silk_cost
    if st.c_un is not None:  # detection terms: a labelled frame instead of a (possibly in-view) missed one
        fi = np.searchsorted(ev.view["frames"], ev.fr)
        for h in range(model.H):
            U[:, h] += np.bincount(ev.seg, st.c_lab[h, fi] - st.c_un[h, fi], ev.n_seg)
    U[model.ev.forbid] = INF
    return U


@dataclass
class Result:
    seg_lab: np.ndarray
    state: State
    U: np.ndarray
    marg: np.ndarray
    log: list


T_POST = 10.0  # J's per-detection terms treat correlated errors as independent: the posterior is tempered by this
ABREAST_S, ABREAST_L, ABREAST_T = 1.5, 3.0, 5.0  # m, units, s: two paths this close for this long run level


def swap_moves(model: Model, st: State, lab):
    """Abreast-pair swaps: for two horses whose paths stay within ABREAST_S along and ABREAST_L across for >= ABREAST_T
    s, exchange their labels on every segment inside the stretch, widened to the nearest camera cuts (times where >= 3
    labelled segments start together) or the race's ends. -> list of (description, labelling)."""
    ev, tg = model.ev, model.tg
    lo = np.full(ev.n_seg, np.inf)
    hi = np.full(ev.n_seg, -np.inf)
    np.minimum.at(lo, ev.seg, ev.t)
    np.maximum.at(hi, ev.seg, ev.t)
    st_t, cnt = np.unique(np.round(lo[lab >= 0] * 5) / 5, return_counts=True)
    cuts = np.r_[tg[0], st_t[cnt >= 3], tg[-1] + 1.0]
    out = []
    has = st.has_obs
    for a in range(model.H):
        for b in range(a + 1, model.H):
            if not (has[a] and has[b]):
                continue
            close = (np.abs(st.S[a] - st.S[b]) < ABREAST_S) & (np.abs(st.L[a] - st.L[b]) < ABREAST_L)
            if close.sum() * DT < ABREAST_T:
                continue
            edges = np.flatnonzero(np.diff(np.r_[0, close.astype(int), 0]))
            for r0, r1 in zip(edges[::2], edges[1::2]):
                if (r1 - r0) * DT < ABREAST_T:
                    continue
                w0 = cuts[cuts <= tg[r0]].max(initial=tg[0])
                w1 = cuts[cuts >= tg[r1 - 1]].min(initial=tg[-1] + 1.0)
                inside = (lo >= w0 - 0.15) & (hi < w1 - 0.05)
                new = lab.copy()
                new[inside & (lab == a)] = b
                new[inside & (lab == b)] = a
                if not np.array_equal(new, lab):
                    out.append((f"swap #{ev.horses[a]}/#{ev.horses[b]} over {w0:.1f}-{w1:.1f} s", new))
    return out


def posterior(seen, lab, J_best, H):
    """Per segment P(label) (H + 1 columns, the last clutter) over every labelling the solve scored, each weighted by
    exp(-(J - J_best) / (2 T_POST)); the kept labelling has J_best."""
    labs = np.array([x for x, _ in seen]) if seen else np.array([lab])
    J = np.array([j for _, j in seen]) if seen else np.array([J_best])
    w = np.exp(-(J - J.min()) / (2.0 * T_POST))
    P = np.zeros((labs.shape[1], H + 1))
    col = np.where(labs >= 0, labs, H)
    for i in range(len(w)):
        P[np.arange(labs.shape[1]), col[i]] += w[i]
    return P / P.sum(1, keepdims=True)


def solve(model: Model, start=(), extra_s=None, extra_l=None, s_shift=None, log=print):
    """Start-free global solve, then alternate assignment (MILP on J's linearised increments) and state steps, keeping
    a proposal only when J falls; ends when no proposal lowers J. ``start``: further labellings whose J is compared
    with the start solve's (the lowest is kept)."""
    ev = model.ev
    hist = []
    seen = []  # every labelling the state step scored: (labels, J); the posterior over them gives the marginals

    def look(labs, X0=None):
        st_ = evaluate(model, labs, extra_s, extra_l, X0=X0, s_shift=s_shift)
        seen.append((np.asarray(labs).copy(), st_.J))
        return st_

    U0 = list_votes(model, None, None) + ev.read_cost + ev.silk_cost
    U0[ev.forbid] = INF
    lab = milp(np.where(U0 < 0, U0, 0.0) - 1e-3 * (U0 < 0), ev, allow=U0 < 0)
    st = look(lab)
    hist.append(("start", st.J, int((lab >= 0).sum())))
    log(f"   core: start solve J {st.J:.1f} ({int((lab >= 0).sum())}/{ev.n_seg} segments labelled)")
    for k, other in enumerate(start):
        st2 = look(np.asarray(other))
        log(f"   core: start {k + 1} J {st2.J:.1f}")
        if st2.J < st.J:
            lab, st = np.asarray(other).copy(), st2
    it = 0
    while True:
        it += 1
        U = unaries(model, st, lab)
        prop = milp(U, ev)
        cand = np.zeros(0, int)
        if not np.array_equal(prop, lab):
            st2 = look(prop, st.S)
            nch = int((prop != lab).sum())
            if st2.J < st.J - 1e-6:
                log(f"   core: it{it} {nch} segments relabelled, J {st.J:.1f} -> {st2.J:.1f} {_fmt(st2.parts)}")
                hist.append((it, st2.J, nch))
                lab, st = prop, st2
                continue
            cand = np.flatnonzero(prop != lab)
        # the joint proposal raised J: try its changes one segment at a time, best predicted gain first
        gain = U[np.arange(len(lab)), np.maximum(lab, 0)] * (lab >= 0) - np.where(prop >= 0, U[np.arange(len(prop)),
                                                                                         np.maximum(prop, 0)], 0.0)
        cand = cand[np.argsort(-gain[cand])]
        moved = False
        for k in cand[:5]:
            trial = lab.copy()
            trial[k] = prop[k]
            if prop[k] >= 0:  # keep exclusivity: drop the label from overlapping holders
                for q in np.flatnonzero(trial == prop[k]):
                    if q != k and ev.seg_lo[q] <= ev.seg_hi[k] and ev.seg_hi[q] >= ev.seg_lo[k]:
                        trial[q] = prop[q] if prop[q] != prop[k] else -1
            st2 = look(trial, st.S)
            if st2.J < st.J - 1e-6:
                log(f"   core: it{it} proposal raised J; single move seg {k} -> J {st.J:.1f} -> {st2.J:.1f}")
                hist.append((it, st2.J, 1))
                lab, st = trial, st2
                moved = True
                break
        if not moved:  # window proposals: each horse's nearby labels left out, the list's rank votes not used
            base = segment_base(ev)[:, None] + ev.read_cost + ev.silk_cost
            if st.c_un is not None:
                fi = np.searchsorted(ev.view["frames"], ev.fr)
                for h in range(model.H):
                    base[:, h] += np.bincount(ev.seg, st.c_lab[h, fi] - st.c_un[h, fi], ev.n_seg)
            for delta in WINDOWS:
                Uw = window_kin_costs(model, st, delta) + base
                Uw[ev.forbid] = INF
                pw = milp(Uw, ev)
                if np.array_equal(pw, lab):
                    continue
                st2 = look(pw, st.S)
                if st2.J < st.J - 1e-6:
                    log(f"   core: it{it} window proposal ({delta:.0f} s) relabels {int((pw != lab).sum())} segments, "
                        f"J {st.J:.1f} -> {st2.J:.1f} {_fmt(st2.parts)}")
                    hist.append((it, st2.J, int((pw != lab).sum())))
                    lab, st = pw, st2
                    moved = True
                    break
                log(f"   core: it{it} window proposal ({delta:.0f} s) raised J ({st2.J:.1f})")
        if not moved:  # permutation moves: a cut's label permutation carried over a whole stretch
            best = None
            for T, B, pm in permutation_moves(ev, lab):
                st2 = look(pm, st.S)
                if st2.J < st.J - 1e-6 and (best is None or st2.J < best[1].J):
                    best = (pm, st2, T, B)
            if best is not None:
                pm, st2, T, B = best
                log(f"   core: it{it} permutation move at {T:.1f} s over {min(B, T):.1f}-{max(B, T):.1f} s relabels "
                    f"{int((pm != lab).sum())} segments, J {st.J:.1f} -> {st2.J:.1f} {_fmt(st2.parts)}")
                hist.append((it, st2.J, int((pm != lab).sum())))
                lab, st = pm, st2
                moved = True
        if not moved:  # omission insertions over stretches
            best = None
            for desc, pm in insertion_moves(model, st, lab):
                st2 = look(pm, st.S)
                if st2.J < st.J - 1e-6 and (best is None or st2.J < best[1].J):
                    best = (pm, st2, desc)
            if best is not None:
                pm, st2, desc = best
                log(f"   core: it{it} {desc} relabels {int((pm != lab).sum())} segments, J {st.J:.1f} -> "
                    f"{st2.J:.1f} {_fmt(st2.parts)}")
                hist.append((it, st2.J, int((pm != lab).sum())))
                lab, st = pm, st2
                moved = True
        if not moved:  # abreast-pair swaps: two horses running level exchange labels over a stretch between cuts
            best = None
            for desc, pm in swap_moves(model, st, lab):
                st2 = look(pm, st.S)
                if st2.J < st.J - 1e-6 and (best is None or st2.J < best[1].J):
                    best = (pm, st2, desc)
            if best is not None:
                pm, st2, desc = best
                log(f"   core: it{it} {desc} relabels {int((pm != lab).sum())} segments, J {st.J:.1f} -> "
                    f"{st2.J:.1f} {_fmt(st2.parts)}")
                hist.append((it, st2.J, int((pm != lab).sum())))
                lab, st = pm, st2
                moved = True
        if not moved:
            log(f"   core: it{it} no proposal lowers J; stop at J {st.J:.1f}")
            break
    U = unaries(model, st, lab)
    return Result(lab, st, U, posterior(seen, lab, st.J, model.H), hist)


def _fmt(parts):
    return "(" + ", ".join(f"{k} {v:.0f}" for k, v in parts.items()) + ")"
