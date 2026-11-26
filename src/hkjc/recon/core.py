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
increments at the current state: the block predictive of the segment's chunks against each runner's posterior with
that runner's overlapping segments left out (a one-horse refit), the list's rank-count vote (identity-free counts of
the horses ahead, weighted by each detection's P(horse) and each undetected runner's posterior), the per-detection
horse/clutter prior, disc reads and the silk term. The solve alternates the MILP with the joint state step while J
falls, then a bounded polish: candidate labellings scored by one-horse refits, the best few given full state steps.
"""

from __future__ import annotations

import copy
import hashlib
import time

from dataclasses import dataclass, field
from functools import partial

import numpy as np
from scipy.linalg import cho_solve_banded as _cho_solve_banded
from scipy.linalg import cholesky_banded as _cholesky_banded
from scipy.linalg import solve_triangular
from scipy.special import ndtr

# banded factor and solve without the finiteness scan (the bands are finite by construction)
cholesky_banded = partial(_cholesky_banded, check_finite=False)
cho_solve_banded = partial(_cho_solve_banded, check_finite=False)

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
PF0_FROZEN = 0.9  # start solve: P(frozen) of a row inside a held near-cloth read
P_OMIT_IN, P_OMIT_STAY = 1e-4, 0.999  # per list row, per horse
# order-wrong state of a pair the list states adjacent (its order says nothing), per adjacent fact of that pair;
# fitted on held-out list rows of the development races against masked paths
P_WRONG_IN, P_WRONG_STAY = 3e-3, 0.98
# a slot read whose number has a dropped-digit partner (1 <-> 11, 12, ...) is the partner: the list's one-row flickers
P_DROP = 1e-3
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
    GHOST_CONF (silk render), the disclaimer band at glyph size, or the left edge far outside its frame's field; and
    any box wholly inside the list band (silk render: a list icon; a horse there is hidden, not boxed). P_EDGE for any
    other box overlapping an occluder band (``occluder_edge``); else P_HORSE0."""
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
    if silk and len(rows):  # a box wholly inside the list band is one of its icons: a horse there is hidden
        zone |= rows[:, 2] - rows[:, 4] / 2 >= legend_top(rows[:, 2], panel_h) + 4
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


def build_evidence(horses, T, members, pose, panel_h, silk_style, nums=None, silk=None, panel_w=None, bend=None,
                   s_shift=None):
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
    if s_shift is not None:  # the final product's metric bias moves render s after the (label-free) segmentation
        s = s + s_shift(s)
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
    partner: np.ndarray | None = None  # (H, H) bool: runner columns whose numbers are dropped-digit partners


def drop_partners(horses) -> np.ndarray:
    """(H, H) bool: numbers one of which contains the other's digits (1 <-> 11, 12, ...; 2 <-> 12), a list read that
    can be the other with a digit dropped."""
    s = [str(int(h)) for h in horses]
    return np.array([[a != b and (a in b or b in a) for b in s] for a in s], bool).reshape(len(s), len(s))


def list_rows(leg, t_of_frame, horses) -> ListRows:
    """legend rows (frame, 8 slot numbers; masked reads are 0) -> ListRows on the race clock."""
    col = {int(h): c for c, h in enumerate(horses)}
    H = len(horses)
    need = min(MAX_RANK, H)
    if leg is None or not len(leg):
        z = np.zeros(0)
        return ListRows(z, np.zeros((0, MAX_RANK), int), z.astype(bool), z.astype(bool), drop_partners(horses))
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
    return ListRows(tau[keep], sl[keep], full[keep], same[keep], drop_partners(horses))


def _scan(M, L, reverse=False):
    """Inclusive prefix (or, ``reverse``, suffix) products of 2 x 2 matrices M (C, n, 2, 2) kept normalised, with their
    log scales L (C, n): Hillis-Steele doubling, log2(n) vectorised steps (on the four entries as separate arrays)."""
    m00, m01, m10, m11 = (np.ascontiguousarray(M[..., i, j]) for i, j in ((0, 0), (0, 1), (1, 0), (1, 1)))
    L = L.copy()
    n = M.shape[1]
    k = 1
    while k < n:
        a00, a01, a10, a11 = m00[:, :-k], m01[:, :-k], m10[:, :-k], m11[:, :-k]
        b00, b01, b10, b11 = m00[:, k:], m01[:, k:], m10[:, k:], m11[:, k:]
        p00 = a00 * b00 + a01 * b10
        p01 = a00 * b01 + a01 * b11
        p10 = a10 * b00 + a11 * b10
        p11 = a10 * b01 + a11 * b11
        Ln = L[:, :-k] + L[:, k:]
        sc = np.maximum(np.maximum(np.maximum(p00, p01), np.maximum(p10, p11)), 1e-300)
        Ln = Ln + np.log(sc)
        dst = np.s_[:, :-k] if reverse else np.s_[:, k:]
        m00[dst], m01[dst], m10[dst], m11[dst] = p00 / sc, p01 / sc, p10 / sc, p11 / sc
        L[dst] = Ln
        k *= 2
    out = np.empty_like(M)
    out[..., 0, 0], out[..., 0, 1], out[..., 1, 0], out[..., 1, 1] = m00, m01, m10, m11
    return out, L


def _hmm(log_e0, log_e1, p_in, p_stay):
    """Two-state chains (0 = ok, 1 = fault) starting in 0, one per row of ``log_e0`` / ``log_e1`` (C x n, or one chain
    as a 1-d array); forward-backward by scans of the step matrices. -> (log Z per chain, P(fault) per chain and
    step)."""
    e0, e1 = np.atleast_2d(log_e0).astype(float), np.atleast_2d(log_e1).astype(float)
    C, n = e0.shape
    if not n:
        return (0.0, np.zeros(0)) if np.ndim(log_e0) == 1 else (np.zeros(C), np.zeros((C, 0)))
    m = np.maximum(e0, e1)
    E0, E1 = np.exp(e0 - m), np.exp(e1 - m)
    T = np.array([[1 - p_in, p_in], [1 - p_stay, p_stay]])
    # step matrices M_i[a, b] = T[a, b] E_b(i), i = 1 .. n-1
    M = T[None, None] * np.stack([E0[:, 1:], E1[:, 1:]], -1)[:, :, None, :]
    Lm = m[:, 1:].copy()
    a0 = np.stack([E0[:, 0], np.zeros(C)], -1)  # alpha_0 / exp(m_0)
    if n > 1:
        Pre, Lp = _scan(M, Lm)
        Suf, _ls = _scan(M, Lm, reverse=True)
        alpha = np.concatenate([a0[:, None], a0[:, None, 0, None] * Pre[..., 0, :]], 1)
        la = np.concatenate([np.zeros((C, 1)), Lp], 1) + m[:, :1]
        beta = np.concatenate([Suf.sum(3), np.ones((C, 1, 2))], 1)  # per-step scales cancel in the posterior
    else:
        alpha, la = a0[:, None], m[:, :1]
        beta = np.ones((C, 1, 2))
    zf = alpha[:, -1].sum(1)
    lz = np.log(np.maximum(zf, 1e-300)) + la[:, -1]
    w = alpha * beta
    post = w[..., 1] / np.maximum(w.sum(2), 1e-300)
    post = np.clip(post, 0.0, 1.0)
    return (float(lz[0]), post[0]) if np.ndim(log_e0) == 1 else (lz, post)


def _p_order(d, eps=EPS_LIST):
    """P(stated order | the ahead-minus-behind gap d) with the misread mixture (``eps`` per fact or one value);
    -> (p, -2 log p, d/dd, d2/dd2)."""
    u = d / SIG_AB
    phi = np.exp(-0.5 * u * u) / np.sqrt(2 * np.pi)
    P = ndtr(u)
    p = (1 - eps) * P + 0.5 * eps
    f = -2.0 * np.log(p)
    g = -2.0 * (1 - eps) * phi / (SIG_AB * p)
    h = 2.0 * (1 - eps) * phi * u / (SIG_AB**2 * p) + 0.25 * g * g
    return p, f, g, np.maximum(h, 0.0)


def _f_order(d, eps):
    """-2 log P(stated order | gap d) alone (``_p_order``'s second output)."""
    return -2.0 * np.log((1 - eps) * ndtr(d / SIG_AB) + 0.5 * eps)


@dataclass
class Statements:
    """Pairwise order statements of the list: horse a ahead of horse b at time tau (row r)."""

    row: np.ndarray
    tau: np.ndarray
    a: np.ndarray
    b: np.ndarray
    unl: np.ndarray  # bool: an unlisted horse (b) behind the last listed (a)
    eps: np.ndarray = None  # per fact misread probability (EPS_LIST, more where a read has a dropped-digit partner)
    pair: np.ndarray = None  # per adjacent fact: its unordered pair's order-wrong chain (-1: unlisted facts)
    slot: np.ndarray = None  # the list slot of a (adjacent facts: b sits in slot + 1)


def statements(L: ListRows, H: int) -> Statements:
    """The list rows' pairwise facts: adjacent read slots, and on full rows the last listed horse ahead of every
    unlisted runner; per fact its misread probability: EPS_LIST, and P_DROP more for each of its two horses whose list
    read can be a dropped-digit partner's number (a listed horse with partners; an unlisted runner that partners a
    read of the row)."""
    rr, aa, bb, uu, kk, qq = [], [], [], [], [], []
    need = min(MAX_RANK, H)
    P = L.partner if L.partner is not None else np.zeros((H, H), bool)
    pr = P.any(1)
    for i in range(len(L.tau)):
        sl = L.slots[i, :need]
        for q in range(need - 1):
            if sl[q] >= 0 and sl[q + 1] >= 0 and sl[q] != sl[q + 1]:
                rr.append(i)
                aa.append(sl[q])
                bb.append(sl[q + 1])
                uu.append(False)
                kk.append(int(pr[sl[q]]) + int(pr[sl[q + 1]]))
                qq.append(q)
        if L.full[i] and H > MAX_RANK:
            read = sl[sl >= 0]
            for u in sorted(set(range(H)) - set(sl.tolist())):
                rr.append(i)
                aa.append(sl[need - 1])
                bb.append(u)
                uu.append(True)
                kk.append(int(pr[sl[need - 1]]) + int(P[u, read].any()))
                qq.append(need - 1)
    rr = np.array(rr, int)
    aa, bb, uu = np.array(aa, int), np.array(bb, int), np.array(uu, bool)
    eps = 1.0 - (1.0 - EPS_LIST) * (1.0 - P_DROP) ** np.array(kk, float)
    pair = np.full(len(rr), -1)
    if (~uu).any():
        key = np.minimum(aa, bb) * H + np.maximum(aa, bb)
        pair[~uu] = np.unique(key[~uu], return_inverse=True)[1].ravel()
    return Statements(rr, L.tau[rr] if len(rr) else np.zeros(0), aa, bb, uu, eps, pair, np.array(qq, int))


OMIT_MARGIN = 2.4  # m: an omitted horse explains an unlisted horse only ahead of the last listed one by > a length


def list_chains(L: ListRows, St: Statements, f_stmt, H, delta=None):
    """Fault chains at the statements' -2 log p ``f_stmt`` (``delta``: their ahead-minus-behind gaps), each fact's
    likelihood taken relative to a clearly kept order (q = p / p_max).
    - The list-level frozen chain acts on the adjacent facts: a frozen row (only a row identical to the previous full
      row can be one) makes each adjacent fact a coin flip (1/2).
    - One order-wrong chain per unordered pair the list states adjacent, over that pair's adjacent facts in row order:
      in the wrong state the pair's stated order says nothing (a coin flip), so a list that misplaces a horse for a
      stretch costs the chain's entry, not every fact (emission: the ratio to the kept order, weighted by the row's
      P(not frozen)).
    - One omission chain per horse over every list row on its unlisted facts: an omitted horse explains being ahead of
      the last listed horse by more than OMIT_MARGIN (likelihood max(q(d), 1 - q(d + OMIT_MARGIN))), so a horse level
      with the last listed one gains nothing from the omitted state and the list keeps ordering it; a listed horse is
      not omitted, unless its read can be a dropped-digit partner's number (then at P_DROP).
    -> (J_list, P(frozen) per row, P(omitted) per (horse, row), P(order wrong) per fact (0 for unlisted facts))."""
    R = len(L.tau)
    eps = St.eps if St.eps is not None else np.full(len(St.a), EPS_LIST)
    pmax = 1.0 - 0.5 * eps
    lp = -0.5 * np.asarray(f_stmt) - np.log(pmax)
    adj = ~St.unl
    e_ok = np.bincount(St.row[adj], lp[adj], R) if R else np.zeros(0)
    n_adj = np.bincount(St.row[adj], minlength=R) if R else np.zeros(0)
    e_fz = np.where(L.same, n_adj * np.log(0.5), -690.0)
    lz_f, pf = _hmm(e_ok, e_fz, P_FROZEN_IN, P_FROZEN_STAY)
    ok = 1.0 - pf
    om = np.zeros((H, R))
    pw = np.zeros(len(St.a))
    lz_w = 0.0
    ka = np.flatnonzero(adj)
    if len(ka) and St.pair is not None:
        o = ka[np.argsort(St.pair[ka], kind="stable")]  # by pair, rows ascending within a pair
        cs = St.pair[o]
        cnt = np.bincount(cs)
        pos = np.arange(len(o)) - np.r_[0, np.cumsum(cnt)[:-1]][cs]
        e1 = np.zeros((len(cnt), int(cnt.max())))
        e1[cs, pos] = ok[St.row[o]] * (np.log(0.5 / pmax[o]) - lp[o])
        lzc, post = _hmm(np.zeros_like(e1), e1, P_WRONG_IN, P_WRONG_STAY)
        pw[o] = post[cs, pos]
        lz_w = float(lzc.sum())
    if not R or not St.unl.any():
        return -2.0 * (lz_f + lz_w), pf, om, pw
    k = np.flatnonzero(St.unl)
    d = np.zeros(len(St.a)) if delta is None else np.asarray(delta)
    q_far = 1.0 - _p_order(d[k] + OMIT_MARGIN, eps[k])[0] / pmax[k]
    l_om = np.maximum(lp[k], np.log(np.maximum(q_far, 1e-300)))
    e0 = np.zeros((H, R))
    e1 = np.zeros((H, R))
    np.add.at(e0, (St.b[k], St.row[k]), lp[k] * ok[St.row[k]])
    np.add.at(e1, (St.b[k], St.row[k]), l_om * ok[St.row[k]])
    part = (L.partner if L.partner is not None else np.zeros((H, H), bool)).any(1)
    for q in range(L.slots.shape[1]):
        sl = L.slots[:, q]
        m = sl >= 0
        e1[sl[m], np.flatnonzero(m)] = np.where(part[sl[m]], max(np.log(P_DROP), -690.0) if P_DROP > 0 else -690.0,
                                                -690.0)
    lz, om = _hmm(e0, e1, P_OMIT_IN, P_OMIT_STAY)
    return -2.0 * (lz_f + lz_w + float(lz.sum())), pf, om, pw


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

    def sd(self, A, every=10, cache=None, factors=None):
        """Posterior sd of every horse on the grid (from the per-horse marginal bands); ``cache`` (dict keyed by the
        band's id) reuses rows of bands seen before (the lateral solve's cached bands); ``factors`` (list) receives each
        band's lower Cholesky factor (None where the cache answered)."""
        n = len(self.tg)
        idx = np.unique(np.r_[np.arange(0, n, every), n - 1])
        out = np.zeros((len(A), n))
        for h, ab in enumerate(A):
            if cache is not None and id(ab) in cache and cache[id(ab)][0] is ab:
                out[h] = cache[id(ab)][1]
                if factors is not None:
                    factors.append(None)
                continue
            c = cholesky_banded(ab, lower=True)
            if factors is not None:
                factors.append(c)
            e = np.zeros((n, len(idx)))
            e[idx, np.arange(len(idx))] = 1.0
            v = np.einsum("ij,ij->j", e, cho_solve_banded((c, True), e))
            out[h] = np.sqrt(np.interp(np.arange(n), idx, v))
            if cache is not None:
                cache[id(ab)] = (ab, out[h].copy())
        return out


PART_CPU: dict = {}  # CPU s by part of the state step and the unaries, accumulated over the process (log only)


def _add_cpu(part, t0):
    t = time.process_time()
    PART_CPU[part] = PART_CPU.get(part, 0.0) + t - t0
    return t


EM_ROUNDS = 20  # EM rounds over the fault chains at most; they stop when the weights change < 1e-3 (4 left J
# unconverged by up to 12)


class Model:
    """The objective's state step for one race: grid, priors, list statements."""

    def __init__(self, ev: Evidence, L: ListRows, tg, width, s_ref=None):
        self.ev, self.Lr, self.tg = ev, L, tg
        self.H, self.n = len(ev.horses), len(tg)
        self.width = width
        self.St = statements(L, self.H)
        # the start solve's P(frozen) per row, before any path exists: a near-cloth read held (contract.legend_frozen,
        # from the list alone) is likely frozen; the state step's frozen chain replaces it
        self.pf0 = np.zeros(len(L.tau))
        if len(L.tau):
            from .contract import legend_frozen

            nums = np.where(L.slots >= 0, np.asarray(ev.horses)[np.maximum(L.slots, 0)], 0)
            self.pf0 = np.where(legend_frozen(nums, L.tau, self.H)[0], PF0_FROZEN, 0.0)
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

    def solve_s(self, obs: Obs, X0=None, rounds=EM_ROUNDS, max_it=30):
        """Joint mode of every horse's s under physics, observations and the weighted order factors (EM over the
        fault chains, Newton with line search). -> (X, J_s, J_list, pf, om, per-horse marginal bands, P(order wrong)
        per statement)."""
        H, n, St = self.H, self.n, self.St
        tc = time.process_time()
        A0, r0 = self._joint_base(obs)
        x = cho_solve_banded((cholesky_banded(A0, lower=True), True), r0) if X0 is None else X0.T.ravel().copy()
        tc = _add_cpu("state: joint base", tc)
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
        pw = np.zeros(m_st)
        for _ in range(rounds if m_st else 1):
            if m_st:
                dl = delta(x)
                _J, pf, om, pw = list_chains(self.Lr, St, _p_order(dl, St.eps)[1], H, dl)
                wt_new = (1 - pf[St.row]) * np.where(St.unl, 1 - om[St.b, St.row], 1.0 - pw)
                tc = _add_cpu("state: EM (fault chains)", tc)
                if _ > 0 and np.abs(wt_new - wt).max() < 1e-3:
                    break
                wt = wt_new
            for _it in range(max_it):
                Ax = _band_matvec(A0, x)
                gq = 2.0 * (Ax - r0)
                q0 = float(x @ Ax) - 2.0 * float(r0 @ x)
                grad = gq.copy()
                obj = q0
                cw = np.zeros(m_st)
                if m_st:
                    _p, f, gd, hd = _p_order(delta(x), St.eps)
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
                        on += float((wt * _p_order(d0 + lam * ds, St.eps)[1]).sum())
                    if on <= obj:
                        break
                    lam *= 0.5
                if on > obj:
                    break
                x = x + lam * step
                if obj - on < 1e-4:
                    break
            tc = _add_cpu("state: Newton", tc)
        Xs = x.reshape(n, H).T.copy()
        # J_s: the objective at the mode with the exact chains, + Laplace log det
        J_list = 0.0
        hd = np.zeros(m_st)
        if m_st:
            dl = delta(x)
            _p, f, _g, hd = _p_order(dl, St.eps)
            J_list, pf, om, pw = list_chains(self.Lr, St, f, H, dl)
            wt = (1 - pf[St.row]) * np.where(St.unl, 1 - om[St.b, St.row], 1.0 - pw)
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
        _add_cpu("state: exact chains, log det, bands", tc)
        return Xs, J_s, J_list, pf, om, As, pw

    def _const_s(self, obs):
        """The quadratic objective's constant: sum y^2/sd^2 of the observations and the prior's m0, v0 terms."""
        c = float((obs.y**2 / obs.sd**2).sum()) if len(obs.h) else 0.0
        _ab, r = self.pri_s
        m0 = r[0] * 0  # the start prior's constant, per horse
        sd0 = 1.0 if self.tg[0] < 0.5 else S0_SD
        mm = 0.0 if self.tg[0] < 0.5 else V0 * self.tg[0]
        v0 = 0.0 if self.tg[0] < 0.5 else V0
        return c + self.H * ((mm / sd0) ** 2 + (v0 / V0_SD) ** 2) + m0

    def lat_one(self, t, y, sd):
        """One horse's lateral smoother (box-bounded) on its observations: -> (mode, J term, band)."""
        n = self.n
        ab = self.phys_l + self.pri_l[0]
        r = self.pri_l[1].copy()
        c = (0.5 * self.width / self.width) ** 2
        lb, ub = L_MIN, self.width - L_MIN
        if len(t):
            kk, ww = _interp(self.tg, t)
            a, b, yy = (1 - ww) / sd, ww / sd, y / sd
            np.add.at(ab[0], kk, a * a)
            np.add.at(ab[0], kk + 1, b * b)
            np.add.at(ab[1], kk, a * b)
            np.add.at(r, kk, a * yy)
            np.add.at(r, kk + 1, b * yy)
            c += float((yy * yy).sum())
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
        Jh = _quad_band(ab, x) - 2.0 * float(r @ x) + c + _logdet(ab)[0] - self.ld_prior_l
        return x, Jh, ab

    # -- lateral: per horse, box-bounded
    def solve_l(self, obs: Obs):
        H, n = self.H, self.n
        X = np.zeros((H, n))
        A = []
        J = 0.0
        cache = self.__dict__.setdefault("_lcache", {})
        if len(cache) > 4000:
            cache.clear()
        for h in range(H):
            m = obs.h == h if len(obs.h) else np.zeros(0, bool)
            key = (h, obs.t[m].tobytes(), obs.y[m].tobytes(), obs.sd[m].tobytes()) if m.any() else (h,)
            if key not in cache:  # this horse's lateral observations unchanged: its solve is too
                cache[key] = self.lat_one(obs.t[m], obs.y[m], obs.sd[m]) if m.any() else self.lat_one(
                    np.zeros(0), np.zeros(0), np.zeros(0))
            x, Jh, ab = cache[key]
            X[h] = x
            A.append(ab)
            J += Jh
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
    Xs, J_s, J_list, pf, om, As, pw = model.solve_s(os_, X0)
    tc = time.process_time()
    Xl, J_l, Al = model.solve_l(ol)
    tc = _add_cpu("state: lateral", tc)
    J_lab = label_terms(ev, seg_lab)
    has = np.zeros(model.H, bool)
    has[np.unique(os_.h)] = True
    st = State(model.tg, Xs, Xl, As, Al, pf, om, 0.0, {}, os_, ol, has)
    st.pw = pw
    sdl_cache = model.__dict__.setdefault("_sdl_cache", {})
    if len(sdl_cache) > 4000:
        sdl_cache.clear()
    st.chol_s = []  # the along-track bands' Cholesky factors, shared with kin_costs
    st.sdS, st.sdL = st.sd(As, factors=st.chol_s), st.sd(Al, cache=sdl_cache)
    tc = _add_cpu("state: sd", tc)
    st.pin = in_view(ev, Xs, Xl, st.sdS, st.sdL, model.tg)
    tc = _add_cpu("state: in_view", tc)
    J_det, st.p_det, st.c_un, st.c_lab = miss_terms(ev, st.pin, seg_lab)
    _add_cpu("state: miss terms", tc)
    st.J = J_s + J_l + J_list + J_lab + J_det
    st.parts = dict(J_s=J_s, J_lat=J_l, J_list=J_list, J_lab=J_lab, J_det=J_det)
    return st


# ------------------------------------------------------------------------------------------------------------------
# the assignment step


MIN_GAIN = 2.0  # a proposal is kept when it lowers J by more than this (about 1 nat; below T_POST's resolution)


def _band_add_obs(ab, r, tg, t, y, sd, sign=1.0):
    """Add (sign +1) or remove (-1) point observations to a per-horse 3 x n band and its right-hand side."""
    k, w = _interp(tg, t)
    a, b = (1 - w) / sd, w / sd
    np.add.at(ab[0], k, sign * a * a)
    np.add.at(ab[0], k + 1, sign * b * b)
    np.add.at(ab[1], k, sign * a * b)
    np.add.at(r, k, sign * a * y / sd)
    np.add.at(r, k + 1, sign * b * y / sd)


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
    posterior, along-track and lateral: for its current owner with the segment left out; for a horse holding other
    segments that overlap it in time, with those segments' chunks left out (a one-horse refit: the path the horse would
    have holding this segment instead), by a low-rank downdate of the posterior; segments of equal chunk count are
    scored together."""
    ev, tg = model.ev, model.tg
    S, H = ev.n_seg, model.H
    out = np.zeros((S, H))
    order = np.argsort(ev.ch_seg, kind="stable")
    bnd = np.r_[0, np.flatnonzero(np.diff(ev.ch_seg[order])) + 1, len(order)]
    segs = ev.ch_seg[order[bnd[:-1]]]
    sizes = np.diff(bnd)
    kk, ww = _interp(tg, ev.ch_t)
    n = len(tg)
    k0, w0 = kk[order], ww[order]
    nodes = np.unique(np.r_[k0, k0 + 1])  # grid nodes the chunks interpolate between
    pos = np.full(n, -1)
    pos[nodes] = np.arange(len(nodes))
    E = np.zeros((n, len(nodes)))
    E[nodes, np.arange(len(nodes))] = 1.0
    p0, p1 = pos[k0], pos[k0 + 1]
    groups = [(m_, np.flatnonzero(sizes == m_)) for m_ in np.unique(sizes)]
    lab = np.asarray(seg_lab)
    rcost, rok = np.zeros((S, H, 2)), np.zeros((S, H, 2), bool)
    ccache = model.__dict__.setdefault("_kin_cache", {})  # lateral bands are reused while a horse's labels stay
    if len(ccache) > 64:
        ccache.clear()
    tc = time.process_time()
    for coord in ("s", "l"):
        if coord == "l":
            tc = _add_cpu("unaries: along-track kin", tc)
        A = st.As if coord == "s" else st.Al
        X = st.S if coord == "s" else st.L
        y = (ev.ch_s if coord == "s" else ev.ch_l)[order]
        R = ((ev.ch_sds if coord == "s" else ev.ch_sdl)[order]) ** 2
        for h in range(H):
            hit = ccache.get(id(A[h])) if coord == "l" else None
            if hit is not None and hit[0] is A[h]:
                blocks, W = hit[1], hit[2]
            else:
                cf = st.chol_s[h] if coord == "s" and getattr(st, "chol_s", None) else None
                if cf is None:
                    cf = cholesky_banded(A[h], lower=True)
                W = cho_solve_banded((cf, True), E)[nodes]  # A^-1 on the nodes
                blocks = {}
                for m_, qs in groups:
                    ix = bnd[qs][:, None] + np.arange(m_)[None, :]  # (G, m) chunk positions
                    a0, a1 = p0[ix], p1[ix]
                    wa = w0[ix]
                    C = ((1 - wa)[:, :, None] * (1 - wa)[:, None, :] * W[a0[:, :, None], a0[:, None, :]]
                         + (1 - wa)[:, :, None] * wa[:, None, :] * W[a0[:, :, None], a1[:, None, :]]
                         + wa[:, :, None] * (1 - wa)[:, None, :] * W[a1[:, :, None], a0[:, None, :]]
                         + wa[:, :, None] * wa[:, None, :] * W[a1[:, :, None], a1[:, None, :]])
                    blocks[m_] = 0.5 * (C + C.transpose(0, 2, 1))
                if coord == "l":
                    ccache[id(A[h])] = (A[h], blocks, W)
            mu = X[h][k0] * (1 - w0) + X[h][k0 + 1] * w0
            _refit_block(W, p0, p1, w0, y, R, mu, bnd, segs, lab, h, ev, rcost, rok, coord)
            for m_, qs in groups:
                ix = bnd[qs][:, None] + np.arange(m_)[None, :]
                C = blocks[m_].copy()
                own = lab[segs[qs]] == h
                for g in np.flatnonzero(own):  # leave the segment out for its owner
                    Cg, rr = C[g], R[ix[g]]
                    Ci = np.linalg.inv(Cg + 1e-12 * np.eye(m_))
                    Mx = Ci - np.diag(1.0 / rr)
                    Mx = 0.5 * (Mx + Mx.T)
                    lam, V = np.linalg.eigh(Mx)
                    lam = np.maximum(lam, 1e-8)
                    Cl = (V / lam) @ V.T
                    mu_g = Cl @ (Ci @ mu[ix[g]] - y[ix[g]] / rr)
                    out[segs[qs[g]], h] += _block_cost(Cl, rr, y[ix[g]] - mu_g)
                rest = np.flatnonzero(~own)
                if not len(rest):
                    continue
                ixr = ix[rest]
                P = C[rest] + np.einsum("gi,ij->gij", R[ixr], np.eye(m_))
                try:
                    cp = np.linalg.cholesky(P)
                except np.linalg.LinAlgError:
                    for j, g in enumerate(rest):
                        out[segs[qs[g]], h] += _block_cost(C[g], R[ixr[j]], y[ixr[j]] - mu[ixr[j]])
                    continue
                z = np.linalg.solve(cp, (y[ixr] - mu[ixr])[:, :, None])[:, :, 0]
                ld = 2.0 * np.log(np.diagonal(cp, axis1=1, axis2=2)).sum(1)
                out[segs[qs[rest]], h] += (z * z).sum(1) + ld - np.log(R[ixr]).sum(1)
    _add_cpu("unaries: lateral kin", tc)
    both = rok.sum(2) == 2
    out[both] = rcost.sum(2)[both]
    return out


def _cov(W, p0, p1, w0, P, Q):
    """Posterior covariance of the chunk functionals P x Q from the inverse W on the nodes."""
    wp, wq = w0[P][:, None], w0[Q][None, :]
    a0, a1, b0, b1 = p0[P][:, None], p1[P][:, None], p0[Q][None, :], p1[Q][None, :]
    return ((1 - wp) * (1 - wq) * W[a0, b0] + (1 - wp) * wq * W[a0, b1] + wp * (1 - wq) * W[a1, b0]
            + wp * wq * W[a1, b1])


def _refit_block(W, p0, p1, w0, y, R, mu, bnd, segs, lab, h, ev, rcost, rok, coord):
    """One coordinate of ``kin_costs``' one-horse refits for horse h: every segment not h's that overlaps h's segments
    in time, scored with their chunks removed from h's posterior (Woodbury downdate); into rcost / rok
    [segment, h, coordinate]."""
    own = np.flatnonzero(lab == h)
    if not len(own):
        return
    qpos = np.full(ev.n_seg, -1)
    qpos[segs] = np.arange(len(segs))
    own = own[qpos[own] >= 0]
    lo, hi = ev.seg_lo, ev.seg_hi
    groups = {}
    for q, k in enumerate(segs):
        if lab[k] == h or ev.forbid[k, h]:
            continue
        conf = own[(lo[own] <= hi[k]) & (hi[own] >= lo[k])]
        if len(conf):
            groups.setdefault(conf.tobytes(), (conf, []))[1].append(q)
    ci = 0 if coord == "s" else 1
    for conf, qs in groups.values():
        rm = np.concatenate([np.arange(bnd[qpos[c]], bnd[qpos[c] + 1]) for c in conf])
        sr = np.sqrt(R[rm])
        Sm = np.eye(len(rm)) - _cov(W, p0, p1, w0, rm, rm) / sr[:, None] / sr[None, :]
        try:
            cS = np.linalg.cholesky(0.5 * (Sm + Sm.T))
        except np.linalg.LinAlgError:
            continue
        t2 = solve_triangular(cS, (y[rm] - mu[rm]) / sr, lower=True)
        qs = np.asarray(qs)
        sizes = bnd[qs + 1] - bnd[qs]
        for m_ in np.unique(sizes):  # segments of equal chunk count together
            qm = qs[sizes == m_]
            ix = bnd[qm][:, None] + np.arange(m_)[None, :]  # (G, m)
            flat = ix.ravel()
            T = solve_triangular(cS, (_cov(W, p0, p1, w0, flat, rm) / sr[None, :]).T, lower=True)  # r x G m
            T = T.reshape(len(rm), len(qm), m_)
            a0, a1, wa = p0[ix], p1[ix], w0[ix]
            C = ((1 - wa)[:, :, None] * (1 - wa)[:, None, :] * W[a0[:, :, None], a0[:, None, :]]
                 + (1 - wa)[:, :, None] * wa[:, None, :] * W[a0[:, :, None], a1[:, None, :]]
                 + wa[:, :, None] * (1 - wa)[:, None, :] * W[a1[:, :, None], a0[:, None, :]]
                 + wa[:, :, None] * wa[:, None, :] * W[a1[:, :, None], a1[:, None, :]])
            C = C + np.einsum("rgi,rgj->gij", T, T)
            C = 0.5 * (C + C.transpose(0, 2, 1))
            res = y[ix] - (mu[ix] - np.einsum("rgi,r->gi", T, t2))
            Rg = R[ix]
            try:
                cp = np.linalg.cholesky(C + np.einsum("gi,ij->gij", Rg, np.eye(m_)))
                z = np.linalg.solve(cp, res[:, :, None])[:, :, 0]
                v = (z * z).sum(1) + 2.0 * np.log(np.diagonal(cp, axis1=1, axis2=2)).sum(1) - np.log(Rg).sum(1)
            except np.linalg.LinAlgError:
                v = np.array([_block_cost(C[g], Rg[g], res[g]) for g in range(len(qm))])
            ok = v < INF / 2
            rcost[segs[qm[ok]], h, ci] = v[ok]
            rok[segs[qm[ok]], h, ci] = True


def _block_cost(C, rr, r):
    """-2 log predictive ratio of a chunk block with prior covariance C, noise variances rr, residual r (INF when the
    predictive covariance is not positive definite)."""
    try:
        cp = np.linalg.cholesky(C + np.diag(rr))
    except np.linalg.LinAlgError:
        return INF
    z = np.linalg.solve(cp, r)
    return float(z @ z) + 2.0 * float(np.log(np.diag(cp)).sum()) - float(np.log(rr).sum())


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
    # a slot whose adjacent pair facts are in their order-wrong state places its horse uninformatively
    pws = np.zeros(L.slots.shape)
    St = model.St
    if st is not None and getattr(st, "pw", None) is not None and len(St.a):
        adj = ~St.unl
        for off in (0, 1):
            np.maximum.at(pws, (St.row[adj], St.slot[adj] + off), st.pw[adj])
    part = (L.partner if L.partner is not None else np.zeros((H, H), bool)).any(1)
    eps_h = 1.0 - (1.0 - EPS_LIST) * (1.0 - P_DROP) ** part.astype(float)  # a read with a dropped-digit partner
    for a, b in zip(bnd[:-1], bnd[1:]):
        j = o[a:b]
        tf = t[j[0]]
        r = int(np.argmin(np.abs(L.tau - tf)))
        if abs(L.tau[r] - tf) > 0.15:
            continue
        ok = 1.0 - (st.pf[r] if st is not None else model.pf0[r])
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
            e, wq = eps_h[h], pws[r, q]
            P = ok * ((1 - wq) * ((1 - e) * cnt[:, q] + e / H) + wq / H) + (1 - ok) / H
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


_MILP_MEMO: dict = {}  # input digest -> labelling: an assignment already solved in this process is not solved again
MILP_MEMO_MAX = 256
MILP_NODE_LIMIT = 2  # branch-and-bound nodes: HiGHS stops here (deterministic) and returns its incumbent
MILP_STATS: list = []  # per MILP solved (memo hits excluded): (HiGHS status, relative gap, B&B nodes, wall s); a
# MILP with nodes <= 1 closed at the root, so its wall is the root time


def _digest(*arrays) -> bytes:
    h = hashlib.sha256()
    for a in arrays:
        if a is None:
            h.update(b"none")
            continue
        a = np.ascontiguousarray(a)
        h.update(str((a.dtype.str, a.shape)).encode())
        h.update(a.tobytes())
    return h.digest()


def milp(U, ev: Evidence, allow=None):
    """min sum U x - sum link rewards z over x[segment, horse] binary (a segment has at most one horse; leaving it
    out is clutter, cost 0), one segment per horse per frame, z[link, horse] <= x at both ends. A pair whose cost
    exceeds every link reward its segment could collect is never in an optimum and gets no variable. Results are
    memoised by a digest of every input (U, allow, the links, the segments' frame spans, the exclusions)."""
    key = _digest(U, allow, ev.links, ev.seg_lo, ev.seg_hi, ev.forbid)
    hit = _MILP_MEMO.get(key)
    if hit is not None:
        return hit.copy()
    out = _milp_solve(U, ev, allow)
    if len(_MILP_MEMO) >= MILP_MEMO_MAX:
        _MILP_MEMO.clear()
    _MILP_MEMO[key] = out.copy()
    return out


def _milp_solve(U, ev: Evidence, allow=None):
    from scipy.optimize import Bounds, LinearConstraint, milp as _milp

    S = U.shape[0]
    prob = _milp_problem(U, ev, allow)
    if prob is None:
        return np.full(S, -1)
    c, A, ub, vs, vh = prob
    nx = len(vs)
    t0 = time.perf_counter()
    res = _milp(c, constraints=LinearConstraint(A, -np.inf, ub), integrality=np.ones(len(c)), bounds=Bounds(0, 1),
                options=dict(node_limit=MILP_NODE_LIMIT))
    gap, nodes = getattr(res, "mip_gap", None), getattr(res, "mip_node_count", None)
    MILP_STATS.append((int(res.status), np.nan if gap is None else float(gap), -1 if nodes is None else int(nodes),
                       time.perf_counter() - t0))
    if res.x is None:
        raise RuntimeError(f"segment MILP failed: {res.message}")
    out = np.full(S, -1)
    on = np.round(res.x[:nx]) > 0.5
    out[vs[on]] = vh[on]
    return out


def _milp_problem(U, ev: Evidence, allow=None):
    """The segment MILP's arrays: objective c over the x (segment, horse) then z (link, horse) variables, the
    constraint matrix A (CSR) with upper bounds ub (rows: one horse per segment, one segment per horse per frame, z <= x
    at both link ends, link aggregation), and the x variables' segments and horses; None without any variable."""
    from scipy.sparse import coo_matrix

    S, H = U.shape
    links = ev.links
    reward = np.zeros(S)
    if len(links):
        np.add.at(reward, links[:, 0].astype(int), links[:, 2])
        np.add.at(reward, links[:, 1].astype(int), links[:, 2])
    ok = np.isfinite(U) & (U < INF / 2) & (~ev.forbid) & (U <= reward[:, None])
    if allow is not None:
        ok &= allow
    vs, vh = np.nonzero(ok)  # x variables
    nx = len(vs)
    if not nx:
        return None
    vid = np.full((S, H), -1)
    vid[vs, vh] = np.arange(nx)
    rows, cols = [vs.copy()], [np.arange(nx)]  # one horse per segment
    r = S
    lo, hi = np.asarray(ev.seg_lo), np.asarray(ev.seg_hi)
    # one segment per horse per frame: per first frame x of a segment (ascending), per horse (ascending), the variables
    # of the segments covering x (ascending), when there are at least two
    xs = np.unique(lo)
    cover = (lo[None, :] <= xs[:, None]) & (hi[None, :] >= xs[:, None])  # (frames x, segments)
    has = cover[:, :, None] & (vid >= 0)[None, :, :]  # (x, segment, horse)
    keep = has.sum(1) > 1  # (x, horse) rows that exist
    ex, eh, es = np.nonzero(has.transpose(0, 2, 1) & keep[:, :, None])  # entries in (x, horse, segment) order
    row_id = np.cumsum(keep.ravel()) - 1
    rows.append(r + row_id[ex * H + eh])
    cols.append(vid[es, eh])
    r += int(keep.sum())
    n_ub1 = r
    # z only where both ends can hold the horse, in (link, horse) order
    la, lb = links[:, 0].astype(int) if len(links) else np.zeros(0, int), links[:, 1].astype(int) if len(links) else \
        np.zeros(0, int)
    zl, zh = np.nonzero((vid[la] >= 0) & (vid[lb] >= 0)) if len(links) else (np.zeros(0, int), np.zeros(0, int))
    zc = -links[zl, 2] if len(links) else np.zeros(0)
    za, zb = vid[la[zl], zh], vid[lb[zl], zh]
    nz = len(zl)
    zi = nx + np.arange(nz)
    if nz:
        rr = r + np.arange(2 * nz)
        rows += [rr, rr]
        cols += [np.repeat(zi, 2), np.ravel(np.c_[za, zb])]
        r += 2 * nz
    rows = np.concatenate(rows)
    cols = np.concatenate(cols)
    vals = np.ones(len(rows))
    if nz:
        vals[-2 * nz:] = -1.0  # the x entries of the link rows
    # link aggregation (valid, tightens the LP): a segment's links with one horse to partners that pairwise overlap in
    # frames can hold that horse at most once together, so their z sum to at most the segment's x (greedy cliques of
    # partners, in order of first frame); groups (segment, horse) in order of their first link, forward then backward
    ar, ac, av = [], [], []
    for own, other in ((la[zl], lb[zl]), (lb[zl], la[zl])):
        code = own * H + zh
        order = np.argsort(code, kind="stable")
        bnd = np.r_[0, np.flatnonzero(np.diff(code[order])) + 1, nz]
        firsts = [(order[i], i, j) for i, j in zip(bnd[:-1], bnd[1:]) if j - i > 1]
        for _f, i, j in sorted(firsts):
            mem = order[i:j]
            part = other[mem]
            o2 = np.lexsort((mem, lo[part]))
            rest = list(zip(part[o2].tolist(), mem[o2].tolist()))
            a_, h_ = int(own[mem[0]]), int(zh[mem[0]])
            while len(rest) > 1:
                cl, kp = [rest[0]], []
                for p in rest[1:]:
                    if all(lo[p[0]] <= hi[q[0]] and lo[q[0]] <= hi[p[0]] for q in cl):
                        cl.append(p)
                    else:
                        kp.append(p)
                rest = kp
                if len(cl) > 1:
                    ar += [r] * (len(cl) + 1)
                    ac += [nx + k for _b, k in cl] + [vid[a_, h_]]
                    av += [1.0] * len(cl) + [-1.0]
                    r += 1
    if ar:
        rows, cols, vals = np.r_[rows, ar], np.r_[cols, ac], np.r_[vals, av]
    A = coo_matrix((vals, (rows, cols)), shape=(r, nx + nz)).tocsr()
    c = np.r_[U[vs, vh], np.asarray(zc, float)]
    return c, A, np.r_[np.ones(n_ub1), np.zeros(r - n_ub1)], vs, vh


def unaries(model: Model, st: State, seg_lab):
    """The assignment step's increments of J per (segment, horse), relative to clutter, at the state: the kinematic
    block (``kin_costs``: one-horse refits), the list's rank-count vote, the detection terms, the segment base, reads
    and silk."""
    ev = model.ev
    lab = np.asarray(seg_lab)
    U = kin_costs(model, st, lab)
    tc = time.process_time()
    U = U + list_votes(model, st, lab)
    tc = _add_cpu("unaries: list votes", tc)
    U += segment_base(ev)[:, None] + ev.read_cost + ev.silk_cost
    if st.c_un is not None:  # detection terms: a labelled frame instead of a (possibly in-view) missed one
        fi = np.searchsorted(ev.view["frames"], ev.fr)
        for h in range(model.H):
            U[:, h] += np.bincount(ev.seg, st.c_lab[h, fi] - st.c_un[h, fi], ev.n_seg)
    U[ev.forbid] = INF
    _add_cpu("unaries: detection terms, base", tc)
    return U


@dataclass
class Result:
    seg_lab: np.ndarray
    state: State
    U: np.ndarray
    marg: np.ndarray | None  # None for a solve asked for no marginals
    log: list
    ends: list = field(default_factory=list)  # the DESCENTS lowest alternation ends (labellings)


REFIT_ROUNDS = 30  # alternation proposals (MILP + full state step) per descent at most
DESCENTS = 2  # alternation ends that are polished: the lowest by J
LIST_SPLIT = 7.0  # s: block length of the split-list inits (off the 10 s grid of the legend masking cells)
POLISH_STEPS = 16  # polish full state steps per descent at most
EXACT_TAIL, TAIL_SLACK = 8, 50.0  # a polish round with no predicted gain gives full steps to its best 8 within +50
NEWTON_IT = 10  # one-horse refit: Newton steps at most
LIST_EXACT, LIST_EM = 30.0, 3  # a refit whose frozen-weight list change exceeds this re-estimates the fault weights
# (LIST_EM rounds) and takes the list term exactly
REFIT_SWEEPS, SWEEP_IT = 2, 3  # Gauss-Seidel sweeps over a candidate's horses after their first refits, Newton steps
REFIT_TOP, SWEEP_TOP = 200, 60  # polish candidates re-scored by one-horse refits (the lowest by the closed-form
# score), and of those again with the sweeps (the lowest by their refit score)
SEG_SLACK = 50.0  # single-segment polish candidates: labels whose unary is at most this above the current one
POST_K, POST_BAND = 20, 20.0  # alternatives given a full state step for the probabilities: the K lowest, any within
# POST_BAND J of the kept labelling
T_PROB = 50.0  # the probability output's temperature: fitted on a race-hashed half of the dev null-pair races (0333)
POST_PRUNE = 2.0 * T_PROB * np.log(1e4)  # J: an alternative whose closed-form screen is above this (weight < 1e-4)
# keeps that screen as its dJ
POST_FULL = 100.0  # J: of the POST_K lowest, only those screened below this get a full state step
T_POST = 10.0  # J's per-detection terms treat correlated errors as independent: the posterior is tempered by this
ABREAST_S, ABREAST_L, ABREAST_T = 1.5, 3.0, 5.0  # m, units, s: two paths this close for this long run level
EXCH_S, EXCH_L = 3.0, 4.0  # m, units: two paths this close at a camera cut may exchange labels from it on
CHAIN_NEAR, CHAIN_K, CHAIN_DT = 10.0, 8, 1.0  # m, chains, s: an inconsistent chain trades labels with the
# CHAIN_K chains whose boxes are closest along the track (median, within CHAIN_NEAR) within CHAIN_DT of its change
LONG_N, REL_GATE = 10, 15.0  # detections, m: the single relabels' segments and horses


class Refit:
    """One-horse refits at a state: a horse re-solved alone with a changed set of segments (its along-track mode under
    physics, prior, its observations and its order statements against the other horses' paths at the state, the
    fault weights frozen at the state; its lateral smoother exactly), the posterior sds held at the state."""

    def __init__(self, model: Model, st: State, lab, extra_s=None, extra_l=None, s_shift=None):
        ev, St = model.ev, model.St
        self.model, self.st, self.lab = model, st, np.asarray(lab)
        self.ys = ev.ch_s if s_shift is None else ev.ch_s + s_shift(ev.ch_s)
        self.ex_s = extra_s if extra_s is not None and len(extra_s.h) else None
        self.ex_l = extra_l if extra_l is not None and len(extra_l.h) else None
        self.wt = ((1 - st.pf[St.row]) * np.where(St.unl, 1 - st.om[St.b, St.row], 1.0 - st.pw)
                   if len(St.a) else np.zeros(0))
        self.order = np.argsort(ev.ch_seg, kind="stable")
        self.cb = np.r_[0, np.cumsum(np.bincount(ev.ch_seg, minlength=ev.n_seg))]
        self.memo, self.lmemo, self.ld0 = {}, {}, {}
        self.smemo = {}  # (screen, labelling bytes) -> score: a labelling scored at this state is not scored again
        self.qmemo = {}  # (horse, dropped chunks, added chunks) -> horse_quick's result at this state
        self.q_calls = self.q_hits = 0
        self.stq = []  # per horse: its statements, the sign of its side, the other horse
        for h in range(model.H):
            ia, ib = np.flatnonzero(St.a == h), np.flatnonzero(St.b == h)
            self.stq.append((np.r_[ia, ib], np.r_[np.ones(len(ia)), -np.ones(len(ib))], np.r_[St.b[ia], St.a[ib]]))
        # label terms by segment (the change of J_lab is a sum over changed segments and their links)
        self.seg_cost = segment_base(ev)[:, None] + ev.read_cost + ev.silk_cost
        self.seg_links = [[] for _ in range(ev.n_seg)]
        for i, (a, b, _w) in enumerate(ev.links):
            self.seg_links[int(a)].append(i)
            self.seg_links[int(b)].append(i)
        # detection terms by frame: per segment its frames; per frame the in-view sum, labelled count, unlabelled sum
        if st.pin.shape[1]:
            fi = np.searchsorted(ev.view["frames"], ev.fr)
            o = np.argsort(ev.seg, kind="stable")
            db = np.r_[0, np.cumsum(np.bincount(ev.seg, minlength=ev.n_seg))]
            self.seg_fr = [np.unique(fi[o[db[k]:db[k + 1]]]) for k in range(ev.n_seg)]
            self.labd = np.zeros(st.pin.shape, bool)
            for k in np.flatnonzero(self.lab >= 0):
                self.labd[self.lab[k], self.seg_fr[k]] = True
            self.p_sum = st.pin.sum(0)
            self.n_lab = self.labd.sum(0)
            self.u_sum = (st.pin * ~self.labd).sum(0)

    def d_lab(self, ch, new):
        """J_lab(new) - J_lab(state's labelling); ``ch`` the changed segments."""
        ev, lab = self.model.ev, self.lab
        o, n = lab[ch], new[ch]
        d = float(np.where(n >= 0, self.seg_cost[ch, np.maximum(n, 0)], 0.0).sum()
                  - np.where(o >= 0, self.seg_cost[ch, np.maximum(o, 0)], 0.0).sum())
        if ev.forbid[ch[n >= 0], n[n >= 0]].any():
            d += INF
        li = sorted({i for k in ch for i in self.seg_links[k]})
        if li:
            L = ev.links[li]
            a, b, w = L[:, 0].astype(int), L[:, 1].astype(int), L[:, 2]
            d += float(w[(new[a] != new[b]) | (new[a] < 0)].sum() - w[(lab[a] != lab[b]) | (lab[a] < 0)].sum())
        return d

    def view_rows(self, hs, S, L):
        """``in_view`` for the rows ``hs`` (the frames' grid interpolation kept from the first call)."""
        st, ev = self.st, self.model.ev
        V = ev.view
        if not hasattr(self, "_vk"):
            self._vk = _interp(self.model.tg, V["t"])
        k, w = self._vk

        def at(X):
            return X[:, k] * (1 - w) + X[:, k + 1] * w

        s_, l_, ss, sl = at(S[hs]), at(L[hs]), at(st.sdS[hs]), at(st.sdL[hs])
        A = V["A"]
        cx = s_ * A[None, :, 0, 0] + l_ * A[None, :, 1, 0] + A[None, :, 2, 0]
        cy = s_ * A[None, :, 0, 1] + l_ * A[None, :, 1, 1] + A[None, :, 2, 1]
        ux = np.hypot(ss * A[None, :, 0, 0], sl * A[None, :, 1, 0]) + VIEW_PX
        uy = np.hypot(ss * A[None, :, 0, 1], sl * A[None, :, 1, 1]) + VIEW_PX
        (x0, x1), (y0, y1) = V["x"], V["y"]
        return ndtr((cx - x0) / ux) * ndtr((x1 - cx) / ux) * ndtr((cy - y0) / uy) * ndtr((y1 - cy) / uy)

    def d_det(self, hs, new, S, L):
        """J_det at the paths S, L (rows ``hs`` changed) and labelling ``new`` minus the state's J_det."""
        st, ev = self.st, self.model.ev
        if not st.pin.shape[1]:
            return 0.0
        rows = self.view_rows(hs, S, L)
        lb = np.zeros(rows.shape, bool)
        for i, h in enumerate(hs):
            for k in np.flatnonzero(new == h):
                lb[i, self.seg_fr[k]] = True
        p_sum = self.p_sum - st.pin[hs].sum(0) + rows.sum(0)
        n_lab = self.n_lab - self.labd[hs].sum(0) + lb.sum(0)
        u_sum = self.u_sum - (st.pin[hs] * ~self.labd[hs]).sum(0) + (rows * ~lb).sum(0)
        V = ev.view
        p = np.clip((V["n"] + 1.0) / (p_sum + 2.0), P_DET_MIN, V["cap"])
        return float((n_lab * (-2.0 * np.log(p)) + u_sum * (-2.0 * np.log(1.0 - p))).sum()) - st.parts["J_det"]

    def chunks(self, segs):
        segs = np.asarray(segs, int)
        if not len(segs):
            return np.zeros(0, int)
        return np.concatenate([self.order[self.cb[k]:self.cb[k + 1]] for k in segs])

    def horse(self, h, segs, S=None, wt_all=None):
        """Horse h holding the segments ``segs``, the other horses at ``S`` (default: the state's paths) ->
        (quadratic + log det part of J_s, J_lat, s mode, lat mode)."""
        segs = np.sort(np.asarray(segs, int))
        key = (int(h), segs.tobytes())
        if S is None and wt_all is None and key in self.memo:
            return self.memo[key]
        m, ev, st, St = self.model, self.model.ev, self.st, self.model.St
        S_ctx = st.S if S is None else S
        tg = m.tg
        c = self.chunks(segs)
        t, y, sd = ev.ch_t[c], self.ys[c], ev.ch_sds[c]
        if self.ex_s is not None:
            e = self.ex_s.h == h
            t, y, sd = np.r_[t, self.ex_s.t[e]], np.r_[y, self.ex_s.y[e]], np.r_[sd, self.ex_s.sd[e]]
        ab = m.phys_s + m.pri_s[0]
        r = m.pri_s[1].copy()
        _band_add_obs(ab, r, tg, t, y, sd)
        cst = float(((y / sd) ** 2).sum())
        q, sg, oth = self.stq[h]
        k, w = m.st_k[q], m.st_w[q]
        xo = S_ctx[oth, k] * (1 - w) + S_ctx[oth, k + 1] * w
        wt, eps = (self.wt if wt_all is None else wt_all)[q], St.eps[q]
        n = m.n

        def obj_at(x):
            quad = _quad_band(ab, x) - 2.0 * float(r @ x) + cst
            d = sg * (x[k] * (1 - w) + x[k + 1] * w - xo)
            return quad + float((wt * _f_order(d, eps)).sum()), quad, d

        x = (st.S[h] if S is None else S[h]).copy()
        obj, quad, d = obj_at(x)
        Hb = ab
        for _ in range(NEWTON_IT if S is None else SWEEP_IT):
            _p, _f, gd, hd = _p_order(d, eps)
            g = 2.0 * (_band_matvec(ab, x) - r)
            g += np.bincount(k, wt * gd * sg * (1 - w), n) + np.bincount(k + 1, wt * gd * sg * w, n)
            cw = 0.5 * wt * hd
            Hb = ab.copy()
            Hb[0] += np.bincount(k, cw * (1 - w) ** 2, n) + np.bincount(k + 1, cw * w * w, n)
            Hb[1] += np.bincount(k, cw * w * (1 - w), n)
            step = cho_solve_banded((cholesky_banded(Hb, lower=True), True), -0.5 * g)
            if float(-g @ step) < 1e-6:
                break
            lam, done = 1.0, False
            while lam > 1e-4:
                o2, q2, d2 = obj_at(x + lam * step)
                if o2 <= obj:
                    done = True
                    break
                lam *= 0.5
            if not done:
                break
            x, gain = x + lam * step, obj - o2
            obj, quad, d = o2, q2, d2
            if gain < 1e-4:
                break
        J_q = quad + _logdet(Hb)[0]
        xl, J_l = self.lat(h, segs)
        out = (J_q, J_l, x, xl)
        if S is None and wt_all is None:
            if len(self.memo) > 4000:
                self.memo.clear()
            self.memo[key] = out
        return out

    def list_part(self, S, hs):
        """Frozen-weight list cost of the statements touching horses ``hs`` at the paths ``S``."""
        m, St = self.model, self.model.St
        on = np.zeros(len(St.a), bool)
        for h in hs:
            on[self.stq[h][0]] = True
        q = np.flatnonzero(on)
        k, w = m.st_k[q], m.st_w[q]
        d = S[St.a[q], k] * (1 - w) + S[St.a[q], k + 1] * w - S[St.b[q], k] * (1 - w) - S[St.b[q], k + 1] * w
        return float((self.wt[q] * _f_order(d, St.eps[q])).sum())

    def horse_quick(self, h, drop, add):
        """Closed-form along-track change of horse h's J_s when its render chunks ``drop`` leave and ``add`` join, under
        the state's marginal band (the order factors as the quadratic at the state) -> (dJ_s, s mode); memoised per
        (horse, dropped chunks, added chunks) at this state."""
        key = (int(h), np.asarray(drop).tobytes(), np.asarray(add).tobytes())
        self.q_calls += 1
        hit = self.qmemo.get(key)
        if hit is not None:
            self.q_hits += 1
            return hit
        if len(self.qmemo) > 20000:
            self.qmemo.clear()
        out = self.qmemo[key] = self._horse_quick(h, drop, add)
        return out

    def _horse_quick(self, h, drop, add):
        st, tg, ev = self.st, self.model.tg, self.model.ev
        A, X = st.As[h], st.S[h]
        ld0 = self.ld0.get(h)
        if ld0 is None:
            ld0 = self.ld0[h] = _logdet(A)[0]
        ab, r = A.copy(), _band_matvec(A, X)
        q0 = float(X @ r)
        dc = 0.0
        for c, sgn in ((drop, -1.0), (add, 1.0)):
            if len(c):
                _band_add_obs(ab, r, tg, ev.ch_t[c], self.ys[c], ev.ch_sds[c], sgn)
                dc += sgn * float(((self.ys[c] / ev.ch_sds[c]) ** 2).sum())
        ab[0] = np.maximum(ab[0], 1e-12)
        try:
            cf = cholesky_banded(ab, lower=True)
        except np.linalg.LinAlgError:
            return INF, X
        x = cho_solve_banded((cf, True), r)
        return dc - float(r @ x) + q0 + 2.0 * float(np.log(cf[0]).sum()) - ld0, x

    def _memo(self, key, f):
        hit = self.smemo.get(key)
        if hit is None:
            if len(self.smemo) > 20000:
                self.smemo.clear()
            hit = self.smemo[key] = f()
        return hit

    def score_quick(self, new):
        """Closed-form estimate of J(new) - J(state's labelling): ``horse_quick`` per changed horse, its lateral
        smoother exactly, J_lab exactly, J_det at the new paths (memoised per labelling at this state)."""
        return self._memo(("quick", np.asarray(new).tobytes()), lambda: self._score_quick(new))

    def _score_quick(self, new):
        st, lab = self.st, self.lab
        ch = np.flatnonzero(new != lab)
        if not len(ch):
            return 0.0
        hs = np.unique(np.r_[lab[ch], new[ch]])
        hs = hs[hs >= 0]
        dJ = self.d_lab(ch, new)
        S, L = st.S.copy(), st.L.copy()
        for h in hs:
            drop = self.chunks(ch[(lab[ch] == h)])
            add = self.chunks(ch[(new[ch] == h)])
            d_s, S[h] = self.horse_quick(h, drop, add)
            Jn, Jb = self.lat(h, np.flatnonzero(new == h)), self.lat(h, np.flatnonzero(lab == h))
            dJ += d_s + Jn[1] - Jb[1]
            L[h] = Jn[0]
        return float(dJ + self.d_det(hs, new, S, L))

    def lat(self, h, segs):
        """(lateral mode, J_lat term) of horse h holding ``segs``."""
        segs = np.sort(np.asarray(segs, int))
        key = (int(h), segs.tobytes())
        if key not in self.lmemo:  # the lateral smoother does not depend on the other horses
            ev = self.model.ev
            c = self.chunks(segs)
            tl, yl, sdl = ev.ch_t[c], ev.ch_l[c], ev.ch_sdl[c]
            if self.ex_l is not None:
                e = self.ex_l.h == h
                tl, yl, sdl = np.r_[tl, self.ex_l.t[e]], np.r_[yl, self.ex_l.y[e]], np.r_[sdl, self.ex_l.sd[e]]
            if len(self.lmemo) > 4000:
                self.lmemo.clear()
            self.lmemo[key] = self.model.lat_one(tl, yl, sdl)[:2]
        return self.lmemo[key]

    def score(self, new, sweeps=True):
        """One-horse-refit estimate of J(new) - J(state's labelling): every horse whose segments change re-solved alone
        (REFIT_SWEEPS Gauss-Seidel sweeps over them when several change, each against the others' re-solved paths), the
        list's statements touching them at the re-solved paths with the fault weights frozen (above LIST_EXACT: the
        list's chains exactly; with sweeps, first the weights re-estimated at the re-solved paths and the horses
        re-solved under them, LIST_EM rounds), J_lab exactly, J_det at the re-solved paths (memoised per labelling and
        sweeps at this state)."""
        return self._memo(("refit", bool(sweeps), np.asarray(new).tobytes()), lambda: self._score(new, sweeps))

    def _score(self, new, sweeps):
        st, lab = self.st, self.lab
        ch = np.flatnonzero(new != lab)
        if not len(ch):
            return 0.0
        hs = np.unique(np.r_[lab[ch], new[ch]])
        hs = hs[hs >= 0]
        dJ = self.d_lab(ch, new)
        S0, S, L = st.S.copy(), st.S.copy(), st.L.copy()
        segs = {h: np.flatnonzero(new == h) for h in hs}
        cur = {}
        for h in hs:
            Jb = self.horse(h, np.flatnonzero(lab == h))
            dJ -= Jb[0] + Jb[1]
            S0[h] = Jb[2]
            cur[h] = self.horse(h, segs[h])
            S[h], L[h] = cur[h][2], cur[h][3]
        for _ in range(REFIT_SWEEPS if sweeps and len(hs) > 1 else 0):
            for h in hs:
                cur[h] = self.horse(h, segs[h], S)
                S[h], L[h] = cur[h][2], cur[h][3]
        dJ += sum(cur[h][0] + cur[h][1] for h in hs)
        lp = self.list_part(S, hs) - self.list_part(S0, hs)
        if lp > LIST_EXACT:  # a large list cost may switch fault chains: their weights re-estimated at the new paths,
            # the moved horses re-solved under them, the list term exactly
            m, St = self.model, self.model.St
            dJ -= sum(cur[h][0] for h in hs)
            for _ in range(LIST_EM if sweeps else 0):
                d = m._stmt_delta(S)
                _J, pf, om, pw = list_chains(m.Lr, St, _p_order(d, St.eps)[1], m.H, d)
                wt2 = (1 - pf[St.row]) * np.where(St.unl, 1 - om[St.b, St.row], 1.0 - pw)
                for h in hs:
                    cur[h] = self.horse(h, segs[h], S, wt2)
                    S[h] = cur[h][2]
            dJ += sum(cur[h][0] for h in hs)
            d = m._stmt_delta(S)
            lp = list_chains(m.Lr, St, _p_order(d, St.eps)[1], m.H, d)[0] - st.parts["J_list"]
        dJ += lp
        return float(dJ + self.d_det(hs, new, S, L))


def crossovers(ev: Evidence, lab, other):
    """Labellings that take ``other``'s labels over a run of consecutive parts (in time) of the segments where the two
    differ; a part links differing segments that overlap in time and share a label (old or new), so every run keeps
    per-frame exclusivity. -> list of (description, labelling)."""
    ch = np.flatnonzero(lab != other)
    if not len(ch):
        return []
    par = np.arange(len(ch))

    def root(i):
        while par[i] != i:
            par[i] = par[par[i]]
            i = par[i]
        return i

    for i in range(len(ch)):
        a = ch[i]
        la = {int(lab[a]), int(other[a])} - {-1}
        for j in range(i + 1, len(ch)):
            b = ch[j]
            if ev.seg_lo[a] <= ev.seg_hi[b] and ev.seg_hi[a] >= ev.seg_lo[b] and la & {int(lab[b]), int(other[b])}:
                par[root(i)] = root(j)
    r = np.array([root(i) for i in range(len(ch))])
    parts = sorted((ch[r == u] for u in np.unique(r)), key=lambda g: ev.seg_lo[g].min())
    out = []
    for i in range(len(parts)):
        for j in range(i, len(parts)):
            g = np.concatenate(parts[i:j + 1])
            new = lab.copy()
            new[g] = other[g]
            out.append((f"crossover of parts {i + 1}-{j + 1} of {len(parts)} ({len(g)} segments)", new))
    return out


def pair_exchanges(ev: Evidence, lab, st: State):
    """Every two labelled horses exchange labels on every segment inside a window: the whole race, or from a camera
    cut (a time where >= 3 labelled segments start together) to the race's end or from its start to the cut, where
    the two paths are within EXCH_S along and EXCH_L across at the cut (elsewhere the exchange makes both paths
    jump there). -> list of (description, labelling)."""
    tg = st.tg
    lo = np.full(ev.n_seg, np.inf)
    hi = np.full(ev.n_seg, -np.inf)
    np.minimum.at(lo, ev.seg, ev.t)
    np.maximum.at(hi, ev.seg, ev.t)
    st_t, cnt = np.unique(np.round(lo[lab >= 0] * 5) / 5, return_counts=True)
    cuts = st_t[cnt >= 3]
    t0, t1 = tg[0] - 1.0, tg[-1] + 1.0
    wins = [(t0, t1, None)] + [(c, t1, c) for c in cuts] + [(t0, c, c) for c in cuts]
    present = np.unique(lab[lab >= 0])
    out, seen = [], {lab.tobytes()}
    for x0, x1, tc in wins:
        inside = (lo >= x0 - 0.15) & (hi < x1 - 0.05)
        k = None if tc is None else int(np.clip(np.searchsorted(tg, tc), 0, len(tg) - 1))
        for ia, a in enumerate(present):
            ma = inside & (lab == a)
            for b in present[ia + 1:]:
                if k is not None and (abs(st.S[a, k] - st.S[b, k]) > EXCH_S or abs(st.L[a, k] - st.L[b, k]) > EXCH_L):
                    continue
                mb = inside & (lab == b)
                if not (ma.any() or mb.any()):
                    continue
                new = lab.copy()
                new[ma], new[mb] = b, a
                key = new.tobytes()
                if key not in seen:
                    seen.add(key)
                    out.append((f"exchange #{ev.horses[a]}/#{ev.horses[b]} over {max(x0, tg[0]):.1f}-"
                                f"{min(x1, tg[-1]):.1f} s", new))
    return out


def _seg_bounds(ev: Evidence):
    lo = np.full(ev.n_seg, np.inf)
    hi = np.full(ev.n_seg, -np.inf)
    np.minimum.at(lo, ev.seg, ev.t)
    np.maximum.at(hi, ev.seg, ev.t)
    return lo, hi


def _seg_first_s(ev: Evidence):
    o = np.argsort(ev.t, kind="stable")
    first = np.full(ev.n_seg, -1)
    first[ev.seg[o[::-1]]] = o[::-1]
    return ev.s[first]


def tracker_chains(ev: Evidence):
    """Chains of segments joined by unambiguous tracker links (a link a -> b where a has no other outgoing link and b
    no other incoming one), through segments with P(horse) >= 0.5 only. -> (successor per segment, -1 none; list of
    chains, each a list of segments in time order)."""
    n = np.bincount(ev.seg, minlength=ev.n_seg)
    ph = np.bincount(ev.seg, ev.p_horse, ev.n_seg) / np.maximum(n, 1)
    succ = np.full(ev.n_seg, -1)
    pred = np.full(ev.n_seg, -1)
    if len(ev.links):
        la, lb = ev.links[:, 0].astype(int), ev.links[:, 1].astype(int)
        n_out, n_in = np.bincount(la, minlength=ev.n_seg), np.bincount(lb, minlength=ev.n_seg)
        for x, y in zip(la, lb):
            if n_out[x] == 1 and n_in[y] == 1 and ph[x] >= 0.5 and ph[y] >= 0.5:
                succ[x], pred[y] = y, x
    chains = []
    for h in np.flatnonzero((pred < 0) & (ph >= 0.5)):
        c = [int(h)]
        while succ[c[-1]] >= 0:
            c.append(int(succ[c[-1]]))
        chains.append(c)
    return succ, chains


def chain_moves(ev: Evidence, lab, st: State):
    """Moves of whole tracker chains (tracker_chains), carrying every linked continuation, clutter included. A chain
    is inconsistent where its label changes along it (a horse to another or to clutter); for each inconsistent chain c
    and the CHAIN_K chains whose boxes are closest to c's along the track in the frames within CHAIN_DT of its change
    point (median, within CHAIN_NEAR; a gate on detections and time overlap, not on the fitted paths): c's label carried over its continuation, the exchange of c's and d's labels and the two 3-cycles of c, d, e,
    over the whole chains (each takes the other's majority label) and from c's change point onwards. A horse that ends
    up on a moved chain loses its other overlapping segments. -> list of (description, labelling)."""
    lo, hi = _seg_bounds(ev)
    _succ, chains = tracker_chains(ev)
    nd = np.bincount(ev.seg, minlength=ev.n_seg)
    cid = np.full(ev.n_seg, -1)
    for i, c in enumerate(chains):
        cid[c] = i

    def major(c, segs=None):
        segs = c if segs is None else segs
        ls = lab[segs]
        if not (ls >= 0).any():
            return -1
        return int(np.bincount(ls[ls >= 0], weights=nd[segs][ls >= 0]).argmax())

    # per chain: frames and along-track positions of its boxes
    fr_t = dict(zip(ev.fr.tolist(), ev.t.tolist()))
    fr_s = {}
    for i, c in enumerate(chains):
        k = np.flatnonzero(np.isin(ev.seg, c))
        fr_s[i] = dict(zip(ev.fr[k].tolist(), ev.s[k].tolist()))
    incons = []
    for i, c in enumerate(chains):
        ls = lab[c]
        for q in range(1, len(c)):
            if ls[q] != ls[q - 1] and (ls[q] >= 0 or ls[q - 1] >= 0):
                incons.append((i, q))
                break
    out, seen = [], {lab.tobytes()}

    def near_chains(i, T):  # the CHAIN_K chains whose boxes are closest along the track to c's around its change
        fi = {f: x for f, x in fr_s[i].items() if abs(fr_t[f] - T) <= CHAIN_DT}
        res = []
        for j in range(len(chains)):
            if j == i:
                continue
            common = fi.keys() & fr_s[j].keys()
            if common:
                d = float(np.median([abs(fi[f] - fr_s[j][f]) for f in common]))
                if d <= CHAIN_NEAR:
                    res.append((d, j))
        return [j for _d, j in sorted(res)[:CHAIN_K]]

    def apply(assign):  # assign: {chain index: (segments, new label)}
        new = lab.copy()
        moved = np.zeros(ev.n_seg, bool)
        for segs, y in assign.values():
            new[segs] = y
            moved[segs] = True
        for segs, y in assign.values():
            if y < 0:
                continue
            t0, t1 = ev.seg_lo[segs].min(), ev.seg_hi[segs].max()
            new[(~moved) & (new == y) & (ev.seg_lo <= t1) & (ev.seg_hi >= t0)] = -1
        return new

    def add(desc, new):
        key = new.tobytes()
        if key not in seen:
            seen.add(key)
            out.append((desc, new))

    H = ev.horses
    for i, q in incons:
        c = chains[i]
        T = lo[c[q]]
        nb = near_chains(i, T)
        for whole in (True, False):
            def part(j):
                cj = chains[j]
                return cj if whole else [x for x in cj if lo[x] >= T - 0.11]
            segs_i = part(i)
            Li = major(c) if whole else (int(lab[c[q - 1]]) if lab[c[q - 1]] >= 0 else major(c))
            span = "whole chains" if whole else f"from {T:.1f} s"
            # c made consistent: its label carried over its continuation
            if Li >= 0:
                add(f"chain {i} carried as #{H[Li]} {span}", apply({i: (segs_i, Li)}))
            for a_, j in enumerate(nb):
                sj = part(j)
                if not sj:
                    continue
                Lj = major(chains[j], sj)
                if Lj == Li:
                    continue
                add(f"chain exchange {i}/{j} {span}", apply({i: (segs_i, Lj), j: (sj, Li)}))
                for e in nb[a_ + 1:]:
                    se = part(e)
                    if not se:
                        continue
                    Le = major(chains[e], se)
                    if len({Li, Lj, Le}) < 3:
                        continue
                    add(f"chain cycle {i}->{j}->{e} {span}", apply({i: (segs_i, Lj), j: (sj, Le), e: (se, Li)}))
                    add(f"chain cycle {i}->{e}->{j} {span}", apply({i: (segs_i, Le), e: (se, Lj), j: (sj, Li)}))
    return out


def single_relabels(ev: Evidence, lab, st: State):
    """Relabels of long segments (>= LONG_N detections; clutter ones with P(horse) >= 0.5): each segment, with its
    unambiguous tracker continuation (tracker_chains), to clutter or to every horse whose path is within REL_GATE along
    the track of the segment's first position; a horse's overlapping segments become clutter.
    -> list of (segment, label, labelling)."""
    succ, _chains = tracker_chains(ev)
    lo, hi = _seg_bounds(ev)
    n = np.bincount(ev.seg, minlength=ev.n_seg)
    ph = np.bincount(ev.seg, ev.p_horse, ev.n_seg) / np.maximum(n, 1)
    s0 = _seg_first_s(ev)
    tg = st.tg
    out = []
    for x in np.flatnonzero((n >= LONG_N) & ((lab >= 0) | (ph >= 0.5))):
        k = int(np.clip(np.searchsorted(tg, lo[x]), 0, len(tg) - 1))
        hs = np.flatnonzero(np.abs(st.S[:, k] - s0[x]) <= REL_GATE)
        segs = [int(x)]
        while succ[segs[-1]] >= 0:
            segs.append(int(succ[segs[-1]]))
        t0, t1 = ev.seg_lo[segs].min(), ev.seg_hi[segs].max()
        for y in list(hs) + [-1]:
            if y == lab[x] and all(lab[z] == y for z in segs):
                continue
            new = lab.copy()
            if y >= 0:
                new[(lab == y) & (ev.seg_lo <= t1) & (ev.seg_hi >= t0)] = -1
            new[segs] = y
            out.append((int(x), int(y), new))
    return out


def swap_moves(model: Model, st: State, lab):
    """Abreast-pair swaps: for two horses whose paths stay within ABREAST_S along and ABREAST_L across for >= ABREAST_T
    s, exchange their labels on every segment inside the stretch, widened to the nearest camera cuts (times where >= 3
    labelled segments start together) or the race's ends, and from there to the race's end or from its start.
    -> list of (description, labelling)."""
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
                # the stretch itself, and the exchange carried on to the race's end or back to its start (two
                # horses that ran level may have been told apart wrongly for the rest of the race)
                for x0, x1 in ((w0, w1), (w0, tg[-1] + 1.0), (tg[0], w1)):
                    inside = (lo >= x0 - 0.15) & (hi < x1 - 0.05)
                    new = lab.copy()
                    new[inside & (lab == a)] = b
                    new[inside & (lab == b)] = a
                    if not np.array_equal(new, lab) and not any(np.array_equal(new, o) for _d, o in out):
                        out.append((f"swap #{ev.horses[a]}/#{ev.horses[b]} over {x0:.1f}-{x1:.1f} s", new))
    return out


def alternative_posterior(lab, alts, H):
    """Per segment P(label) (H + 1 columns, the last clutter) from a fixed alternative set around the kept labelling:
    weight 1 for the kept label, exp(-dJ / (2 T_PROB)) for each alternative's label on every segment it changes.
    ``alts``: list of (labelling, dJ). A segment no alternative changes has P 1 on its label."""
    S = len(lab)
    P = np.zeros((S, H + 1))
    P[np.arange(S), np.where(lab >= 0, lab, H)] = 1.0
    for alt, dj in alts:
        ch = np.flatnonzero(alt != lab)
        if len(ch):
            P[ch, np.where(alt[ch] >= 0, alt[ch], H)] += np.exp(-max(dj, 0.0) / (2.0 * T_PROB))
    return P / P.sum(1, keepdims=True)


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


def polish_candidates(model: Model, st: State, lab, U, pool=()):
    """The polish's candidate labellings: ``pool`` (description, labelling) first; every single segment moved to a
    horse (its overlapping segments become clutter) or to clutter whose unary at the state is less than SEG_SLACK
    above its current one; the single relabels, pair exchanges, tracker-chain moves, abreast swaps, permutation moves
    and insertions at the state; distinct, each different from ``lab``. -> list of (description, labelling)."""
    ev = model.ev
    H = ev.horses
    cands = list(pool)
    Uc = np.c_[U, np.zeros(len(U))]
    cur = Uc[np.arange(len(lab)), np.where(lab >= 0, lab, -1)]
    for k, y in zip(*np.nonzero(Uc < cur[:, None] + SEG_SLACK)):
        y = int(y) if y < model.H else -1
        if y == lab[k]:
            continue
        new = lab.copy()
        if y >= 0:
            new[(lab == y) & (ev.seg_lo <= ev.seg_hi[k]) & (ev.seg_hi >= ev.seg_lo[k])] = -1
        new[k] = y
        cands.append((f"segment {k} -> {'#' + str(H[y]) if y >= 0 else 'clutter'}", new))
    cands += [(f"relabel seg {x} -> {'#' + str(H[y]) if y >= 0 else 'clutter'}", pm)
              for x, y, pm in single_relabels(ev, lab, st)]
    cands += pair_exchanges(ev, lab, st) + chain_moves(ev, lab, st) + swap_moves(model, st, lab)
    cands += [(f"permutation move at {T:.1f} s over {min(B, T):.1f}-{max(B, T):.1f} s", pm)
              for T, B, pm in permutation_moves(ev, lab)]
    cands += insertion_moves(model, st, lab)
    out, seen = [], {np.asarray(lab).tobytes()}
    for d, pm in cands:
        key = pm.tobytes()
        if key not in seen:
            seen.add(key)
            out.append((d, pm))
    return out


def solve(model: Model, start=(), extra_s=None, extra_l=None, s_shift=None, log=print, polish=True, marginals=True):
    """Global solve. Starts: five init MILPs (list votes with reads and silk; list votes; reads and silk; the list
    rows of alternate LIST_SPLIT s blocks of each parity with reads and silk) and the caller's labellings. Every start
    alternates: the exact MILP assignment over segments on unaries at the current state (one-horse-refit kinematic
    block), then a full state step of its proposal, kept when J falls by more than MIN_GAIN, until the MILP returns the
    current labelling or its proposal raises J (REFIT_ROUNDS proposals at most per descent). The DESCENTS lowest
    distinct alternation ends then descend with a bounded polish (POLISH_STEPS full state steps at most per descent):
    candidates (the pool: the rejected proposal's label-connected parts; single segments whose unaries prefer another
    label; the move generators' labellings) scored by one-horse refits (generator candidates first in closed form,
    ``Refit.score_quick``, the REFIT_TOP lowest of those by refits), the SWEEP_TOP lowest again with Gauss-Seidel
    sweeps over their horses; the predicted-negative ones with disjoint horses go together into one full state step,
    on rejection the single best ones one at a time, then the round's other predicted-negative ones whose horses no
    accepted change touched; an accepted step resumes the alternation. Then the lowest end descends again with the
    crossover runs toward every other end (``crossovers``) as its pool, while that lowers J. The pre-solve without
    the finish camera (``polish`` False) keeps the lowest alternation end; its DESCENTS lowest alternation ends are
    further starts of the solve after it."""
    ev = model.ev
    seen = []  # every labelling the state step scored: (labels, J); the posterior over them gives the marginals
    memo = {}
    last_rf = []  # the latest polish round's Refit (its state and memoised one-horse refits)
    cnt = dict(full=0, rounds=0, steps=0, acc_r=0, acc_p=0)
    cpu = dict(state=0.0, unaries=0.0, milp=0.0, scores=0.0)
    pol = dict(candidates=0.0, refit_setup=0.0, closed_form=0.0, refits=0.0, sweeps=0.0)  # parts of cpu["scores"]
    q_count = [0, 0]  # closed-form horse terms requested / memo hits
    part0 = dict(PART_CPU)
    n_milp0 = len(MILP_STATS)

    def look(labs):  # the state step from scratch: J a function of the labelling alone
        key = np.asarray(labs).tobytes()
        if key in memo:
            return memo[key]
        t0 = time.process_time()
        st_ = evaluate(model, labs, extra_s, extra_l, s_shift=s_shift)
        cpu["state"] += time.process_time() - t0
        cnt["full"] += 1
        seen.append((np.asarray(labs).copy(), st_.J))
        memo[key] = st_
        return st_

    hist = []
    # inits: MILPs on the list votes with reads and silk, the list votes alone, reads and silk alone, and the list
    # rows of alternate LIST_SPLIT s blocks (each parity) with reads and silk (a list stretch the full vote gets wrong
    # can be outvoted only from a start that leaves it out); then the caller's labellings
    lv0 = list_votes(model, None, None)
    app = ev.read_cost + ev.silk_cost
    vot = [("init", lv0 + app), ("list-only init", lv0), ("appearance-only init", app)]
    for par in (0, 1):
        keep = (np.floor(model.Lr.tau / LIST_SPLIT).astype(int) % 2) == par
        m2 = copy.copy(model)
        m2.Lr = ListRows(model.Lr.tau, np.where(keep[:, None], model.Lr.slots, -1), model.Lr.full & keep,
                         model.Lr.same & keep)
        vot.append((f"split-list init {par}", list_votes(m2, None, None) + app))
    inits = []
    for name, U0 in vot:
        U0 = np.array(U0, float)
        U0[ev.forbid] = INF
        inits.append((name, milp(np.where(U0 < 0, U0, 0.0) - 1e-3 * (U0 < 0), ev, allow=U0 < 0)))
    inits += [(f"start {k + 1}", np.asarray(o).copy()) for k, o in enumerate(start)]
    ucache = {}

    def unaries_of(st_, labs):
        key = (id(st_), np.asarray(labs).tobytes())
        if key not in ucache or ucache[key][0] is not st_:
            ucache.clear()
            t0 = time.process_time()
            ucache[key] = (st_, unaries(model, st_, labs))
            cpu["unaries"] += time.process_time() - t0
        return ucache[key][1]

    def descend(lab, st, pool, polish=polish):
        rounds = steps = 0
        pcache, pkeys_h = {}, {}  # polish scores by candidate change, and the horses each changes

        def changed(a, b):  # forget the polish scores of candidates touching horses whose labels changed
            ch = np.flatnonzero(a != b)
            hs = (set(a[ch].tolist()) | set(b[ch].tolist())) - {-1}
            for kk in [kk for kk, v in pkeys_h.items() if v & hs]:
                del pkeys_h[kk], pcache[kk]

        fresh = bool(pool)
        while True:
            while not fresh and rounds < REFIT_ROUNDS:  # alternation: exact assignment, joint state step
                U = unaries_of(st, lab)
                t0 = time.process_time()
                prop = milp(U, ev)
                cpu["milp"] += time.process_time() - t0
                if np.array_equal(prop, lab):
                    log(f"   core: assignment fixed point at J {st.J:.1f}")
                    break
                rounds += 1
                cnt["rounds"] += 1
                st2 = look(prop)
                nch = int((prop != lab).sum())
                if st2.J < st.J - MIN_GAIN:
                    log(f"   core: round {rounds}: {nch} segments relabelled, J {st.J:.1f} -> {st2.J:.1f} "
                        f"{_fmt(st2.parts)}")
                    hist.append((f"round {rounds}", st2.J, nch))
                    cnt["acc_r"] += 1
                    changed(lab, prop)
                    lab, st = prop, st2
                    continue
                log(f"   core: round {rounds}: proposal ({nch} segments) raised J ({st2.J:.1f}); its parts to the "
                    f"polish")
                pool = [(f"part of round {rounds}'s proposal: {d}", pm) for d, pm in crossovers(ev, lab, prop)]
                break
            if not polish or steps >= POLISH_STEPS:
                break
            U = unaries_of(st, lab)
            pool = list({pm.tobytes(): (d, pm) for d, pm in pool
                         if pm.tobytes() not in memo and not np.array_equal(pm, lab)}.values())
            npool = len(pool)
            t0 = time.process_time()
            cands = polish_candidates(model, st, lab, U, pool)
            cands = cands[:npool] + [(d, pm) for d, pm in cands[npool:] if pm.tobytes() not in memo]
            t1 = time.process_time()
            pol["candidates"] += t1 - t0
            if not cands:
                break
            rf = Refit(model, st, lab, extra_s, extra_l, s_shift)
            last_rf[:] = [rf]
            lab_r = lab
            # a candidate's score is kept from an earlier polish round while none of its horses changed since
            keys, hsets, pred, todo = [], [], np.zeros(len(cands)), []
            for i, (_d, pm) in enumerate(cands):
                ch = np.flatnonzero(pm != lab)
                keys.append((ch.tobytes(), lab[ch].tobytes(), pm[ch].tobytes()))
                hsets.append(set(lab[ch].tolist()) | set(pm[ch].tolist()))
                if keys[-1] in pcache:
                    pred[i] = pcache[keys[-1]]
                else:
                    todo.append(i)
            todo = np.array(todo, int)
            gen = todo[todo >= npool]  # the pool (listed first) is refit-scored whole: the closed form misjudges
            # multi-horse moves
            t2 = time.process_time()
            pol["refit_setup"] += t2 - t1
            for i in gen:
                pred[i] = rf.score_quick(cands[i][1])
            t3 = time.process_time()
            pol["closed_form"] += t3 - t2
            q_count[0] += rf.q_calls
            q_count[1] += rf.q_hits
            for i in np.r_[todo[todo < npool], gen[np.argsort(pred[gen], kind="stable")[:REFIT_TOP]]].astype(int):
                pred[i] = rf.score(cands[i][1], sweeps=False)
            t4 = time.process_time()
            pol["refits"] += t4 - t3
            for i in todo[np.argsort(pred[todo], kind="stable")[:SWEEP_TOP]]:
                pred[i] = rf.score(cands[i][1])
            pol["sweeps"] += time.process_time() - t4
            for i in todo:
                pcache[keys[i]] = pred[i]
                pkeys_h[keys[i]] = hsets[i]
            cpu["scores"] += time.process_time() - t0
            neg = [i for i in np.argsort(pred, kind="stable") if pred[i] < -MIN_GAIN]
            log(f"   core: polish at J {st.J:.1f}: {len(cands)} candidates, {len(todo)} scored, {len(neg)} predicted "
                f"to lower J" + (f" (best {pred[neg[0]]:+.1f}: {cands[neg[0]][0]})" if neg else ""))
            if not neg and len(todo) < len(cands):  # kept scores may be stale: all scored afresh before ending
                pcache.clear()
                pkeys_h.clear()
                fresh = True
                continue
            pool, fresh = [], False
            tail = not neg
            if tail:  # none predicted to lower J: the EXACT_TAIL best scores within TAIL_SLACK get full steps
                neg = [i for i in np.argsort(pred, kind="stable") if pred[i] < TAIL_SLACK][:EXACT_TAIL]
                if not neg:
                    break
            # the predicted-negative candidates with disjoint horses and segments, together
            batch, used_h, used_s = [], set(), np.zeros(len(lab), bool)
            for i in neg if not tail else []:
                ch = np.flatnonzero(cands[i][1] != lab)
                hs = set(lab[ch].tolist()) | set(cands[i][1][ch].tolist())
                hs.discard(-1)
                if hs & used_h or used_s[ch].any():
                    continue
                batch.append(i)
                used_h |= hs
                used_s[ch] = True
            trials = []
            if len(batch) > 1:
                comb = lab.copy()
                for i in batch:
                    ch = cands[i][1] != lab
                    comb[ch] = cands[i][1][ch]
                trials.append((f"{len(batch)} disjoint candidates together (predicted {pred[batch].sum():+.1f})",
                               comb, None))
            trials += [(f"{cands[i][0]} (predicted {pred[i]:+.1f})", cands[i][1], keys[i]) for i in neg]
            moved = False
            for desc, pm, kk in trials:
                if steps >= POLISH_STEPS:
                    break
                if pm.tobytes() in memo:
                    continue
                steps += 1
                cnt["steps"] += 1
                st2 = look(pm)
                if st2.J < st.J - MIN_GAIN:
                    log(f"   core: polish step {steps}: {desc}, {int((pm != lab).sum())} segments, J {st.J:.1f} -> "
                        f"{st2.J:.1f} {_fmt(st2.parts)}")
                    hist.append((f"polish {steps}", st2.J, int((pm != lab).sum())))
                    cnt["acc_p"] += 1
                    changed(lab, pm)
                    lab, st = pm, st2
                    moved = True
                    break
                log(f"   core: polish step {steps}: {desc} raised J ({st2.J:.1f})")
                if kk is not None:  # a rejected change is not tried again while its horses keep their labels
                    pcache[kk] = np.inf
            if not moved:
                break
            # the round's other predicted-negative candidates whose horses no accepted change touched, each applied
            # on the current labelling
            touched = set(np.unique(np.r_[lab_r[lab_r != lab], lab[lab_r != lab]]).tolist()) - {-1}
            for i in neg:
                if steps >= POLISH_STEPS:
                    break
                if hsets[i] - {-1} & touched:
                    continue
                ch = np.flatnonzero(cands[i][1] != lab_r)
                pm = lab.copy()
                pm[ch] = cands[i][1][ch]
                if pm.tobytes() in memo:
                    continue
                steps += 1
                cnt["steps"] += 1
                st2 = look(pm)
                if st2.J < st.J - MIN_GAIN:
                    log(f"   core: polish step {steps}: {cands[i][0]} (predicted {pred[i]:+.1f}), {len(ch)} segments, "
                        f"J {st.J:.1f} -> {st2.J:.1f}")
                    hist.append((f"polish {steps}", st2.J, len(ch)))
                    cnt["acc_p"] += 1
                    changed(lab, pm)
                    touched |= hsets[i] - {-1}
                    lab, st = pm, st2
                else:
                    pcache[keys[i]] = np.inf
        return lab, st

    starts, done = [], set()
    for name, lab0 in inits:
        if lab0.tobytes() in done:
            continue
        done.add(lab0.tobytes())
        st0 = look(lab0)
        hist.append((name, st0.J, int((lab0 >= 0).sum())))
        log(f"   core: {name} J {st0.J:.1f} ({int((lab0 >= 0).sum())}/{ev.n_seg} segments labelled)")
        starts.append((name, lab0, st0))
    alt = []  # every start's alternation end (no polish); the DESCENTS lowest distinct ends are polished
    for name, lab0, st0 in starts:
        la, sa = descend(lab0, st0, [], polish=False)
        log(f"   core: {name} alternates to J {sa.J:.1f}")
        if all(not np.array_equal(la, o[1]) for o in alt):
            alt.append((name, la, sa))
    alt.sort(key=lambda x: x[2].J)
    ends = []
    for name, la, sa in alt[:DESCENTS] if polish else alt[:1]:
        e = descend(la, sa, []) if polish else (la, sa)
        if polish:
            log(f"   core: {name}'s end polishes to J {e[1].J:.1f}")
        if all(not np.array_equal(e[0], o[0]) for o in ends):
            ends.append(e)
    lab, st = min(ends, key=lambda e: e[1].J)
    others = [e for e in ends if not np.array_equal(e[0], lab)]
    while polish and others:  # the lowest end with the crossover runs toward the other ends as its pool
        pool = [(f"crossover toward an end at J {sb.J:.1f}: {d}", pm)
                for lb, sb in others for d, pm in crossovers(ev, lab, lb)]
        lab2, st2 = descend(lab, st, pool)
        if st2.J >= st.J - MIN_GAIN:
            break
        log(f"   core: crossovers J {st.J:.1f} -> {st2.J:.1f}")
        lab, st = lab2, st2
        others = [e for e in others if not np.array_equal(e[0], lab)]
    ms = MILP_STATS[n_milp0:]
    stopped = [(round(g, 4), round(w)) for stt, g, _nd, w in ms if stt == 1]
    log(f"   core: MILPs solved {len(ms)}: stopped at the {MILP_NODE_LIMIT}-node limit {len(stopped)}"
        + (f" (gap, wall s: {stopped})" if stopped else "")
        + (f"; max nodes {max(nd for _s, _g, nd, _w in ms)}, max gap {max(g for _s, g, _n, _w in ms):.2e}, max wall "
           f"{max(w for _s, _g, _n, w in ms):.1f} s, max root-solved wall "
           f"{max((w for _s, _g, nd, w in ms if nd <= 1), default=0.0):.1f} s" if ms else ""))
    log("   core: state and unaries CPU s by part " + ", ".join(f"{k} {v - part0.get(k, 0.0):.1f}"
                                                            for k, v in PART_CPU.items()))
    log("   core: polish scores CPU s " + ", ".join(f"{k} {v:.1f}" for k, v in pol.items())
        + f"; closed-form horse terms {q_count[0]}, memo hits {q_count[1]}")
    log(f"   core: end J {st.J:.1f}; {len(ends)} start ends; alternation proposals {cnt['rounds']} (kept "
        f"{cnt['acc_r']}), polish steps {cnt['steps']} (kept {cnt['acc_p']}), full state steps {cnt['full']}; CPU s "
        + ", ".join(f"{k} {v:.1f}" for k, v in cpu.items()))
    U = unaries_of(st, lab)
    ends = [a[1] for a in alt[:DESCENTS]]
    if not polish:  # a preliminary solve: its marginals are not a product's
        return Result(lab, st, U, posterior(seen, lab, st.J, model.H), hist, ends)
    if not marginals:  # a product whose probabilities nothing reads
        return Result(lab, st, U, None, hist, ends)
    t0 = time.process_time()
    rf = last_rf[0] if last_rf and last_rf[0].st is st and np.array_equal(last_rf[0].lab, lab) else None
    known = {k: v.J for k, v in memo.items()}
    marg, _alts = probability_output(model, lab, extra_s, extra_l, s_shift, st=st, rf=rf, known=known, log=log)
    log(f"   core: probability output CPU s {time.process_time() - t0:.1f}")
    return Result(lab, st, U, marg, hist, ends)


def alternatives(model: Model, lab, st: State):
    """The fixed alternative set around a labelling: the chain-aware single relabels of every segment with >= LONG_N
    detections (single_relabels) and the pair exchanges over the whole race, cut -> end and start -> cut
    (pair_exchanges). -> list of labellings."""
    return [pm for _x, _y, pm in single_relabels(model.ev, lab, st)] + [pm for _d, pm in pair_exchanges(model.ev, lab, st)]


def probability_output(model: Model, lab, extra_s=None, extra_l=None, s_shift=None, st=None, rf=None, known=None,
                       log=print):
    """Per segment P(label) for a final labelling and its evidence, independent of how the labelling was found: the
    alternatives (alternatives()) screened in closed form (Refit.score_quick); those screened at most POST_PRUNE
    re-screened by their one-horse-refit scores without sweeps (Refit.score); a full state step from scratch for those
    screened below POST_BAND and for the POST_K lowest screened below POST_FULL (``known``: labelling bytes -> J of
    full state steps already taken); combined by alternative_posterior at T_PROB. ``rf``: a Refit at (lab, st) to reuse.
    -> (P (segments x H + 1), list of (labelling, dJ))."""
    lab = np.asarray(lab)
    if st is None:
        st = evaluate(model, lab, extra_s, extra_l, s_shift=s_shift)
    known = {} if known is None else known
    alts = alternatives(model, lab, st)
    if rf is None:
        rf = Refit(model, st, lab, extra_s, extra_l, s_shift)
    qd = np.array([rf.score_quick(pm) for pm in alts])
    near = np.flatnonzero(qd <= POST_PRUNE)
    for i in near:
        qd[i] = rf.score(alts[i], sweeps=False)
    low = np.argsort(qd, kind="stable")[:POST_K]
    full = set(low[qd[low] < POST_FULL].tolist()) | set(np.flatnonzero(qd < POST_BAND).tolist())
    dj = qd.tolist()
    n_new = 0
    for i in sorted(full):
        J = known.get(alts[i].tobytes())
        if J is None:
            J = evaluate(model, alts[i], extra_s, extra_l, s_shift=s_shift).J
            n_new += 1
        dj[i] = J - st.J
    log(f"   core: probabilities from {len(alts)} alternatives ({len(near)} refit-scored, {len(full)} with a full "
        f"state step, {n_new} new)")
    pairs = list(zip(alts, dj))
    return alternative_posterior(lab, pairs, model.H), pairs


def _fmt(parts):
    return "(" + ", ".join(f"{k} {v:.0f}" for k, v in parts.items()) + ")"
