"""Output contract for modelling: the flag and uncertainty columns and artifacts ``fuse_final`` writes beside the paths.

Nothing here changes a path: the contract columns are computed from the finished paths and appended after them.

paths.parquet gains
  valid        the node lies in a validity-flagged stretch of the horse (``flag_stretches``)
  free         no other horse within one length along and one render abreast spacing across at the node
  pos_ok       position flag: the horse is observed and firm at the node, and its video-only path agrees with the
               official times at the bounding visible official boundaries (``position_crossings``, ``node_pos_ok``)
  sd_s_axis, sd_s_horse, sd_s_gap, sd_s_rel, sd_s_total   along-track error terms sd_s omits (``uncertainty_columns``):
               the shared axis, a per-horse persistent term and a long-gap term; field-relative and total sd
det_margins_final.npz  per tracks row: the final E-step's assignment margin (``assoc.assign_frames``); a margin, not a
               calibrated probability
contract.json          the definitions behind the columns, the run's flagged stretches, the running-order list windows
               held in near-cloth order, the per-crossing position verdicts and the noise table
"""

from __future__ import annotations

import numpy as np


# ------------------------------------------------------------------------------------------------- validity flag
# A detection is flagged when its video-only and final labels agree (E1), it is >= START_PAD s after the first render
# detection and before the line (E2, E3), the horse's final path is firm there and overlaps no other firm path (E4),
# and on the circle render its tracklet's majority confident disc read is the label (E5). Flagged detections of a horse
# bridged over gaps <= BRIDGE s form stretches of >= MIN_DET detections and >= MIN_DUR s; on the silk render a stretch
# also needs the legend to confirm the horse's rank on >= MIN_CONF and >= CONF_SHARE of its legend-read frames. Legend
# frames where the list is held in near-cloth order (``legend_frozen``) are not reads, and detections inside such a window
# are not flagged, so no stretch spans it.
BRIDGE, MIN_DUR, MIN_DET, START_PAD = 0.5, 2.0, 10, 1.0
MIN_READS = 3
LEG_WIN, TIE, CONF_SHARE, MIN_CONF = 6, 0.6, 0.8, 3
OVL_DS, OVL_DLAT, FIRM_SD = 2.0, 0.7, 1.5
# free-running: one length along, the render's abreast spacing across (render lateral units)
FREE_DS, FREE_DLAT = 2.4, 2.6


def _grid(P):
    """Paths on their shared grid: t, horses, S, L, SD, firm mask, overlap mask (E4)."""
    import polars as pl

    P = P.select("horse_no", "t", "s", "lat", "sd_s").with_columns(pl.col("t").round(2))
    W = P.pivot(on="horse_no", index="t", values=["s", "lat", "sd_s"]).sort("t")
    hs = sorted({int(c.split("_")[-1]) for c in W.columns if c.startswith("s_")})
    S = np.stack([W[f"s_{h}"].to_numpy() for h in hs], 1).astype(float)
    L = np.stack([W[f"lat_{h}"].to_numpy() for h in hs], 1).astype(float)
    SD = np.stack([W[f"sd_s_{h}"].to_numpy() for h in hs], 1).astype(float)
    Dm = np.nanmax(S)
    ok = np.isfinite(S) & np.isfinite(L) & (SD < FIRM_SD) & (S > 5) & (S < Dm - 1)
    hit = np.zeros_like(ok)
    for a in range(len(hs)):
        for b in range(a + 1, len(hs)):
            c = ok[:, a] & ok[:, b] & (np.abs(S[:, a] - S[:, b]) < OVL_DS) & (np.abs(L[:, a] - L[:, b]) < OVL_DLAT)
            hit[:, a] |= c
            hit[:, b] |= c
    return W["t"].to_numpy(), hs, S, L, ok, hit


# Frozen running-order list (label-free, list reads only). A read is near-cloth when every slot the field can fill is read
# (min(8, field) >= FROZEN_MIN_READ slots) and it has at most FROZEN_MAX_INV inversions against ascending order (cloth
# order or one adjacent swap, e.g. 2,1,3,...,8). The same near-cloth read held over >= FROZEN_HOLD s of race time (rows
# with an unread slot neither break nor extend the hold; another full read breaks it) marks every legend row from its first
# to its last read. A list frozen in any other order cannot be told from a strung-out field by its reads, and is not caught.
FROZEN_MIN_READ, FROZEN_MAX_INV, FROZEN_HOLD = 6, 1, 5.0


def legend_frozen(slots, t, n_field):
    """Per legend row (``slots``: rows x 8 numbers, 0 = unread; ``t``: race time of each row, ascending): the row lies in a
    held near-cloth read. -> (mask, windows: list of dict(t0, t1, read))."""
    S = np.asarray(slots, int)
    t = np.asarray(t, float)
    need = min(8, int(n_field))
    mark = np.zeros(len(S), bool)
    wins = []
    if need < FROZEN_MIN_READ or not len(S):
        return mark, wins
    full = (S[:, :need] > 0).sum(1) == need

    def near(r):
        r = r[:need]
        return int((r[:, None] > r[None, :])[np.triu_indices(need, 1)].sum()) <= FROZEN_MAX_INV

    a = b = None  # first and last row of the current near-cloth hold
    for i in [*np.flatnonzero(full), None]:
        if i is not None and a is not None and np.array_equal(S[i], S[a]):
            b = i
            continue
        if a is not None and t[b] - t[a] >= FROZEN_HOLD:
            mark[a:b + 1] = True
            wins.append(dict(t0=float(t[a]), t1=float(t[b]), read=[int(x) for x in S[a]]))
        a = b = i if i is not None and near(S[i]) else None
    return mark, wins


def legend_frozen_run(run, leg=None, T=None):
    """``legend_frozen`` on run ``run``'s legend rows (``leg``, default as the stages see them: ``legend_rows``), timed on
    the tracks' frame -> race-time map. -> (mask in ``leg``'s row order, windows)."""
    leg = run.legend_rows() if leg is None else leg
    T = np.load(run.p("tracks.npz"), allow_pickle=True) if T is None else T
    fr, t = T["rows"][:, 0].astype(float), T["t"].astype(float)
    mask = np.zeros(len(leg), bool)
    if not len(leg) or not len(fr):
        return mask, []
    o = np.argsort(fr, kind="stable")
    q = np.argsort(leg[:, 0], kind="stable")
    lt = np.interp(leg[q, 0].astype(float), fr[o], t[o])
    m, wins = legend_frozen(leg[q, 1:9], lt, len(run.info.runners))
    mask[q] = m
    return mask, wins


def frozen_windows(run) -> list:
    """The run's legend windows held in near-cloth order (``legend_frozen``); none on the circle render."""
    return [] if run.p("numbers.npz").exists() else legend_frozen_run(run)[1]


def flag_stretches(run, P):
    """Validity-flagged stretches of run ``run`` (a RaceRun with fuse_final's inputs) given its final paths ``P``
    (polars, the paths.parquet columns). -> (stretches: list of dict(horse, tier, t0, t1, n_det, ...),
    rows: per stretch the tracks rows in it)."""
    D = float(run.info.distance)
    T = np.load(run.p("tracks.npz"), allow_pickle=True)
    Lv = np.load(run.p("det_labels.npz"))["labels"]
    Lf = np.load(run.p("det_labels_final.npz"))["labels"]
    t, s = T["t"].astype(float), D - T["d"].astype(float)
    fr = T["rows"][:, 0].astype(int)
    N = len(t)
    circle = run.p("numbers.npz").exists()
    lab = Lv > 0
    e1 = lab & (Lv == Lf)
    e23 = (t >= float(t.min()) + START_PAD) & (s < D)
    gt, hs, S, _L, ok, hit = _grid(P)
    hcol = {h: c for c, h in enumerate(hs)}
    node = np.clip(np.searchsorted(gt, np.round(t, 1) - 1e-6), 0, len(gt) - 1)
    alt = np.clip(node - 1, 0, len(gt) - 1)
    node = np.where(np.abs(gt[alt] - t) < np.abs(gt[node] - t), alt, node)
    col = np.array([hcol.get(int(h), -1) for h in Lf])
    e4 = np.zeros(N, bool)
    m = col >= 0
    e4[m] = ok[node[m], col[m]] & ~hit[node[m], col[m]]
    base = e1 & e23 & e4
    frag = np.full(N, -1)
    for i, mem in enumerate(T["members"]):
        frag[np.asarray(mem, int)] = i
    if circle:
        reads = run._number_reads(T)
        maj = np.zeros(len(T["members"]), int)
        for i, mem in enumerate(T["members"]):
            r = reads[np.asarray(mem, int)]
            r = r[r > 0]
            if len(r) >= MIN_READS:
                c = np.bincount(r)
                top = np.flatnonzero(c == c.max())
                if len(top) == 1:
                    maj[i] = int(top[0])
        base &= (frag >= 0) & (maj[np.maximum(frag, 0)] == Lf)
    else:
        leg = run.legend_rows()
        frozen, wins = legend_frozen_run(run, leg, T)
        for w in wins:  # no stretch spans a frozen window: each side stands on its own
            base &= ~((t >= w["t0"]) & (t <= w["t1"]))
        lf_ = leg[:, 0].astype(int) if len(leg) else np.zeros(0, int)
        o = np.argsort(lf_)
        lf_, leg, frozen = lf_[o], leg[o], frozen[o]
        order = np.argsort(fr, kind="stable")
        _u, start = np.unique(fr[order], return_index=True)
        end = np.r_[start[1:], len(order)]
        rank = np.zeros(N, int)
        for a, b in zip(start, end):
            j = order[a:b]
            rank[j] = 1 + (s[j][None, :] > s[j][:, None] + TIE).sum(1)
    out, rows = [], []
    for h in np.unique(Lf[base]):
        idx = np.flatnonzero(base & (Lf == h))
        idx = idx[np.argsort(t[idx], kind="stable")]
        cut = np.flatnonzero(np.diff(t[idx]) > BRIDGE) + 1
        for g in np.split(idx, cut):
            if len(g) < MIN_DET or t[g[-1]] - t[g[0]] < MIN_DUR:
                continue
            nconf = nread = 0
            if not circle:
                for i in g:
                    if not len(lf_):
                        break
                    p = np.searchsorted(lf_, fr[i])
                    q = min((q for q in (p - 1, p) if 0 <= q < len(lf_)), key=lambda q: abs(lf_[q] - fr[i]))
                    if abs(lf_[q] - fr[i]) > LEG_WIN or frozen[q]:
                        continue
                    row = leg[q, 1:9]
                    if not (row > 0).any():
                        continue
                    nread += 1
                    r = rank[i]
                    nconf += bool(1 <= r <= 8 and int(row[r - 1]) == int(h))
                if not (nconf >= MIN_CONF and nconf >= CONF_SHARE * nread):
                    continue
            out.append(dict(horse=int(h), tier=1 if circle else 2, t0=float(t[g[0]]), t1=float(t[g[-1]]),
                            s0=float(s[g[0]]), s1=float(s[g[-1]]), n_det=len(g), n_leg_read=nread, n_leg_conf=nconf))
            rows.append(g)
    return out, rows


def node_valid(stretches, horse, t) -> np.ndarray:
    """Per node of ``horse`` at times ``t``: inside one of its flagged stretches [t0, t1]."""
    t = np.asarray(t, float)
    v = np.zeros(t.shape, bool)
    for r in stretches:
        if r["horse"] == horse:
            v |= (t >= r["t0"]) & (t <= r["t1"])
    return v


def node_free(P) -> dict:
    """{horse: per-node free-running flag} on the shared grid of paths ``P`` (polars): no other horse's finite node at
    the same t within |ds| < FREE_DS and |dlat| < FREE_DLAT. A node without a finite position is not free."""
    import polars as pl

    W = P.select("horse_no", "t", "s", "lat").with_columns(pl.col("t").round(2))
    W = W.pivot(on="horse_no", index="t", values=["s", "lat"]).sort("t")
    hs = sorted({int(c.split("_")[-1]) for c in W.columns if c.startswith("s_")})
    S = np.stack([W[f"s_{h}"].to_numpy() for h in hs], 1).astype(float)
    L = np.stack([W[f"lat_{h}"].to_numpy() for h in hs], 1).astype(float)
    fin = np.isfinite(S) & np.isfinite(L)
    near = np.zeros(S.shape, bool)
    for a in range(len(hs)):
        for b in range(a + 1, len(hs)):
            c = fin[:, a] & fin[:, b] & (np.abs(S[:, a] - S[:, b]) < FREE_DS) & (np.abs(L[:, a] - L[:, b]) < FREE_DLAT)
            near[:, a] |= c
            near[:, b] |= c
    tt = W["t"].to_numpy()
    return {h: (tt, fin[:, c] & ~near[:, c]) for c, h in enumerate(hs)}


# ------------------------------------------------------------------------------------------------------ position flag
# A crossing (a horse at an official boundary whose time the run sees) passes when the horse is observed there (a render
# detection with its final label within POS_WIN s of the final path's crossing time), the final path is firm there
# (sd_s < FIRM_SD) and the horse's video-only path agrees with the official time within POS_TAU m, field-relative: the
# video-only crossing error (t_video - t_official) x video speed, minus the boundary's median over >= POS_MIN_FIELD horses.
# Node rule: a node is pos_ok when the horse is observed (a final-labelled detection within POS_WIN s) and firm at the node
# and the identity check holds at the horse's bounding visible boundaries (the last one at or behind the node's s and the
# first one ahead of it; the single neighbouring one before the first and after the last; no visible boundary -> False).
POS_WIN, POS_TAU, POS_MIN_FIELD = 0.5, 1.25, 3


def _cross(t, s, smark):
    """First crossing of ``smark`` by the path (t, s): (time, index pair a, b, interpolation weight) or None."""
    k = np.flatnonzero(s >= smark)
    if not len(k) or k[0] == 0:
        return None
    a, b = k[0] - 1, k[0]
    f = (smark - s[a]) / max(s[b] - s[a], 1e-9)
    return t[a] + f * (t[b] - t[a]), a, b, f


def _near(sorted_t, t, win):
    """Per time in ``t``: an element of ``sorted_t`` lies within ``win``."""
    t = np.asarray(t, float)
    if not len(sorted_t):
        return np.zeros(t.shape, bool)
    i = np.clip(np.searchsorted(sorted_t, t), 1, len(sorted_t) - 1) if len(sorted_t) > 1 else np.zeros(t.shape, int)
    d = np.abs(sorted_t[i] - t)
    if len(sorted_t) > 1:
        d = np.minimum(d, np.abs(sorted_t[i - 1] - t))
    return d <= win


def _final_dets(run):
    """{horse: sorted race times of its final-labelled render detections}."""
    T = np.load(run.p("tracks.npz"), allow_pickle=True)
    Lf = np.load(run.p("det_labels_final.npz"))["labels"]
    t = T["t"].astype(float)
    return {int(h): np.sort(t[Lf == h]) for h in np.unique(Lf[Lf > 0])}


def position_crossings(run, P, Pv, dets=None) -> list:
    """Per horse and official boundary the run sees (``run.info.official_boundaries``, the masked view): the crossing
    verdict of the final paths ``P`` and the video-only paths ``Pv`` (polars: horse_no, t, s[, sd_s]). -> list of dict(horse,
    j, d (distance to go), s_mark, t_off, t_c, sd_c, observed, firm, r_v (m), id_ok, pass)."""
    D = float(run.info.distance)
    bnd, cum = run.info.official_boundaries()
    dets = _final_dets(run) if dets is None else dets

    def by_horse(X, cols):
        X = X.sort("horse_no", "t", maintain_order=True)
        return {int(h): [g[c].to_numpy().astype(float) for c in cols] for (h,), g in X.group_by(["horse_no"],
                                                                                                  maintain_order=True)}

    G = by_horse(P, ("t", "s", "sd_s"))
    GV = by_horse(Pv, ("t", "s"))
    out = []
    for j, b in enumerate(bnd):
        smark = D - float(b)
        recs = []
        for h, c in cum.items():
            to = float(c[j])
            if not np.isfinite(to):
                continue
            r = dict(horse=int(h), j=j, d=float(b), s_mark=smark, t_off=to, t_c=np.nan, sd_c=np.nan, observed=False,
                     firm=False, e_v=np.nan)
            if int(h) in G:
                t, s, sd = G[int(h)]
                x = _cross(t, s, smark)
                if x is not None:
                    tc, a, bb, f = x
                    r.update(t_c=float(tc), sd_c=float(sd[a] + f * (sd[bb] - sd[a])))
                    r["observed"] = bool(_near(dets.get(int(h), np.zeros(0)), [tc], POS_WIN)[0])
                    r["firm"] = bool(r["sd_c"] < FIRM_SD)
            if int(h) in GV:
                t, s = GV[int(h)]
                x = _cross(t, s, smark)
                if x is not None:
                    tcv, a, bb, _f = x
                    v = (s[bb] - s[a]) / (t[bb] - t[a])
                    r["e_v"] = float((tcv - to) * v)
            recs.append(r)
        ev = np.array([r["e_v"] for r in recs], float)
        ok = np.isfinite(ev)
        med = float(np.median(ev[ok])) if ok.sum() >= POS_MIN_FIELD else np.nan
        for r in recs:
            r["r_v"] = r.pop("e_v") - med
            r["id_ok"] = bool(np.abs(r["r_v"]) <= POS_TAU)
            r["pass"] = r["observed"] and r["firm"] and r["id_ok"]
        out += recs
    return out


def node_pos_ok(crossings, P, dets) -> np.ndarray:
    """Per row of the final paths ``P`` (polars: horse_no, t, s, sd_s): the position flag by the node rule, from the
    crossing verdicts ``crossings`` (``position_crossings``) and ``dets`` ({horse: sorted final-labelled detection
    times})."""
    h_ = P["horse_no"].to_numpy()
    t_, s_, sd_ = (P[c].to_numpy().astype(float) for c in ("t", "s", "sd_s"))
    out = np.zeros(len(h_), bool)
    for h in np.unique(h_):
        cr = sorted((c for c in crossings if c["horse"] == int(h)), key=lambda c: c["s_mark"])
        if not cr:
            continue
        m = h_ == h
        marks = np.array([c["s_mark"] for c in cr])
        idok = np.array([c["id_ok"] for c in cr])
        k = np.searchsorted(marks, s_[m], side="right")  # boundaries at or behind the node; k - 1 behind, k ahead
        behind = np.where(k > 0, idok[np.maximum(k - 1, 0)], True)
        ahead = np.where(k < len(cr), idok[np.minimum(k, len(cr) - 1)], True)
        node = _near(dets.get(int(h), np.zeros(0)), t_[m], POS_WIN) & (sd_[m] < FIRM_SD)
        out[m] = node & behind & ahead
    return out


def position_flag(run, P):
    """Position flag of run ``run`` (its tracks, final labels and paths_video.parquet) given its final paths ``P``
    (polars). -> (crossing verdicts, per-row pos_ok of ``P``)."""
    import polars as pl

    dets = _final_dets(run)
    Pv = pl.read_parquet(run.p("paths_video.parquet"), columns=["horse_no", "t", "s"])
    cr = position_crossings(run, P, Pv, dets)
    return cr, node_pos_ok(cr, P, dets)


def flag_run(rd, key, mask=None, conflicts=False):
    """Research-script entry (same signature as the scorecard's flag.py): stretches of run dir ``rd`` (race ``key``),
    ``mask`` None | ("discs", phase) | ("legend", phase) with 10 s blocks. -> (stretches DataFrame, rows DataFrame,
    {})."""
    from pathlib import Path

    import polars as pl

    from ..data import Mask, race_info
    from .pipeline import RaceRun

    d_, v, n = key.split("_")
    mk = Mask()
    if mask is not None:
        mk = Mask(discs=10.0, discs_phase=mask[1]) if mask[0] == "discs" else Mask(legend=10.0, legend_phase=mask[1])
    run = RaceRun(race_info(d_, v, int(n)), root=str(Path(rd).parent), log=lambda *a, **k: None, mask=mk)
    P = pl.read_parquet(run.p("paths.parquet"))
    st, rows = flag_stretches(run, P)
    if not st:
        return pl.DataFrame(), pl.DataFrame(), {}
    T = np.load(run.p("tracks.npz"), allow_pickle=True)
    D = float(run.info.distance)
    ST = pl.DataFrame([dict(r, race=key, stretch_id=i) for i, r in enumerate(st)])
    DT = pl.concat([pl.DataFrame(dict(horse=[r["horse"]] * len(g), stretch_id=[i] * len(g), row=g,
                                      t=T["t"][g].astype(float), s=D - T["d"][g].astype(float),
                                      lat=T["lat"][g].astype(float), tier=[r["tier"]] * len(g)))
                    for i, (r, g) in enumerate(zip(st, rows))])
    return ST, DT, {}


# ------------------------------------------------------------------------------------------------------ uncertainty
# Along-track error terms the posterior sd_s omits, added at write time (the paths are unchanged). They do not meet the
# calibration targets yet (RECON 8).
# Shared axis: a race-level latent b(s) common to every horse at the same s. Its core is a stationary OU process along s
# (variance AXIS_S2 m^2, correlation length AXIS_ELL m) conditioned on its values at the visible anchors (the official
# boundaries the metric bias is set at); a scale mixture gives it a heavy tail (weight AXIS_PI at AXIS_K x the core sd).
# sd_s_axis is the mixture sd; NaN without an anchor.
AXIS_S2, AXIS_ELL = 7.092734000380029, 799.9999999998456
AXIS_PI, AXIS_K = 0.10216067886921987, 4.057275999777488
# Per-horse persistent term by render style (correlation 1 along the path, independent across horses).
HORSE_VAR = {"circle": 0.8768334661112646, "silk": 0.4673140522042648}  # m^2
# Long-gap term: an integrated random walk from the horse's last labelled detection, var = GAP_SA^2 dt^3 / 3 (dt the time
# since that detection; the time to its first one before it). Too wide for gaps over about 5 s (RECON 8).
GAP_SA = 4.102187414573527  # m/s^2/sqrt(s)


def axis_core_var(s, anchors) -> np.ndarray:
    """Core (OU) variance of the shared axis at s (metres run), conditioned on the anchors; NaN without anchors."""
    s = np.asarray(s, float)
    a = np.sort(np.asarray(anchors, float))
    if not len(a):
        return np.full(s.shape, np.nan)
    k = np.searchsorted(a, s)
    lo = np.where(k > 0, s - a[np.maximum(k - 1, 0)], np.inf)
    hi = np.where(k < len(a), a[np.minimum(k, len(a) - 1)] - s, np.inf)
    ra = np.where(np.isfinite(lo), np.exp(-np.where(np.isfinite(lo), lo, 0.0) / AXIS_ELL), 0.0)
    rb = np.where(np.isfinite(hi), np.exp(-np.where(np.isfinite(hi), hi, 0.0) / AXIS_ELL), 0.0)
    rl = ra * rb
    return AXIS_S2 * (1.0 - (ra ** 2 + rb ** 2 - 2.0 * ra * rb * rl) / (1.0 - rl ** 2))


def axis_var(s, anchors) -> np.ndarray:
    """Variance of the shared axis at s: the scale mixture's (core x (1 - AXIS_PI + AXIS_PI AXIS_K^2))."""
    return axis_core_var(s, anchors) * (1.0 - AXIS_PI + AXIS_PI * AXIS_K ** 2)


def gap_dt(t, label_t) -> np.ndarray:
    """Per node time t: seconds since the horse's last labelled detection (``label_t``, sorted); before its first one,
    the time to it; inf without labelled detections."""
    t = np.asarray(t, float)
    label_t = np.asarray(label_t, float)
    if not len(label_t):
        return np.full(t.shape, np.inf)
    k = np.searchsorted(label_t, t, side="right")
    return np.where(k > 0, t - label_t[np.maximum(k - 1, 0)], label_t[0] - t)


def uncertainty_columns(s, t, sd_s, anchors, style, label_t) -> dict:
    """sd_s_axis, sd_s_horse, sd_s_gap, sd_s_rel = sqrt(sd_s^2 + horse^2 + gap^2) (field-relative) and
    sd_s_total = sqrt(sd_s_rel^2 + axis^2) for one path."""
    sd_s = np.asarray(sd_s, float)
    ax = np.sqrt(axis_var(s, anchors))
    hz = np.full(sd_s.shape, np.sqrt(HORSE_VAR[style]))
    dt = gap_dt(t, label_t)
    gp = np.where(np.isfinite(dt), np.sqrt(GAP_SA ** 2 * np.where(np.isfinite(dt), dt, 0.0) ** 3 / 3.0), np.nan)
    rel = np.sqrt(sd_s ** 2 + hz ** 2 + gp ** 2)
    return {"sd_s_axis": ax, "sd_s_horse": hz, "sd_s_gap": gp, "sd_s_rel": rel, "sd_s_total": np.sqrt(rel ** 2 + ax ** 2)}


# ------------------------------------------------------------------------------------------------------ noise table
# Per-detection observation noise of the detection-only state-space model, by feed
# and region; raw render coordinates (along metres, lateral render units). Per row: independent white sd per detection,
# and a frame-common AR(1) term shared by every horse of the frame (sd, time constant); the bend factor is folded into
# the bend rows. Film and finish-camera rows are timing errors (s). Silk-render figures rest on validity-flagged
# detections with few horses per frame, so its along-track split between the shared term and per-horse noise is not
# identified (``split_identified`` False): use white and common together as one per-detection noise.
def _nr(feed, region, axis, white, common, tau, source, split=True):
    return dict(feed=feed, region=region, axis=axis, white_sd=white, common_sd=common, common_tau_s=tau, source=source,
                split_identified=split)


NOISE_TABLE: list = [
    _nr("render disc", "straight", "along", 0.0191, 0.696, 26.8, "disc render, tier-1 flagged detections"),
    _nr("render disc", "bend", "along", 0.0191 * 1.179, 0.696 * 1.146, 26.8, "disc render"),
    _nr("render disc", "straight", "lateral", 0.0186, 0.0624, 14.3, "disc render"),
    _nr("render disc", "bend", "lateral", 0.0186 * 1.152, 0.0624 * 7.58, 14.3, "disc render"),
    _nr("render silk", "straight", "along", 0.0462, 0.123, 0.04, "silk render, tier-2 flagged detections", split=False),
    _nr("render silk", "bend", "along", 0.0462 * 1.474, 0.123 * 0.344, 0.04, "silk render", split=False),
    _nr("render silk", "straight", "lateral", 0.0436, 0.716, 18.0, "silk render"),
    _nr("render silk", "bend", "lateral", 0.0436 * 0.330, 0.716 * 0.666, 18.0, "silk render"),
    dict(feed="film marker crossing", region="home straight", axis="time", bias_s=-0.113, sd_s=0.09,
         source="400 m marker"),
    dict(feed="finish camera", region="line", axis="time", sd_s=0.023, source="within race"),
]


def _finite(x):
    """JSON-safe: non-finite floats -> None, recursively."""
    if isinstance(x, dict):
        return {k: _finite(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_finite(v) for v in x]
    if isinstance(x, (float, np.floating)):
        return float(x) if np.isfinite(x) else None
    return x


def describe(stretches, frozen=(), crossings=(), anchors=None) -> dict:
    """contract.json: the definitions behind the columns, the run's flagged stretches, the legend windows held in
    near-cloth order (``frozen``), the position flag's crossing verdicts, the uncertainty terms with the run's axis
    ``anchors`` and the noise table."""
    return {
        "uncertainty": {
            "anchors_s": list(anchors or []),
            "axis": {"core_var_m2": AXIS_S2, "corr_length_m": AXIS_ELL, "mix_weight": AXIS_PI, "mix_scale": AXIS_K,
                     "form": "b(s) common to every horse at the same s; core = stationary OU along s (variance "
                     "core_var_m2, correlation exp(-|ds| / corr_length_m)) conditioned on the anchors; scale mixture "
                     "(1 - mix_weight) N(0, v) + mix_weight N(0, mix_scale^2 v); sd_s_axis = the mixture sd"},
            "horse": {"var_m2": HORSE_VAR, "form": "persistent per horse (correlation 1 along the path, independent "
                      "across horses)"},
            "gap": {"sa": GAP_SA, "form": "var = sa^2 dt^3 / 3, dt = seconds since the horse's last labelled "
                    "detection (before its first: the time to it); too wide for gaps over about 5 s"},
            "status": "does not meet the calibration targets",
        },
        "flag": {"stretches": stretches, "rule": "E1 video == final label, E2 >= 1 s after the first render detection, E3 "
                 "before the line, E4 firm and no overlap on the final paths, E5 tracklet disc-read majority (circle); "
                 "silk: legend rank confirmation; stretches >= 10 detections and >= 2 s, gaps <= 0.5 s",
                 "legend_frozen": {"windows": list(frozen), "min_read": FROZEN_MIN_READ, "max_inversions": FROZEN_MAX_INV,
                                   "hold_s": FROZEN_HOLD, "rule": "the same near-cloth list read (all slots the field "
                                   "fills, <= 1 inversion against ascending order) held >= hold_s: its frames are not "
                                   "legend reads for the silk gate, and no stretch spans the window"}},
        "free": {"ds_m": FREE_DS, "dlat_render": FREE_DLAT},
        "pos_ok": {"crossings": _finite(list(crossings)), "win_s": POS_WIN, "tau_m": POS_TAU, "firm_sd_m": FIRM_SD,
                   "min_field": POS_MIN_FIELD,
                   "crossing_rule": "pass = observed (final-labelled detection within win_s of the final path's "
                   "crossing) and firm (sd_c < firm_sd_m) and id_ok (|r_v| <= tau_m; r_v = the video-only path's "
                   "crossing error (t - t_off) x speed minus the boundary median over >= min_field horses)",
                   "node_rule": "observed and firm at the node, and id_ok at the horse's bounding visible boundaries "
                   "(behind: s_mark <= s; ahead: s_mark > s; one of them before the first / after the last; none: "
                   "False)"},
        "noise_table": NOISE_TABLE,
        "margins": {"file": "det_margins_final.npz", "units": "-2 log-likelihood",
                    "meaning": "assignment margin, not a calibrated probability"},
    }
