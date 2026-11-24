"""Output contract for modelling: the flag and uncertainty columns and artifacts ``fuse_final`` writes beside the paths.

Nothing here changes a path: the contract columns are computed from the finished paths and appended after them.

paths.parquet gains
  valid        the node lies in a validity-flagged stretch of the horse (``flag_stretches``)
  free         no other horse within one length along and one render abreast spacing across at the node
  pos_ok       position flag: the horse is observed and firm at the node, and its video-only path agrees with the
               official times at the bounding visible official boundaries (``position_crossings``, ``node_pos_ok``)
  sd_s_axis, sd_s_horse, sd_s_gap, sd_s_ident, sd_s_rel, sd_s_total   along-track error terms sd_s omits
               (``uncertainty_columns``): the shared axis, a per-horse persistent term, a long-gap term and an identity
               term; field-relative and total sd
  id_robust    identity-robust: in the node's section unit, no horse within 5 m / 2.5 render units of the horse has
               more than tR of its detections relabelled between the video-only and final labels, and the horse's
               read support is >= tP (``id_robust``; thresholds from one cross-fit, not yet confirmed on a fresh set)
  sd_lat_m_horse, sd_lat_m_node, sd_lat_m_scale, sd_lat_m_total   the lat_m map's own error, PROVISIONAL
               (``LAT_M_ERR``, Sha Tin bend and home straight only, NaN elsewhere): a per-horse persistent part, a
               per-node remainder, the race-level scale (shared per race and segment; defined wherever lat_m is) and
               the total with sd_lat_m
  lane_units, lane   lateral in units of the race's own abreast pitch from its rail runner (scale-free, every season)
               and its category 1 rail / 2 two-wide / 3 three-plus (``lane_columns``)
  enclosed     walled in ahead and on both sides (rail counts inside), lateral in the race's abreast pitch, held for
               >= 1 s (``box_columns``; a broader state than a 1 m-lane metre rule)
  boxed_in     held up behind in the final 600 m: an enclosed run in which the horse closed on the horse in front and
               then sat within a length of it (``box_columns``)
det_margins_final.npz  per tracks row: the final E-step's assignment margin (``assoc.assign_frames``); a margin, not a
               calibrated probability
contract.json          the definitions behind the columns, the run's flagged stretches, the running-order list windows
               held in near-cloth order, the per-crossing position verdicts, the noise table, id_robust's
               per-section and per-horse bend verdicts with the run's yield, the run's lane pitch and origin, and the
               enclosed rule
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
HORSE_VAR = {"circle": 1.0441935443052455, "silk": 0.4673909584957436}  # m^2
# Long-gap term: from the horse's last labelled detection (dt the time since it; the time to its first one before it) the
# horse coasts with white acceleration of the detection-level scale GAP_SA, an integrated random walk v = GAP_SA^2 dt^3 / 3;
# the field holds an unseen horse near its neighbours, so it saturates at GAP_CAP: var = v / (1 + v / GAP_CAP^2).
GAP_SA, GAP_CAP = 0.67, 12.539699709556691  # m/s^2/sqrt(s), m
# Identity term: with probability IDENT_EPS the label at a node belongs to a neighbour, and the error is then minus the
# offset to that neighbour (one of the IDENT_K nearest other paths along s within FREE_DLAT across, equally likely). The
# column carries the mixture's variance, IDENT_EPS x the mean squared offset to those neighbours (0 with none).
IDENT_EPS, IDENT_K = 0.006813608765177593, 2


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


def gap_var(dt) -> np.ndarray:
    """Long-gap variance at dt seconds from the last labelled detection (NaN where dt is not finite)."""
    dt = np.asarray(dt, float)
    v = GAP_SA ** 2 * np.where(np.isfinite(dt), dt, 0.0) ** 3 / 3.0
    return np.where(np.isfinite(dt), v / (1.0 + v / GAP_CAP ** 2), np.nan)


def ident_var(P) -> dict:
    """{horse: (t, identity variance per node)} on the shared grid of paths ``P`` (polars): IDENT_EPS x the mean squared
    s offset to the IDENT_K nearest other finite paths along s with |dlat| < FREE_DLAT at the same t (0 with none)."""
    import polars as pl

    W = P.select("horse_no", "t", "s", "lat").with_columns(pl.col("t").round(2))
    W = W.pivot(on="horse_no", index="t", values=["s", "lat"]).sort("t")
    hs = sorted({int(c.split("_")[-1]) for c in W.columns if c.startswith("s_")})
    S = np.stack([W[f"s_{h}"].to_numpy() for h in hs], 1).astype(float)
    L = np.stack([W[f"lat_{h}"].to_numpy() for h in hs], 1).astype(float)
    fin = np.isfinite(S) & np.isfinite(L)
    tt = W["t"].to_numpy()
    out = {}
    for a, h in enumerate(hs):
        ds = S[:, [a]] - S
        ok = fin[:, [a]] & fin & (np.abs(L[:, [a]] - L) < FREE_DLAT)
        ok[:, a] = False
        d2 = np.sort(np.where(ok, ds ** 2, np.inf), axis=1)[:, :IDENT_K]
        n = np.isfinite(d2).sum(1)
        out[h] = (tt, IDENT_EPS * np.where(n > 0, np.where(np.isfinite(d2), d2, 0.0).sum(1) / np.maximum(n, 1), 0.0))
    return out


def uncertainty_columns(s, t, sd_s, anchors, style, label_t, ident=0.0) -> dict:
    """sd_s_axis, sd_s_horse, sd_s_gap, sd_s_ident (``ident``: the identity variance per node, ``ident_var``),
    sd_s_rel = sqrt(sd_s^2 + horse^2 + gap^2 + ident^2) (field-relative) and sd_s_total = sqrt(sd_s_rel^2 + axis^2) for
    one path."""
    sd_s = np.asarray(sd_s, float)
    ax = np.sqrt(axis_var(s, anchors))
    hz = np.full(sd_s.shape, np.sqrt(HORSE_VAR[style]))
    gp = np.sqrt(gap_var(gap_dt(t, label_t)))
    idn = np.sqrt(np.broadcast_to(np.asarray(ident, float), sd_s.shape))
    rel = np.sqrt(sd_s ** 2 + hz ** 2 + gp ** 2 + idn ** 2)
    return {"sd_s_axis": ax, "sd_s_horse": hz, "sd_s_gap": gp, "sd_s_ident": idn, "sd_s_rel": rel,
            "sd_s_total": np.sqrt(rel ** 2 + ax ** 2)}


# ------------------------------------------------------------------------------------------------ lateral-metre error
# The lat_m map's own error (m), PROVISIONAL design values, not fitted parameters: Sha Tin only, from a comparison with the
# drone on 7-9 races. By (venue, segment of RaceRun.lat_m_segment): (sd of a per-horse part persistent along the segment,
# sd of the per-node remainder about it).
# - Sha Tin bend: the persistent sd is the p90 of the per-horse bend-mean error / 1.645 (0.60 m); the per-node remainder
#   is 0.25 m. Held out by race the total is still narrow at the tail.
# - Sha Tin home straight: the map's leave-race-out residual p90 / 1.645 (0.58 m), not split and carried whole as the
#   persistent part. Its per-node 0.0 means "not split", not a measured zero.
# Every other segment and every Happy Valley segment has no value, so the three columns are NaN there while lat_m may be
# defined.
LAT_M_ERR = {("ST", "bend"): (0.98 / 1.645, 0.25), ("ST", "home"): (0.95 / 1.645, 0.0)}
# Race-level lateral scale: the render's per-race scale error, which no tested observable predicts (field size, the
# abreast-gap ruler, the drawn width, the drone box ruler). Error = z x sigma x (lat_m - a), z ~ N(0, 1) ONE per race and
# segment (lat_m_segment), shared by every horse and node of that race and segment; a = the map intercept. sigma (log-scale
# sd) is the noise-free race-to-race sd of the render/drone lateral ratio within venue x render style: start and straight
# LAT_M_SCALE_STRAIGHT; bend and home a ramp in x = d - home straight length, LAT_M_SCALE_RAMP (x, sigma) at both ends,
# linear between (the lat_m map's ramp).
LAT_M_SCALE_STRAIGHT = 0.15
LAT_M_SCALE_RAMP = ((0.0, 0.11), (400.0, 0.26))


def lat_m_scale_sd(seg, x) -> np.ndarray:
    """sigma of the race-level scale per node: segment ``seg`` and x = d - home straight length (m)."""
    seg = np.asarray(seg).astype(str)
    (x0, s0), (x1, s1) = LAT_M_SCALE_RAMP
    w = np.clip((np.asarray(x, float) - x0) / (x1 - x0), 0.0, 1.0)
    ramp = np.where(w <= 0.0, s0, np.where(w >= 1.0, s1, s0 + (s1 - s0) * w))
    return np.where(np.isin(seg, ["start", "straight"]), LAT_M_SCALE_STRAIGHT,
                    np.where(np.isin(seg, ["bend", "home"]), ramp, np.nan))


def lat_m_err(venue, seg, sd_lat_m, dev, x) -> dict:
    """sd_lat_m_horse, sd_lat_m_node, sd_lat_m_scale and sd_lat_m_total = sqrt(sd_lat_m^2 + horse^2 + node^2 + scale^2)
    per node from its segment ``seg`` (``RaceRun.lat_m_segment``), ``dev`` = lat_m - a (the map intercept) and x = d - home
    straight length. sd_lat_m_scale = sigma x |dev| is the race-level scale term, shared by every horse and node of the
    race and segment (not independent per node); it is defined wherever lat_m is. horse, node and the total are NaN where
    lat_m is undefined (``sd_lat_m`` NaN) or the segment has no LAT_M_ERR value."""
    seg = np.asarray(seg)
    hn = np.array([LAT_M_ERR.get((venue, str(x)), (np.nan, np.nan)) for x in seg.ravel()], float).reshape(seg.shape + (2,))
    sd = np.asarray(sd_lat_m, float)
    ok = np.isfinite(sd)
    h = np.where(ok, hn[..., 0], np.nan)
    n = np.where(ok, hn[..., 1], np.nan)
    sc = np.where(ok, lat_m_scale_sd(seg, x) * np.abs(np.asarray(dev, float)), np.nan)
    return {"sd_lat_m_horse": h, "sd_lat_m_node": n, "sd_lat_m_scale": sc,
            "sd_lat_m_total": np.sqrt(sd ** 2 + h ** 2 + n ** 2 + sc ** 2)}


# ------------------------------------------------------------------------------------------------------ lane category
# lane_units: lateral counted in the race's own abreast pitch from its rail runner, so the render's drawing scale cancels
# and no metres are needed (every season). Per race and segment class (lat_m_segment's bend / home / straight; start nodes
# use the straight class), on firm nodes (sd_lat < LANE_FIRM) at whole multiples of LANE_GRID s:
#   pitch  = median over nodes of the nearest-abreast lateral gap (another firm node at the same time within LANE_ABREAST
#            m along); with < LANE_MIN_GAPS gaps, the pooled median LANE_PITCH_POOL for venue x period x class;
#   origin = median over time steps of the innermost firm node's lat;
#   pitch3 = median inner gap of the depth-3 nodes (third from the inside among the firm nodes of the class at the same
#            time within LANE_ABREAST m along; the gap to the nearest of those inside it); the race pitch where the class
#            has < LANE_MIN_GAPS such gaps or a pooled pitch. The lane beyond the second is about 0.94 pitch wide.
# lane_units = 1 + (lat - origin) / pitch up to 2, then 2 + (lat - origin - pitch) / pitch3, on firm in-race nodes (NaN
# elsewhere); lane = 1 rail (< 1.5), 2 two-wide (< 2.5), 3 three-plus. Scored against the official running comments'
# explicit lane statements (window means of lane_units, 340 races, 2023-09 to 2026-11): agreement 0.83, 0.90 in
# 2026/27; stated three-plus horses read low (recall 0.78).
LANE_GRID, LANE_FIRM, LANE_ABREAST, LANE_MIN_GAPS = 0.5, 1.5, 1.0, 20
LANE_CUTS = (1.5, 2.5)
# period start dates (render eras); a race before the first has no pooled pitch
LANE_PERIODS = (("20230901", "2023/24"), ("20240901", "2024/25"), ("20250901", "2025/26 pre-drone"),
                ("20260107", "2026 silk"), ("20260901", "2026/27 disc"))
# median of the race pitches (>= LANE_MIN_GAPS gaps) per venue x period x class over the 459 scored races (render units)
LANE_PITCH_POOL = {
    ("ST", "2023/24"): {"bend": 1.4782105789052054, "home": 2.22870791167592, "straight": 1.848585200313603},
    ("ST", "2024/25"): {"bend": 1.7024693291082102, "home": 2.3872162687991367, "straight": 1.9687545485925788},
    ("ST", "2025/26 pre-drone"): {"bend": 1.8955454317250395, "home": 2.3992178787966676, "straight": 2.1057563102337387},
    ("ST", "2026 silk"): {"bend": 2.190847062604993, "home": 2.6282802457775274, "straight": 2.448576447690632},
    ("ST", "2026/27 disc"): {"bend": 2.125265605877618, "home": 2.847586753518297, "straight": 2.34348215450569},
    ("HV", "2023/24"): {"bend": 1.5387919617741668, "home": 2.323477400927679, "straight": 1.7300944698216623},
    ("HV", "2024/25"): {"bend": 1.7988209095119068, "home": 2.5029744606568176, "straight": 1.8277507889349727},
    ("HV", "2025/26 pre-drone"): {"bend": 2.065614824943134, "home": 2.5146546558742564, "straight": 1.9876326717133765},
    ("HV", "2026 silk"): {"bend": 2.1081422779068983, "home": 2.6788714958951045, "straight": 2.2166205541601136},
    ("HV", "2026/27 disc"): {"bend": 2.2787977140836935, "home": 3.2011608211831355, "straight": 2.2264745890481237},
}


def lane_period(date) -> str | None:
    """The LANE_PERIODS label of a race date (YYYYMMDD), None before the first."""
    out = None
    for d0, lab in LANE_PERIODS:
        if str(date) >= d0:
            out = lab
    return out


def lane_columns(t, d, lat, sd_lat, seg, venue, date):
    """({"lane_units", "lane"}, per-class pitch and origin) per node; ``seg`` from ``RaceRun.lat_m_segment`` ("" outside
    the race), ``date`` YYYYMMDD. lane is 1 / 2 / 3 as float, NaN where lane_units is NaN."""
    t, d, lat, sd_lat = (np.asarray(x, float) for x in (t, d, lat, sd_lat))
    seg = np.asarray(seg).astype(str)
    firm = sd_lat < LANE_FIRM
    tk = np.round(t / LANE_GRID).astype(np.int64)
    grid = np.abs(t / LANE_GRID - tk) < 1e-6
    gap, gap3 = np.full(len(t), np.nan), np.full(len(t), np.nan)
    gi = np.flatnonzero(grid)
    order = gi[np.argsort(tk[gi], kind="stable")]
    for idx in np.split(order, np.flatnonzero(np.diff(tk[order])) + 1):
        if len(idx) < 2:
            continue
        ab = (np.abs(d[idx][:, None] - d[idx][None, :]) < LANE_ABREAST) & ~np.eye(len(idx), dtype=bool)
        ab &= firm[idx][:, None] & firm[idx][None, :]
        g = np.where(ab, np.abs(lat[idx][:, None] - lat[idx][None, :]), np.inf).min(1)
        gap[idx] = np.where(np.isfinite(g), g, np.nan)
        inside = ab & (seg[idx][:, None] == seg[idx][None, :]) & (lat[idx][None, :] < lat[idx][:, None])
        dep3 = inside.sum(1) == 2
        gap3[idx[dep3]] = lat[idx][dep3] - np.where(inside, lat[idx][None, :], -np.inf).max(1)[dep3]
    cls = {}
    for c in ("bend", "home", "straight"):
        sel = grid & firm & (seg == c)
        if not sel.any():
            continue
        g, g3 = gap[sel & np.isfinite(gap)], gap3[sel & np.isfinite(gap3)]
        k, lo = tk[sel], lat[sel]
        o = np.argsort(k, kind="stable")
        cut = np.flatnonzero(np.diff(k[o])) + 1
        inner = np.array([v.min() for v in np.split(lo[o], cut)])
        cls[c] = {"pitch": float(np.median(g)) if len(g) else None, "n_gaps": int(len(g)), "origin": float(np.median(inner)),
                  "pitch3": float(np.median(g3)) if len(g3) else None, "n_gaps3": int(len(g3))}
    pool = LANE_PITCH_POOL.get((venue, lane_period(date)), {})
    used = {}
    for c in ("bend", "home", "straight"):
        v = cls.get(c)
        if v and v["n_gaps"] >= LANE_MIN_GAPS:
            own3 = v["n_gaps3"] >= LANE_MIN_GAPS
            used[c] = {"pitch": v["pitch"], "origin": v["origin"], "pitch_source": "race", "n_gaps": v["n_gaps"],
                       "pitch3": v["pitch3"] if own3 else v["pitch"], "pitch3_source": "race" if own3 else "pitch",
                       "n_gaps3": v["n_gaps3"]}
        else:
            o = v["origin"] if v else (cls.get("straight") or cls.get("bend") or {}).get("origin", np.nan)
            p = pool.get(c, np.nan)
            used[c] = {"pitch": p, "origin": o, "pitch_source": "pooled", "n_gaps": v["n_gaps"] if v else 0,
                       "pitch3": p, "pitch3_source": "pitch", "n_gaps3": v["n_gaps3"] if v else 0}
    units = np.full(len(t), np.nan)
    for c in ("start", "bend", "home", "straight"):
        u = used["straight" if c == "start" else c]
        sel = firm & (seg == c)
        x = 1.0 + (lat[sel] - u["origin"]) / u["pitch"]
        units[sel] = np.where(x <= 2.0, x, 2.0 + (x - 2.0) * u["pitch"] / u["pitch3"])
    lane = np.where(np.isfinite(units), np.where(units < LANE_CUTS[0], 1.0, np.where(units < LANE_CUTS[1], 2.0, 3.0)),
                    np.nan)
    return {"lane_units": units, "lane": lane}, used


def describe_lane(used, period=None) -> dict:
    """contract.json's ``lane`` entry: the race's pitch and origin per segment class and the rule."""
    return {"classes": used, "cuts": list(LANE_CUTS), "grid_s": LANE_GRID, "firm_sd_lat": LANE_FIRM,
            "abreast_m": LANE_ABREAST, "min_gaps": LANE_MIN_GAPS, "period": period,
            "rule": "per race and segment class (bend / home / straight; start nodes use straight), on firm nodes "
                    "(sd_lat < firm_sd_lat) at multiples of grid_s: pitch = median nearest-abreast lateral gap (another "
                    "firm node at the same time within abreast_m along), or the pooled venue x period x class median "
                    "(pitch_source 'pooled') with < min_gaps gaps; origin = median over time of the innermost firm "
                    "node's lat; pitch3 = median inner gap of the depth-3 nodes (third from the inside among the "
                    "class's firm nodes at the same time within abreast_m along), or the pitch (pitch3_source 'pitch') "
                    "with < min_gaps such gaps or a pooled pitch; lane_units = 1 + (lat - origin) / pitch up to 2, then "
                    "2 + (lat - origin - pitch) / pitch3, on firm in-race nodes, NaN elsewhere; lane = 1 rail "
                    "(< cuts[0]), 2 two-wide (< cuts[1]), 3 three-plus",
            "status": "agreement with the official running comments' explicit lane statements (window means, 340 "
                      "races 2023-09..2026-11) 0.83 overall, 0.90 in 2026/27; stated three-plus horses read low"}



# ------------------------------------------------------------------------------------------------------------ enclosed
# enclosed: the horse is walled in ahead and on both sides, with lateral counted in the race's abreast pitch (every season,
# no metres). On the final paths' BOX_GRID s node grid, at nodes where lane_units is finite (firm, in-race), with
# dS = s_j - s_i (m) and dL = (lat_j - lat_i) / pitch of the horse's segment class (``lane_columns``; start uses straight):
#   ahead    some horse with 0 < dS <= 2 BOX_LEN and |dL| < 1;
#   outside  some horse with |dS| < BOX_LEN and 0 < dL < 2;
#   inside   some horse with |dS| < BOX_LEN and 0 < -dL < 2, or the rail (lane_units < 1.5);
# only neighbours with finite lane_units count. Enclosed nodes form runs; runs closer than BOX_BRIDGE_N nodes are joined
# when every node between has finite lane_units, and runs of >= BOX_MIN_N nodes are kept (the column is True on them,
# bridged nodes included). False wherever lane_units is NaN, so False can mean undefined. One pitch is about 1.2-1.4 m
# of lat_m off the home straight and 1.8 m in it (derived: drone-era race pitch x the lat_m slope), so this is a broader
# state than the same rule with 1 m lanes in metres: 30 % of drone-era lat_m nodes against 16 % (29 % of in-race
# horse-time over 458 turf-oval races, 2023-09..2026-11).
BOX_GRID, BOX_LEN, BOX_BRIDGE_N, BOX_MIN_N = 0.1, 2.4, 5, 10
# boxed_in: held up behind, a costly box-in. An enclosed run with >= BOX_LATE_N nodes at d <= BOX_LATE_D
# (its late part) is boxed in when the horse closes on the horse in front and then sits on its heels, all in the late
# window W (nodes with finite lane_units and d <= BOX_LATE_D):
#   closing node t: t and t - BOX_CLOSE_N nodes both in W, t in the late part or the BOX_CLOSE_N nodes before it, and
#                   the gap to the nearest ahead wall j at t shrank by >= -BOX_CLOSE m over those BOX_CLOSE_N nodes;
#   held: at the first closing node, >= BOX_HOLD_N of the BOX_HOLD_WIN nodes from t have 0 < dS_ij <= BOX_LEN and
#         |dL| < 1 with both horses in W.
# The column is true on the late part of each such run (False where lane_units is NaN). Scored per run against the
# stewards' late blocked-run statements (566 on 458 turf-oval races, 2023-09..2026-11): it finds 0.52 of them at 1.9x
# chance and its runs overlap one 2.0x as often as chance; the closing term was chosen on these same statements, so the
# figures are not confirmed on a fresh set, and node timing inside the late part is not validated. It fires for about a
# third of runners (precision 0.125). Across masked reruns its horse x race reliability is low on the silk render (kappa
# 0.69; 0.88 where id_robust) and high on the disc render.
BOX_LATE_D, BOX_LATE_N, BOX_CLOSE_N, BOX_CLOSE, BOX_HOLD_N, BOX_HOLD_WIN = 600.0, 5, 10, -0.3, 10, 30


def _box_grid(horse_no, t, *cols):
    """Rows -> (row horse index, row time index, horses, grids H x T of ``cols`` with NaN where a horse has no row)."""
    horse_no = np.asarray(horse_no).astype(np.int64)
    horses = np.unique(horse_no)
    k = np.round(np.asarray(t, float) / BOX_GRID).astype(np.int64)
    hi, ki = np.searchsorted(horses, horse_no), k - k.min()
    grids = []
    for c in cols:
        g = np.full((len(horses), int(ki.max()) + 1), np.nan)
        g[hi, ki] = np.asarray(c, float)
        grids.append(g)
    return hi, ki, horses, grids


def _runs(b):
    x = np.diff(np.r_[0, b.astype(np.int8), 0])
    return list(zip(np.flatnonzero(x == 1), np.flatnonzero(x == -1)))


def _episodes(node, valid):
    """Per horse [(a, b_exclusive)] of ``node`` runs bridged over gaps < BOX_BRIDGE_N valid nodes, kept >= BOX_MIN_N."""
    out = []
    for i in range(node.shape[0]):
        r = _runs(node[i] & valid[i])
        eps = []
        if r:
            merged = [list(r[0])]
            for a, e in r[1:]:
                if a - merged[-1][1] < BOX_BRIDGE_N and valid[i, merged[-1][1]:a].all():
                    merged[-1][1] = e
                else:
                    merged.append([a, e])
            eps = [(a, e) for a, e in merged if e - a >= BOX_MIN_N]
        out.append(eps)
    return out


def _walls(S, LAT, LU, PITCH):
    """(ahead, enclosed) node masks H x T and the nearest ahead-wall horse index (-1 where none)."""
    H = S.shape[0]
    ok = np.isfinite(LU) & np.isfinite(S)
    dS = S[None, :, :] - S[:, None, :]
    dL = (LAT[None, :, :] - LAT[:, None, :]) / PITCH[:, None, :]
    okj = ok[None, :, :] & ok[:, None, :] & ~np.eye(H, dtype=bool)[:, :, None]
    with np.errstate(invalid="ignore"):
        ahead_j = okj & (dS > 0) & (dS <= 2 * BOX_LEN) & (np.abs(dL) < 1.0)
        side = okj & (np.abs(dS) < BOX_LEN)
        out = (side & (dL > 0) & (dL < 2.0)).any(1)
        ins = (side & (-dL > 0) & (-dL < 2.0)).any(1)
        ahead = ahead_j.any(1)
        enc = ahead & out & (ins | (LU < 1.5))
    nearest = np.where(ahead, np.argmin(np.where(ahead_j, dS, np.inf), axis=1), -1)
    return ahead & ok, enc & ok, nearest


def _held_up(i, la, lb, S, LAT, PITCH, W, nearest) -> bool:
    """The boxed_in test of one enclosed run's late part [la, lb) of horse ``i``."""
    T = S.shape[1]
    for t in range(max(la - BOX_CLOSE_N, BOX_CLOSE_N), lb):
        j = nearest[i, t]
        if j < 0 or not (W[i, t] and W[i, t - BOX_CLOSE_N]):
            continue
        d0, d1 = S[j, t] - S[i, t], S[j, t - BOX_CLOSE_N] - S[i, t - BOX_CLOSE_N]
        if not (np.isfinite(d0) and np.isfinite(d1) and d0 - d1 <= BOX_CLOSE):
            continue
        sl = slice(t, min(t + BOX_HOLD_WIN, T))
        with np.errstate(invalid="ignore"):
            dS = S[j, sl] - S[i, sl]
            dL = (LAT[j, sl] - LAT[i, sl]) / PITCH[i, sl]
            held = (dS > 0) & (dS <= BOX_LEN) & (np.abs(dL) < 1.0) & W[i, sl] & W[j, sl]
        return int(held.sum()) >= BOX_HOLD_N
    return False


def box_columns(horse_no, t, s, d, lat, lane_units, seg, used) -> dict:
    """{"enclosed", "boxed_in"} per row (bool); ``lane_units``, ``used`` from ``lane_columns``, ``seg`` from
    ``lat_m_segment``, ``d`` the distance to go."""
    seg = np.asarray(seg).astype(str)
    cls = np.where(seg == "start", "straight", seg)
    pitch = np.array([used[c]["pitch"] if c in used else np.nan for c in cls], float)
    hi, ki, _h, (S, LAT, LU, PITCH, D) = _box_grid(horse_no, t, s, lat, lane_units, pitch, d)
    _ahead, enc, nearest = _walls(S, LAT, LU, PITCH)
    valid = np.isfinite(LU)
    with np.errstate(invalid="ignore"):
        W = valid & (D <= BOX_LATE_D)
    E, B = np.zeros(S.shape, bool), np.zeros(S.shape, bool)
    for i, eps in enumerate(_episodes(enc, valid)):
        for a, b in eps:
            E[i, a:b] = True
            late = np.flatnonzero(W[i, a:b]) + a
            if len(late) >= BOX_LATE_N and _held_up(i, late[0], late[-1] + 1, S, LAT, PITCH, W, nearest):
                B[i, late[0]:late[-1] + 1] = True
    return {"enclosed": E[hi, ki], "boxed_in": B[hi, ki]}


def describe_box() -> dict:
    """contract.json's ``box`` entry: the enclosed and boxed_in rules and their constants."""
    return {"grid_s": BOX_GRID, "len_m": BOX_LEN, "bridge_nodes": BOX_BRIDGE_N, "min_nodes": BOX_MIN_N,
            "late_d_m": BOX_LATE_D, "late_nodes": BOX_LATE_N, "close_nodes": BOX_CLOSE_N, "close_m": BOX_CLOSE,
            "hold_nodes": BOX_HOLD_N, "hold_window_nodes": BOX_HOLD_WIN,
            "boxed_in": "held up behind: an enclosed run with >= late_nodes nodes in the late window W (finite lane_units, "
                        "d <= late_d_m) is boxed in when at its first closing node t (t and t - close_nodes in W, t in "
                        "the run's late part or the close_nodes before it, the gap to the nearest ahead wall j shrank by "
                        ">= -close_m over close_nodes) the horse then spends >= hold_nodes of the hold_window_nodes from t "
                        "within len_m behind j and |dL| < 1, both in W; true on the run's late part",
            "boxed_in_status": "scored per run against the stewards' late blocked-run statements (566, 458 turf-oval "
                               "races 2023-09..2026-11): recall 0.52 (1.9x chance), precision 0.125 (2.0x chance); not "
                               "confirmed on a fresh set; fires for about a third of runners; masked-rerun kappa 0.69 "
                               "silk (0.88 where id_robust), 0.96 disc; false where lane_units is NaN",
            "enclosed": "at nodes with finite lane_units, dS = s_j - s_i (m), dL = (lat_j - lat_i) / pitch of the horse's "
                        "segment class (lane): ahead = some horse with 0 < dS <= 2 len_m and |dL| < 1; outside = some "
                        "horse with |dS| < len_m and 0 < dL < 2; inside = the same with 0 < -dL < 2, or lane_units < 1.5 "
                        "(rail); enclosed = ahead and outside and inside, neighbours with finite lane_units only; runs "
                        "joined over gaps < bridge_nodes nodes with finite lane_units, kept when >= min_nodes nodes",
            "status": "lane units, not metres: one pitch is about 1.2-1.4 m off the home straight and 1.8 m in it "
                      "(derived), so a broader state than a 1 m-lane metre rule (30 % against 16 % of drone-era lat_m "
                      "nodes; 29 % of in-race horse-time, 2023-09..2026-11); false where lane_units is NaN"}

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


def describe(stretches, frozen=(), crossings=(), anchors=None, idr=None, lane=None, box=None) -> dict:
    """contract.json: the definitions behind the columns, the run's flagged stretches, the legend windows held in
    near-cloth order (``frozen``), the position flag's crossing verdicts, the uncertainty terms with the run's axis
    ``anchors``, the noise table, ``idr``, id_robust's (sections, bends, yield) (``describe_id_robust``), and ``lane``,
    the run's lane pitch and origin (``describe_lane``), and ``box``, the enclosed rule (``describe_box``)."""
    return {
        "uncertainty": {
            "anchors_s": list(anchors or []),
            "axis": {"core_var_m2": AXIS_S2, "corr_length_m": AXIS_ELL, "mix_weight": AXIS_PI, "mix_scale": AXIS_K,
                     "form": "b(s) common to every horse at the same s; core = stationary OU along s (variance "
                     "core_var_m2, correlation exp(-|ds| / corr_length_m)) conditioned on the anchors; scale mixture "
                     "(1 - mix_weight) N(0, v) + mix_weight N(0, mix_scale^2 v); sd_s_axis = the mixture sd"},
            "horse": {"var_m2": HORSE_VAR, "form": "persistent per horse (correlation 1 along the path, independent "
                      "across horses)"},
            "gap": {"sa": GAP_SA, "cap_m": GAP_CAP, "form": "v = sa^2 dt^3 / 3, var = v / (1 + v / cap_m^2), dt = "
                    "seconds since the horse's last labelled detection (before its first: the time to it)"},
            "ident": {"eps": IDENT_EPS, "k": IDENT_K, "dlat_render": FREE_DLAT, "form": "with probability eps the label "
                      "belongs to one of the k nearest other paths along s within dlat_render across (equally likely), "
                      "error = minus the offset to it; sd_s_ident^2 = eps x mean squared offset (0 with none)"},
            "status": "does not meet the calibration targets",
        },
        "lat_m_err": {
            "sd_m": {f"{v} {g}": {"horse": h, "node": n} for (v, g), (h, n) in LAT_M_ERR.items()},
            "form": "the lat_m map's own error by (venue, segment): a per-horse part persistent along the segment, a "
                    "per-node remainder and the race-level scale; sd_lat_m_total = sqrt(sd_lat_m^2 + horse^2 + node^2 + "
                    "scale^2); NaN in every segment without a horse/node value; the home straight is not split (node 0 "
                    "= not split)",
            "status": "provisional design values (Sha Tin only, 7-9-race drone comparison; narrow at the tail held out)",
            "race_scale": {"sigma_start_straight": LAT_M_SCALE_STRAIGHT,
                           "sigma_ramp_x_m": [list(v) for v in LAT_M_SCALE_RAMP],
                           "form": "shared per race and segment: error = z x sigma x (lat_m - a), z ~ N(0, 1) one per "
                                   "race x segment, common to every horse and node in it (do not treat as independent "
                                   "per node: a race mean keeps all of it); sigma = start/straight value, or in bend "
                                   "and home linear in x = d - home straight length between the ramp ends; "
                                   "sd_lat_m_scale = sigma x |lat_m - a|, defined wherever lat_m is",
                           "source": "noise-free race-to-race sd of log(render sd / drone sd) within venue x render "
                                     "style, drone-era q3d frames (0304)"},
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
        **({} if idr is None else {"id_robust": describe_id_robust(*idr)}),
        **({} if lane is None else {"lane": _finite(lane)}),
        **({} if box is None else {"box": box}),
    }


# ------------------------------------------------------------------------------------------------- identity robustness
# id_robust: per horse x node, the node's window passes both identity tests below. Windows: a section unit (the horse's
# run nodes between two official boundaries, >= ID_MIN_NODES nodes) for the node column; and, for bend descriptors, the
# horse's bend window (all its run nodes on the bends of the race, every bend pooled), reported per horse in contract.json.
# Run nodes: finite s, before the line, at or after the field's render start (median over horses of each horse's first
# observed node after 0.5 s).
# Relabel share of horse j in a window: of the render detections labelled j by the video-only labels (det_labels) or by
# the final labels (det_labels_final), the share labelled j by exactly one of them. Section window: detections within
# ID_DET_PAD s of the unit's first and last node time; bend window: detections on a bend, at or after the render start.
# Rmax: the maximum relabel share over the focal horse and every horse within ID_NEAR_DS m along and ID_NEAR_DLAT render
# units across of it at any node of the window (about 1.5 lanes; render units, so not a fixed metric distance on bends).
# Read support: circle render, the share of the focal horse's final-labelled window detections with a confident disc read
# (``_number_reads``) naming it; silk render, the share of legend rows (``legend_rows``) timed inside the window (the bend
# window: between the horse's first and last bend node) that list it in a slot. Legend rows of a frozen list count as
# rows here (the thresholds were fitted so). No read source (no numbers.npz, no legend rows): read support is not tested.
# A window is robust when Rmax <= tR and read support >= tP.
# Thresholds (ID_THRESHOLDS) by render style and window type, by a stated rule: the median of the two folds' selected
# thresholds of one cross-fit (journal 0286 addendum: legend- and disc-masked reruns, 2-fold race split by md5 parity,
# each fold's set the maximum-yield one whose fit-fold kappa / ICC clears its bar + 0.05; pack exposure was never cut, so
# it is dropped). Read support is a different quantity per render style, so each style has its own pair. Not yet
# confirmed on a fresh set: neither fold was scored with these medians, and the folds' out-of-fold kappa / ICC do not
# describe them. On disc bends the rule is nearly no cut (one fold selected none).
ID_NEAR_DS, ID_NEAR_DLAT, ID_MIN_NODES, ID_DET_PAD = 5.0, 2.5, 20, 0.05
ID_FOLDS = {  # (style, window): ((tR, tP) fit on fold 0, (tR, tP) fit on fold 1)
    ("silk", "section"): ((0.0, 0.8184110007639422), (0.0991358024691349, 0.9861425218689994)),
    ("circle", "section"): ((0.0, 0.9701098421473776), (0.03524229074889868, 0.0)),
    ("silk", "bend"): ((0.020498506880355347, 0.2518796992481203), (0.022099447513812154, 0.0)),
    ("circle", "bend"): ((1.0, 0.0), (0.7394366197183099, 0.0)),
}
ID_THRESHOLDS = {k: tuple(float(np.median([a[i], b[i]])) for i in (0, 1)) for k, (a, b) in ID_FOLDS.items()}


def relabel_share(lv, lf, h, sel) -> float:
    """Of the detections in ``sel`` labelled ``h`` by ``lv`` or ``lf``, the share labelled ``h`` by exactly one."""
    a, b = (lv == h) & sel, (lf == h) & sel
    un = int((a | b).sum())
    return float((a ^ b).sum() / un) if un else 0.0


def bend_windows(geometry, D) -> list:
    """(lo, hi) distance-to-go intervals of race distance ``D`` that lie on a bend of ``geometry`` (laps tiled)."""
    out, m = [], 0
    while geometry.bend_segments and m * geometry.circumference < D:
        for a, b in geometry.bend_segments:
            lo, hi = max(a + m * geometry.circumference, 0.0), min(b + m * geometry.circumference, D)
            if hi > lo:
                out.append((lo, hi))
        m += 1
    return sorted(out)


def _section_ends(run, D):
    """Distance to go at the official section ends (400 m apart, the line last)."""
    bnd = run.info.official_boundaries()[0]
    return np.asarray(bnd if len(bnd) else [400.0 * k for k in range(int((D - 1) // 400), -1, -1)], float)


def id_signals(run, P) -> dict:
    """Identity-robustness signals of run ``run`` (tracks, video-only and final labels, disc reads or legend rows) on its
    final paths ``P`` (polars: horse_no, t, s, lat, n_obs). -> dict(horses, tg, sections: list of dict(horse, section,
    t0, t1, n_nodes, R, Rmax, P), section_masks: {(i, section): node mask on the horses x tg grid}, bends: list of
    dict(horse, R, Rmax, P, n_nodes)). Read support P is NaN where no read source exists."""
    from . import video
    from .pipeline import FPS

    D = float(run.info.distance)
    horses = sorted(int(h) for h in P["horse_no"].unique().to_list())
    tg = np.round(np.sort(P["t"].unique().to_numpy()), 3)
    S, L, N = (np.full((len(horses), len(tg)), np.nan) for _ in range(3))
    hn = P["horse_no"].to_numpy()
    for i, h in enumerate(horses):
        m = hn == h
        j = np.searchsorted(tg, np.round(P["t"].to_numpy()[m], 3))
        S[i, j], L[i, j], N[i, j] = (P[c].to_numpy()[m] for c in ("s", "lat", "n_obs"))
    firsts = [tg[(N[i] > 0) & (tg > 0.5)] for i in range(len(horses))]
    t_r = float(np.nanmedian([f.min() if len(f) else np.nan for f in firsts]))
    with np.errstate(invalid="ignore"):
        runm = np.isfinite(S) & (S < D) & (tg >= t_r)[None, :]
        sec = ((D - S)[:, :, None] < _section_ends(run, D)[:-1][None, None, :]).sum(2)
    T = np.load(run.p("tracks.npz"), allow_pickle=True)
    tt, td = T["t"].astype(float), T["d"].astype(float)
    lv = np.load(run.p("det_labels.npz"))["labels"]
    lf = np.load(run.p("det_labels_final.npz"))["labels"]
    num = run._number_reads(T)
    leg = leg_t = None
    if num is None and run.p("legend.npz").exists():
        leg = run.legend_rows()
        if len(leg):
            grid = video.tick_grid(np.load(run.p("frames_idx.npz"))["idx"])
            leg_t = (video.tick_frame(leg[:, 0], grid) - run.clock_f0()) / FPS

    def support(h, sel, t0, t1):
        if num is not None:
            m = (lf == h) & sel
            return float((num[m] == h).mean()) if m.sum() else 0.0
        if leg_t is not None:
            w = (leg_t >= t0) & (leg_t <= t1)
            return float((leg[w, 1:] == h).any(1).mean()) if w.sum() else 0.0
        return float("nan")

    def near(i, m):
        with np.errstate(invalid="ignore"):
            z = ((np.abs(S[:, m] - S[i, m]) <= ID_NEAR_DS) & (np.abs(L[:, m] - L[i, m]) <= ID_NEAR_DLAT)).any(1)
        z[i] = True
        return np.flatnonzero(z)

    secs, masks = [], {}
    for i, h in enumerate(horses):
        for k in np.unique(sec[i][runm[i]]):
            m = runm[i] & (sec[i] == k)
            if m.sum() < ID_MIN_NODES:
                continue
            masks[(i, int(k))] = m
            t0, t1 = float(tg[m].min()), float(tg[m].max())
            sel = (tt >= t0 - ID_DET_PAD) & (tt <= t1 + ID_DET_PAD)
            secs.append(dict(horse=h, section=int(k), t0=t0, t1=t1, n_nodes=int(m.sum()), R=relabel_share(lv, lf, h, sel),
                             Rmax=max(relabel_share(lv, lf, horses[j], sel) for j in near(i, m)),
                             P=support(h, sel, t0, t1)))
    onb = np.zeros(S.shape, bool)
    detb = np.zeros(len(td), bool)
    for lo, hi in bend_windows(run.info.geometry, D):
        with np.errstate(invalid="ignore"):
            onb |= (D - S >= lo) & (D - S < hi)
        detb |= (td >= lo) & (td < hi)
    onb &= runm
    detb &= tt >= t_r
    bends = []
    if onb.any():
        Rh = [relabel_share(lv, lf, h, detb) for h in horses]
        for i, h in enumerate(horses):
            m = onb[i]
            if not m.any():
                bends.append(dict(horse=h, R=Rh[i], Rmax=Rh[i], P=float("nan"), n_nodes=0))
                continue
            bends.append(dict(horse=h, R=Rh[i], Rmax=max(Rh[j] for j in near(i, m)),
                              P=support(h, detb, float(tg[m].min()), float(tg[m].max())), n_nodes=int(m.sum())))
    return dict(horses=horses, tg=tg, sections=secs, section_masks=masks, bends=bends)


def id_robust(run, P, thresholds=None):
    """``id_robust`` of run ``run`` on its final paths ``P`` (``id_signals``); ``thresholds`` {(style, window): (tR, tP)},
    default ``ID_THRESHOLDS``. -> (per row of ``P``: its node's section unit is robust (no unit: False); sections with
    their verdict ``robust``; bends with ``robust`` (None without a bend node); yield dict)."""
    th = ID_THRESHOLDS if thresholds is None else thresholds
    style = "circle" if run.p("numbers.npz").exists() else "silk"
    sig = id_signals(run, P)

    def ok(r, window):
        tR, tP = th[(style, window)]
        return bool(r["Rmax"] <= tR and (not np.isfinite(r["P"]) or r["P"] >= tP))

    horses, tg = sig["horses"], sig["tg"]
    V = np.zeros((len(horses), len(tg)), bool)
    inunit = np.zeros_like(V)
    secs = []
    for r in sig["sections"]:
        r = dict(r, robust=ok(r, "section"))
        i = horses.index(r["horse"])
        m = sig["section_masks"][(i, r["section"])]
        V[i] |= m & r["robust"]
        inunit[i] |= m
        secs.append(r)
    bends = [dict(r, robust=ok(r, "bend") if r["n_nodes"] else None) for r in sig["bends"]]
    node = V[np.searchsorted(horses, P["horse_no"].to_numpy()), np.searchsorted(tg, np.round(P["t"].to_numpy(), 3))]
    nb = [b for b in bends if b["robust"] is not None]
    yld = dict(style=style, section_units=len(secs), section_units_robust=sum(r["robust"] for r in secs),
               unit_nodes=int(inunit.sum()), unit_nodes_robust=int(V.sum()), rows=len(node),
               rows_robust=int(node.sum()), bend_horses=len(nb), bend_horses_robust=sum(b["robust"] for b in nb))
    return node, secs, bends, yld


def describe_id_robust(secs, bends, yld) -> dict:
    """contract.json's ``id_robust`` entry: definitions, thresholds, the run's per-window verdicts and its yield."""
    return {
        "thresholds": [{"style": s, "window": w, "tR": tR, "tP": tP} for (s, w), (tR, tP) in ID_THRESHOLDS.items()],
        "threshold_rule": "median of the two folds' selected thresholds of one cross-fit, per render style and window "
        "type; not yet confirmed on a fresh set",
        "near_ds_m": ID_NEAR_DS, "near_dlat_render": ID_NEAR_DLAT, "min_nodes": ID_MIN_NODES, "det_pad_s": ID_DET_PAD,
        "rule": "robust = Rmax <= tR and read support >= tP (no read source: support not tested). Rmax: the maximum over "
        "the focal horse and every horse within near_ds_m along and near_dlat_render across at a window node of its "
        "relabel share (of its window detections, the share labelled it by exactly one of the video-only and final "
        "labels). Read support: circle, the share of the focal's final-labelled window detections with a confident disc "
        "read naming it; silk, the share of legend rows in the window listing it",
        "windows": "column: the node's section unit (run nodes between official section ends, >= min_nodes; no unit: "
        "false); bends: per horse, all its run nodes on bends pooled, for descriptors over bends",
        "yield": yld,
        "sections": _finite(secs),
        "bends": _finite(bends),
    }
