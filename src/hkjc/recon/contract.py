"""Output contract for modelling (``HKJC_CONTRACT=1``, opt-in): extra columns and artifacts written by ``fuse_final``.

Nothing here changes a path: every column the product writes without the switch is bitwise the same with it.

paths.parquet gains
  valid        the node lies in a validity-flagged stretch of the horse (``flag_stretches``)
  free         no other horse within one length along and one render abreast spacing across at the node
det_margins_final.npz  per tracks row: the final E-step's assignment margin (``assoc.assign_frames``); a margin, not a
               calibrated probability
contract.json          the definitions behind the columns, the run's flagged stretches and the noise table
"""

from __future__ import annotations

import os

import numpy as np


def contract_on() -> bool:
    """``HKJC_CONTRACT=1`` (opt-in): fuse_final writes the contract columns and artifacts. Off: as before."""
    return os.environ.get("HKJC_CONTRACT", "0") == "1"


# ------------------------------------------------------------------------------------------------- validity flag
# A detection is flagged when its video-only and final labels agree (E1), it is >= START_PAD s after the first render
# detection and before the line (E2, E3), the horse's final path is firm there and overlaps no other firm path (E4),
# and on the circle render its tracklet's majority confident disc read is the label (E5). Flagged detections of a horse
# bridged over gaps <= BRIDGE s form stretches of >= MIN_DET detections and >= MIN_DUR s; on the silk render a stretch
# also needs the legend to confirm the horse's rank on >= MIN_CONF and >= CONF_SHARE of its legend-read frames.
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
        lf_ = leg[:, 0].astype(int) if len(leg) else np.zeros(0, int)
        o = np.argsort(lf_)
        lf_, leg = lf_[o], leg[o]
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
                    if abs(lf_[q] - fr[i]) > LEG_WIN:
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


def describe(stretches) -> dict:
    """contract.json: the definitions behind the columns, the run's flagged stretches and the noise table."""
    return {
        "flag": {"stretches": stretches, "rule": "E1 video == final label, E2 >= 1 s after the first render detection, E3 "
                 "before the line, E4 firm and no overlap on the final paths, E5 tracklet disc-read majority (circle); "
                 "silk: legend rank confirmation; stretches >= 10 detections and >= 2 s, gaps <= 0.5 s"},
        "free": {"ds_m": FREE_DS, "dlat_render": FREE_DLAT},
        "noise_table": NOISE_TABLE,
        "margins": {"file": "det_margins_final.npz", "units": "-2 log-likelihood",
                    "meaning": "assignment margin, not a calibrated probability"},
    }
