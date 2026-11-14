"""Bayesian path reconstruction: every feed is an incomplete, noisy PROPOSAL of each horse's path.

Latent per horse: positions on a common race-clock grid t_k (``DT`` = 0.1 s)
    s_k  metres run along the racing rail (s = distance - d),   l_k  metres outward of the racing rail.
Prior (physics): bounded acceleration — second differences ~ N(0, (a_sd*DT^2)^2), much stiffer
laterally (horses drift sideways slowly) than longitudinally.
Likelihood: each observation (t, s?, l?) from feed f with its own noise sd; linear interpolation
between grid nodes. Geometry: l is box-bounded to [l_min, width - l_min] (outside the racing
rail, inside the outer rail).
MAP = sparse bounded linear least squares (``scipy.optimize.lsq_linear``); posterior sd from the
unconstrained normal equations (a sound local approximation when the bound is inactive).

Association: tracklets without an identity are scored against each horse's posterior path
(chi^2 of their observations) and assigned by Hungarian under a no-time-overlap rule, then the
paths are re-fused (``fuse_race``).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

DT = 0.1


@dataclass
class Obs:
    """Observations of one horse (or one anonymous tracklet) from one feed."""

    t: np.ndarray  # race-clock seconds
    s: np.ndarray | None  # metres run (NaN = unobserved)
    lat: np.ndarray | None  # lateral metres (NaN = unobserved)
    sd_s: float
    sd_l: float
    feed: str
    horse: int = 0  # 0 = anonymous tracklet
    tid: int = -1


@dataclass
class Prior:
    a_sd_s: float = 1.2  # m/s^2 longitudinal accel sd once galloping (|a| < ~2 m/s^2)
    a_sd_start: float = 6.0  # m/s^2 during the standing start (0 -> ~16 m/s in ~3 s)
    t_start: float = 6.0  # s: duration of the start phase (race clock)
    a_sd_l: float = 0.25  # m/s^2 lateral accel sd
    l_min: float = 0.3  # m: a horse centre is at least this far outside the racing rail
    width: float = 30.0  # m: racing rail -> outer rail
    vl_sd: float = 2.0  # lateral-velocity prior (m/s): sideways drift is slow
    # (no along-track velocity prior: it decelerated paths toward a stop across end-of-race data gaps;
    #  the acceleration prior alone extrapolates at constant speed)


@dataclass
class Path:
    horse: int
    t: np.ndarray
    s: np.ndarray
    lat: np.ndarray
    sd_s: np.ndarray
    sd_l: np.ndarray
    n_obs: np.ndarray  # observations within +-DT of each node (coverage)
    feeds: dict = field(default_factory=dict)


def _interp_rows(t, t0, n):
    """Linear-interpolation weights of times t onto grid nodes t0 + k*DT -> (rows, cols, vals)."""
    x = (np.asarray(t) - t0) / DT
    k = np.clip(np.floor(x).astype(int), 0, n - 2)
    w = np.clip(x - k, 0.0, 1.0)
    r = np.arange(len(x))
    return np.r_[r, r], np.r_[k, k + 1], np.r_[1 - w, w]


def smooth(obs: list, prior: Prior, t_grid: np.ndarray, *, want_sd: bool = True):
    """Fuse a list of ``Obs`` (all assumed to be the same horse) into one MAP path on ``t_grid``."""
    from scipy.sparse import coo_matrix, diags, vstack

    n = len(t_grid)
    t0 = t_grid[0]
    blocks_s, rhs_s, blocks_l, rhs_l = [], [], [], []
    for o in obs:
        for val, sd, blocks, rhs in (
            (o.s, o.sd_s, blocks_s, rhs_s),
            (o.lat, o.sd_l, blocks_l, rhs_l),
        ):
            if val is None:
                continue
            m = np.isfinite(val) & (o.t >= t_grid[0]) & (o.t <= t_grid[-1])
            if not m.any():
                continue
            r, c, v = _interp_rows(o.t[m], t0, n)
            blocks.append(coo_matrix((v / sd, (r, c)), shape=(m.sum(), n)))
            rhs.append(val[m] / sd)
    D2 = diags([1.0, -2.0, 1.0], [0, 1, 2], shape=(n - 2, n))
    # time-dependent longitudinal accel sd: the standing start accelerates ~5-8 m/s^2
    a_s = np.where(t_grid[1:-1] < prior.t_start, prior.a_sd_start, prior.a_sd_s)
    D1 = diags([-1.0, 1.0], [0, 1], shape=(n - 1, n))

    def solve(blocks, rhs, a_sd, lb, ub, v_sd=None):
        """Exact sparse MAP (normal equations, LU). Box bounds by an active set: violating nodes are
        pinned to their bound with a stiff penalty and the system re-solved (bounds are rarely
        active, so this converges in 1-3 rounds). Iterative LS solvers do NOT converge here — the
        acceleration prior makes the system stiff (rejected: lsq_linear/lsmr, 1.5 m error)."""
        from scipy.sparse.linalg import splu

        if not blocks:
            return None, None
        a_sd = np.broadcast_to(np.asarray(a_sd, float), (n - 2,))
        reg = [diags(1.0 / (a_sd * DT * DT)) @ D2] + ([D1 / (v_sd * DT)] if v_sd else [])
        A = vstack(blocks + reg).tocsr()
        b = np.r_[np.concatenate(rhs), np.zeros(sum(m.shape[0] for m in reg))]
        N0 = (A.T @ A).tocsc() + 1e-9 * diags(np.ones(n))
        r0 = A.T @ b
        pen = np.zeros(n)
        tgt = np.zeros(n)
        for _ in range(8):
            lu = splu((N0 + diags(pen)).tocsc())
            x = lu.solve(r0 + pen * tgt)
            if lb is None:
                break
            lo_v, hi_v = (x < lb - 1e-6) & (pen == 0), (x > ub + 1e-6) & (pen == 0)
            if not (lo_v.any() or hi_v.any()):
                break
            pen[lo_v | hi_v] = 1e4 * float(N0.diagonal().max())  # stiff vs the prior itself
            tgt[lo_v] = lb
            tgt[hi_v] = ub
        sd = None
        if want_sd:
            lu = splu(N0)  # posterior of the unconstrained problem
            idx = np.unique(np.r_[np.arange(0, n, 10), n - 1])
            e = np.zeros((n, len(idx)))
            e[idx, np.arange(len(idx))] = 1.0
            var = np.einsum("ij,ij->j", e, lu.solve(e))
            sd = np.sqrt(np.interp(np.arange(n), idx, var))
        return x, sd

    s, sd_s = solve(blocks_s, rhs_s, a_s, None, None)
    lat, sd_l = solve(
        blocks_l, rhs_l, prior.a_sd_l, prior.l_min, prior.width - prior.l_min, v_sd=prior.vl_sd
    )
    cnt = np.zeros(n)
    for o in obs:
        k = np.round((o.t - t0) / DT).astype(int)
        k = k[(k >= 0) & (k < n)]
        np.add.at(cnt, k, 1)
    return s, sd_s, lat, sd_l, cnt


def chi2_against(o: Obs, path: Path):
    """Mean chi^2 per observation of an anonymous tracklet against a horse's posterior path."""
    m = (o.t >= path.t[0]) & (o.t <= path.t[-1])
    if m.sum() < 3:
        return np.inf
    ps = np.interp(o.t[m], path.t, path.s)
    pl = np.interp(o.t[m], path.t, path.lat)
    vs = np.interp(o.t[m], path.t, path.sd_s) ** 2 + o.sd_s**2
    vl = np.interp(o.t[m], path.t, path.sd_l) ** 2 + o.sd_l**2
    r = 0.0
    if o.s is not None:
        r += np.nanmean((o.s[m] - ps) ** 2 / vs)
    if o.lat is not None:
        r += np.nanmean((o.lat[m] - pl) ** 2 / vl)
    return float(r)


def _overlaps(a: Obs, b_list: list) -> bool:
    for b in b_list:
        if not (a.t.max() < b.t.min() or b.t.max() < a.t.min()):
            return True
    return False


def _chains(orphans, *, gap_max=8.0):
    """Greedy stitch of anonymous tracklets into kinematically consistent, non-overlapping chains."""
    chains = []
    for o in sorted(orphans, key=lambda o: o.t[0]):
        best = None
        for ch in chains:
            last = ch[-1]
            dt = o.t[0] - last.t[-1]
            if not (0 < dt <= gap_max):
                continue
            v = (
                np.polyfit(last.t[-min(8, len(last.t)) :], last.s[-min(8, len(last.t)) :], 1)[0]
                if len(last.t) >= 3
                else 16.0
            )
            ds = abs(o.s[0] - (last.s[-1] + v * dt))
            dl = abs(o.lat[0] - last.lat[-1])
            if ds < 2.0 + 1.0 * dt and dl < 1.0 + 0.3 * dt:
                c = ds / (2.0 + dt) + dl / (1.0 + 0.3 * dt)
                if best is None or c < best[0]:
                    best = (c, ch)
        (best[1].append(o) if best else chains.append([o]))
    return chains


def _chain_vs_anchors(chain, anchors):
    """chi^2 per anchor of a chain's (linearly extrapolated) s at other-feed anchor observations."""
    t = np.concatenate([o.t for o in chain])
    s = np.concatenate([o.s for o in chain])
    o = np.argsort(t)
    t, s = t[o], s[o]
    r = []
    for a in anchors:
        for ta, sa in zip(a.t, a.s):
            k = slice(-min(15, len(t)), None) if ta > t[-1] else slice(0, min(15, len(t)))
            if ta < t[0] - 10 or ta > t[-1] + 10:
                continue
            b, c = np.polyfit(t[k], s[k], 1)
            r.append(((b * ta + c - sa) / max(a.sd_s, 0.5)) ** 2)
    return float(np.mean(r)) if r else np.inf


def fuse_race(
    obs: list,
    horses: list,
    prior: Prior,
    *,
    t_range=None,
    assoc_rounds: int = 2,
    chi2_max: float = 9.0,
    margin: float = 2.0,
    anchor_check: bool = True,
    stats: dict | None = None,
):
    """Fuse all feeds' observations into per-horse paths.

    ``obs``: list of ``Obs`` — numbered (horse>0) and anonymous (horse==0) tracklets from any feed.
    Anonymous tracklets are associated by posterior chi^2 (best must be < ``chi2_max`` and beat the
    runner-up by ``margin``x, and must not overlap in time with the horse's tracklets from the SAME
    feed), then everything is re-fused. Returns ({horse: Path}, assignment {tid: horse})."""
    ts = np.concatenate([o.t for o in obs])
    lo, hi = (ts.min(), ts.max()) if t_range is None else t_range
    t_grid = np.arange(lo, hi + DT / 2, DT)
    assign = {o.tid: o.horse for o in obs if o.horse > 0}
    paths = {}
    for rnd in range(assoc_rounds + 1):
        mine = {h: [o for o in obs if assign.get(o.tid, o.horse) == h] for h in horses}
        paths = {}
        for h in horses:
            if not mine[h]:
                continue
            s, sd_s, lat, sd_l, cnt = smooth(mine[h], prior, t_grid)
            if s is None or lat is None:
                continue
            feeds = {}
            for o in mine[h]:
                feeds[o.feed] = feeds.get(o.feed, 0) + len(o.t)
            paths[h] = Path(h, t_grid, s, lat, sd_s, sd_l, cnt, feeds)
        if rnd == assoc_rounds:
            break
        changed = False
        anon = sorted(
            [o for o in obs if o.horse == 0 and o.tid not in assign], key=lambda o: -len(o.t)
        )
        for o in anon:
            sc = sorted(
                (
                    (chi2_against(o, p), h)
                    for h, p in paths.items()
                    if not _overlaps(o, [x for x in mine[h] if x.feed == o.feed])
                ),
                key=lambda x: x[0],
            )
            if not sc or not np.isfinite(sc[0][0]) or sc[0][0] > chi2_max:
                continue
            if len(sc) > 1 and sc[1][0] < margin * sc[0][0]:
                continue
            assign[o.tid] = sc[0][1]
            mine[sc[0][1]].append(o)
            changed = True
        if not changed:
            break
    # Orphans: anonymous tracklets of horses that have NO numbered tracklets yet (nothing to score
    # them against above). Stitch them into chains and give each chain to a path-less horse by its
    # other-feed anchors (e.g. finish-camera crossings), or by elimination when one horse is missing.
    orphans = [
        o
        for o in obs
        if o.horse == 0 and o.tid not in assign and len(o.t) >= 5 and o.lat is not None
    ]
    covered = {h for h, p in paths.items() if p.n_obs.sum() > 0.3 * len(p.t)}
    missing = [h for h in horses if h not in covered]
    if orphans and missing:
        chains = sorted(_chains(orphans), key=lambda ch: -sum(len(o.t) for o in ch))
        anchors = {
            h: [o for o in obs if assign.get(o.tid, o.horse) == h and o.feed != "render"]
            for h in missing
        }
        newly = {}
        for ch in chains:
            sc = sorted(
                ((_chain_vs_anchors(ch, anchors[h]), h) for h in missing if h not in newly),
                key=lambda x: x[0],
            )
            if (
                sc
                and np.isfinite(sc[0][0])
                and sc[0][0] < chi2_max
                and (len(sc) == 1 or sc[1][0] > margin * sc[0][0])
            ):
                newly[sc[0][1]] = ch
            elif len(missing) - len(newly) == 1 and not np.isfinite(sc[0][0]) if sc else False:
                newly[sc[0][1]] = ch
        for h, ch in newly.items():
            for o in ch:
                assign[o.tid] = h
        if newly:
            mine = {h: [o for o in obs if assign.get(o.tid, o.horse) == h] for h in horses}
            for h in newly:
                s_, sd_s, lat, sd_l, cnt = smooth(mine[h], prior, t_grid)
                if s_ is not None and lat is not None:
                    feeds = {}
                    for o in mine[h]:
                        feeds[o.feed] = feeds.get(o.feed, 0) + len(o.t)
                    paths[h] = Path(h, t_grid, s_, lat, sd_s, sd_l, cnt, feeds)
    if anchor_check:
        bad = _anchor_conflicts(obs, assign, paths, prior, t_grid)
        if stats is not None:
            stats["demoted"] = stats.get("demoted", 0) + len(bad)
        if bad:  # demote the mislabelled tracklets to anonymous and re-fuse once
            obs2 = [replace(o, horse=0) if o.tid in bad else o for o in obs]
            return fuse_race(
                obs2,
                horses,
                prior,
                t_range=t_range,
                assoc_rounds=assoc_rounds,
                chi2_max=chi2_max,
                margin=margin,
                anchor_check=False,
            )
    return paths, assign


ANCHOR_FEEDS = ("finishcam", "official")


def _anchor_resid(anchors, path):
    """[(feed, rounded s, residual m, sd m)] of a horse's independent-feed anchors vs its path."""
    out = []
    for a in anchors:
        m = np.isfinite(a.s) & (a.t >= path.t[0]) & (a.t <= path.t[-1])
        for ta, sa in zip(a.t[m], a.s[m]):
            ps = np.interp(ta, path.t, path.s)
            sd = np.sqrt(a.sd_s**2 + np.interp(ta, path.t, path.sd_s) ** 2)
            out.append((a.feed, round(float(sa)), float(sa - ps), float(sd)))
    return out


def _anchor_z(anchors, path, common=None):
    """Largest HORSE-SPECIFIC |z| of a horse's anchors: the field's median residual at the same
    anchor (``common`` {(feed, s): m}) is subtracted, so a common-mode render bias (e.g. +0.4 s for
    every horse at R08's first boundary) is not mistaken for a mislabelled tracklet."""
    common = common or {}
    z = [abs(r - common.get((f, s_), 0.0)) / sd for f, s_, r, sd in _anchor_resid(anchors, path)]
    return max(z) if z else 0.0


def _anchor_conflicts(obs, assign, paths, prior, t_grid, *, z_max=4.0, max_drop=3):
    """IDENTITY-OUTLIER rejection driven by independent feeds. A render tracklet with a wrong label
    drags its horse's path away from that horse's finish-camera crossing (R08 #8: 0.32 s early; the
    camera's own gap was right to 0.06 s). When an anchor disagrees at > z_max sigma, the numbered
    render tracklet whose removal best restores it (and at least halves the conflict) is returned
    for demotion to anonymous; association then re-places it by chi^2. -> set of tids."""
    bad = set()
    per = {}
    for h, p in paths.items():
        anchors = [o for o in obs if assign.get(o.tid, o.horse) == h and o.feed in ANCHOR_FEEDS]
        for f, s_, r, _sd in _anchor_resid([a for a in anchors if a.s is not None], p):
            per.setdefault((f, s_), []).append(r)
    common = {k: float(np.median(v)) for k, v in per.items() if len(v) >= 3}
    for h, p in paths.items():
        mine = [o for o in obs if assign.get(o.tid, o.horse) == h]
        anchors = [o for o in mine if o.feed in ANCHOR_FEEDS and o.s is not None]
        if not anchors:
            continue
        z0 = _anchor_z(anchors, p, common)
        n_render0 = sum(len(o.t) for o in mine if o.feed == "render")
        for _ in range(max_drop):
            if z0 <= z_max:
                break
            best = None
            n_render = sum(len(o.t) for o in mine if o.feed == "render")
            for o in mine:
                if o.feed != "render" or o.horse == 0:
                    continue
                if n_render - len(o.t) < 0.5 * n_render0:
                    continue  # never strip a horse below half its render data (R03: 2 path-less)
                rest = [x for x in mine if x is not o]
                s_, sd_s, lat, sd_l, cnt = smooth(rest, prior, t_grid)
                if s_ is None:
                    continue
                z = _anchor_z(anchors, Path(h, t_grid, s_, lat, sd_s, sd_l, cnt), common)
                if best is None or z < best[0]:
                    best = (z, o)
            if best is None or best[0] > 0.5 * z0:
                break
            bad.add(best[1].tid)
            mine = [x for x in mine if x is not best[1]]
            z0 = best[0]
    return bad
