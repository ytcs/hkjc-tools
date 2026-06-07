"""HOLY-GRAIL 4-feed FUSION — STEP 2: the state-space SMOOTHER + reconstructed-race trajectories.

Task #80 / step 2 (journal thread 0060). This is the CENTREPIECE the per-feed work laddered to:
fuse the three VALIDATED lateral feeds into a single per-horse latent (arclen, lateral) trajectory
on the shared clock, under a FIRST-PRINCIPLES PHYSICS motion prior (bounded longitudinal accel +
bounded lateral velocity — the principled home of the 'no teleport' / jitter concern, applied
JOINTLY in the smoother, NOT as a per-feed pre-filter).

INPUTS (REUSE read-only, never rebuilt, never mutated):
  - AERIAL render   data_cache/cv_s0d_aerial_render.parquet  (#82/0075):
        full-race; ARCLEN via the field odometer (frame-shared, valid on 198 frames, low noise),
        y_lat ORDERING faithful (rp +0.45) but absolute metric §4b-null (high noise; worse >1300m
        foreshortening, down-weight there). Identity 11/12 via the legend.
  - DRONE numbered  data_cache/cv_drone_numbered.parquet     (#87/0080):
        time-anchored on the shared clock (vid_t -> race_t affine -> arclen); y_lat ORDERING
        cross-validated vs aerial (rho +0.93) but metric §4b-null. Identity 10/12.
  - PATROL numbered data_cache/cv_patrol_numbered.parquet    (#86/0082):
        the METRIC ANCHOR — the ONLY feed with within/between<1 (fixed camera, head-on); y_lat in
        METRES, validated tight/wide +4.1m vs stewards. Window arclen ~1058-1279m, 8/12 numbered.

THE MODEL (per horse, independent — field interactions are future work):
  latent z_h(t) = [s, s_dot, l, l_dot]  on a shared race_t grid (5 Hz).
  MOTION PRIOR (physics):
    - s: near-constant cruise with BOUNDED longitudinal accel (small accel process noise).
    - l: BOUNDED LATERAL velocity (a horse drifts sideways slowly vs forward) — small l_dot
      process noise => the smoothness prior that defeats per-frame jitter.
  OBSERVATION MODELS (each feed over ITS coverage window, with its own noise):
    - ARCLEN obs: aerial odometer (low noise), drone time-anchor (moderate), patrol arclen (moderate).
    - LATERAL obs: PATROL = LOW-noise metric (anchors absolute scale on its window);
      AERIAL + DRONE = HIGH metric-noise lateral whose ORDERING is trustworthy but absolute metres
      biased -> inflated metric variance (+ extra inflation for aerial on >1300m). PATROL pins the
      scale; aerial/drone extend coverage; the motion prior carries the metric where only ordering
      feeds see the horse, so the band WIDENS honestly there.
  INFERENCE: a numpy Kalman forward filter + RTS backward smoother per horse; propagate the state
  covariance (the uncertainty bands MUST widen where coverage is thin / only-ordering feeds).

DISCIPLINE: study-only. src/hkjc untouched. Inputs read-only (not mutated). Official data
(sections / finish-RP / stewards comments) is VALIDATION-ONLY, never a model input.
"""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

# ------------------------------------------------------------------------------------------------
# Paths
AERIAL_PANEL = "data_cache/cv_s0d_aerial_render.parquet"
DRONE_PANEL = "data_cache/cv_drone_numbered.parquet"
PATROL_PANEL = "data_cache/cv_patrol_numbered.parquet"
OUT_TRAJ = "data_cache/cv_fusion_trajectories.parquet"
FIG_RECON = "data_cache/figs/cv_fusion_recon.png"

# ------------------------------------------------------------------------------------------------
# Shared clock / geometry  (official sections — defines race_t, NOT a model input beyond the clock).
DIST = 1650.0
TRACK_WIDTH = 21.3  # "C" rail width (m)
OFFICIAL_SECTIONS = [27.65, 23.30, 23.75, 24.15]  # s  (sum 98.85 == winner 1:38.85)
WINNER_FINISH_RACE_T = sum(OFFICIAL_SECTIONS)  # 98.85 s
# distance-run boundaries for the 1650m race (first call at the 1200m-to-go mark == 450m run).
_D_BOUNDS = np.array([0.0, 450.0, 850.0, 1250.0, 1650.0])
_T_CUM = np.concatenate([[0.0], np.cumsum(OFFICIAL_SECTIONS)])  # 0,27.65,50.95,74.70,98.85

# DRONE shared-clock anchor (0080): race_t = (vid_t - 3.0) * 0.8931
DRONE_T0, DRONE_RATE = 3.0, 0.8931


def arclen_to_race_t(arclen_m: np.ndarray) -> np.ndarray:
    return np.interp(np.asarray(arclen_m, float), _D_BOUNDS, _T_CUM)


def race_t_to_arclen(race_t: np.ndarray) -> np.ndarray:
    return np.interp(np.asarray(race_t, float), _T_CUM, _D_BOUNDS)


# ------------------------------------------------------------------------------------------------
# OFFICIAL VALIDATION-ONLY tables (never a model input).
OFFICIAL_RP = {  # running positions at the 4 calls; lower == nearer the lead.
    3: [7, 7, 6, 1], 12: [1, 1, 1, 2], 6: [11, 11, 10, 3], 2: [9, 9, 9, 4],
    9: [4, 4, 3, 5], 8: [6, 2, 2, 6], 11: [8, 8, 8, 7], 4: [5, 6, 5, 8],
    7: [12, 12, 11, 9], 10: [3, 3, 4, 10], 5: [2, 5, 7, 11], 1: [10, 10, 12, 12],
}
FINISH_ORDER = [3, 12, 6, 2, 9, 8, 11, 4, 7, 10, 5, 1]
RAIL_HORSES = {12, 8, 4, 11}   # inside/rail-commented (stewards)
WIDE_HORSES = {2, 6, 9, 1}     # wide-commented (stewards)


# ------------------------------------------------------------------------------------------------
# OBSERVATION INGEST — each feed -> (runner, race_t, arclen?, ylat?, with per-obs noise tag).
# ------------------------------------------------------------------------------------------------
@dataclass
class Obs:
    runner: int
    race_t: float
    arclen: float | None
    ylat: float | None
    feed: str
    sd_arclen: float
    sd_ylat: float


# --- noise levels (metres). The metric hierarchy lives HERE (PHILOSOPHY: patrol pins the scale).
SD_ARC_AERIAL = 3.0     # odometer field arclen — low noise, but it is the FIELD position (shared)
SD_ARC_DRONE = 8.0      # time-anchored arclen — moderate
SD_ARC_PATROL = 6.0     # patrol arclen from its homography window — moderate
SD_YLAT_PATROL = 1.2    # METRIC ANCHOR — low noise (0076 within/between 0.06-0.16)
SD_YLAT_AERIAL = 6.0    # ordering trustworthy, absolute metric biased -> high (inflated) variance
SD_YLAT_AERIAL_RUNIN = 14.0  # >1300m foreshortening (0075) -> extra inflation
SD_YLAT_DRONE = 7.0     # ordering trustworthy, metric §4b-null -> high variance
SD_YLAT_DRONE_RUNIN = 11.0   # run-in (>1300m): no metric anchor sees it, drone metric §4b-null (0080)


def load_aerial() -> tuple[list[Obs], pl.DataFrame]:
    """AERIAL: per-row race_t from the FIELD ODOMETER (frame-shared arclen, valid 198 frames),
    interpolated over vid-time t to every row; y_lat is the high-metric-noise lateral obs."""
    a = pl.read_parquet(AERIAL_PANEL)
    # field odometer: median x_arclen per frame on the 198 valid-arclen frames -> frame/t -> arclen.
    od = (
        a.filter(pl.col("x_arclen").is_not_nan())
        .group_by("frame")
        .agg(pl.col("x_arclen").median().alias("arc"), pl.col("t").first().alias("t"))
        .sort("t")
    )
    od_t, od_arc = od["t"].to_numpy(), od["arc"].to_numpy()
    t_all = a["t"].to_numpy()
    arc_all = np.interp(t_all, od_t, od_arc)  # field arclen at every aerial row
    race_t_all = arclen_to_race_t(arc_all)
    a = a.with_columns(
        pl.Series("arclen", arc_all), pl.Series("race_t", race_t_all)
    )
    obs: list[Obs] = []
    rn = a["runner"].to_numpy()
    yl = a["y_lat"].to_numpy()
    for i in range(a.height):
        runin = arc_all[i] > 1300.0
        obs.append(
            Obs(
                runner=int(rn[i]), race_t=float(race_t_all[i]),
                arclen=float(arc_all[i]), ylat=float(yl[i]), feed="aerial",
                sd_arclen=SD_ARC_AERIAL,
                sd_ylat=SD_YLAT_AERIAL_RUNIN if runin else SD_YLAT_AERIAL,
            )
        )
    return obs, a


def load_drone() -> list[Obs]:
    """DRONE: vid_t -> race_t (affine 0080) -> arclen; y_lat high-metric-noise lateral obs."""
    d = pl.read_parquet(DRONE_PANEL)
    vt = d["vid_t"].to_numpy()
    race_t = (vt - DRONE_T0) * DRONE_RATE
    arc = race_t_to_arclen(race_t)
    rn = d["number"].to_numpy()
    yl = d["y_lat"].to_numpy()
    obs: list[Obs] = []
    for i in range(d.height):
        runin = arc[i] > 1300.0
        obs.append(
            Obs(
                runner=int(rn[i]), race_t=float(race_t[i]),
                arclen=float(arc[i]), ylat=float(yl[i]), feed="drone",
                sd_arclen=SD_ARC_DRONE,
                sd_ylat=SD_YLAT_DRONE_RUNIN if runin else SD_YLAT_DRONE,
            )
        )
    return obs


def load_patrol() -> list[Obs]:
    """PATROL: already on race_t + arclen + y_lat_m (metres). The METRIC ANCHOR (low ylat noise).
    Confidence column down-weights the ambiguous pink-silk family (scale the ylat sd by 1/conf)."""
    p = pl.read_parquet(PATROL_PANEL)
    rn = p["runner"].to_numpy()
    rt = p["race_t"].to_numpy()
    arc = p["arclen"].to_numpy()
    yl = p["y_lat_m"].to_numpy()
    conf = p["confidence"].to_numpy()
    obs: list[Obs] = []
    for i in range(p.height):
        # low confidence -> inflate ylat sd (cap inflation x4); confident reads keep the anchor sd.
        c = float(np.clip(conf[i], 0.25, 1.0))
        obs.append(
            Obs(
                runner=int(rn[i]), race_t=float(rt[i]),
                arclen=float(arc[i]), ylat=float(yl[i]), feed="patrol",
                sd_arclen=SD_ARC_PATROL, sd_ylat=SD_YLAT_PATROL / c,
            )
        )
    return obs


# ------------------------------------------------------------------------------------------------
# THE SMOOTHER — per-horse Kalman forward filter + RTS backward smoother.
# State z = [s, s_dot, l, l_dot]; constant-velocity transition with bounded-accel process noise.
# ------------------------------------------------------------------------------------------------
GRID_HZ = 5.0
DT = 1.0 / GRID_HZ
# Process noise — the PHYSICS motion prior (bounded accel). Continuous white-accel model:
#  s: cruise ~16-17 m/s, small longitudinal accel allowed (the EOM scale).
SIGMA_A_LONG = 0.8   # m/s^2  longitudinal accel std (bounded — no teleport, smooth pace)
#  l: a horse drifts sideways SLOWLY vs forward motion -> tight lateral-accel cap.
SIGMA_A_LAT = 0.35   # m/s^2  lateral accel std (bounded lateral velocity ~<1-2 m/s emerges)


def _process_block(dt: float, sa: float) -> tuple[np.ndarray, np.ndarray]:
    """1-D constant-velocity transition F (2x2) and white-accel process cov Q (2x2)."""
    f = np.array([[1.0, dt], [0.0, 1.0]])
    q = (sa**2) * np.array(
        [[dt**3 / 3.0, dt**2 / 2.0], [dt**2 / 2.0, dt]]
    )
    return f, q


def smooth_horse(
    grid: np.ndarray, obs: list[Obs]
) -> dict[str, np.ndarray]:
    """RTS smoother for one horse on the shared race_t grid. Returns smoothed s,l + their sds."""
    n = len(grid)
    fL, qL = _process_block(DT, SIGMA_A_LONG)
    fT, qT = _process_block(DT, SIGMA_A_LAT)
    # block-diagonal 4-state: [s,s_dot | l,l_dot]
    f = np.zeros((4, 4))
    f[:2, :2], f[2:, 2:] = fL, fT
    q = np.zeros((4, 4))
    q[:2, :2], q[2:, 2:] = qL, qT

    # bin observations to nearest grid node.
    obs_by_node: dict[int, list[Obs]] = {}
    for o in obs:
        k = int(round((o.race_t - grid[0]) / DT))
        if 0 <= k < n:
            obs_by_node.setdefault(k, []).append(o)

    # init from the earliest arclen obs (or race_t->arclen field guess).
    first = min(obs, key=lambda o: o.race_t)
    s0 = first.arclen if first.arclen is not None else float(race_t_to_arclen(first.race_t))
    l0 = first.ylat if first.ylat is not None else TRACK_WIDTH / 2.0
    x = np.array([s0, 16.5, l0, 0.0])
    p = np.diag([50.0**2, 5.0**2, 12.0**2, 2.0**2])

    xp = np.zeros((n, 4))
    pp = np.zeros((n, 4, 4))
    xf = np.zeros((n, 4))
    pf = np.zeros((n, 4, 4))

    for k in range(n):
        if k == 0:
            x_pred, p_pred = x, p
        else:
            x_pred = f @ xf[k - 1]
            p_pred = f @ pf[k - 1] @ f.T + q
        xp[k], pp[k] = x_pred, p_pred
        x_upd, p_upd = x_pred.copy(), p_pred.copy()
        for o in obs_by_node.get(k, []):
            # arclen obs -> measures s (index 0)
            if o.arclen is not None:
                h = np.array([1.0, 0.0, 0.0, 0.0])
                r = o.sd_arclen**2
                x_upd, p_upd = _kf_update(x_upd, p_upd, h, o.arclen, r)
            # lateral obs -> measures l (index 2)
            if o.ylat is not None:
                h = np.array([0.0, 0.0, 1.0, 0.0])
                r = o.sd_ylat**2
                x_upd, p_upd = _kf_update(x_upd, p_upd, h, o.ylat, r)
        xf[k], pf[k] = x_upd, p_upd

    # RTS backward pass.
    xs = xf.copy()
    ps = pf.copy()
    for k in range(n - 2, -1, -1):
        p_pred = pp[k + 1]
        try:
            c = pf[k] @ f.T @ np.linalg.inv(p_pred)
        except np.linalg.LinAlgError:
            c = pf[k] @ f.T @ np.linalg.pinv(p_pred)
        xs[k] = xf[k] + c @ (xs[k + 1] - xp[k + 1])
        ps[k] = pf[k] + c @ (ps[k + 1] - p_pred) @ c.T

    return {
        "s": xs[:, 0], "sd_s": np.sqrt(np.clip(ps[:, 0, 0], 0, None)),
        "l": xs[:, 2], "sd_l": np.sqrt(np.clip(ps[:, 2, 2], 0, None)),
        "node_has_obs": np.array([k in obs_by_node for k in range(n)]),
    }


def _kf_update(
    x: np.ndarray, p: np.ndarray, h: np.ndarray, z: float, r: float
) -> tuple[np.ndarray, np.ndarray]:
    y = z - h @ x
    s = h @ p @ h + r
    k = (p @ h) / s
    x_new = x + k * y
    p_new = (np.eye(4) - np.outer(k, h)) @ p
    return x_new, p_new


# ------------------------------------------------------------------------------------------------
# DRIVER
# ------------------------------------------------------------------------------------------------
def coverage_window(obs: list[Obs]) -> tuple[float, float]:
    rts = [o.race_t for o in obs]
    return min(rts), max(rts)


def run() -> None:
    aer_obs, _aer_df = load_aerial()
    dro_obs = load_drone()
    pat_obs = load_patrol()
    all_obs = aer_obs + dro_obs + pat_obs

    runners = sorted({o.runner for o in all_obs})
    print(f"[ingest] aerial {len(aer_obs)} obs ({len({o.runner for o in aer_obs})} runners), "
          f"drone {len(dro_obs)} ({len({o.runner for o in dro_obs})}), "
          f"patrol {len(pat_obs)} ({len({o.runner for o in pat_obs})})")
    print(f"[ingest] union identity: {len(runners)} runners {runners}")

    # shared grid spanning the union coverage.
    t_lo = min(o.race_t for o in all_obs)
    t_hi = max(o.race_t for o in all_obs)
    grid = np.arange(t_lo, t_hi + DT, DT)
    print(f"[grid] race_t {t_lo:.2f}..{t_hi:.2f}s @ {GRID_HZ:.0f}Hz, {len(grid)} nodes")

    rows = []
    smoothed: dict[int, dict] = {}
    for rn in runners:
        ho = [o for o in all_obs if o.runner == rn]
        clo, chi = coverage_window(ho)
        res = smooth_horse(grid, ho)
        smoothed[rn] = res
        # only emit grid nodes within (a small pad of) this horse's coverage window.
        pad = 1.0
        mask = (grid >= clo - pad) & (grid <= chi + pad)
        for i in np.where(mask)[0]:
            rows.append({
                "runner": rn, "race_t": float(grid[i]),
                "arclen": float(res["s"][i]), "y_lat_m": float(res["l"][i]),
                "sd_arclen": float(res["sd_s"][i]), "sd_ylat": float(res["sd_l"][i]),
            })
        feeds = sorted({o.feed for o in ho})
        print(f"  runner {rn:2d}: cover race_t {clo:5.1f}-{chi:5.1f}s  "
              f"arclen {res['s'][mask].min():6.0f}-{res['s'][mask].max():6.0f}m  feeds {feeds}")

    traj = pl.DataFrame(rows)
    traj.write_parquet(OUT_TRAJ)
    print(f"[panel] {OUT_TRAJ}  ({traj.height} rows, {traj['runner'].n_unique()} runners)")

    validate(smoothed, grid, runners)
    make_figure(smoothed, grid, runners, all_obs)


# ------------------------------------------------------------------------------------------------
# VALIDATION (official = validation only)
# ------------------------------------------------------------------------------------------------
def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3:
        return float("nan")
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean()
    rb -= rb.mean()
    den = np.sqrt((ra**2).sum() * (rb**2).sum())
    return float((ra * rb).sum() / den) if den > 0 else float("nan")


def _median_ylat_band(
    smoothed: dict, grid: np.ndarray, runners: list[int], a_lo: float, a_hi: float
) -> dict[int, float]:
    """Median fused y_lat per runner over an ARCLEN band (mapped to race_t via the grid arclen)."""
    out: dict[int, float] = {}
    for rn in runners:
        s = smoothed[rn]["s"]
        m = (s >= a_lo) & (s <= a_hi)
        if m.sum() >= 2:
            out[rn] = float(np.median(smoothed[rn]["l"][m]))
    return out


def _mean_sd_ylat_band(
    smoothed: dict, runners: list[int], a_lo: float, a_hi: float
) -> float:
    vals = []
    for rn in runners:
        s = smoothed[rn]["s"]
        m = (s >= a_lo) & (s <= a_hi)
        if m.sum() >= 1:
            vals.append(np.median(smoothed[rn]["sd_l"][m]))
    return float(np.mean(vals)) if vals else float("nan")


def validate(smoothed: dict, grid: np.ndarray, runners: list[int]) -> None:
    print("\n" + "=" * 78)
    print("VALIDATION (official = validation only, never a model input)")
    print("=" * 78)

    # ---- fused tight-vs-wide gap vs stewards, over the clean body (<1300m) ----
    body = _median_ylat_band(smoothed, grid, runners, 150.0, 1300.0)
    rail_v = [body[h] for h in RAIL_HORSES if h in body]
    wide_v = [body[h] for h in WIDE_HORSES if h in body]
    n_cov = len([h for h in (RAIL_HORSES | WIDE_HORSES) if h in body])
    if rail_v and wide_v:
        gap = float(np.mean(wide_v) - np.mean(rail_v))
        print(f"[tight/wide <1300m] rail {sorted(RAIL_HORSES & set(body))} mean {np.mean(rail_v):5.2f}m | "
              f"wide {sorted(WIDE_HORSES & set(body))} mean {np.mean(wide_v):5.2f}m | "
              f"GAP wide-rail = {gap:+.2f}m  (stewards/patrol ref +4.1m; covers {n_cov}/8 commented)")
    # per-runner band y_lat (clean body) for the record.
    print("  per-runner fused median y_lat (<1300m):")
    for rn in sorted(body):
        tag = "rail" if rn in RAIL_HORSES else ("wide" if rn in WIDE_HORSES else "mid ")
        print(f"    #{rn:2d} [{tag}] {body[rn]:5.2f}m")

    # ---- fused LATERAL (rail-vs-wide) vs official mean running position (0075 reference: +0.45) ----
    mean_rp = {h: float(np.mean(OFFICIAL_RP[h])) for h in OFFICIAL_RP}
    common = [h for h in body if h in mean_rp]
    if len(common) >= 3:
        yl = np.array([body[h] for h in common])
        rp = np.array([mean_rp[h] for h in common])
        print(f"\n[fused y_lat(<1300m) vs official mean-RP]  Spearman {_spearman(yl, rp):+.2f}  "
              f"(n={len(common)}; NB mean-RP conflates lead/back with rail/wide — the rail-vs-wide "
              f"GAP above is the cleaner validated axis)")

    # ---- fused running ORDER vs official RP at the section poles ----
    # HONEST CAVEAT: the only LONGITUDINAL observation is the aerial FIELD odometer (frame-shared
    # distance-run common to all horses) + coarse drone arclen. Per-horse STAGGER is essentially
    # UNOBSERVED (std ~1.2m across the field at a pole) -> the fused lead-order is dominated by
    # sub-metre smoother artifacts and is NOT a meaningful order signal. Reported for completeness;
    # a real per-horse longitudinal order needs the RF MARGINS feed (the un-built 4th feed).
    print("\n[order vs official RP at the section poles — see caveat: longitudinal is FIELD-shared]")
    poles = {"1200m(450m)": 450.0, "800m(850m)": 850.0, "400m(1250m)": 1250.0, "finish(1640m)": 1640.0}
    rp_call = {0: {h: OFFICIAL_RP[h][0] for h in OFFICIAL_RP},
               1: {h: OFFICIAL_RP[h][1] for h in OFFICIAL_RP},
               2: {h: OFFICIAL_RP[h][2] for h in OFFICIAL_RP},
               3: {h: OFFICIAL_RP[h][3] for h in OFFICIAL_RP}}
    for ci, (name, a_mark) in enumerate(poles.items()):
        # fused arclen ordering: who is most advanced at the race_t where field arclen ~ a_mark.
        target_t = float(arclen_to_race_t(np.array([a_mark]))[0])
        ki = int(np.clip(round((target_t - grid[0]) / DT), 0, len(grid) - 1))
        fused_s = {rn: smoothed[rn]["s"][ki] for rn in runners
                   if smoothed[rn]["node_has_obs"][:ki + 1].any()}
        common = [h for h in fused_s if h in rp_call[ci]]
        if len(common) >= 3:
            # lead order: higher arclen == nearer the front == lower RP.
            fs = np.array([-fused_s[h] for h in common])  # neg so rank1=leader
            rp = np.array([rp_call[ci][h] for h in common], float)
            rho = _spearman(fs, rp)
            print(f"  {name:14s} race_t {target_t:5.1f}s  n={len(common):2d}  "
                  f"rho(fused lead-order, official RP) = {rho:+.2f}")

    # ---- per-band lateral uncertainty (the 'how good is each segment' quantifier) ----
    print("\n[per-band fused y_lat uncertainty — the segment-quality census]")
    bands = [("start/accel 150-450m", 150, 450), ("backstr/bend 450-850m", 450, 850),
             ("bend->straight 850-1300m", 850, 1300), ("RUN-IN >1300m", 1300, 1650)]
    for name, lo, hi in bands:
        sd = _mean_sd_ylat_band(smoothed, runners, lo, hi)
        ncov = len([rn for rn in runners
                    if ((smoothed[rn]["s"] >= lo) & (smoothed[rn]["s"] <= hi)).sum() >= 1])
        print(f"  {name:26s}  mean fused sd(y_lat) = {sd:5.2f}m  ({ncov} runners present)")


# ------------------------------------------------------------------------------------------------
# THE RECON FIGURE
# ------------------------------------------------------------------------------------------------
def make_figure(
    smoothed: dict, grid: np.ndarray, runners: list[int], all_obs: list[Obs]
) -> None:
    cmap = plt.colormaps["tab20"]
    colors = {rn: cmap(i % 20) for i, rn in enumerate(runners)}
    fig, axes = plt.subplots(2, 1, figsize=(15, 13))

    # coverage bands (race_t) for the feed-annotation strip.
    feed_cov = {}
    for feed in ("aerial", "drone", "patrol"):
        fo = [o for o in all_obs if o.feed == feed]
        feed_cov[feed] = (min(o.race_t for o in fo), max(o.race_t for o in fo))

    # ---- PANEL 1: arclen vs race_t (the longitudinal reconstruction) ----
    ax = axes[0]
    for rn in runners:
        s, sd, has = smoothed[rn]["s"], smoothed[rn]["sd_s"], smoothed[rn]["node_has_obs"]
        # only draw where this horse has support (between first & last obs node).
        idx = np.where(has)[0]
        if len(idx) == 0:
            continue
        m = slice(idx.min(), idx.max() + 1)
        ax.plot(grid[m], s[m], color=colors[rn], lw=1.4, label=f"#{rn}")
        ax.fill_between(grid[m], s[m] - 2 * sd[m], s[m] + 2 * sd[m],
                        color=colors[rn], alpha=0.10, lw=0)
    for a_mark in (450, 850, 1250, 1640):
        tt = float(arclen_to_race_t(np.array([a_mark]))[0])
        ax.axvline(tt, color="0.7", ls=":", lw=0.8)
    ax.set_xlabel("race_t (s)  [shared clock: official sections]")
    ax.set_ylabel("arclen (m, distance run)")
    ax.set_title("FUSION RECON — longitudinal: arclen vs race_t (12 horses, ±2σ bands)")
    ax.legend(ncol=6, fontsize=7, loc="upper left")

    # ---- PANEL 2: y_lat vs arclen (the lateral/control reconstruction) ----
    ax = axes[1]
    for rn in runners:
        s, lat, sdl, has = (smoothed[rn]["s"], smoothed[rn]["l"],
                            smoothed[rn]["sd_l"], smoothed[rn]["node_has_obs"])
        idx = np.where(has)[0]
        if len(idx) == 0:
            continue
        m = slice(idx.min(), idx.max() + 1)
        tag = "rail" if rn in RAIL_HORSES else ("wide" if rn in WIDE_HORSES else "")
        ax.plot(s[m], lat[m], color=colors[rn], lw=1.4,
                label=f"#{rn}{(' ' + tag) if tag else ''}")
        ax.fill_between(s[m], lat[m] - 2 * sdl[m], lat[m] + 2 * sdl[m],
                        color=colors[rn], alpha=0.10, lw=0)
    ax.axvline(1300, color="crimson", ls="--", lw=1.2)
    ax.text(1305, ax.get_ylim()[1] * 0.97, "RUN-IN >1300m\n(inflated y_lat uncertainty:\naerial foreshortens)",
            color="crimson", fontsize=8, va="top")
    # patrol metric-anchor window band.
    pat = [o for o in all_obs if o.feed == "patrol"]
    pa_lo, pa_hi = min(o.arclen for o in pat), max(o.arclen for o in pat)
    ax.axvspan(pa_lo, pa_hi, color="gold", alpha=0.12)
    ax.text((pa_lo + pa_hi) / 2, 0.5, "PATROL metric anchor\n(low-noise, absolute m)",
            color="darkgoldenrod", fontsize=8, ha="center")
    ax.set_xlabel("arclen (m, distance run)")
    ax.set_ylabel("y_lat (m from rail)")
    ax.set_title("FUSION RECON — lateral/control: y_lat vs arclen (±2σ; bands widen on the run-in)")
    ax.legend(ncol=6, fontsize=7, loc="upper right")

    # ---- feed-coverage annotation across panel 1 (which feed covers which race_t band) ----
    ann = (f"FEED COVERAGE (race_t):  "
           f"aerial {feed_cov['aerial'][0]:.0f}-{feed_cov['aerial'][1]:.0f}s [full-race, odometer] | "
           f"drone {feed_cov['drone'][0]:.0f}-{feed_cov['drone'][1]:.0f}s [3 windows] | "
           f"patrol {feed_cov['patrol'][0]:.0f}-{feed_cov['patrol'][1]:.0f}s [METRIC ANCHOR, head-on]")
    fig.text(0.5, 0.005, ann, ha="center", fontsize=9, color="0.2")

    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(FIG_RECON, dpi=110)
    print(f"\n[fig] {FIG_RECON}")


if __name__ == "__main__":
    run()
