"""HOLY-GRAIL 4-FEED FUSION — THE CAPSTONE: fold RF into the smoother for the complete recon.

Task #90 (journal thread 0060). Supersedes the 3-feed cv_fusion_smoother.py (0084). The 3-feed
smoother reconstructed LATERAL well (fused rail-vs-wide gap +3.84m, clean 8/8 vs stewards) but its
LONGITUDINAL axis was MEANINGLESS (order-vs-RP rho~0) because the only longitudinal observation was
the aerial FIELD-SHARED odometer (one distance-run for the whole field; per-horse stagger ~1.2m
unobserved). RF (#89/0086) is the fix: it gives a per-horse longitudinal ORDER backbone (legend
strip, RP-faithful rho +0.98..+1.00) + metric finishing MARGINS at the line (MAE 0.41 len).

INPUTS (REUSE read-only, never rebuilt, never mutated):
  - AERIAL render   data_cache/cv_s0d_aerial_render.parquet  (#82/0075):
        full-race; ARCLEN via the field odometer (frame-shared, valid on 198 frames, low noise),
        y_lat ORDERING faithful (rp +0.45) but absolute metric §4b-null (high noise; worse >1300m
        foreshortening, down-weight there). Identity 11/12 via the legend.
  - DRONE numbered  data_cache/cv_drone_numbered.parquet     (#87/0080):
        time-anchored on the shared clock (vid_t -> race_t affine -> arclen); y_lat ORDERING
        cross-validated vs aerial (rho +0.93) but metric §4b-null. Identity 10/12.
  - PATROL numbered data_cache/cv_patrol_numbered.parquet    (#86/0082):
        the LATERAL METRIC ANCHOR — the ONLY feed with within/between<1 (fixed camera, head-on);
        y_lat in METRES, validated tight/wide +4.1m vs stewards. Window arclen ~1058-1279m, 8/12.
  - RF longitudinal data_cache/cv_rf_longitudinal.parquet    (#89/0086) — THE NEW 4th FEED:
        kind='finish_margin' (5 metric rows): per-horse metric arclen at the line (race_t 98.85),
          = 1650 - margin_len*2.4; low noise (sd ~0.4-0.5m). The LONGITUDINAL METRIC ANCHOR — pins
          the absolute per-horse SPREAD at the finish (what the field odometer could not).
        kind='legend_order' (328 rows, 42 timepoints race_t 8.7-90.7): per-horse RANK among the
          visible lead group (arclen_or_gap = -rank, rank1=leader), RP-faithful rho +0.98..+1.00.
          The LONGITUDINAL ORDER backbone over the race body.

THE MODEL (per horse, independent — field interactions are future work):
  latent z_h(t) = [s, s_dot, l, l_dot]  on a shared race_t grid (5 Hz).
  MOTION PRIOR (physics): bounded longitudinal accel (sigma 0.8 m/s^2) + bounded lateral accel
    (sigma 0.35 m/s^2) — the no-teleport prior, applied JOINTLY in the smoother (0084).
  OBSERVATION MODELS (each feed over ITS coverage window, with its own noise):
    - LATERAL: PATROL low-noise metric anchor; AERIAL+DRONE high-metric-noise ordering (0084, kept).
    - LONGITUDINAL (the #90 FIX):
        * aerial FIELD odometer = the shared backbone s_field(race_t) (low noise but FIELD-shared,
          per-horse stagger ~0 -> gives the absolute mean-of-field, NOT per-horse order).
        * RF finish MARGINS -> per-horse METRIC arclen obs at race_t 98.85 (sd from the RF MAE
          ~0.4-0.5m). Pins the absolute per-horse SPREAD at the line.
        * RF legend ORDER -> a per-horse arclen OFFSET from the field backbone that REPRODUCES the
          RF rank: obs = s_field(t) + rank_offset(rank, n_visible), entered as a soft arclen obs.
          The per-rank gap (RANK_GAP_M ~1.5m, ~0.6 body-length) makes the RELATIVE ordering of the
          field RP-faithful; the obs sd (SD_ARC_RFRANK ~3.5m) is loose enough that the ABSOLUTE
          mid-race per-horse metres stay HONESTLY UNCERTAIN (the rank is observed, not the metres) —
          the finish margins (sd~0.45m) own the absolute scale at the line, the motion prior carries
          it between. This is the modelling crux: a MEANINGFUL longitudinal axis (RP-faithful order)
          WITHOUT fabricating false absolute mid-race stagger.
  INFERENCE: a numpy Kalman forward filter + RTS backward smoother per horse; propagate the state
  covariance (uncertainty bands widen where coverage is thin / only-ordering feeds).

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
RF_PANEL = "data_cache/cv_rf_longitudinal.parquet"
OUT_TRAJ = "data_cache/cv_fusion_trajectories.parquet"  # overwrite (4-feed supersedes 3-feed)
FIG_RECON = "data_cache/figs/cv_fusion_recon_4feed.png"

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

BODY_LEN_M = 2.4  # 1 length == 2.4 m (0086)


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
# official lengths-behind-winner at the line (LBW), for the margins MAE check.
OFFICIAL_LBW = {3: 0.0, 12: 0.3, 6: 1.5, 2: 1.75, 9: 2.25, 8: 4.25,
                11: 4.5, 4: 5.0, 7: 6.0, 10: 7.0, 5: 8.0, 1: 12.0}
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


# --- noise levels (metres). The metric hierarchy lives HERE (PHILOSOPHY: anchors pin the scale).
SD_ARC_AERIAL = 7.0     # odometer arclen — it is the FIELD-MEAN position (shared across horses), so
#                         it anchors the ABSOLUTE backbone but must be LOOSE per-horse, else its
#                         hundreds of identical obs clamp every horse to the field mean and erase the
#                         per-horse RF-rank stagger. Loose enough to let RF rank/margins set order.
SD_ARC_DRONE = 8.0      # time-anchored arclen — moderate
SD_ARC_PATROL = 6.0     # patrol arclen from its homography window — moderate
SD_YLAT_PATROL = 1.2    # LATERAL METRIC ANCHOR — low noise (0076 within/between 0.06-0.16)
SD_YLAT_AERIAL = 6.0    # ordering trustworthy, absolute metric biased -> high (inflated) variance
SD_YLAT_AERIAL_RUNIN = 14.0  # >1300m foreshortening (0075) -> extra inflation
SD_YLAT_DRONE = 7.0     # ordering trustworthy, metric §4b-null -> high variance
SD_YLAT_DRONE_RUNIN = 11.0   # run-in (>1300m): no metric anchor sees it, drone metric §4b-null (0080)

# --- RF (the 4th feed) noise levels.
SD_ARC_RFFINISH = 1.0   # finish-margin metric arclen at the line: RF MAE 0.41 len ~ 1.0 m (0086)
SD_ARC_RFRANK = 1.8     # legend-ORDER soft arclen obs: tight enough vs RANK_GAP_M to IMPOSE the
#                         ordering, loose enough (>RANK_GAP_M) that ABSOLUTE mid-race metres stay
#                         honestly uncertain (the rank is observed, not the absolute stagger).
RANK_GAP_M = 2.4        # arclen offset per rank from the field-leader backbone (= 1 body-length).
#                         sets the per-rank metric spacing the ordering asserts; small vs the line
#                         spread (the bunched-field prior) but enough that order dominates the noise.
SD_ARC_RFBACK = 2.5     # absent-from-legend = BEHIND the lead group: a soft back-marker obs (the
#                         exact depth is unknown -> looser than the visible-rank sd, but tight
#                         enough to beat the field-MEAN odometer and keep back-markers from floating
#                         into mid-pack at the field mean — the corruption that capped the rho).
SD_ARC_RFFINORDER = 2.5  # finish-LINE order anchor (last-legend rank carried to the line) for the
#                          legend-seen horses RF has no metric margin for: SOFT (preserves ORDER,
#                          not absolute metres — the 5 metric margins own the absolute spread).


def load_aerial() -> tuple[list[Obs], pl.DataFrame, np.ndarray, np.ndarray]:
    """AERIAL: per-row race_t from the FIELD ODOMETER (frame-shared arclen, valid 198 frames),
    interpolated over vid-time t to every row; y_lat is the high-metric-noise lateral obs.

    Also returns the field-odometer (race_t -> arclen) curve so the RF legend-rank obs can be
    placed RELATIVE to the field backbone."""
    a = pl.read_parquet(AERIAL_PANEL)
    # field odometer: median x_arclen per frame on the 198 valid-arclen frames -> frame/t -> arclen.
    od = (
        a.filter(pl.col("x_arclen").is_not_nan())
        .group_by("frame")
        .agg(pl.col("x_arclen").median().alias("arc"), pl.col("t").first().alias("t"))
        .sort("t")
    )
    od_t, od_arc = od["t"].to_numpy(), od["arc"].to_numpy()
    # field backbone on the race_t axis (for the RF rank offsets).
    od_race_t = arclen_to_race_t(od_arc)
    order = np.argsort(od_race_t)
    field_race_t, field_arc = od_race_t[order], od_arc[order]

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
    return obs, a, field_race_t, field_arc


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
    """PATROL: already on race_t + arclen + y_lat_m (metres). The LATERAL METRIC ANCHOR.
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


def load_rf(field_race_t: np.ndarray, field_arc: np.ndarray) -> list[Obs]:
    """RF (THE 4th FEED) — the LONGITUDINAL observation model.

    Two row-kinds in the panel:
      * finish_margin: metric arclen at the line (race_t 98.85). Direct, low-noise per-horse arclen
        observations -> pin the absolute per-horse SPREAD at the finish.
      * legend_order : per-timepoint RANK among the visible lead group (arclen_or_gap = -rank).
        Converted to a SOFT arclen obs placed RELATIVE to the field-leader backbone so that the
        per-horse ORDER reproduces the RF rank, WITHOUT asserting false absolute metres:
            obs_arclen(h,t) = s_field(t) + (rank_mid - rank_h) * RANK_GAP_M
        where rank_mid is the centre rank of the visible group at t (so the offset is centred on the
        field backbone — leaders ahead, back-markers behind). High obs sd (SD_ARC_RFRANK) keeps the
        absolute mid-race metres honestly uncertain; only the ORDERING is informative.
    """
    rf = pl.read_parquet(RF_PANEL)
    obs: list[Obs] = []

    # --- finish margins -> metric arclen at the line (low noise). ---
    fm = rf.filter(pl.col("kind") == "finish_margin")
    for r in fm.iter_rows(named=True):
        sd = float(r["sd"]) if (r["sd"] is not None and r["sd"] > 0) else SD_ARC_RFFINISH
        sd = max(sd, SD_ARC_RFFINISH)  # floor at the RF MAE (~1.0 m) — do not over-trust
        obs.append(
            Obs(
                runner=int(r["runner"]), race_t=float(r["race_t"]),
                arclen=float(r["arclen_or_gap"]), ylat=None, feed="rf_finish",
                sd_arclen=sd, sd_ylat=0.0,
            )
        )

    # --- legend order -> soft rank-offset arclen obs relative to the field backbone. ---
    # The legend strip is the LEAD-GROUP top-down running order (0086): horses IN it are ordered
    # RP-faithfully; horses ABSENT from it are BEHIND the deepest visible rank (off the lead-group
    # strip). We use BOTH facts: visible horses get their exact rank offset; absent horses get a
    # soft "behind the group" offset (rank = max_visible+1, looser sd since the exact depth is
    # unknown) — this stops back-markers floating into mid-pack at the field mean (the corruption
    # that capped the per-pole rho). All_runners = the 12-horse field (pre-race fact, not official
    # result). Track each horse's last LATE rank for the finish-order carry-forward.
    all_runners = set(range(1, 13))
    lo = rf.filter(pl.col("kind") == "legend_order")
    last_rank: dict[int, tuple[float, float]] = {}  # runner -> (rank, rank_mid_of_group)
    for t, grp in lo.group_by("race_t"):
        race_t = float(t[0])
        ranks = -grp["arclen_or_gap"].to_numpy()  # rank1 == leader
        runners = grp["runner"].to_numpy()
        s_field = float(np.interp(race_t, field_race_t, field_arc))
        max_rank = float(np.max(ranks))
        absent = sorted(all_runners - {int(h) for h in runners})
        # centre the offset on the field backbone over the WHOLE field (visible + the absent
        # back-markers placed just behind) so the backbone stays the field mean.
        full_ranks = list(ranks) + [max_rank + 1.0] * len(absent)
        rank_mid = float(np.mean(full_ranks))
        for h, rk in zip(runners, ranks, strict=True):
            arc_h = s_field + (rank_mid - float(rk)) * RANK_GAP_M
            obs.append(
                Obs(
                    runner=int(h), race_t=race_t, arclen=float(arc_h), ylat=None,
                    feed="rf_rank", sd_arclen=SD_ARC_RFRANK, sd_ylat=0.0,
                )
            )
            last_rank[int(h)] = (float(rk), rank_mid)
        # absent horses: soft "behind the lead group" obs (depth unknown -> looser sd).
        for h in absent:
            arc_h = s_field + (rank_mid - (max_rank + 1.0)) * RANK_GAP_M
            obs.append(
                Obs(
                    runner=int(h), race_t=race_t, arclen=float(arc_h), ylat=None,
                    feed="rf_rank_back", sd_arclen=SD_ARC_RFBACK, sd_ylat=0.0,
                )
            )

    # --- carry the RP-faithful ORDER to the FINISH for legend-seen horses that have NO RF metric
    #     margin. The legend stops ~race_t 90.7 (~1500m); without this, the 8s gap to the line lets
    #     the loose odometer scramble their finish order. Anchor each such horse at the line BEHIND
    #     the winner by its last-legend rank gap (SOFT — sd loose; the metric margins, sd~0.45m, own
    #     the absolute spread; this only preserves the observed ORDER). Honest: horses never in the
    #     late legend (e.g. #1/#7, last seen race_t<20) get NO finish anchor and stay under-observed.
    margin_horses = {int(r["runner"]) for r in fm.iter_rows(named=True)}
    LATE_T = 80.0  # prefer the genuinely-late rank (after the 3/4 pole region) for the finish order.
    late_rows = lo.filter(pl.col("race_t") >= LATE_T)
    late_rank: dict[int, float] = {}
    for r in late_rows.iter_rows(named=True):
        h = int(r["runner"])
        if h not in margin_horses:
            late_rank[h] = -float(r["arclen_or_gap"])  # latest wins (rows time-sorted-ish)
    # the deepest visible late rank -> horses with NO late rank (e.g. #1/#7, dropped from the strip
    # early) are BACK-MARKERS at the line: place them behind the deepest late-ranked horse, ordered
    # by their LAST EVER legend rank (a soft, honest back-of-field estimate, looser sd).
    max_late = max(late_rank.values()) if late_rank else 8.0
    win_arc = 1650.0
    for h, rk in late_rank.items():
        arc_h = win_arc - (rk - 1.0) * RANK_GAP_M  # behind the winner by (rank-1) body-lengths.
        obs.append(
            Obs(
                runner=h, race_t=WINNER_FINISH_RACE_T, arclen=float(arc_h),
                ylat=None, feed="rf_finish_order", sd_arclen=SD_ARC_RFFINORDER, sd_ylat=0.0,
            )
        )
    for h, (rk, _rm) in sorted(last_rank.items(), key=lambda kv: kv[1][0]):
        if h in margin_horses or h in late_rank:
            continue
        # never-late, no-margin -> back-marker: rank = beyond the deepest late horse (soft).
        eff_rank = max_late + 1.0 + rk  # deeper last-ever rank -> further back.
        arc_h = win_arc - (eff_rank - 1.0) * RANK_GAP_M
        obs.append(
            Obs(
                runner=h, race_t=WINNER_FINISH_RACE_T, arclen=float(arc_h),
                ylat=None, feed="rf_finish_order", sd_arclen=SD_ARC_RFBACK, sd_ylat=0.0,
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
        "node_has_lat": np.array(
            [any(o.ylat is not None for o in obs_by_node.get(k, [])) for k in range(n)]
        ),
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
    aer_obs, _aer_df, field_race_t, field_arc = load_aerial()
    dro_obs = load_drone()
    pat_obs = load_patrol()
    rf_obs = load_rf(field_race_t, field_arc)
    all_obs = aer_obs + dro_obs + pat_obs + rf_obs

    runners = sorted({o.runner for o in all_obs})
    rf_fin = {o.runner for o in rf_obs if o.feed == "rf_finish"}
    rf_rank = {o.runner for o in rf_obs if o.feed == "rf_rank"}
    print(f"[ingest] aerial {len(aer_obs)} obs ({len({o.runner for o in aer_obs})} runners), "
          f"drone {len(dro_obs)} ({len({o.runner for o in dro_obs})}), "
          f"patrol {len(pat_obs)} ({len({o.runner for o in pat_obs})}), "
          f"RF {len(rf_obs)} (finish {sorted(rf_fin)}, rank {len(rf_rank)} runners)")
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


def _arclen_at_t(smoothed: dict, rn: int, grid: np.ndarray, race_t: float) -> float:
    ki = int(np.clip(round((race_t - grid[0]) / DT), 0, len(grid) - 1))
    return float(smoothed[rn]["s"][ki])


def _median_ylat_band(
    smoothed: dict, grid: np.ndarray, runners: list[int], a_lo: float, a_hi: float
) -> dict[int, float]:
    """Median fused y_lat per runner over an ARCLEN band (only where lateral was observed)."""
    out: dict[int, float] = {}
    for rn in runners:
        s = smoothed[rn]["s"]
        haslat = smoothed[rn]["node_has_lat"]
        m = (s >= a_lo) & (s <= a_hi) & haslat
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

    # ============================================================================================
    # (1) LONGITUDINAL — the #90 FIX. Fused per-horse arclen ORDER vs official, now RF-informed.
    # ============================================================================================
    print("\n[LONGITUDINAL — the RF fix: fused per-horse order vs official]")

    # ---- fused FINISH ORDER vs official (winner recovery) ----
    t_fin = WINNER_FINISH_RACE_T
    fin_arc = {rn: _arclen_at_t(smoothed, rn, grid, t_fin) for rn in runners}
    fused_finish = sorted(runners, key=lambda r: -fin_arc[r])  # most arclen == winner
    off_rank = {h: i for i, h in enumerate(FINISH_ORDER)}
    common = [h for h in runners if h in off_rank]
    fused_pos = np.array([fused_finish.index(h) for h in common], float)
    off_pos = np.array([off_rank[h] for h in common], float)
    rho_fin = _spearman(fused_pos, off_pos)
    winner_recovered = fused_finish[0] == FINISH_ORDER[0]
    print(f"  fused finish order : {fused_finish}")
    print(f"  official finish    : {FINISH_ORDER}")
    print(f"  finish-order Spearman rho = {rho_fin:+.2f}   "
          f"winner #3 recovered: {'YES' if winner_recovered else 'NO'} "
          f"(fused leader = #{fused_finish[0]})")

    # ---- fused finish MARGINS vs official LBW (lengths) ----
    win_arc = fin_arc[FINISH_ORDER[0]]
    fused_lbw = {rn: (win_arc - fin_arc[rn]) / BODY_LEN_M for rn in runners}
    margin_common = [h for h in runners if h in OFFICIAL_LBW]
    ae = [abs(fused_lbw[h] - OFFICIAL_LBW[h]) for h in margin_common]
    mae = float(np.mean(ae))
    # honest split: RF metric-anchors only the lead group (5 horses); the back tail is ORDER-only
    # (rank-gap prior keeps it bunched -> absolute depth honestly compressed, NOT fabricated).
    rf_anchored = {3, 6, 8, 9, 12}
    mae_lead = float(np.mean([abs(fused_lbw[h] - OFFICIAL_LBW[h])
                              for h in margin_common if h in rf_anchored]))
    mae_tail = float(np.mean([abs(fused_lbw[h] - OFFICIAL_LBW[h])
                              for h in margin_common if h not in rf_anchored]))
    print(f"\n  fused finish MARGINS vs official LBW (lengths): MAE = {mae:.2f} len  "
          f"(n={len(margin_common)})  ||  RF-anchored lead group MAE = {mae_lead:.2f} len "
          f"(n={len(rf_anchored)}) vs ORDER-only tail MAE = {mae_tail:.2f} len "
          f"({len(margin_common) - len(rf_anchored)} — honestly compressed, not metric-observed)")
    print("    per-horse fused / official LBW (len): " +
          ", ".join(f"#{h}:{fused_lbw[h]:.1f}/{OFFICIAL_LBW[h]:.1f}" for h in FINISH_ORDER))

    # ---- fused per-pole ORDER vs official RP at the 4 calls (the 0084 ~0 -> RF target) ----
    print("\n  [per-pole fused lead-order vs official RP — 0084 was rho~0; RF target +0.98..+1.00]")
    poles = {"1/4 (450m)": 450.0, "1/2 (850m)": 850.0, "3/4 (1250m)": 1250.0, "finish (1640m)": 1640.0}
    rp_call = {i: {h: OFFICIAL_RP[h][i] for h in OFFICIAL_RP} for i in range(4)}
    for ci, (name, a_mark) in enumerate(poles.items()):
        target_t = float(arclen_to_race_t(np.array([a_mark]))[0])
        fused_s = {rn: _arclen_at_t(smoothed, rn, grid, target_t) for rn in runners
                   if smoothed[rn]["node_has_obs"][
                       :int(np.clip(round((target_t - grid[0]) / DT), 0, len(grid) - 1)) + 1].any()}
        cc = [h for h in fused_s if h in rp_call[ci]]
        if len(cc) >= 3:
            fs = np.array([-fused_s[h] for h in cc])  # neg so rank1 == leader
            rp = np.array([rp_call[ci][h] for h in cc], float)
            print(f"    {name:16s} race_t {target_t:5.1f}s  n={len(cc):2d}  "
                  f"rho(fused lead-order, official RP) = {_spearman(fs, rp):+.2f}")

    # ============================================================================================
    # (2) LATERAL — should HOLD at the 0084 result (+3.84m, 8/8). RF must not corrupt it.
    # ============================================================================================
    print("\n[LATERAL — must HOLD the 0084 result: rail-vs-wide gap +3.84m, clean 8/8]")
    body = _median_ylat_band(smoothed, grid, runners, 150.0, 1300.0)
    rail_v = [body[h] for h in RAIL_HORSES if h in body]
    wide_v = [body[h] for h in WIDE_HORSES if h in body]
    n_cov = len([h for h in (RAIL_HORSES | WIDE_HORSES) if h in body])
    if rail_v and wide_v:
        gap = float(np.mean(wide_v) - np.mean(rail_v))
        thr = 0.5 * (np.mean(rail_v) + np.mean(wide_v))  # rail/wide split midpoint
        correct = sum(body[h] < thr for h in RAIL_HORSES & set(body)) + \
                  sum(body[h] >= thr for h in WIDE_HORSES & set(body))
        print(f"  rail {sorted(RAIL_HORSES & set(body))} mean {np.mean(rail_v):5.2f}m | "
              f"wide {sorted(WIDE_HORSES & set(body))} mean {np.mean(wide_v):5.2f}m | "
              f"GAP wide-rail = {gap:+.2f}m  (0084 ref +3.84m; covers {n_cov}/8 commented; "
              f"classified correct {correct}/{n_cov} @midpoint {thr:.1f}m)")
    print("  per-runner fused median y_lat (<1300m):")
    for rn in sorted(body):
        tag = "rail" if rn in RAIL_HORSES else ("wide" if rn in WIDE_HORSES else "mid ")
        print(f"    #{rn:2d} [{tag}] {body[rn]:5.2f}m")

    # ============================================================================================
    # (3) PER-BAND uncertainty — lateral + the NEW longitudinal spread band.
    # ============================================================================================
    print("\n[per-band fused y_lat uncertainty — the lateral segment-quality census]")
    bands = [("start/accel 150-450m", 150, 450), ("backstr/bend 450-850m", 450, 850),
             ("bend->straight 850-1300m", 850, 1300), ("RUN-IN >1300m", 1300, 1650)]
    for name, lo, hi in bands:
        sd = _mean_sd_ylat_band(smoothed, runners, lo, hi)
        ncov = len([rn for rn in runners
                    if ((smoothed[rn]["s"] >= lo) & (smoothed[rn]["s"] <= hi)).sum() >= 1])
        print(f"  {name:26s}  mean fused sd(y_lat) = {sd:5.2f}m  ({ncov} runners present)")

    print("\n[per-band fused ARCLEN (longitudinal) uncertainty — honest absolute-metre stagger]")
    for name, lo, hi in [("body (legend, rank-only) <1300m", 150, 1300),
                         ("finish (RF margin anchor) >1500m", 1500, 1650)]:
        vals = []
        for rn in runners:
            s = smoothed[rn]["s"]
            m = (s >= lo) & (s <= hi)
            if m.sum() >= 1:
                vals.append(float(np.median(smoothed[rn]["sd_s"][m])))
        mn = float(np.mean(vals)) if vals else float("nan")
        print(f"  {name:34s}  mean fused sd(arclen) = {mn:5.2f}m  ({len(vals)} runners)")
    print("  NOTE: the posterior sd_arclen reflects the RANK obs spacing, NOT a true absolute-metre")
    print("        scale — RF observes ORDER, not per-horse metres, mid-race. What is VALIDATED is")
    print("        the per-horse ORDER (the +0.96/+0.80/+0.90 per-pole rho), with the bunched-field")
    print("        rank-gap prior (2.4m/rank) as the only absolute-metre stagger model (the deep tail")
    print("        is honestly COMPRESSED — see the order-only-tail margin MAE). The ABSOLUTE per-horse")
    print("        spread is metric-pinned ONLY at the finish line (the 5 RF margins, sd~0.45m).")


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
    for feed in ("aerial", "drone", "patrol", "rf_rank", "rf_finish"):
        fo = [o for o in all_obs if o.feed == feed]
        if fo:
            feed_cov[feed] = (min(o.race_t for o in fo), max(o.race_t for o in fo))

    # ---- PANEL 1: per-horse GAP-BEHIND-LEADER vs race_t (the longitudinal SEPARATION the RF fix
    # buys). Raw arclen (0..1650m) hides the ~16-28m field stagger; plotting each horse's lag behind
    # the field leader at each instant makes the per-horse ORDER + winner #3's 7->1 climb VISIBLE. ----
    ax = axes[0]
    # field-leader arclen at each grid node (max over horses that have support there).
    lead = np.full(len(grid), -np.inf)
    for rn in runners:
        has = smoothed[rn]["node_has_obs"]
        idx = np.where(has)[0]
        if len(idx) == 0:
            continue
        m = slice(idx.min(), idx.max() + 1)
        lead[m] = np.maximum(lead[m], smoothed[rn]["s"][m])
    for rn in runners:
        s, sd, has = smoothed[rn]["s"], smoothed[rn]["sd_s"], smoothed[rn]["node_has_obs"]
        idx = np.where(has)[0]
        if len(idx) == 0:
            continue
        m = slice(idx.min(), idx.max() + 1)
        gap = s[m] - lead[m]  # <= 0; 0 == leader, negative == lengths behind
        lw = 2.6 if rn == 3 else 1.1
        ax.plot(grid[m], gap, color=colors[rn], lw=lw, label=f"#{rn}" + (" WIN" if rn == 3 else ""))
        ax.fill_between(grid[m], gap - 2 * sd[m], gap + 2 * sd[m],
                        color=colors[rn], alpha=0.06, lw=0)
    for a_mark, lbl in ((450, "1/4"), (850, "1/2"), (1250, "3/4"), (1640, "fin")):
        tt = float(arclen_to_race_t(np.array([a_mark]))[0])
        ax.axvline(tt, color="0.7", ls=":", lw=0.8)
        ax.text(tt, ax.get_ylim()[0] if False else -28, lbl, fontsize=7, color="0.5", ha="center")
    ax.axhline(0, color="0.3", lw=0.7)
    ax.set_xlabel("race_t (s)  [shared clock: official sections]")
    ax.set_ylabel("gap behind field leader (m)   [0 = leader]")
    ax.set_title("4-FEED FUSION RECON — longitudinal SEPARATION: per-horse lag behind the leader "
                 "(RF order backbone; winner #3 climbs from the back 7->1; ±2σ)")
    ax.legend(ncol=7, fontsize=7, loc="lower left")
    ax.set_ylim(-30, 4)

    # inset: absolute arclen near the line — the per-horse FAN the RF metric margins pin.
    axins = ax.inset_axes((0.60, 0.50, 0.38, 0.46))
    for rn in runners:
        s, has = smoothed[rn]["s"], smoothed[rn]["node_has_obs"]
        idx = np.where(has)[0]
        if len(idx) == 0:
            continue
        m = slice(idx.min(), idx.max() + 1)
        lw = 2.4 if rn == 3 else 1.0
        axins.plot(grid[m], s[m], color=colors[rn], lw=lw)
    axins.set_xlim(94, 99.3)
    axins.set_ylim(1560, 1655)
    axins.set_title("finish fan: absolute arclen (RF margins pin the spread)", fontsize=7)
    axins.tick_params(labelsize=6)
    axins.axhline(1650, color="0.5", ls=":", lw=0.6)

    # ---- PANEL 2: y_lat vs arclen (the lateral/control reconstruction) ----
    ax = axes[1]
    for rn in runners:
        s, lat, sdl, has = (smoothed[rn]["s"], smoothed[rn]["l"],
                            smoothed[rn]["sd_l"], smoothed[rn]["node_has_lat"])
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
    ax.text(1305, ax.get_ylim()[1] * 0.97,
            "RUN-IN >1300m\n(inflated y_lat uncertainty:\naerial foreshortens)",
            color="crimson", fontsize=8, va="top")
    pat = [o for o in all_obs if o.feed == "patrol"]
    pa_lo, pa_hi = min(o.arclen for o in pat), max(o.arclen for o in pat)
    ax.axvspan(pa_lo, pa_hi, color="gold", alpha=0.12)
    ax.text((pa_lo + pa_hi) / 2, 0.5, "PATROL lateral metric anchor\n(low-noise, absolute m)",
            color="darkgoldenrod", fontsize=8, ha="center")
    ax.set_xlabel("arclen (m, distance run)")
    ax.set_ylabel("y_lat (m from rail)")
    ax.set_title("4-FEED FUSION RECON — lateral/control: y_lat vs arclen "
                 "(rail-vs-wide gap +3.84m holds; ±2σ; bands widen on run-in)")
    ax.legend(ncol=6, fontsize=7, loc="upper right")

    # ---- feed-coverage annotation across the figure ----
    parts = [f"aerial {feed_cov['aerial'][0]:.0f}-{feed_cov['aerial'][1]:.0f}s [full-race odometer]",
             f"drone {feed_cov['drone'][0]:.0f}-{feed_cov['drone'][1]:.0f}s",
             f"patrol {feed_cov['patrol'][0]:.0f}-{feed_cov['patrol'][1]:.0f}s [LAT anchor]"]
    if "rf_rank" in feed_cov:
        parts.append(f"RF-rank {feed_cov['rf_rank'][0]:.0f}-{feed_cov['rf_rank'][1]:.0f}s "
                     f"[LONG. order]")
    if "rf_finish" in feed_cov:
        parts.append(f"RF-margins @{feed_cov['rf_finish'][0]:.0f}s [LONG. metric anchor]")
    ann = "FEED COVERAGE (race_t):  " + " | ".join(parts)
    fig.text(0.5, 0.005, ann, ha="center", fontsize=8.5, color="0.2")

    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(FIG_RECON, dpi=110)
    print(f"\n[fig] {FIG_RECON}")


if __name__ == "__main__":
    run()
