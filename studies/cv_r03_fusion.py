"""R03 4-FEED FUSION — GENERALIZE cv_fusion_4feed (#90/0087) to HV R03 1200m (#91).

REUSE strategy: import the R1 fusion SMOOTHER unchanged (Obs / smooth_horse / _kf_update /
_process_block / _spearman / the RTS machinery + noise levels) and re-point the R1 module
clock/geometry/validation tables at R03:
  * DIST 1650->1200, sections 4->3, _D_BOUNDS/_T_CUM re-derived, winner-t 98.85->69.29.
  * the 4 input parquets -> the R03-suffixed ones.
  * the drone clock anchor (re-derived).
  * the official RP / FINISH_ORDER / LBW / rail-wide tables -> from the R03 official parse.
  * RF winner reference #3 -> #7 (the load_rf finish-order carry hard-coded #3).

Writes data_cache/cv_fusion_trajectories_R03.parquet + fig cv_fusion_recon_R03.png.
Official data VALIDATION-only.

    PYTHONHASHSEED=0 uv run --group bayes python studies/cv_r03_fusion.py
"""
from __future__ import annotations

import sys

import numpy as np
import polars as pl

sys.path.insert(0, "studies")
import cv_fusion_4feed as F1  # noqa: E402,N812  (R1 fusion smoother, reused)
from cv_fusion_4feed import (  # noqa: E402
    DT,
    GRID_HZ,
    Obs,
    coverage_window,
    smooth_horse,
)
from cv_r03_common import (  # noqa: E402
    DIST,
    OFFICIAL_SECTIONS,
    TRACK_WIDTH_M,
    WINNER_FINISH_RACE_T,
    lateral_truth,
    parse_official,
)

OUT_TRAJ = "data_cache/cv_fusion_trajectories_R03.parquet"
FIG_RECON = "data_cache/figs/cv_fusion_recon_R03.png"

AERIAL_PANEL = "data_cache/cv_r03_aerial_render.parquet"
DRONE_PANEL = "data_cache/cv_r03_drone_numbered.parquet"
PATROL_PANEL = "data_cache/cv_r03_patrol_numbered.parquet"
RF_PANEL = "data_cache/cv_r03_rf_longitudinal.parquet"

# [GENERALIZE] R03 clock/geometry (sections 4->3; 1200m calls at 0/400/800/1200).
_D_BOUNDS = np.array([0.0, 400.0, 800.0, 1200.0])
_T_CUM = np.concatenate([[0.0], np.cumsum(OFFICIAL_SECTIONS)])  # 0,23.54,46.19,69.29
DRONE_START_VID_T, DRONE_FINISH_VID_T = 3.0, 73.0
# [GENERALIZE] run-in foreshorten threshold: R1 used 1300m on 1650m (~79%); 79% of 1200m ~ 950m.
RUNIN_THRESH = 950.0


def arclen_to_race_t(a):
    return np.interp(np.asarray(a, float), _D_BOUNDS, _T_CUM)


def race_t_to_arclen(t):
    return np.interp(np.asarray(t, float), _T_CUM, _D_BOUNDS)


def _setup_r03(off) -> None:
    """[GENERALIZE] re-point all R1 fusion-module constants/curves at R03."""
    F1.DIST = DIST
    F1.TRACK_WIDTH = TRACK_WIDTH_M
    F1.OFFICIAL_SECTIONS = OFFICIAL_SECTIONS
    F1.WINNER_FINISH_RACE_T = WINNER_FINISH_RACE_T
    F1._D_BOUNDS = _D_BOUNDS
    F1._T_CUM = _T_CUM
    F1.arclen_to_race_t = arclen_to_race_t
    F1.race_t_to_arclen = race_t_to_arclen
    F1.AERIAL_PANEL = AERIAL_PANEL
    F1.DRONE_PANEL = DRONE_PANEL
    F1.PATROL_PANEL = PATROL_PANEL
    F1.RF_PANEL = RF_PANEL
    F1.DRONE_T0 = DRONE_START_VID_T
    F1.DRONE_RATE = WINNER_FINISH_RACE_T / (DRONE_FINISH_VID_T - DRONE_START_VID_T)
    F1.OFFICIAL_RP = dict(off.rp)
    F1.FINISH_ORDER = list(off.finish_order)
    F1.OFFICIAL_LBW = dict(off.lbw)
    rail_t, wide_t = lateral_truth(off)
    F1.RAIL_HORSES = rail_t
    F1.WIDE_HORSES = wide_t


def load_drone_r03() -> list[Obs]:
    """DRONE: race_t already in the R03 panel (cv_r03_drone wrote race_t+arclen+y_lat)."""
    try:
        d = pl.read_parquet(DRONE_PANEL)
    except Exception:
        return []
    d = d.filter(pl.col("number") > 0)
    if d.height == 0:
        return []
    obs = []
    for r in d.iter_rows(named=True):
        runin = r["arclen"] > RUNIN_THRESH
        obs.append(Obs(runner=int(r["number"]), race_t=float(r["race_t"]),
                       arclen=float(r["arclen"]), ylat=float(r["y_lat"]), feed="drone",
                       sd_arclen=F1.SD_ARC_DRONE,
                       sd_ylat=F1.SD_YLAT_DRONE_RUNIN if runin else F1.SD_YLAT_DRONE))
    return obs


def load_patrol_r03() -> list[Obs]:
    try:
        p = pl.read_parquet(PATROL_PANEL)
    except Exception:
        return []
    obs = []
    for r in p.iter_rows(named=True):
        c = float(np.clip(r.get("confidence", 1.0), 0.25, 1.0))
        obs.append(Obs(runner=int(r["runner"]), race_t=float(r["race_t"]),
                       arclen=float(r["arclen"]), ylat=float(r["y_lat_m"]), feed="patrol",
                       sd_arclen=F1.SD_ARC_PATROL, sd_ylat=F1.SD_YLAT_PATROL / c))
    return obs


def run():
    off = parse_official()
    _setup_r03(off)
    print("=" * 80)
    print("R03 4-FEED FUSION #91 — RTS smoother GENERALIZED to HV 1200m")
    print("=" * 80)
    print(f"winner #{off.finish_order[0]} {off.horse[off.finish_order[0]]} | "
          f"sections {OFFICIAL_SECTIONS} | winner_t {WINNER_FINISH_RACE_T:.2f}s")

    aer_obs, _aer_df, field_race_t, field_arc = F1.load_aerial()
    dro_obs = load_drone_r03()
    pat_obs = load_patrol_r03()
    F1.DIST = DIST
    rf_obs = _load_rf_r03(off, field_race_t, field_arc)

    all_obs = aer_obs + dro_obs + pat_obs + rf_obs
    runners = sorted({o.runner for o in all_obs})
    print(f"[ingest] aerial {len(aer_obs)} ({len({o.runner for o in aer_obs})}r) | "
          f"drone {len(dro_obs)} ({len({o.runner for o in dro_obs})}r) | "
          f"patrol {len(pat_obs)} ({len({o.runner for o in pat_obs})}r) | "
          f"RF {len(rf_obs)} | union {len(runners)} runners")

    t_lo = min(o.race_t for o in all_obs)
    t_hi = max(o.race_t for o in all_obs)
    grid = np.arange(t_lo, t_hi + DT, DT)
    print(f"[grid] race_t {t_lo:.2f}..{t_hi:.2f}s @ {GRID_HZ:.0f}Hz, {len(grid)} nodes")

    rows, smoothed = [], {}
    for rn in runners:
        ho = [o for o in all_obs if o.runner == rn]
        clo, chi = coverage_window(ho)
        res = smooth_horse(grid, ho)
        smoothed[rn] = res
        mask = (grid >= clo - 1.0) & (grid <= chi + 1.0)
        for i in np.where(mask)[0]:
            rows.append(dict(runner=rn, race_t=float(grid[i]), arclen=float(res["s"][i]),
                             y_lat_m=float(res["l"][i]), sd_arclen=float(res["sd_s"][i]),
                             sd_ylat=float(res["sd_l"][i])))
    traj = pl.DataFrame(rows)
    traj.write_parquet(OUT_TRAJ)
    print(f"[panel] {OUT_TRAJ} ({traj.height} rows, {traj['runner'].n_unique()} runners)")

    validate(smoothed, grid, runners, off)
    make_figure(smoothed, grid, runners, all_obs, off)


def _load_rf_r03(off, field_race_t, field_arc):
    """R03 RF obs. Reuse R1 load_rf's structure but with win_arc=1200 + winner-relative carry."""
    rf = pl.read_parquet(RF_PANEL)
    obs: list[Obs] = []
    fm = rf.filter(pl.col("kind") == "finish_margin")
    for r in fm.iter_rows(named=True):
        sd = max(float(r["sd"]) if r["sd"] and r["sd"] > 0 else F1.SD_ARC_RFFINISH,
                 F1.SD_ARC_RFFINISH)
        obs.append(Obs(runner=int(r["runner"]), race_t=float(r["race_t"]),
                       arclen=float(r["arclen_or_gap"]), ylat=None, feed="rf_finish",
                       sd_arclen=sd, sd_ylat=0.0))
    margin_horses = {int(r["runner"]) for r in fm.iter_rows(named=True)}

    lo = rf.filter(pl.col("kind") == "legend_order")
    all_runners = set(range(1, 13))
    last_rank: dict[int, tuple[float, float]] = {}
    for t, grp in lo.group_by("race_t"):
        race_t = float(t[0])
        ranks = -grp["arclen_or_gap"].to_numpy()
        rr = grp["runner"].to_numpy()
        s_field = float(np.interp(race_t, field_race_t, field_arc))
        max_rank = float(np.max(ranks))
        absent = sorted(all_runners - {int(h) for h in rr})
        full_ranks = list(ranks) + [max_rank + 1.0] * len(absent)
        rank_mid = float(np.mean(full_ranks))
        for h, rk in zip(rr, ranks, strict=True):
            obs.append(Obs(runner=int(h), race_t=race_t,
                           arclen=s_field + (rank_mid - float(rk)) * F1.RANK_GAP_M,
                           ylat=None, feed="rf_rank", sd_arclen=F1.SD_ARC_RFRANK, sd_ylat=0.0))
            last_rank[int(h)] = (float(rk), rank_mid)
        for h in absent:
            obs.append(Obs(runner=int(h), race_t=race_t,
                           arclen=s_field + (rank_mid - (max_rank + 1.0)) * F1.RANK_GAP_M,
                           ylat=None, feed="rf_rank_back", sd_arclen=F1.SD_ARC_RFBACK, sd_ylat=0.0))

    LATE_T = WINNER_FINISH_RACE_T * 0.80
    late_rows = lo.filter(pl.col("race_t") >= LATE_T)
    late_rank: dict[int, float] = {}
    for r in late_rows.iter_rows(named=True):
        h = int(r["runner"])
        if h not in margin_horses:
            late_rank[h] = -float(r["arclen_or_gap"])
    max_late = max(late_rank.values()) if late_rank else 8.0
    win_arc = DIST
    for h, rk in late_rank.items():
        obs.append(Obs(runner=h, race_t=WINNER_FINISH_RACE_T,
                       arclen=win_arc - (rk - 1.0) * F1.RANK_GAP_M, ylat=None,
                       feed="rf_finish_order", sd_arclen=F1.SD_ARC_RFFINORDER, sd_ylat=0.0))
    for h, (rk, _rm) in sorted(last_rank.items(), key=lambda kv: kv[1][0]):
        if h in margin_horses or h in late_rank:
            continue
        eff = max_late + 1.0 + rk
        obs.append(Obs(runner=h, race_t=WINNER_FINISH_RACE_T,
                       arclen=win_arc - (eff - 1.0) * F1.RANK_GAP_M, ylat=None,
                       feed="rf_finish_order", sd_arclen=F1.SD_ARC_RFBACK, sd_ylat=0.0))
    return obs


def _spearman(a, b):
    return F1._spearman(np.asarray(a, float), np.asarray(b, float))


def _arclen_at_t(smoothed, rn, grid, race_t):
    ki = int(np.clip(round((race_t - grid[0]) / DT), 0, len(grid) - 1))
    return float(smoothed[rn]["s"][ki])


def validate(smoothed, grid, runners, off):
    print("\n" + "=" * 78 + "\nVALIDATION (official = validation only)\n" + "=" * 78)
    FINISH_ORDER = off.finish_order

    fin_arc = {rn: _arclen_at_t(smoothed, rn, grid, WINNER_FINISH_RACE_T) for rn in runners}
    fused_finish = sorted(runners, key=lambda r: -fin_arc[r])
    off_rank = {h: i for i, h in enumerate(FINISH_ORDER)}
    common = [h for h in runners if h in off_rank]
    rho_fin = _spearman([fused_finish.index(h) for h in common], [off_rank[h] for h in common])
    win_ok = fused_finish[0] == FINISH_ORDER[0]
    print(f"\n[LONGITUDINAL]\n  fused finish : {fused_finish}")
    print(f"  official     : {FINISH_ORDER}")
    print(f"  finish-order rho = {rho_fin:+.2f}   winner #{FINISH_ORDER[0]} recovered: "
          f"{'YES' if win_ok else 'NO'} (fused leader #{fused_finish[0]})")

    win_arc = fin_arc[FINISH_ORDER[0]]
    fused_lbw = {rn: (win_arc - fin_arc[rn]) / 2.4 for rn in runners}
    mc = [h for h in runners if h in off.lbw]
    mae = float(np.mean([abs(fused_lbw[h] - off.lbw[h]) for h in mc]))
    rf_anchored = {int(r["runner"]) for r in
                   pl.read_parquet(RF_PANEL).filter(pl.col("kind") == "finish_margin")
                   .iter_rows(named=True)}
    lead = [h for h in mc if h in rf_anchored]
    mae_lead = (float(np.mean([abs(fused_lbw[h] - off.lbw[h]) for h in lead]))
                if lead else float("nan"))
    print(f"\n  finish MARGINS vs official LBW: MAE {mae:.2f}len (all) | "
          f"RF-anchored lead {sorted(rf_anchored)} MAE {mae_lead:.2f}len (R1: 0.72 lead)")
    print("   per-horse fused/official: " +
          ", ".join(f"#{h}:{fused_lbw[h]:.1f}/{off.lbw[h]:.1f}" for h in FINISH_ORDER))

    print("\n  [per-pole fused order vs official RP — 3 calls (1200m sprint)]")
    poles = {"1/4 (400m)": 400.0, "1/2 (800m)": 800.0, "late (1100m)": 1100.0}
    for ci, (name, arc) in enumerate(poles.items()):
        target_t = float(arclen_to_race_t(np.array([arc]))[0])
        ki = int(np.clip(round((target_t - grid[0]) / DT), 0, len(grid) - 1))
        fs = {rn: _arclen_at_t(smoothed, rn, grid, target_t) for rn in runners
              if smoothed[rn]["node_has_obs"][:ki + 1].any()}
        cc = [h for h in fs if h in off.rp and len(off.rp[h]) > ci]
        if len(cc) >= 3:
            rho = _spearman([-fs[h] for h in cc], [off.rp[h][ci] for h in cc])
            print(f"    {name:14s} race_t {target_t:4.1f}s n={len(cc):2d}  "
                  f"rho(fused, RP) = {rho:+.2f}")

    rail_t, wide_t = lateral_truth(off)
    print(f"\n[LATERAL — rail/tight {sorted(rail_t)} vs wide {sorted(wide_t)} (stewards-derived)]")
    body = {}
    # classification band 100-1100m INCLUDES the patrol metric anchor (~950-1050m on the sprint),
    # which is the only low-noise lateral source (R1's anchor sat <1300m; on 1200m it sits ~1000m).
    for rn in runners:
        s = smoothed[rn]["s"]
        haslat = smoothed[rn]["node_has_lat"]
        m = (s >= 100.0) & (s <= 1100.0) & haslat
        if m.sum() >= 2:
            body[rn] = float(np.median(smoothed[rn]["l"][m]))
    rv = [body[h] for h in rail_t if h in body]
    wv = [body[h] for h in wide_t if h in body]
    if rv and wv:
        gap = np.mean(wv) - np.mean(rv)
        thr = 0.5 * (np.mean(rv) + np.mean(wv))
        correct = sum(body[h] < thr for h in rail_t & set(body)) + \
                  sum(body[h] >= thr for h in wide_t & set(body))
        ncov = len((rail_t | wide_t) & set(body))
        print(f"  rail mean {np.mean(rv):.2f}m | wide mean {np.mean(wv):.2f}m | "
              f"GAP {gap:+.2f}m  (R1 +4.26m; {ncov} covered, {correct}/{ncov} correct)")
    for rn in sorted(body):
        tag = "rail" if rn in rail_t else ("wide" if rn in wide_t else "mid ")
        print(f"    #{rn:2d} [{tag}] {body[rn]:5.2f}m")

    print("\n[per-band fused y_lat uncertainty]")
    for name, lo, hi in [("start 100-400m", 100, 400), ("mid 400-800m", 400, 800),
                         ("bend->str 800-950m", 800, 950), ("RUN-IN >950m", 950, 1200)]:
        vals = []
        for rn in runners:
            s = smoothed[rn]["s"]
            m = (s >= lo) & (s <= hi)
            if m.sum() >= 1:
                vals.append(float(np.median(smoothed[rn]["sd_l"][m])))
        print((f"  {name:22s} mean sd(y_lat) = {np.mean(vals):.2f}m ({len(vals)} runners)")
              if vals else f"  {name:22s} no coverage")


def make_figure(smoothed, grid, runners, all_obs, off):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rail_t, wide_t = lateral_truth(off)
    cmap = plt.colormaps["tab20"]
    colors = {rn: cmap(i % 20) for i, rn in enumerate(runners)}
    fig, axes = plt.subplots(2, 1, figsize=(15, 13))
    winner = off.finish_order[0]

    ax = axes[0]
    lead = np.full(len(grid), -np.inf)
    for rn in runners:
        idx = np.where(smoothed[rn]["node_has_obs"])[0]
        if len(idx):
            m = slice(idx.min(), idx.max() + 1)
            lead[m] = np.maximum(lead[m], smoothed[rn]["s"][m])
    for rn in runners:
        s, sd, has = smoothed[rn]["s"], smoothed[rn]["sd_s"], smoothed[rn]["node_has_obs"]
        idx = np.where(has)[0]
        if not len(idx):
            continue
        m = slice(idx.min(), idx.max() + 1)
        gap = s[m] - lead[m]
        lw = 2.6 if rn == winner else 1.0
        ax.plot(grid[m], gap, color=colors[rn], lw=lw,
                label=f"#{rn}" + (" WIN" if rn == winner else ""))
        ax.fill_between(grid[m], gap - 2 * sd[m], gap + 2 * sd[m],
                        color=colors[rn], alpha=0.06, lw=0)
    ax.axhline(0, color="0.3", lw=0.7)
    ax.set_xlabel("race_t (s)")
    ax.set_ylabel("gap behind leader (m)")
    ax.set_title(f"R03 4-FEED FUSION — longitudinal (winner #{winner} climbs; +/-2sd)")
    ax.legend(ncol=7, fontsize=7, loc="lower left")
    ax.set_ylim(-30, 4)

    ax = axes[1]
    for rn in runners:
        s, lat, sdl, has = (smoothed[rn]["s"], smoothed[rn]["l"], smoothed[rn]["sd_l"],
                            smoothed[rn]["node_has_lat"])
        idx = np.where(has)[0]
        if not len(idx):
            continue
        m = slice(idx.min(), idx.max() + 1)
        tag = "rail" if rn in rail_t else ("wide" if rn in wide_t else "")
        ax.plot(s[m], lat[m], color=colors[rn], lw=1.4, label=f"#{rn}{(' ' + tag) if tag else ''}")
        ax.fill_between(s[m], lat[m] - 2 * sdl[m], lat[m] + 2 * sdl[m],
                        color=colors[rn], alpha=0.10, lw=0)
    ax.axvline(RUNIN_THRESH, color="crimson", ls="--", lw=1.2)
    pat = [o for o in all_obs if o.feed == "patrol"]
    if pat:
        ax.axvspan(min(o.arclen for o in pat), max(o.arclen for o in pat), color="gold", alpha=0.12)
    ax.set_xlabel("arclen (m)")
    ax.set_ylabel("y_lat (m from rail)")
    ax.set_title("R03 4-FEED FUSION — lateral/control (gold=patrol anchor; +/-2sd)")
    ax.legend(ncol=6, fontsize=7, loc="upper right")
    fig.tight_layout()
    fig.savefig(FIG_RECON, dpi=110)
    print(f"\n[fig] {FIG_RECON}")


if __name__ == "__main__":
    run()
