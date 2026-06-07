"""R08 4-FEED FUSION — REUSE cv_fusion_4feed (#90/0087) on HV R08 1800m route (#95).

HELD-OUT route generalization. Imports the R1 fusion SMOOTHER unchanged (Obs / smooth_horse /
the RTS machinery + noise levels) and re-points ONLY the R08 race PARAMETERS (DIST 1650->1800,
sections 4->5, _D_BOUNDS/_T_CUM route boundaries, winner-t 108.62, the 4 input parquets, the
drone clock anchor, the official RP/FINISH_ORDER/LBW/rail-wide tables). NO smoother noise level
or algorithm is tuned for R08.

Includes the FORESHORTENING RECON (the metric-gap angle): quantify how badly the AERIAL render
lateral degrades on R08's BENDS vs its STRAIGHTS (within/between dispersion per bend-vs-straight
band) and how much patrol+draw+drone compensate — the route is the foreshortening stress case.

The DRAW -> y_lat(0) anchor (#93) is added in the v2 runner (cv_identity_v2_rerun pattern); this
module exposes load_patrol_r08 for that wrapper, exactly like cv_r03_fusion.

Writes data_cache/cv_fusion_trajectories_R08.parquet + fig cv_fusion_recon_R08.png.
Official data VALIDATION-only.

    PYTHONHASHSEED=0 uv run --group bayes --group vision python studies/cv_r08_fusion.py
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
from cv_r08_common import (  # noqa: E402
    DIST,
    OFFICIAL_SECTIONS,
    TRACK_WIDTH_M,
    WINNER_FINISH_RACE_T,
    _D_BOUNDS,
    _T_CUM,
    lateral_truth,
    parse_official,
)

OUT_TRAJ = "data_cache/cv_fusion_trajectories_R08.parquet"
FIG_RECON = "data_cache/figs/cv_fusion_recon_R08.png"

AERIAL_PANEL = "data_cache/cv_r08_aerial_render.parquet"
DRONE_PANEL = "data_cache/cv_r08_drone_numbered.parquet"
PATROL_PANEL = "data_cache/cv_r08_patrol_numbered.parquet"
RF_PANEL = "data_cache/cv_r08_rf_longitudinal.parquet"

DRONE_START_VID_T, DRONE_FINISH_VID_T = 3.0, 120.5
# run-in foreshorten threshold: R1 1300m/1650 (~79%), R03 950m/1200 (~79%). 79% of 1800 ~ 1420m.
RUNIN_THRESH = 1420.0


def arclen_to_race_t(a):
    return np.interp(np.asarray(a, float), _D_BOUNDS, _T_CUM)


def race_t_to_arclen(t):
    return np.interp(np.asarray(t, float), _T_CUM, _D_BOUNDS)


def _setup_r08(off) -> None:
    """[GENERALIZE] re-point all R1 fusion-module constants/curves at R08 (parameters only)."""
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


def load_drone_r08() -> list[Obs]:
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


def load_patrol_r08() -> list[Obs]:
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
    _setup_r08(off)
    print("=" * 80)
    print("R08 4-FEED FUSION #95 — RTS smoother on HV 1800m ROUTE (held out, no tuning)")
    print("=" * 80)
    print(f"winner #{off.finish_order[0]} {off.horse[off.finish_order[0]]} | "
          f"sections {OFFICIAL_SECTIONS} | winner_t {WINNER_FINISH_RACE_T:.2f}s")

    aer_obs, aer_df, field_race_t, field_arc = F1.load_aerial()
    dro_obs = load_drone_r08()
    pat_obs = load_patrol_r08()
    F1.DIST = DIST
    rf_obs = _load_rf_r08(off, field_race_t, field_arc)

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
    foreshorten_recon(aer_df, off)
    make_figure(smoothed, grid, runners, all_obs, off)


def _load_rf_r08(off, field_race_t, field_arc):
    """R08 RF obs. Reuse R03's structure with win_arc=1800 + winner-relative carry."""
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
    all_runners = set(off.finish_order)
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
          f"RF-anchored lead {sorted(rf_anchored)} MAE {mae_lead:.2f}len (R1 0.72, R03 1.70)")
    print("   per-horse fused/official: " +
          ", ".join(f"#{h}:{fused_lbw[h]:.1f}/{off.lbw[h]:.1f}" for h in FINISH_ORDER))

    print("\n  [per-pole fused order vs official RP — 5 calls (1800m route)]")
    poles = {"c1 (200m)": 200.0, "c2 (600m)": 600.0, "c3 (1000m)": 1000.0,
             "c4 (1400m)": 1400.0, "late (1700m)": 1700.0}
    for ci, (name, arc) in enumerate(poles.items()):
        target_t = float(arclen_to_race_t(np.array([arc]))[0])
        ki = int(np.clip(round((target_t - grid[0]) / DT), 0, len(grid) - 1))
        fs = {rn: _arclen_at_t(smoothed, rn, grid, target_t) for rn in runners
              if smoothed[rn]["node_has_obs"][:ki + 1].any()}
        cc = [h for h in fs if h in off.rp and len(off.rp[h]) > ci]
        if len(cc) >= 3:
            rho = _spearman([-fs[h] for h in cc], [off.rp[h][ci] for h in cc])
            print(f"    {name:14s} race_t {target_t:5.1f}s n={len(cc):2d}  "
                  f"rho(fused, RP) = {rho:+.2f}")

    rail_t, wide_t = lateral_truth(off)
    print(f"\n[LATERAL — rail/tight {sorted(rail_t)} vs wide {sorted(wide_t)} (stewards-derived)]")
    body = {}
    # classification band 100-1500m INCLUDES the patrol metric anchor (the route home straight
    # entry ~1400-1600m); R1 used <1300m on 1650, R03 <1100m on 1200.
    for rn in runners:
        s = smoothed[rn]["s"]
        haslat = smoothed[rn]["node_has_lat"]
        m = (s >= 100.0) & (s <= 1550.0) & haslat
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
              f"GAP {gap:+.2f}m  (R1 +4.26m, R03 +5.81m; {ncov} covered, {correct}/{ncov} correct) "
              f"SIGN {'CORRECT' if gap > 0 else 'WRONG'}")
    for rn in sorted(body):
        tag = "rail" if rn in rail_t else ("wide" if rn in wide_t else "mid ")
        print(f"    #{rn:2d} [{tag}] {body[rn]:5.2f}m")

    print("\n[per-band fused y_lat uncertainty]")
    for name, lo, hi in [("start 100-600m", 100, 600), ("mid 600-1100m", 600, 1100),
                         ("bend->str 1100-1420m", 1100, 1420), ("RUN-IN >1420m", 1420, 1800)]:
        vals = []
        for rn in runners:
            s = smoothed[rn]["s"]
            m = (s >= lo) & (s <= hi)
            if m.sum() >= 1:
                vals.append(float(np.median(smoothed[rn]["sd_l"][m])))
        print((f"  {name:24s} mean sd(y_lat) = {np.mean(vals):.2f}m ({len(vals)} runners)")
              if vals else f"  {name:24s} no coverage")


def foreshorten_recon(aer_df, off):
    """FORESHORTENING RECON (the metric-gap angle). Quantify how badly the AERIAL render lateral
    degrades on R08's BENDS vs its STRAIGHTS.

    The route (1800m) is mostly turning. We partition the AERIAL render's per-frame y_lat by
    arclen BAND (HV "C" 1450m circuit: a 1800m race runs ~1.24 laps, so the back-straight + home-
    straight are short windows between long bends). For each band we compute the within-runner
    lateral dispersion (frame-to-frame jitter of the SAME runner's y_lat — should be ~0 on a clean
    feed) vs the between-runner spread. A within/between ratio >> on bends vs straights localizes
    where the oblique foreshortening bites. Aerial-ONLY (the metric-gap is the render's; patrol/draw
    compensate by anchoring, quantified in the per-band fused sd above).
    """
    print("\n" + "=" * 78 + "\nFORESHORTENING RECON — aerial render bend-vs-straight degradation\n"
          + "=" * 78)
    if aer_df is None or aer_df.height == 0:
        print("  (no aerial panel — skipped)")
        return
    df = aer_df.filter(pl.col("x_arclen").is_finite())
    if df.height == 0:
        print("  (no arclen-tagged aerial rows — skipped)")
        return
    # HV "C" 1450m circuit. A 1800m route ~ 1.24 laps. The STRAIGHTS are short; classify each band
    # as straight vs bend by the render rail straightness if available, else by a circuit model.
    # Circuit model (arclen along the 1450m lap): straights ~ [0-150] of each ~725m half. We use a
    # simple geometry-agnostic proxy: split into 200m bands and within each, measure the per-runner
    # frame-jitter (within) and the cross-runner spread (between); also report a straightness proxy
    # from the per-runner y_lat AUTOCORR (high = smooth/graphics-faithful, low = noisy/foreshortened).
    bands = [(lo, lo + 200) for lo in range(0, int(DIST), 200)]
    print(f"  {'band (m)':14s} {'within':>7s} {'between':>8s} {'ratio':>6s} {'ac1':>6s} {'n':>4s}")
    ratios = []
    for lo, hi in bands:
        b = df.filter((pl.col("x_arclen") >= lo) & (pl.col("x_arclen") < hi))
        if b.height < 10:
            continue
        withins, acs, meds = [], [], []
        for rn in b["runner"].unique().to_list():
            s = b.filter(pl.col("runner") == rn).sort("frame")
            y = s["y_lat"].to_numpy()
            if len(y) < 4:
                continue
            withins.append(float(np.std(np.diff(y)) / np.sqrt(2)))  # frame-to-frame jitter sd
            meds.append(float(np.median(y)))
            yc = y - y.mean()
            denom = float(np.dot(yc, yc))
            acs.append(float(np.dot(yc[:-1], yc[1:]) / denom) if denom > 1e-9 else 0.0)
        if len(meds) < 2 or not withins:
            continue
        within = float(np.median(withins))
        between = float(np.std(meds))
        ratio = within / (between + 1e-6)
        ac1 = float(np.median(acs)) if acs else float("nan")
        ratios.append((lo, hi, ratio, ac1))
        print(f"  {f'{lo}-{hi}':14s} {within:7.2f} {between:8.2f} {ratio:6.2f} {ac1:+6.2f} "
              f"{len(meds):4d}")
    if ratios:
        # split into "straighter" (high ac1, body) vs "bend/run-in" (low ac1 or late arclen) bands
        early = [r for r in ratios if r[1] <= 0.55 * DIST]
        late = [r for r in ratios if r[0] >= 0.80 * DIST]
        if early and late:
            re_ = float(np.mean([r[2] for r in early]))
            rl_ = float(np.mean([r[2] for r in late]))
            print(f"\n  BODY (<{0.55 * DIST:.0f}m) mean within/between {re_:.2f} | "
                  f"RUN-IN (>{0.80 * DIST:.0f}m) {rl_:.2f}  -> degradation x{rl_ / (re_ + 1e-6):.2f}")
        ac_all = [r[3] for r in ratios if r[3] == r[3]]
        if ac_all:
            print(f"  aerial y_lat lag-1 autocorr across bands: median {np.median(ac_all):+.2f} "
                  f"(R1/R03 body +0.90/+0.93; low/negative = foreshortened-noisy band)")
    print("  INTERPRETATION: ratio >> 1 and low ac1 mark bands where the oblique render "
          "foreshortens the lateral; the fusion leans on patrol (head-on) + draw (start anchor) "
          "+ drone there — see the per-band FUSED sd above for the compensated uncertainty.")


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
    ax.set_title(f"R08 4-FEED FUSION — longitudinal (winner #{winner} comes from behind; +/-2sd)")
    ax.legend(ncol=6, fontsize=7, loc="lower left")
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
    ax.set_title("R08 4-FEED FUSION — lateral/control (gold=patrol anchor; +/-2sd)")
    ax.legend(ncol=6, fontsize=7, loc="upper right")
    fig.tight_layout()
    fig.savefig(FIG_RECON, dpi=110)
    print(f"\n[fig] {FIG_RECON}")


if __name__ == "__main__":
    run()
