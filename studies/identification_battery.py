"""DEDICATED-SURFACE IDENTIFICATION BATTERY for the EOM physical constants (journal DRAFT 0048).

Identifies each EOM constant (CP, W′_0, P_max, τ) on the DEDICATED surface where it DOMINATES the
dynamics, and CROSS-CHECKS the same constant measured multiple independent ways. The headline is
the decisive physics test: does CP from the route CRUISE agree with CP from the distance–time
(HILL) slope? Cross-surface consistency == the physics is real; disagreement == missing physics.

    uv run python studies/identification_battery.py             # ST + HV, full
    IDB_VENUE=ST uv run python studies/identification_battery.py # one venue

NO regression of outcome on features, NO market odds, NO win, NO pool. Every estimator inverts the
SAME EOM physics (``hkjc.model.dynamics`` + ``identify``). Mass + going are KNOWN inputs.
Prints a per-constant table + the headline consistency; writes the CP_cruise-vs-CP_hill scatter.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import polars as pl

from hkjc.model.dynamics import (
    body_mass_kg, cp_for_cruise, cruise_velocity, make_resistances,
)
from hkjc.model.identify import (
    Run, bland_altman, cp_cruise_run, hill_fit, hill_wprime_joules, pmax_for_ceiling,
    pmax_opening_run, settled_cruise_velocity, sit_kick_run, sprint_ceiling_velocity,
    tau_from_sit_kick, wprime_spent_run,
)

CACHE = Path("data_cache")
OUT = Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)
ROUTE_MIN = 1800.0           # m, a "route" (settled aerobic cruise present)
SPRINT_MAX = 1200.0          # m, a "sprint" (P_max-dominated)


# ----------------------------------------------------------------------- data assembly
def _going_mass_frame() -> pl.DataFrame:
    rr = pl.read_parquet(CACHE / "results_rich.parquet")
    return (rr.with_columns(
                pl.col("Date").cast(pl.Utf8).str.replace_all("/", "-").alias("date"),
                pl.col("Venue").cast(pl.Utf8).alias("venue"),
                pl.col("RaceNumber").cast(pl.Int32).alias("race_no"),
                pl.col("HorseNo").cast(pl.Int64, strict=False).alias("runner"),
                pl.col("Going").cast(pl.Utf8).alias("going"),
                pl.col("DeclarHorseWt").cast(pl.Float64, strict=False).alias("bodywt_lb"),
                pl.col("ActWt").cast(pl.Float64, strict=False).alias("carriedwt_lb"))
            .select("date", "venue", "race_no", "runner", "going", "bodywt_lb", "carriedwt_lb")
            .unique(["date", "venue", "race_no", "runner"]))


def load_runs(venue: str) -> list[Run]:
    """Assemble the leak-free per-run kinematic frame (velocities/times/dist + KNOWN mass+going)."""
    traj = pl.read_parquet(CACHE / "trajectory_dataset_full.parquet").with_columns(
        pl.col("date").cast(pl.Utf8).str.replace_all("/", "-").alias("date"))
    gm = _going_mass_frame()
    agg = (traj.filter(pl.col("venue") == venue)
           .sort(["date", "venue", "race_no", "runner", "section"])
           .group_by(["date", "venue", "race_no", "runner"])
           .agg(pl.col("horse_id").first(), pl.col("distance").first().alias("distance"),
                pl.col("section_dist"), pl.col("section_time"), pl.col("velocity"),
                pl.col("n_sections").first()))
    agg = agg.join(gm, on=["date", "venue", "race_no", "runner"], how="left")
    runs = []
    for r in agg.to_dicts():
        m = body_mass_kg(r.get("bodywt_lb") or 1100.0, r.get("carriedwt_lb") or 0.0)
        runs.append(Run(
            horse_id=r["horse_id"], distance=float(r["distance"]),
            section_dist=np.asarray(r["section_dist"], float),
            section_time=np.asarray(r["section_time"], float),
            velocity=np.asarray(r["velocity"], float),
            mass=m, going=r.get("going") or "GOOD"))
    return runs


# ----------------------------------------------------------------- per-horse estimators
def cp_cruise_by_horse(runs: list[Run]) -> dict[str, dict]:
    """Per-horse CP_cruise: pooled over the horse's ROUTE runs (≥ROUTE_MIN), force-balance root."""
    by: dict[str, list] = {}
    for r in runs:
        if r.distance < ROUTE_MIN:
            continue
        cp = cp_cruise_run(r, level="min")
        if cp is None:
            continue
        vc_obs = settled_cruise_velocity(r)
        by.setdefault(r.horse_id, []).append((cp, vc_obs, r.mass, r.going))
    out = {}
    for h, lst in by.items():
        cps = np.array([x[0] for x in lst])
        vcs = np.array([x[1] for x in lst])
        out[h] = {"cp_cruise": float(np.mean(cps)), "v_settle": float(np.mean(vcs)),
                  "n_route": len(lst), "mass": float(np.mean([x[2] for x in lst]))}
    return out


def hill_by_horse(runs: list[Run], *, min_dist: int = 3) -> dict[str, dict]:
    """Per-horse Hill fit d = CV·t + D' across the horse's races at ≥min_dist distinct distances."""
    by: dict[str, list] = {}
    for r in runs:
        by.setdefault(r.horse_id, []).append(r)
    out = {}
    for h, lst in by.items():
        d = np.array([x.distance for x in lst])
        t = np.array([x.total_time for x in lst])
        fit = hill_fit(d, t, min_dist=min_dist)
        if fit is None:
            continue
        # CV → CP_power via the force balance at CV, using the horse's mean mass/going
        mass = float(np.mean([x.mass for x in lst]))
        res = make_resistances(mass, "GOOD")
        cp_hill = cp_for_cruise(fit.cv, res) if 8.0 < fit.cv < 22.0 else float("nan")
        w_hill = hill_wprime_joules(fit, res) if np.isfinite(cp_hill) else float("nan")
        out[h] = {"cv": fit.cv, "d_prime": fit.d_prime, "cp_hill": cp_hill, "w_hill_J": w_hill,
                  "n_dist": fit.n_dist, "n_run": fit.n_run, "r2": fit.r2, "cv_se": fit.cv_se,
                  "span_m": fit.span_m, "mass": mass}
    return out


# ----------------------------------------------------------------------------- the report
def run_venue(venue: str, runs: list[Run]) -> dict:
    print(f"\n{'='*80}\n[{venue}] DEDICATED-SURFACE IDENTIFICATION BATTERY\n{'='*80}")
    print(f"loaded {len(runs)} runs; "
          f"{sum(r.distance >= ROUTE_MIN for r in runs)} route (≥{ROUTE_MIN:.0f}m), "
          f"{sum(r.distance <= SPRINT_MAX for r in runs)} sprint (≤{SPRINT_MAX:.0f}m)")

    # ---- CP (a) cruise, (b) hill ------------------------------------------------------
    cpc = cp_cruise_by_horse(runs)
    hill = hill_by_horse(runs, min_dist=3)
    print(f"\nCP_cruise: {len(cpc)} horses (≥1 route run)  |  "
          f"Hill: {len(hill)} horses (≥3 distinct distances)")

    # headline consistency: v_c(CP_cruise) [via v_settle] vs CV (both critical velocity, m/s)
    common = sorted(set(cpc) & set(hill))
    vc_cruise, cv_hill, cp_cruise_W, cp_hill_W = [], [], [], []
    for h in common:
        res = make_resistances(cpc[h]["mass"], "GOOD")
        vc_c = cruise_velocity(cpc[h]["cp_cruise"], res)   # the cruise-derived critical velocity
        vc_cruise.append(vc_c)
        cv_hill.append(hill[h]["cv"])
        cp_cruise_W.append(cpc[h]["cp_cruise"])
        cp_hill_W.append(hill[h]["cp_hill"])
    vc_cruise = np.array(vc_cruise)
    cv_hill = np.array(cv_hill)
    cp_cruise_W = np.array(cp_cruise_W)
    cp_hill_W = np.array(cp_hill_W)
    ba_v = bland_altman(vc_cruise, cv_hill)
    ba_p = bland_altman(cp_cruise_W / 1e3, cp_hill_W / 1e3)
    print(f"\n  --- HEADLINE: CP_cruise vs CP_hill (n={ba_v['n']} horses with BOTH) ---")
    print(f"  velocity-space (m/s):  v_c(cruise) med={np.nanmedian(vc_cruise):.2f}  "
          f"CV(hill) med={np.nanmedian(cv_hill):.2f}")
    print(f"     corr={ba_v['corr']:+.3f}  bias(cruise−hill)={ba_v['bias']:+.3f}  "
          f"MAD={ba_v['mad']:.3f}  95%LoA=[{ba_v['loa_lo']:+.2f},{ba_v['loa_hi']:+.2f}]")
    print(f"  power-space (kW):      CP(cruise) med={np.nanmedian(cp_cruise_W)/1e3:.1f}  "
          f"CP(hill) med={np.nanmedian(cp_hill_W)/1e3:.1f}")
    print(f"     corr={ba_p['corr']:+.3f}  bias={ba_p['bias']:+.2f}kW  MAD={ba_p['mad']:.2f}kW")

    # CP_cruise sensitivity to the settled-level choice (min vs median) — aliasing probe
    sens = []
    for r in runs:
        if r.distance < ROUTE_MIN:
            continue
        a = cp_cruise_run(r, level="min")
        b = cp_cruise_run(r, level="median")
        if a and b:
            sens.append((a, b))
    if sens:
        a = np.array([x[0] for x in sens]) / 1e3
        b = np.array([x[1] for x in sens]) / 1e3
        print(f"  CP_cruise level sensitivity (min vs median interior): "
              f"med min={np.median(a):.1f}kW median={np.median(b):.1f}kW "
              f"Δ={np.median(b - a):+.1f}kW ({np.median((b - a) / a) * 100:+.0f}%)")

    # ---- P_max (a) opening transient, (b) sprint ceiling ------------------------------
    pmax_open, pmax_ceil = [], []
    for r in runs:
        po = pmax_opening_run(r)
        if po and 0 < po < 2e5:
            pmax_open.append((r.horse_id, po, r.mass))
        if r.distance <= SPRINT_MAX:
            vpk = sprint_ceiling_velocity(r)
            pc = pmax_for_ceiling(vpk, r.res)
            if 0 < pc < 2e5:
                pmax_ceil.append((r.horse_id, pc, r.mass))
    po_arr = np.array([x[1] for x in pmax_open])
    pc_arr = np.array([x[1] for x in pmax_ceil])
    print(f"\n  --- P_max: opening transient (n={len(po_arr)}) vs sprint ceiling (n={len(pc_arr)}) ---")
    print(f"  opening:  med={np.median(po_arr)/1e3:.1f}kW "
          f"[{np.percentile(po_arr,10)/1e3:.1f},{np.percentile(po_arr,90)/1e3:.1f}]  "
          f"({np.median(po_arr/np.array([x[2] for x in pmax_open])):.0f} W/kg)")
    print(f"  ceiling:  med={np.median(pc_arr)/1e3:.1f}kW "
          f"[{np.percentile(pc_arr,10)/1e3:.1f},{np.percentile(pc_arr,90)/1e3:.1f}]  "
          f"({np.median(pc_arr/np.array([x[2] for x in pmax_ceil])):.0f} W/kg)")
    # per-horse cross-check
    po_h: dict[str, list] = {}
    pc_h: dict[str, list] = {}
    for hid, p, _ in pmax_open:
        po_h.setdefault(hid, []).append(p)
    for hid, p, _ in pmax_ceil:
        pc_h.setdefault(hid, []).append(p)
    ch = sorted(set(po_h) & set(pc_h))
    pao = np.array([np.mean(po_h[h]) for h in ch]) / 1e3
    pac = np.array([np.mean(pc_h[h]) for h in ch]) / 1e3
    ba_pm = bland_altman(pao, pac)
    print(f"  per-horse consistency (n={ba_pm['n']}): corr={ba_pm['corr']:+.3f} "
          f"bias(open−ceil)={ba_pm['bias']:+.1f}kW MAD={ba_pm['mad']:.1f}kW")

    # ---- W′_0 (a) hill intercept, (b) reserve spent above CP --------------------------
    w_hill_h = {h: hill[h]["w_hill_J"] for h in hill if np.isfinite(hill[h]["w_hill_J"])}
    w_spent_h: dict[str, list] = {}
    for r in runs:
        if r.horse_id in cpc:                        # need a CP for this horse
            ws = wprime_spent_run(r, cpc[r.horse_id]["cp_cruise"])
            if ws and ws > 0:
                w_spent_h.setdefault(r.horse_id, []).append(ws)
    w_spent_mean = {h: float(np.mean(v)) for h, v in w_spent_h.items()}
    wh = sorted(set(w_hill_h) & set(w_spent_mean))
    wa = np.array([w_hill_h[h] for h in wh]) / 1e3
    wb = np.array([w_spent_mean[h] for h in wh]) / 1e3
    ba_w = bland_altman(wa, wb)
    print(f"\n  --- W′_0: Hill intercept (n={len(w_hill_h)}) vs reserve-spent (n={len(w_spent_mean)}) ---")
    if len(w_hill_h):
        print(f"  Hill D'→J:  med={np.nanmedian(list(w_hill_h.values()))/1e3:.0f}kJ")
    if len(w_spent_mean):
        print(f"  spent>CP:   med={np.nanmedian(list(w_spent_mean.values()))/1e3:.0f}kJ")
    print(f"  per-horse consistency (n={ba_w['n']}): corr={ba_w['corr']:+.3f} "
          f"bias(hill−spent)={ba_w['bias']:+.0f}kJ MAD={ba_w['mad']:.0f}kJ")

    # ---- τ — sit-and-kick closers -----------------------------------------------------
    taus = []
    n_route_checked = 0
    n_sit = 0
    for r in runs:
        if r.distance < ROUTE_MIN or r.horse_id not in cpc:
            continue
        n_route_checked += 1
        cp = cpc[r.horse_id]["cp_cruise"]
        sk = sit_kick_run(r, cp)
        if sk is None:
            continue
        n_sit += 1
        # W′_0 estimate for this horse (Hill or spent) for the (1−W′/W′_0) term scale
        w0 = w_hill_h.get(r.horse_id) or w_spent_mean.get(r.horse_id) or 2.0e5
        tau = tau_from_sit_kick(sk, w0, r.res)
        if tau and np.isfinite(tau):
            taus.append(tau)
    taus = np.array(taus)
    print("\n  --- τ: sit-and-kick closers (route runs that settled below v_c then kicked) ---")
    print(f"  {n_sit}/{n_route_checked} route runs were sit-and-kick; τ estimable on {len(taus)}")
    if len(taus):
        print(f"  τ med={np.median(taus):.0f}s [{np.percentile(taus,25):.0f},"
              f"{np.percentile(taus,75):.0f}]  (EOM prior 120s; #52: in-race negligible)")

    return {"venue": venue, "vc_cruise": vc_cruise, "cv_hill": cv_hill,
            "cp_cruise_W": cp_cruise_W, "cp_hill_W": cp_hill_W, "common": common,
            "cpc": cpc, "hill": hill, "ba_v": ba_v, "ba_p": ba_p, "ba_pm": ba_pm, "ba_w": ba_w,
            "taus": taus, "w_hill_h": w_hill_h, "w_spent_mean": w_spent_mean}


def reemergence(res: dict, runs: list[Run]) -> None:
    """RE-EMERGENCE probe: does the physically-measured CP predict held-out per-horse form?

    Lightweight, descriptive, leak-free: split each horse's route runs in time; does the
    CP_cruise measured on the FIRST half correlate with the settled cruise velocity in the
    SECOND half (held-out)? A physical constant should persist → predict the horse's own future
    sustainable level. Correlation only (no outcome regression, no market).
    """
    venue = res["venue"]
    by: dict[str, list] = {}
    for r in runs:
        if r.distance >= ROUTE_MIN:
            v = settled_cruise_velocity(r)
            if v is not None:
                by.setdefault(r.horse_id, []).append(v)
    early_cp, late_v = [], []
    for h, vs in by.items():
        if len(vs) < 2 or h not in res["cpc"]:
            continue
        half = len(vs) // 2
        early_cp.append(np.mean(vs[:half]) if half else vs[0])
        late_v.append(np.mean(vs[half:]))
    if len(early_cp) >= 5:
        c = float(np.corrcoef(early_cp, late_v)[0, 1])
        print(f"\n  [{venue}] RE-EMERGENCE: early-half settled cruise vs held-out late-half "
              f"(n={len(early_cp)}): corr={c:+.3f}  (the physical CP persists → predicts own form)")
    # CV (hill) vs aggregate sustainable level
    cvs, levs = [], []
    for h in res["common"]:
        cvs.append(res["hill"][h]["cv"])
        levs.append(res["cpc"][h]["v_settle"])
    if len(cvs) >= 5:
        c2 = float(np.corrcoef(cvs, levs)[0, 1])
        print(f"  [{venue}] CV(hill) vs mean settled cruise (n={len(cvs)}): corr={c2:+.3f}")


def plot_headline(results: list[dict]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(results), figsize=(6 * len(results), 5.5), squeeze=False)
    for ax, res in zip(axes[0], results):
        x = res["cv_hill"]
        y = res["vc_cruise"]
        m = np.isfinite(x) & np.isfinite(y)
        ax.scatter(x[m], y[m], s=14, alpha=0.45, edgecolor="none")
        lo, hi = 12.0, 19.0
        ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="identity (perfect agreement)")
        ba = res["ba_v"]
        ax.set_xlabel("CV from distance–time Hill slope  (m/s)")
        ax.set_ylabel("v_c from route cruise force-balance  (m/s)")
        ax.set_title(f"[{res['venue']}] CP: cruise vs Hill  (n={ba['n']})\n"
                     f"corr={ba['corr']:+.2f}  bias={ba['bias']:+.2f}  MAD={ba['mad']:.2f} m/s")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(alpha=0.25)
    fig.tight_layout()
    p = OUT / "identification_cp_cruise_vs_hill.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    print(f"\nwrote {p}")


def main() -> None:
    venues = [os.environ["IDB_VENUE"]] if os.environ.get("IDB_VENUE") else ["ST", "HV"]
    results = []
    for v in venues:
        runs = load_runs(v)
        res = run_venue(v, runs)
        reemergence(res, runs)
        results.append(res)
    plot_headline(results)


if __name__ == "__main__":
    main()
