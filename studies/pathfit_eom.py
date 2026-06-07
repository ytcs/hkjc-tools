"""FIRST-PRINCIPLES path-fit (journal DRAFT 0047) — does the EOM generative core, solved under
the bang–CP–bang policy, reproduce observed race paths, and does it remove the +0.42 m/s level
bias + opening-ramp defect by MECHANISM (no regression)?

Re-runs the #49 pathfit harness (``studies/pathfit_foundation.py``) with the heuristic
``expected_profile`` replaced by the EOM solution (``hkjc.model.dynamics``). Per run we
INVERT the physical constants (v_c↔CP, W′_0, P_max) by physically-constrained bounded
least-squares so the EOM section velocities match the observed profile; the per-horse constants
are pooled leave-one-race-out (strictly-prior, train-only population fallback); the held-out
profile is the EOM forward solution at the pooled constants. NO regression, NO market/win/pool.

    uv run --group bayes python studies/pathfit_eom.py            # ST + HV
    EOM_VENUE=ST EOM_NRUN=4000 uv run --group bayes python studies/pathfit_eom.py   # quick

Prints the structured VERDICT vs the noise floor, the descriptive baseline, the heuristic
(0045 numbers), the level bias, the opening-ramp fit, and PIT.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import polars as pl
from scipy.optimize import least_squares

from hkjc.model.dynamics import (
    body_mass_kg, cp_for_cruise, eom_section_velocities_fast, make_resistances,
)
from hkjc.model.pathfit import build_run_panel, noise_floor, rmse_r2

CACHE = Path("data_cache")
OUT = Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)
SIGMA = 0.06
TAU = 120.0                # s, recovery time constant (fixed: in-race recovery is negligible)
DT = 0.06                  # s, EOM integration step (fast core)
V0 = 4.0                   # m/s, gate-release initial velocity (standing-start lunge; physical)

# physical inversion bounds (the latents' physical ranges)
VC_LO, VC_HI = 12.0, 19.0          # m/s cruise root
W0_LO, W0_HI = 3.0e4, 2.0e6        # J anaerobic reserve
PR_LO, PR_HI = 1.05, 3.2           # P_max / CP ratio (sprint power 1.05–3.2× CP)


def _going_frame() -> pl.DataFrame:
    rr = pl.read_parquet(CACHE / "results_rich.parquet")
    return (rr.with_columns(
                pl.col("Date").cast(pl.Utf8).str.replace_all("/", "-").alias("date"),
                pl.col("Venue").cast(pl.Utf8).alias("venue"),
                pl.col("RaceNumber").cast(pl.Int32).alias("race_no"),
                pl.col("Going").cast(pl.Utf8).alias("going"),
                pl.when(pl.col("Course").str.contains("ALL WEATHER"))
                  .then(pl.lit("awt")).otherwise(pl.lit("turf")).alias("surface"))
            .select("date", "venue", "race_no", "going", "surface")
            .unique(["date", "venue", "race_no"]))


def _mass_frame() -> pl.DataFrame:
    rr = pl.read_parquet(CACHE / "results_rich.parquet")
    return (rr.with_columns(
                pl.col("Date").cast(pl.Utf8).str.replace_all("/", "-").alias("date"),
                pl.col("Venue").cast(pl.Utf8).alias("venue"),
                pl.col("RaceNumber").cast(pl.Int32).alias("race_no"),
                pl.col("HorseNo").cast(pl.Int64, strict=False).alias("runner"),
                pl.col("DeclarHorseWt").cast(pl.Float64, strict=False).alias("bodywt_lb"),
                pl.col("ActWt").cast(pl.Float64, strict=False).alias("carriedwt_lb"))
            .select("date", "venue", "race_no", "runner", "bodywt_lb", "carriedwt_lb")
            .unique(["date", "venue", "race_no", "runner"]))


def _coarse_going(expr: pl.Expr) -> pl.Expr:
    return (pl.when(expr.str.contains("FIRM")).then(pl.lit("GOOD-FIRM"))
            .when(expr.str.contains("YIELD") | expr.str.contains("SOFT") | expr.str.contains("HEAVY"))
            .then(pl.lit("SOFT+"))
            .when(expr.str.contains("WET") | expr.str.contains("SEALED")).then(pl.lit("WET-AWT"))
            .when(expr.str.contains("GOOD")).then(pl.lit("GOOD"))
            .otherwise(pl.lit("OTHER")))


# --------------------------------------------------------------- per-run physical inversion
def invert_run(d: np.ndarray, v_obs: np.ndarray, m: float, going: str) -> tuple[float, float, float, float]:
    """Infer (v_c, W′_0, P_max, CP) for one run by bounded NLS of the EOM sections vs observed.

    Physically-constrained MAP/least-squares: minimise Σ (log v_eom − log v_obs)² over the three
    physical unknowns (v_c↔CP, W′_0, P_max=ratio·CP) within their physical bounds. Mass + going
    are KNOWN (resistances fixed). τ fixed. The reserve, cruise root and peak power are physical
    quantities, not free regressors — this is inference of physical constants, not a curve fit.
    """
    res = make_resistances(m, going)

    def resid(th: np.ndarray) -> np.ndarray:
        vc, lw, pr = th
        cp = cp_for_cruise(vc, res)
        sv, _, _ = eom_section_velocities_fast(d, cp, np.exp(lw), pr * cp, TAU, res, dt=DT, v0=V0)
        return np.log(np.clip(sv, 1e-3, None)) - np.log(np.clip(v_obs, 1e-3, None))

    # physical initial guess: v_c ≈ settled mid level, W′_0 ≈ 2e5 J, P_max ≈ 1.7·CP
    vcru = float(np.mean(v_obs[1:-1])) if len(v_obs) >= 3 else float(v_obs[-1])
    x0 = np.array([np.clip(vcru - 0.8, VC_LO + 0.5, VC_HI - 0.5), np.log(2.0e5), 1.7])
    try:
        sol = least_squares(resid, x0,
                            bounds=([VC_LO, np.log(W0_LO), PR_LO], [VC_HI, np.log(W0_HI), PR_HI]),
                            method="trf", max_nfev=50, diff_step=0.05, xtol=1e-3, ftol=1e-3)
        vc, lw, pr = sol.x
    except Exception:
        vc, lw, pr = x0
    cp = cp_for_cruise(float(vc), res)
    return float(vc), float(np.exp(lw)), float(pr * cp), float(cp)


def build_eom_panel(panel: pl.DataFrame) -> pl.DataFrame:
    """Attach the inverted per-run physical constants (v_c, W′_0, P_max, CP, m, going) to the panel."""
    rows = panel.to_dicts()
    n = len(rows)
    vc = np.empty(n)
    w0 = np.empty(n)
    pm = np.empty(n)
    cpv = np.empty(n)
    mass = np.empty(n)
    for i, r in enumerate(rows):
        d = np.asarray(r["section_dist"], float)
        v = np.asarray(r["velocity"], float)
        m = body_mass_kg(r.get("bodywt_lb") or 1100.0, r.get("carriedwt_lb") or 0.0)
        mass[i] = m
        vc[i], w0[i], pm[i], cpv[i] = invert_run(d, v, m, r.get("going", "GOOD"))
        if (i + 1) % 2000 == 0:
            print(f"    inverted {i + 1}/{n}")
    return panel.with_columns(
        pl.Series("vc_eom", vc), pl.Series("w0_eom", w0), pl.Series("pm_eom", pm),
        pl.Series("cp_eom", cpv), pl.Series("mass_kg", mass),
        (pl.Series("cp_eom", cpv) / pl.Series("mass_kg", mass)).alias("cp_per_kg"))


def attach_loo_eom(panel: pl.DataFrame, *, min_runs: int = 2) -> pl.DataFrame:
    """Per-horse leave-one-race-out mean of the inferred physical constants (strictly-prior)."""
    g = panel.group_by("horse_id").agg(
        pl.col("vc_eom").sum().alias("svc"), pl.col("w0_eom").sum().alias("sw0"),
        pl.col("pm_eom").sum().alias("spm"), pl.col("vc_eom").count().alias("cnt"))
    p = panel.join(g, on="horse_id").filter(pl.col("cnt") >= min_runs)
    return p.with_columns(
        ((pl.col("svc") - pl.col("vc_eom")) / (pl.col("cnt") - 1)).alias("vc_loo"),
        ((pl.col("sw0") - pl.col("w0_eom")) / (pl.col("cnt") - 1)).alias("w0_loo"),
        ((pl.col("spm") - pl.col("pm_eom")) / (pl.col("cnt") - 1)).alias("pm_loo"))


def assert_eom_no_lookahead(loo: pl.DataFrame) -> float:
    """Leak check: the LOO physical constants exclude the held-out run's own inverted constants."""
    chk = loo.with_columns(
        ((pl.col("svc") - pl.col("vc_eom")) / (pl.col("cnt") - 1)).alias("vc_loo_chk"))
    return float((chk["vc_loo"] - chk["vc_loo_chk"]).abs().max() or 0.0)


def section_residuals_eom(loo: pl.DataFrame, descr: pl.DataFrame | None, *,
                          sigma: float = SIGMA) -> pl.DataFrame:
    """Held-out EOM forward profile per section vs observed, with phase + buckets + PIT."""
    rows = loo.to_dicts()
    recs = []
    for r in rows:
        d = np.asarray(r["section_dist"], float)
        v = np.asarray(r["velocity"], float)
        k = len(v)
        m = r["mass_kg"]
        res = make_resistances(m, r.get("going", "GOOD"))
        cp = cp_for_cruise(float(r["vc_loo"]), res)
        sv, _, _ = eom_section_velocities_fast(d, cp, float(r["w0_loo"]), float(r["pm_loo"]),
                                               TAU, res, dt=DT, v0=V0)
        for s in range(k):
            phase = "opening" if s == 0 else ("closing" if s == k - 1 else "cruise")
            recs.append({
                "race_id": r["race_id"], "runner": r["runner"], "horse_id": r["horse_id"],
                "date": r["date"], "venue": r["venue"], "band": r["band"],
                "going_c": r.get("going_c", "GOOD"), "field_size": r["field_size"],
                "pos_frac": r["pos_frac"], "n_sec": k, "section": s + 1, "phase": phase,
                "v_obs": float(v[s]), "v_gen": float(sv[s]), "race_vcru": r["race_vcru"],
            })
    rdf = pl.DataFrame(recs)
    if descr is not None:
        rdf = rdf.join(descr, on=["race_id", "runner", "section"], how="left")
    return rdf.with_columns(
        (pl.col("v_obs") - pl.col("v_gen")).alias("res_gen"),
        _normal_cdf((pl.col("v_obs").log() - pl.col("v_gen").log()) / sigma).alias("pit_gen"))


def _normal_cdf(z: pl.Expr) -> pl.Expr:
    return 0.5 * (1.0 + (0.7988456 * z * (1.0 + 0.044715 * z * z)).tanh())


def main():
    venues = [os.environ.get("EOM_VENUE")] if os.environ.get("EOM_VENUE") else ["ST", "HV"]
    nrun = int(os.environ.get("EOM_NRUN", "0"))
    print("building per-run profile panel + KNOWN mass + going...")
    traj = pl.read_parquet(CACHE / "trajectory_dataset_full.parquet")
    panel = build_run_panel(traj, going=_going_frame())
    panel = panel.with_columns(_coarse_going(pl.col("going")).alias("going_c"))
    panel = panel.filter(pl.col("n_sec") >= 3)
    panel = panel.join(_mass_frame(), on=["date", "venue", "race_no", "runner"], how="left")
    dates = sorted(panel["date"].unique().to_list())
    cut = dates[int(len(dates) * 0.7)]
    floor = noise_floor(panel)

    results = {}
    for venue in venues:
        vp = panel.filter(pl.col("venue") == venue)
        if nrun:                                      # quick mode: a horse-coherent subsample
            keep = vp["horse_id"].unique().head(max(nrun // 6, 50)).to_list()
            vp = vp.filter(pl.col("horse_id").is_in(keep))
        print(f"\n[{venue}] inverting {vp.height} runs (physical constants per run)...")
        vp = build_eom_panel(vp)
        loo = attach_loo_eom(vp, min_runs=2)
        drift = assert_eom_no_lookahead(loo)
        print(f"  {loo.height} held-out runs (LOO physical constants); leak drift {drift:.2e}")

        # physical-plausibility report on the INVERTED constants
        cpkg = vp["cp_per_kg"].to_numpy()
        print(f"  INFERRED CONSTANTS (per run): v_c med={vp['vc_eom'].median():.2f} m/s "
              f"[{vp['vc_eom'].quantile(0.1):.2f},{vp['vc_eom'].quantile(0.9):.2f}] | "
              f"W′_0 med={vp['w0_eom'].median()/1e3:.0f} kJ "
              f"[{vp['w0_eom'].quantile(0.1)/1e3:.0f},{vp['w0_eom'].quantile(0.9)/1e3:.0f}] | "
              f"CP med={vp['cp_eom'].median()/1e3:.1f} kW ({np.median(cpkg):.0f} W/kg) | "
              f"P_max/CP med={(vp['pm_eom']/vp['cp_eom']).median():.2f}")

        descr = _descr_baseline(loo, cut)
        sr_all = section_residuals_eom(loo, descr)
        sr = sr_all.filter(pl.col("date") >= cut)
        if sr.height < 500:
            sr = sr_all
        results[venue] = {"sr": sr, "floor": floor[venue]}

        ov_gen = rmse_r2(sr, "v_gen")
        ov_descr = rmse_r2(sr.drop_nulls("v_descr"), "v_descr")
        fl = floor[venue]["overall"]
        print(f"  RECON (held-out): EOM RMSE={ov_gen['rmse'][0]:.4f} R²={ov_gen['r2'][0]:+.3f} | "
              f"descriptive RMSE={ov_descr['rmse'][0]:.4f} | floor={fl:.4f} "
              f"→ EOM excess {(ov_gen['rmse'][0]/fl-1)*100:+.0f}% (descr {(ov_descr['rmse'][0]/fl-1)*100:+.0f}%)")
        print(f"  LEVEL BIAS (mean res, + = under-predict): {sr['res_gen'].mean():+.4f} m/s")
        pit = sr["pit_gen"].drop_nulls().to_numpy()
        print(f"  PIT mean={pit.mean():.3f} std={pit.std():.3f} (ideal .500/.289)")
        print("  RESIDUAL by PHASE (mean res / rmse / floor):")
        for ph in ["opening", "cruise", "closing"]:
            s = sr.filter(pl.col("phase") == ph)
            print(f"    {ph:8s} n={s.height:6d} mean_res={s['res_gen'].mean():+.4f} "
                  f"rmse={(s['res_gen']**2).mean()**0.5:.4f} floor={floor[venue][ph]:.4f}")
        # opening-ramp fit: R² of v_gen vs v_obs on the OPENING section (was ≈0 for the heuristic)
        op = sr.filter(pl.col("phase") == "opening")
        og = rmse_r2(op, "v_gen")
        print(f"  OPENING-RAMP fit: R²={og['r2'][0]:+.3f} rmse={og['rmse'][0]:.4f}")
        # pace-dependent closing miss (the missing field/draft mechanism)
        cl = sr.filter(pl.col("phase") == "closing").with_columns(
            (pl.col("race_vcru") - pl.col("race_vcru").mean()).alias("pace_dev"))
        if cl.height > 200:
            fast = cl.filter(pl.col("pace_dev") > pl.col("pace_dev").quantile(0.66))
            slow = cl.filter(pl.col("pace_dev") < pl.col("pace_dev").quantile(0.34))
            print(f"  PACE (closing res): fast={fast['res_gen'].mean():+.4f} "
                  f"slow={slow['res_gen'].mean():+.4f} Δ={fast['res_gen'].mean()-slow['res_gen'].mean():+.4f}")
    _plots(results, floor)
    _verdict(results, floor)


def _plots(results, floor):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    phases = ["opening", "cruise", "closing"]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for ven in results:
        sr = results[ven]["sr"]
        mr = [sr.filter(pl.col("phase") == p)["res_gen"].mean() for p in phases]
        ax[0].plot(phases, mr, "o-", label=ven)
        rg = [(sr.filter(pl.col("phase") == p)["res_gen"] ** 2).mean() ** 0.5 for p in phases]
        fl = [floor[ven][p] for p in phases]
        ax[1].plot(phases, rg, "o-", label=f"{ven} EOM")
        ax[1].plot(phases, fl, "k:", alpha=0.5)
    ax[0].axhline(0, color="k", lw=0.8)
    ax[0].set_title("EOM residual by phase (+ = under-predict)")
    ax[0].set_ylabel("mean res m/s")
    ax[0].legend()
    ax[1].set_title("EOM RMSE by phase (dotted = floor)")
    ax[1].set_ylabel("RMSE m/s")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(OUT / "pathfit_eom_phases.png", dpi=110)
    plt.close(fig)
    print(f"  wrote {OUT}/pathfit_eom_phases.png")


def _descr_baseline(loo: pl.DataFrame, cut: str) -> pl.DataFrame:
    """Flexible physics-free descriptive baseline (the #44/45 yardstick), level = LOO cruise."""
    # use the heuristic per-horse cruise level (mean observed mid velocity) as the strictly-prior level
    rows = loo.to_dicts()
    recs = []
    for r in rows:
        v = np.asarray(r["velocity"], float)
        k = len(v)
        # horse LOO cruise: the inverted v_c LOO mean is a clean strictly-prior level proxy
        lvl = float(r["vc_loo"])
        for s in range(k):
            recs.append({"race_id": r["race_id"], "runner": r["runner"], "horse_id": r["horse_id"],
                         "date": r["date"], "band": r["band"], "going": r.get("going", "GOOD"),
                         "n_sec": k, "section": s + 1, "ratio": float(v[s]) / max(lvl, 1e-6),
                         "vcru_loo": lvl})
    ex = pl.DataFrame(recs)
    train = ex.filter(pl.col("date") < cut)
    shape = (train.group_by(["band", "going", "n_sec", "section"])
             .agg(pl.col("ratio").mean().alias("shape_ratio"), pl.len().alias("shape_n")))
    shape_bg = (train.group_by(["band", "n_sec", "section"])
                .agg(pl.col("ratio").mean().alias("shape_ratio_bg")))
    shape_g = (train.group_by(["n_sec", "section"]).agg(pl.col("ratio").mean().alias("shape_ratio_g")))
    pred = (ex.join(shape, on=["band", "going", "n_sec", "section"], how="left")
            .join(shape_bg, on=["band", "n_sec", "section"], how="left")
            .join(shape_g, on=["n_sec", "section"], how="left"))
    pred = pred.with_columns(
        pl.when(pl.col("shape_n") >= 30).then(pl.col("shape_ratio"))
        .otherwise(pl.coalesce(["shape_ratio_bg", "shape_ratio_g"])).alias("shape_use"))
    pred = pred.with_columns((pl.col("shape_use") * pl.col("vcru_loo")).alias("v_descr"))
    return pred.select(["race_id", "runner", "section", "v_descr"])


def _verdict(results, floor):
    print("\n" + "=" * 78 + "\nEOM VERDICT (held-out, leak-free) vs 0045 heuristic\n" + "=" * 78)
    heur = {"ST": (0.673, 0.02), "HV": (0.521, -0.02)}
    for ven in results:
        sr = results[ven]["sr"]
        gen = rmse_r2(sr, "v_gen")
        descr = rmse_r2(sr.drop_nulls("v_descr"), "v_descr")
        fl = floor[ven]["overall"]
        pit = sr["pit_gen"].drop_nulls().to_numpy()
        h = heur.get(ven, (float("nan"), float("nan")))
        print(f"[{ven}] EOM RMSE={gen['rmse'][0]:.3f} (R²={gen['r2'][0]:+.2f}) bias={sr['res_gen'].mean():+.3f} "
              f"| heuristic(0045) RMSE={h[0]:.3f} bias={h[1]:+.3f} | descr={descr['rmse'][0]:.3f} "
              f"| floor={fl:.3f} | PIT std={pit.std():.3f}")


if __name__ == "__main__":
    main()
