"""FOUNDATION evaluation (journal DRAFT 0044) — is the generative path model a GOOD path
model in ABSOLUTE terms, before any market/win gate?

Runs the five-axis path-fit evaluation of ``hkjc.model.pathfit`` on the full trajectory
surface, leak-free held-out (leave-one-race-out within horse; train-only descriptive baseline &
noise floor). NO market odds, NO win gate, NO pool book — pure profile reconstruction quality.

    uv run --group bayes python studies/pathfit_foundation.py

Writes plots to studies/out/pathfit_*.png and prints the structured VERDICT numbers.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from hkjc.model.pathfit import (
    assert_path_no_lookahead, attach_loo_theta, build_run_panel, descriptive_baseline,
    noise_floor, rmse_r2, section_residual_frame,
)

CACHE = Path("data_cache")
OUT = Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)
SIGMA = 0.06   # the production log-velocity observation noise (profile_loglik default)


def _going_frame() -> pl.DataFrame:
    rr = pl.read_parquet(CACHE / "results_rich.parquet")
    g = (rr.with_columns(
            pl.col("Date").cast(pl.Utf8).str.replace_all("/", "-").alias("date"),
            pl.col("Venue").cast(pl.Utf8).alias("venue"),
            pl.col("RaceNumber").cast(pl.Int32).alias("race_no"),
            pl.col("Going").cast(pl.Utf8).alias("going"),
            pl.when(pl.col("Course").str.contains("ALL WEATHER"))
              .then(pl.lit("awt")).otherwise(pl.lit("turf")).alias("surface"))
         .select("date", "venue", "race_no", "going", "surface")
         .unique(["date", "venue", "race_no"]))
    return g


def _coarse_going(expr: pl.Expr) -> pl.Expr:
    # collapse the many going strings into GOOD / SOFT-ish / FIRM-ish / WET(awt)
    return (pl.when(expr.str.contains("FIRM")).then(pl.lit("GOOD-FIRM"))
            .when(expr.str.contains("YIELD") | expr.str.contains("SOFT") | expr.str.contains("HEAVY"))
            .then(pl.lit("SOFT+"))
            .when(expr.str.contains("WET") | expr.str.contains("SEALED")).then(pl.lit("WET-AWT"))
            .when(expr.str.contains("GOOD")).then(pl.lit("GOOD"))
            .otherwise(pl.lit("OTHER")))


def main():
    print("building per-run profile panel (v_c, W′_0, a + conditions)...")
    traj = pl.read_parquet(CACHE / "trajectory_dataset_full.parquet")
    panel = build_run_panel(traj, going=_going_frame())
    panel = panel.with_columns(_coarse_going(pl.col("going")).alias("going_c"))
    panel = panel.filter(pl.col("n_sec") >= 3)
    print(f"  {panel.height} runs | a median {panel['a_run'].median():.3f}")

    floor = noise_floor(panel)

    # walk-forward style train/test split for the descriptive baseline + floor population means:
    # use a strict date cutoff (last 30% of dates = held-out test); descriptive shape TRAIN-only.
    dates = sorted(panel["date"].unique().to_list())
    cut = dates[int(len(dates) * 0.7)]
    print(f"  train < {cut} <= test")

    results = {}
    for venue in ["ST", "HV"]:
        vp = panel.filter(pl.col("venue") == venue)
        loo = attach_loo_theta(vp, min_runs=2)
        drift = assert_path_no_lookahead(loo)
        print(f"\n[{venue}] {loo.height} held-out runs (LOO θ); leak drift {drift:.2e}")

        # descriptive baseline shape fit TRAIN-only, predicted on all (held-out level strictly-prior)
        descr = descriptive_baseline(loo, train_cut=cut)
        sr_all = section_residual_frame(loo, descr=descr, sigma=SIGMA).df
        # HELD-OUT evaluation set = test dates only (descriptive shape was fit on train only)
        sr = sr_all.filter(pl.col("date") >= cut)
        if sr.height < 500:
            sr = sr_all  # fallback if the test slice is too thin
        results[venue] = {"sr": sr, "sr_all": sr_all, "floor": floor[venue], "loo": loo}

        # ---- (1) absolute reconstruction + gap to floor
        ov_gen = rmse_r2(sr, "v_gen")
        ov_noa = rmse_r2(sr, "v_gen_noa")
        ov_descr = rmse_r2(sr.drop_nulls("v_descr"), "v_descr")
        fl = floor[venue]["overall"]
        print(f"  RECON (held-out, m/s):  generative RMSE={ov_gen['rmse'][0]:.4f} "
              f"R²={ov_gen['r2'][0]:+.3f} | a:=1 RMSE={ov_noa['rmse'][0]:.4f} | "
              f"descriptive RMSE={ov_descr['rmse'][0]:.4f} R²={ov_descr['r2'][0]:+.3f}")
        print(f"  NOISE FLOOR (within-horse) RMSE={fl:.4f}  "
              f"→ gen excess over floor {ov_gen['rmse'][0]-fl:+.4f} ({(ov_gen['rmse'][0]/fl-1)*100:+.0f}%)  "
              f"| descr excess {ov_descr['rmse'][0]-fl:+.4f}")

        # ---- (2) calibration / PIT
        pit = sr["pit_gen"].drop_nulls().to_numpy()
        cov68 = float(np.mean((pit > 0.16) & (pit < 0.84)))
        cov90 = float(np.mean((pit > 0.05) & (pit < 0.95)))
        print(f"  CALIBRATION: PIT mean={pit.mean():.3f} std={pit.std():.3f} "
              f"(ideal 0.500/0.289) | 68% band coverage={cov68:.3f} 90%={cov90:.3f}  "
              f"→ {'OVER' if pit.std() < 0.289 else 'UNDER'}-dispersed")

        # ---- (3) residual structure
        print("  RESIDUAL by PHASE (mean res, m/s; + = model UNDER-predicts):")
        for ph in ["opening", "cruise", "closing"]:
            s = sr.filter(pl.col("phase") == ph)
            print(f"    {ph:8s} n={s.height:6d} mean_res={s['res_gen'].mean():+.4f} "
                  f"std={s['res_gen'].std():.4f} rmse={(s['res_gen']**2).mean()**0.5:.4f} "
                  f"floor={floor[venue][ph]:.4f}")
        print("  RESIDUAL by BAND:")
        for row in rmse_r2(sr, "v_gen", by=["band"]).iter_rows(named=True):
            mb = sr.filter(pl.col("band") == row["band"])["res_gen"].mean()
            print(f"    {row['band']:8s} n={row['n']:6d} rmse={row['rmse']:.4f} "
                  f"r2={row['r2']:+.3f} mean_res={mb:+.4f}")
        print("  RESIDUAL by GOING:")
        for row in rmse_r2(sr, "v_gen", by=["going_c"]).iter_rows(named=True):
            print(f"    {row['going_c']:10s} n={row['n']:6d} rmse={row['rmse']:.4f} r2={row['r2']:+.3f}")
        # run-style tertiles
        sr2 = sr.with_columns(
            pl.when(pl.col("pos_frac") <= 0.34).then(pl.lit("front"))
            .when(pl.col("pos_frac") >= 0.67).then(pl.lit("closer"))
            .otherwise(pl.lit("mid")).alias("style"))
        print("  RESIDUAL by RUN-STYLE (front/mid/closer):")
        for st in ["front", "mid", "closer"]:
            s = sr2.filter(pl.col("style") == st)
            print(f"    {st:7s} n={s.height:6d} mean_res={s['res_gen'].mean():+.4f} "
                  f"rmse={(s['res_gen']**2).mean()**0.5:.4f}")
        # PACE dependence: is the closing miss worse in fast-pace races?
        sr3 = sr.filter(pl.col("phase") == "closing").with_columns(
            (pl.col("race_vcru") - pl.col("race_vcru").mean()).alias("pace_dev"))
        if sr3.height > 200:
            fast = sr3.filter(pl.col("pace_dev") > pl.col("pace_dev").quantile(0.66))
            slow = sr3.filter(pl.col("pace_dev") < pl.col("pace_dev").quantile(0.34))
            print(f"  PACE (closing res): fast-pace mean_res={fast['res_gen'].mean():+.4f} "
                  f"vs slow-pace {slow['res_gen'].mean():+.4f} "
                  f"(Δ={fast['res_gen'].mean()-slow['res_gen'].mean():+.4f})")
        # field size
        print("  RESIDUAL by FIELD SIZE bucket:")
        srf = sr.with_columns(
            pl.when(pl.col("field_size") <= 8).then(pl.lit("small≤8"))
            .when(pl.col("field_size") <= 11).then(pl.lit("med9-11"))
            .otherwise(pl.lit("big≥12")).alias("fsb"))
        for fb in ["small≤8", "med9-11", "big≥12"]:
            s = srf.filter(pl.col("fsb") == fb)
            if s.height:
                print(f"    {fb:8s} n={s.height:6d} rmse={(s['res_gen']**2).mean()**0.5:.4f} "
                      f"mean_res={s['res_gen'].mean():+.4f}")

    _plots(results, floor, panel)
    _verdict(results, floor)


def _plots(results, floor, panel):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # ---- Plot A: predicted vs observed profiles (good + bad), ST 4-section
    vp = panel.filter((pl.col("venue") == "ST") & (pl.col("n_sec") == 4))
    loo = attach_loo_theta(vp, min_runs=2)
    from hkjc.model.energetics_dynamic import expected_profile
    rows = loo.to_dicts()
    errs = []
    for r in rows:
        d = np.asarray(r["section_dist"], float)
        v = np.asarray(r["velocity"], float)
        g = expected_profile(d, r["vc_loo"], r["wp_loo"], r["a_loo"])
        errs.append((float(np.sqrt(np.mean((v - g) ** 2))), r))
    errs.sort(key=lambda x: x[0])
    good = errs[len(errs) // 20]      # 5th pct (good)
    bad = errs[-len(errs) // 20]      # 95th pct (bad)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for a_, (rmse, r), title in [(ax[0], good, "GOOD fit (5th pct)"), (ax[1], bad, "BAD fit (95th pct)")]:
        d = np.asarray(r["section_dist"], float)
        v = np.asarray(r["velocity"], float)
        g = expected_profile(d, r["vc_loo"], r["wp_loo"], r["a_loo"])
        xs = range(1, len(v) + 1)
        a_.plot(xs, v, "o-", label="observed")
        a_.plot(xs, g, "s--", label="generative")
        a_.fill_between(xs, g * np.exp(-SIGMA), g * np.exp(SIGMA), alpha=0.2, label="±1σ band")
        a_.set_title(f"{title}  RMSE={rmse:.3f} m/s")
        a_.set_xlabel("section")
        a_.set_ylabel("m/s")
        a_.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "pathfit_profiles.png", dpi=110)
    plt.close(fig)

    # ---- Plot B: residual-by-phase + RMSE-vs-floor + PIT, both venues
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    phases = ["opening", "cruise", "closing"]
    for ven, c in [("ST", "C0"), ("HV", "C1")]:
        sr = results[ven]["sr"]
        means = [sr.filter(pl.col("phase") == p)["res_gen"].mean() for p in phases]
        ax[0].plot(phases, means, "o-", color=c, label=ven)
    ax[0].axhline(0, color="k", lw=0.8)
    ax[0].set_ylabel("mean residual m/s (+ = under-predict)")
    ax[0].set_title("Residual by phase")
    ax[0].legend()
    # RMSE vs floor
    x = np.arange(len(phases))
    w = 0.35
    for i, (ven, off) in enumerate([("ST", -w / 2), ("HV", w / 2)]):
        sr = results[ven]["sr"]
        rg = [(sr.filter(pl.col("phase") == p)["res_gen"] ** 2).mean() ** 0.5 for p in phases]
        fl = [floor[ven][p] for p in phases]
        ax[1].bar(x + off, rg, w, label=f"{ven} gen", alpha=0.8)
        ax[1].plot(x + off, fl, "k_", ms=18, mew=2)
    ax[1].set_xticks(x)
    ax[1].set_xticklabels(phases)
    ax[1].set_ylabel("RMSE m/s")
    ax[1].set_title("Generative RMSE (bars) vs noise floor (—)")
    ax[1].legend()
    # PIT histogram
    for ven, c in [("ST", "C0"), ("HV", "C1")]:
        pit = results[ven]["sr"]["pit_gen"].drop_nulls().to_numpy()
        ax[2].hist(pit, bins=20, histtype="step", density=True, label=ven, color=c)
    ax[2].axhline(1.0, color="k", ls="--", lw=1)
    ax[2].set_title("PIT (flat=calibrated)")
    ax[2].set_xlabel("PIT")
    ax[2].legend()
    fig.tight_layout()
    fig.savefig(OUT / "pathfit_residual_calibration.png", dpi=110)
    plt.close(fig)

    # ---- Plot C: generative vs descriptive gap by phase
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for j, ven in enumerate(["ST", "HV"]):
        sr = results[ven]["sr"].drop_nulls("v_descr")
        g = [(sr.filter(pl.col("phase") == p)["res_gen"] ** 2).mean() ** 0.5 for p in phases]
        de = [((sr.filter(pl.col("phase") == p)["v_obs"]
                - sr.filter(pl.col("phase") == p)["v_descr"]) ** 2).mean() ** 0.5 for p in phases]
        fl = [floor[ven][p] for p in phases]
        ax[j].plot(phases, g, "o-", label="generative (physics)")
        ax[j].plot(phases, de, "s--", label="descriptive (flexible)")
        ax[j].plot(phases, fl, "k:", label="noise floor")
        ax[j].set_title(f"{ven}: physics vs flexible vs floor")
        ax[j].set_ylabel("RMSE m/s")
        ax[j].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "pathfit_generative_vs_descriptive.png", dpi=110)
    plt.close(fig)
    print(f"\n  wrote {OUT}/pathfit_profiles.png, pathfit_residual_calibration.png, "
          f"pathfit_generative_vs_descriptive.png")


def _verdict(results, floor):
    print("\n" + "=" * 78 + "\nVERDICT SUMMARY (held-out, leak-free; m/s & normalized)\n" + "=" * 78)
    for ven in ["ST", "HV"]:
        sr = results[ven]["sr"]
        gen = rmse_r2(sr, "v_gen")
        descr = rmse_r2(sr.drop_nulls("v_descr"), "v_descr")
        fl = floor[ven]["overall"]
        pit = sr["pit_gen"].drop_nulls().to_numpy()
        print(f"[{ven}] gen RMSE={gen['rmse'][0]:.3f} (R²={gen['r2'][0]:+.2f}) | "
              f"floor={fl:.3f} | excess={(gen['rmse'][0]/fl-1)*100:+.0f}% | "
              f"descr RMSE={descr['rmse'][0]:.3f} | PIT std={pit.std():.3f}")


if __name__ == "__main__":
    main()
