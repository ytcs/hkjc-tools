"""Queue item #33 — empirically validate CV(T) ∝ 1/√L and characterize the D(L) floor.

DESCRIPTIVE population study (NOT a predictive feature → no as-of constraint needed).

Theory (THEORY §3e): from the homogeneous Wald first-passage law with λ = L²/2D, μ = L/v,
    Var(T) = μ³/λ = 2D·L/v³   ⇒   CV(T) = sd(T)/mean(T) = √(2D/(L·v)) ∝ 1/√L.
Longer races → relatively LESS finishing-time variance. Slope of log CV on log L should be −0.5.
Refinement: a distance-INDEPENDENT chunk of variance (bad break, a check ≈ same seconds at any L)
implies D(L) = D_0 + D_phase, i.e. a floor that lifts CV above the 1/√L curve in the sprints.

Method:
  - within-race CV removes the race mean (= field strength + going), leaving the residual
    stochastic spread (exactly the "de-mean by race" the task asks for).
  - aggregate per bucket L × Venue × surface (turf/AWT).
  - regress log CV on log L (race-as-unit), report slope ± CI, R².
  - back out D per bucket from CV² = 2D/(L·v), v = L/mean(T).

Run: uv run python studies/cv_distance_law.py
"""

from __future__ import annotations

import numpy as np
import polars as pl

OUT_PNG = "studies/out/cv_distance_law.png"


def load() -> pl.DataFrame:
    df = pl.read_parquet("data_cache/results_rich.parquet")
    df = df.with_columns(
        (
            pl.col("date").cast(pl.Utf8)
            + "_"
            + pl.col("Venue")
            + "_"
            + pl.col("RaceNumber").cast(pl.Utf8)
        ).alias("race_id"),
        pl.when(pl.col("Course").str.contains("ALL WEATHER"))
        .then(pl.lit("AWT"))
        .otherwise(pl.lit("turf"))
        .alias("surface"),
        pl.col("RaceDistance").cast(pl.Int64).alias("L"),
    )
    # clean: positive finish times, field >= 5 (need a meaningful within-race sd)
    df = df.filter(pl.col("FinishTime") > 0)
    return df


def per_race_cv(df: pl.DataFrame) -> pl.DataFrame:
    """within-race CV(T): de-means by race (removes field strength + going)."""
    g = df.group_by("race_id", "Venue", "surface", "L").agg(
        [
            pl.col("FinishTime").mean().alias("mean_T"),
            pl.col("FinishTime").std().alias("sd_T"),  # sample sd (ddof=1)
            pl.len().alias("n"),
        ]
    )
    g = g.filter(pl.col("n") >= 5, pl.col("sd_T") > 0)
    g = g.with_columns((pl.col("sd_T") / pl.col("mean_T")).alias("cv"))
    # mean velocity for this race (m/s)
    g = g.with_columns((pl.col("L") / pl.col("mean_T")).alias("v"))
    return g


def ols(x: np.ndarray, y: np.ndarray):
    """slope, intercept, slope SE, R² for y = a + b x."""
    n = len(x)
    X = np.column_stack([np.ones(n), x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ beta
    resid = y - yhat
    dof = n - 2
    sigma2 = float(resid @ resid) / dof
    cov = sigma2 * np.linalg.inv(X.T @ X)
    se_b = float(np.sqrt(cov[1, 1]))
    ss_tot = float(((y - y.mean()) ** 2).sum())
    ss_res = float((resid**2).sum())
    r2 = 1.0 - ss_res / ss_tot
    return float(beta[1]), float(beta[0]), se_b, r2


def main() -> None:
    import os

    os.makedirs("studies/out", exist_ok=True)
    df = load()
    g = per_race_cv(df)

    # ---- bucket aggregation (Venue × surface × L) ----
    buckets = (
        g.group_by("Venue", "surface", "L")
        .agg(
            [
                pl.col("cv").mean().alias("cv_mean"),
                pl.col("cv").median().alias("cv_median"),
                pl.col("cv").std().alias("cv_sd"),
                pl.col("v").mean().alias("v_mean"),
                pl.col("mean_T").mean().alias("T_mean"),
                pl.len().alias("n_races"),
            ]
        )
        .sort("Venue", "surface", "L")
    )
    # back out D per bucket: CV² = 2D/(L v) ⇒ D = CV²·L·v / 2  (units m²/s)
    buckets = buckets.with_columns(
        (pl.col("cv_mean") ** 2 * pl.col("L") * pl.col("v_mean") / 2.0).alias("D")
    )

    print("=" * 100)
    print("PER-BUCKET (Venue × surface × distance L)  — within-race CV(T) and implied D")
    print("=" * 100)
    with pl.Config(tbl_rows=40, tbl_cols=20, float_precision=5):
        print(buckets)

    # ---- the scaling regression: log CV on log L (race as the unit, leak-free-irrelevant) ----
    # restrict to buckets with enough races; pool turf only first (cleanest, AWT is more diffusive)
    def regress(sub: pl.DataFrame, label: str):
        sub = sub.filter(pl.col("cv") > 0)
        x = np.log(sub["L"].to_numpy().astype(float))
        y = np.log(sub["cv"].to_numpy())
        b, a, se, r2 = ols(x, y)
        lo, hi = b - 1.96 * se, b + 1.96 * se
        print(
            f"{label:28s}  slope={b:+.3f}  95%CI=[{lo:+.3f},{hi:+.3f}]  "
            f"R²={r2:.3f}  n_races={len(x)}"
        )
        return b, lo, hi, r2

    print("\n" + "=" * 100)
    print("SCALING: regress log CV(T) on log L  (theory predicts slope = −0.5)")
    print("=" * 100)
    regress(g, "ALL (turf+AWT, both venues)")
    regress(g.filter(pl.col("surface") == "turf"), "TURF only")
    regress(g.filter(pl.col("surface") == "AWT"), "AWT only")
    regress(g.filter((pl.col("surface") == "turf") & (pl.col("Venue") == "ST")), "TURF ST")
    regress(g.filter((pl.col("surface") == "turf") & (pl.col("Venue") == "HV")), "TURF HV")
    # weighted bucket-level regression (each bucket = one point, weighted by n_races)
    bsub = buckets.filter(pl.col("n_races") >= 30)
    xb = np.log(bsub["L"].to_numpy().astype(float))
    yb = np.log(bsub["cv_mean"].to_numpy())
    bb, ab, seb, r2b = ols(xb, yb)
    print(
        f"{'BUCKET-LEVEL (n_races≥30)':28s}  slope={bb:+.3f}  "
        f"95%CI=[{bb - 1.96 * seb:+.3f},{bb + 1.96 * seb:+.3f}]  R²={r2b:.3f}  n_buckets={len(xb)}"
    )

    # ---- D(L) trend: regress log D on log L (theory's simplest form: D constant ⇒ slope 0) ----
    print("\n" + "=" * 100)
    print("D(L) TREND: regress log D on log L  (constant-D theory ⇒ slope 0; floor ⇒ D rises with L)")
    print("=" * 100)
    for surf in ["turf", "AWT"]:
        s = buckets.filter((pl.col("surface") == surf) & (pl.col("n_races") >= 30))
        if s.height < 3:
            continue
        xd = np.log(s["L"].to_numpy().astype(float))
        yd = np.log(s["D"].to_numpy())
        bd, ad, sed, r2d = ols(xd, yd)
        print(
            f"{surf:6s}  D-slope={bd:+.3f}  95%CI=[{bd - 1.96 * sed:+.3f},{bd + 1.96 * sed:+.3f}]  "
            f"R²={r2d:.3f}  D@min={s['D'].to_numpy()[0]:.3f}  D@max={s['D'].to_numpy()[-1]:.3f}"
        )

    # ---- plot ----
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # panel A: CV vs L with 1/sqrt(L) reference, by venue/surface
    ax = axes[0]
    markers = {("ST", "turf"): "o", ("HV", "turf"): "s", ("ST", "AWT"): "^"}
    colors = {("ST", "turf"): "C0", ("HV", "turf"): "C1", ("ST", "AWT"): "C2"}
    plotted = buckets.filter(pl.col("n_races") >= 20)
    for (ven, surf), mk in markers.items():
        sub = plotted.filter((pl.col("Venue") == ven) & (pl.col("surface") == surf)).sort("L")
        if sub.height == 0:
            continue
        L = sub["L"].to_numpy().astype(float)
        cv = sub["cv_mean"].to_numpy()
        sd = sub["cv_sd"].to_numpy()
        nn = sub["n_races"].to_numpy()
        ax.errorbar(
            L, cv * 100, yerr=(sd / np.sqrt(nn)) * 100, fmt=mk, color=colors[(ven, surf)],
            label=f"{ven} {surf}", capsize=3, ms=7,
        )
    # 1/sqrt(L) reference anchored at the turf grand mean
    turf = plotted.filter(pl.col("surface") == "turf")
    Lref = np.linspace(950, 2450, 100)
    # anchor at median L of turf
    anchorL = 1400.0
    anchor_cv = (
        turf.filter(pl.col("L") == 1400)["cv_mean"].mean()
        if turf.filter(pl.col("L") == 1400).height
        else turf["cv_mean"].mean()
    )
    ax.plot(
        Lref, anchor_cv * np.sqrt(anchorL / Lref) * 100, "k--", lw=1.5,
        label=r"$1/\sqrt{L}$ ref (anchor 1400m)",
    )
    ax.set_xlabel("race distance L (m)")
    ax.set_ylabel("within-race CV(T)  (%)")
    ax.set_title("A.  CV(T) vs distance  (points = bucket mean ± SE)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # panel B: log-log with fitted slope
    ax = axes[1]
    sub = g.filter(pl.col("surface") == "turf")
    xx = np.log(sub["L"].to_numpy().astype(float))
    yy = np.log(sub["cv"].to_numpy())
    # hexbin for the cloud
    ax.hexbin(xx, yy, gridsize=40, cmap="Blues", mincnt=1, alpha=0.7)
    b, a, se, r2 = ols(xx, yy)
    xline = np.linspace(xx.min(), xx.max(), 50)
    ax.plot(xline, a + b * xline, "r-", lw=2, label=f"fit slope={b:+.3f}±{1.96 * se:.3f}")
    # theory slope -0.5 anchored at data centroid
    yc = yy.mean()
    xc = xx.mean()
    ax.plot(xline, yc - 0.5 * (xline - xc), "k--", lw=2, label="theory slope −0.5")
    ax.set_xlabel("log L")
    ax.set_ylabel("log CV(T)")
    ax.set_title(f"B.  log–log scaling (turf, per-race)  R²={r2:.3f}")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    fig.suptitle(
        "Queue #33 — distance-law of finishing-time variance: CV(T) ∝ 1/√L  (descriptive)",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=120)
    print(f"\nsaved plot → {OUT_PNG}")


if __name__ == "__main__":
    main()
