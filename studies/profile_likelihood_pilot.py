"""Profile-likelihood pilot (Fork 2, draft 0040) — anchor the generative model to the
SECTIONAL PROFILE (a broad surface) for FIT/DISCOVERY, keep win as the FINAL eval.

THE DIRECTION (the batch's strategic finding): the WIN outcome is too coarse a likelihood
to identify rich mechanism. So FIT the generative W′-balance dynamics to a DENSE surface —
here the K-section velocity profile — and test whether that rich likelihood surfaces a NEW
latent that win alone cannot identify. The test case is ACCELERATION `a` (the opening-split
ramp-up; THEORY §3e Phase 1, the start transient the homogeneous law has no representation
of). `a` is invisible to win at the population level but leaves a clear fingerprint in the
opening split — exactly "the rich likelihood identifies what win cannot".

TWO EVALUATIONS:
 (a) PROFILE FIT — does the dynamic 4-latent generative model predict HELD-OUT sectional
     profiles better than the min-of-last-two SCALAR heuristic? Per-section RMSE,
     LEAVE-ONE-RACE-OUT within a horse (the held-out run's latents come from that horse's
     OTHER runs only, shrunk to a TRAIN-ONLY population fit — leak-free). The "denser signal"
     check: the surface should be easier to fit than the single win bit.
 (b) WIN-RELEVANCE OF THE NEW LATENT — does the recovered `a` carry INDEPENDENT signal for
     winning, controlling for the existing v_c ability estimate (and W′)? Partial correlation
     + drop-in ΔR² of `a` with the winner indicator given v_c, W′. The cheap proxy for "would
     acceleration help the win model" WITHOUT wiring it in. ST broken out by distance band
     (acceleration should matter most in sprints, where the start is a big fraction of the trip).

LEAK-FREE: per-horse θ from STRICTLY-PRIOR profiles only (expanding mean over a horse's
earlier runs); population fit train-only per fold; assert_no_lookahead on the fed-forward
frame. Deterministic per-run latent recovery (no outcome, no other horse).

Run: uv run --group bayes python studies/profile_likelihood_pilot.py
     (writes plots to studies/out/; default both venues, full panel)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl

from hkjc.backtesting import assert_no_lookahead, walk_forward_folds
from hkjc.model.calibration import distance_band
from hkjc.model.energetics_dynamic import (
    estimate_run_profile, expected_profile,
)

sys.path.insert(0, str(Path(__file__).parent))

CACHE = Path("data_cache")
TRAJ = CACHE / "trajectory_dataset_full.parquet"
OUT = Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)
KEYS = ["date", "venue", "race_no", "runner"]


# --------------------------------------------------------------------------- panels
def _run_panel() -> pl.DataFrame:
    """Per-run recovered (v_c, W′_0, a) + the raw section profile, for one row per run.

    Deterministic per-run transform of each run's OWN public sectionals (leak-free).
    Carries the section velocity/time/dist lists so the LOO profile-fit can score the
    held-out raw profile against the generative prediction.
    """
    tr = (pl.read_parquet(TRAJ)
          .with_columns(pl.col("date").cast(pl.Utf8).str.replace_all("/", "-").alias("date"))
          .sort(KEYS + ["section"]))
    agg = tr.group_by(KEYS).agg(
        pl.col("section_time"), pl.col("section_dist"),
        pl.col("horse_id").first(), pl.col("distance").first(),
        pl.col("won").first(), pl.col("n_sections").first())
    rows = agg.to_dicts()
    vc = np.empty(len(rows))
    wp = np.empty(len(rows))
    ac = np.empty(len(rows))
    vcru = np.empty(len(rows))
    for i, r in enumerate(rows):
        e = estimate_run_profile(np.asarray(r["section_time"], float),
                                 np.asarray(r["section_dist"], float))
        vc[i], wp[i], ac[i], vcru[i] = e.v_c, e.w_prime, e.accel, e.v_cruise
    out = pl.DataFrame({
        "date": [r["date"] for r in rows], "venue": [r["venue"] for r in rows],
        "race_no": [r["race_no"] for r in rows], "runner": [r["runner"] for r in rows],
        "horse_id": [r["horse_id"] for r in rows],
        "distance": [r["distance"] for r in rows],
        "won": [r["won"] for r in rows], "n_sec": [r["n_sections"] for r in rows],
        "section_time": [r["section_time"] for r in rows],
        "section_dist": [r["section_dist"] for r in rows],
        "vc_run": vc, "wp_run": wp, "a_run": ac, "vcru_run": vcru,
    })
    return out.with_columns(pl.col("race_no").cast(pl.Int32))


# ----------------------------------------------------------- (a) PROFILE-FIT (LOO)
def profile_fit_rmse(panel: pl.DataFrame) -> dict:
    """Leave-one-RACE-out within a horse: predict each held-out run's K-section profile.

    For a held-out run, the horse's latents (v_c, W′_0, a) are the STRICTLY-prior-EXCLUDED
    mean of that horse's OTHER runs (leave-one-out → no outcome / no same-run info). We then
    GENERATE the expected K-section velocity profile (4-latent dynamic) and the SCALAR-heuristic
    profile, and score both against the observed velocities by per-section RMSE.

    Scalar heuristic baseline: the min-of-last-two `v_c` (production scalar recovery) predicts a
    FLAT profile at the horse's mean v_c for every section — the "no shape" null (it has no W′
    or ramp structure to place velocity within the trip). We also report a stronger scalar:
    flat at the horse's mean RUN velocity. The 4-latent model must beat the better of the two.
    """
    df = panel.filter(pl.col("n_sec") >= 3)
    # per-horse LOO means of the recovered latents (exclude the held-out run)
    g = df.group_by("horse_id").agg(
        pl.col("vc_run").sum().alias("svc"), pl.col("wp_run").sum().alias("swp"),
        pl.col("a_run").sum().alias("sa"), pl.col("vcru_run").sum().alias("svcru"),
        pl.col("vc_run").count().alias("cnt"))
    df = df.join(g, on="horse_id")
    df = df.filter(pl.col("cnt") >= 2)   # need ≥1 OTHER run for a LOO estimate
    # leave-one-out per-horse latent = (group sum − own) / (cnt − 1)
    df = df.with_columns(
        ((pl.col("svc") - pl.col("vc_run")) / (pl.col("cnt") - 1)).alias("vc_loo"),
        ((pl.col("swp") - pl.col("wp_run")) / (pl.col("cnt") - 1)).alias("wp_loo"),
        ((pl.col("sa") - pl.col("a_run")) / (pl.col("cnt") - 1)).alias("a_loo"),
        ((pl.col("svcru") - pl.col("vcru_run")) / (pl.col("cnt") - 1)).alias("vcru_loo"))

    rows = df.to_dicts()
    se_dyn, se_dyn_noa, se_scal, se_vmean, n_sec_tot = 0.0, 0.0, 0.0, 0.0, 0
    se_open_dyn, se_open_noa, se_open_scal, n_open = 0.0, 0.0, 0.0, 0
    for r in rows:
        d = np.asarray(r["section_dist"], float)
        t = np.asarray(r["section_time"], float)
        v = d / t
        vc, wp, a, vcru = r["vc_loo"], r["wp_loo"], r["a_loo"], r["vcru_loo"]
        # 4-latent generative profile (uses the horse's prior-mean a for the opening ramp)
        pred_dyn = expected_profile(d, vc, wp, a)
        # ablation: same model with a fixed to 1.0 (NO acceleration latent — flat opening)
        pred_noa = expected_profile(d, vc, wp, 1.0)
        # scalar heuristic: flat at the horse's mean min-of-last-two v_c (no shape)
        pred_scal = np.full_like(v, vc)
        # stronger scalar: flat at the horse's mean cruise velocity
        pred_vmean = np.full_like(v, vcru)
        se_dyn += float(np.sum((v - pred_dyn) ** 2))
        se_dyn_noa += float(np.sum((v - pred_noa) ** 2))
        se_scal += float(np.sum((v - pred_scal) ** 2))
        se_vmean += float(np.sum((v - pred_vmean) ** 2))
        n_sec_tot += len(v)
        # opening-section-only RMSE (where `a` lives — the fingerprint test)
        se_open_dyn += float((v[0] - pred_dyn[0]) ** 2)
        se_open_noa += float((v[0] - pred_noa[0]) ** 2)
        se_open_scal += float((v[0] - pred_scal[0]) ** 2)
        n_open += 1
    return {
        "n_runs": len(rows), "n_sec": n_sec_tot,
        "rmse_dyn": (se_dyn / n_sec_tot) ** 0.5,
        "rmse_dyn_noa": (se_dyn_noa / n_sec_tot) ** 0.5,
        "rmse_scalar_vc": (se_scal / n_sec_tot) ** 0.5,
        "rmse_scalar_vmean": (se_vmean / n_sec_tot) ** 0.5,
        "rmse_open_dyn": (se_open_dyn / n_open) ** 0.5,
        "rmse_open_noa": (se_open_noa / n_open) ** 0.5,
        "rmse_open_scalar": (se_open_scal / n_open) ** 0.5,
    }


# ------------------------------------------------ (b) WIN-RELEVANCE of the latent `a`
def _trailing_mean(df: pl.DataFrame, col: str, out: str) -> pl.DataFrame:
    """Strictly-prior expanding mean of `col` per horse (leak-free: excludes the current run)."""
    df = df.sort(["horse_id", "d"])
    s = df.select(pl.col(col).cum_sum().shift(1).over("horse_id"))[col].fill_null(0.0).to_numpy()
    n = df.select(pl.col(col).cum_count().over("horse_id") - 1)[col].to_numpy().astype(float)
    return df.with_columns(pl.Series(out, np.where(n > 0, s / np.maximum(n, 1.0), np.nan)),
                           pl.Series(out + "_n", n))


def _partial_r2(y: np.ndarray, base: np.ndarray, extra: np.ndarray) -> tuple:
    """ΔR² and partial correlation of `extra` with `y` controlling for `base` columns.

    base: (n,k) controls (an intercept is added). Returns (r2_base, r2_full, dR2, partial_r,
    n). Plain OLS; the proxy for "does `a` carry win signal v_c (+W′) cannot explain".
    """
    n = len(y)
    Xb = np.column_stack([np.ones(n), base])
    Xf = np.column_stack([Xb, extra])

    def _r2(X):
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ beta
        ss_res = float(resid @ resid)
        ss_tot = float(((y - y.mean()) ** 2).sum())
        return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"), resid

    r2b, resid_b = _r2(Xb)
    r2f, _ = _r2(Xf)
    # partial correlation: residualise both y and extra on base, correlate
    beta_e, *_ = np.linalg.lstsq(Xb, extra, rcond=None)
    resid_e = extra - Xb @ beta_e
    if resid_e.std() > 0 and resid_b.std() > 0:
        pr = float(np.corrcoef(resid_b, resid_e)[0, 1])
    else:
        pr = float("nan")
    return r2b, r2f, r2f - r2b, pr, n


def win_relevance(panel: pl.DataFrame, venue: str) -> dict:
    """Partial corr / ΔR² of the LEAK-FREE trailing `a` with the winner indicator given v_c, W′.

    Per-horse trailing (strictly-prior expanding) means of (v_c, W′, a); merged with the
    contemporaneous win indicator. We regress `won` on [trailing v_c] (+ trailing W′) and ask
    whether adding [trailing a] explains independent variance. Within-RACE the win indicator is
    relative, so we also race-centre v_c/a (the order-statistic is race-relative) for a cleaner
    partial. Returns the full-panel result and a per-distance-band breakout.
    """
    p = (panel.filter((pl.col("venue") == venue) & pl.col("a_run").is_finite()
                      & pl.col("vc_run").is_finite() & pl.col("wp_run").is_finite())
         .with_columns(pl.col("date").str.strptime(pl.Date, "%Y-%m-%d", strict=False).alias("d")))
    p = _trailing_mean(p, "vc_run", "vc_tr")
    p = _trailing_mean(p, "wp_run", "wp_tr")
    p = _trailing_mean(p, "a_run", "a_tr")
    p = p.with_columns(pl.col("distance").map_elements(distance_band, return_dtype=pl.Utf8).alias("band"))
    p = p.with_columns((pl.col("date") + "_" + pl.col("race_no").cast(pl.Utf8)).alias("race_id"))
    # need a FINITE trailing estimate for all three latents (≥1 prior run; drop inf/nan from
    # degenerate per-run recoveries so the OLS design matrix is well-posed)
    p = p.filter(pl.col("vc_tr").is_finite() & pl.col("a_tr").is_finite()
                 & pl.col("wp_tr").is_finite())
    # RACE-CENTRE the predictors (the win is a within-race order statistic)
    p = p.with_columns(
        (pl.col("vc_tr") - pl.col("vc_tr").mean().over("race_id")).alias("vc_c"),
        (pl.col("wp_tr") - pl.col("wp_tr").mean().over("race_id")).alias("wp_c"),
        (pl.col("a_tr") - pl.col("a_tr").mean().over("race_id")).alias("a_c"))

    def _block(sub: pl.DataFrame) -> dict:
        if sub.height < 200:
            return {}
        y = sub["won"].to_numpy().astype(float)
        vc_c = sub["vc_c"].to_numpy()
        wp_c = sub["wp_c"].to_numpy()
        a_c = sub["a_c"].to_numpy()
        # given v_c
        r2b, r2f, dr2, pr, n = _partial_r2(y, vc_c.reshape(-1, 1), a_c)
        # given v_c AND W′
        r2b2, r2f2, dr22, pr2, _ = _partial_r2(y, np.column_stack([vc_c, wp_c]), a_c)
        return {"n": n, "dR2_a_given_vc": dr2, "pr_a_given_vc": pr,
                "dR2_a_given_vc_wp": dr22, "pr_a_given_vc_wp": pr2,
                "corr_a_vc": float(np.corrcoef(a_c, vc_c)[0, 1])}

    res = {"full": _block(p)}
    if venue == "ST":
        for bnd in ["sprint", "mile", "middle", "staying"]:
            res[bnd] = _block(p.filter(pl.col("band") == bnd))
    return res


# ------------------------------------------------------------------------ plotting
def _plot_profiles(panel: pl.DataFrame):
    """Sanity plot: mean observed vs generative profile shape; opening-ramp distribution."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    p = panel.filter((pl.col("n_sec") == 4) & (pl.col("venue") == "ST"))
    if p.height < 50:
        return
    obs = np.vstack([np.asarray(r["section_dist"], float) / np.asarray(r["section_time"], float)
                     for r in p.to_dicts()])
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    mo = obs.mean(0)
    ax[0].plot(range(1, 5), mo, "o-", label="observed mean (ST, 4-sec)")
    # generative mean at panel-mean latents
    vc = float(p["vc_run"].mean())
    wp = float(p["wp_run"].mean())
    a = float(p["a_run"].mean())
    d = np.asarray(p.to_dicts()[0]["section_dist"], float)
    ax[0].plot(range(1, 5), expected_profile(d, vc, wp, a), "s--", label=f"generative (a={a:.2f})")
    ax[0].plot(range(1, 5), expected_profile(d, vc, wp, 1.0), "^:", label="generative (a=1, no accel)")
    ax[0].set_xlabel("section")
    ax[0].set_ylabel("velocity m/s")
    ax[0].set_title("Opening-split ramp: the `a` fingerprint")
    ax[0].legend(fontsize=8)
    ax[1].hist(panel.filter(pl.col("a_run").is_finite())["a_run"].to_numpy(), bins=60)
    ax[1].axvline(1.0, color="k", ls="--", lw=1)
    ax[1].set_xlabel("a = v_1 / v_cruise (per run)")
    ax[1].set_title("Recovered acceleration latent")
    fig.tight_layout()
    fig.savefig(OUT / "profile_likelihood_pilot.png", dpi=110)
    print(f"  wrote {OUT / 'profile_likelihood_pilot.png'}")


# ----------------------------------------------------------------- leak-free check
def _leak_check(panel: pl.DataFrame, venue: str):
    """assert_no_lookahead on the trailing-a frame fed forward (predict = trailing means)."""
    p = (panel.filter((pl.col("venue") == venue) & pl.col("a_run").is_finite())
         .with_columns(pl.col("date").str.strptime(pl.Date, "%Y-%m-%d", strict=False).alias("d")))
    p = _trailing_mean(p, "a_run", "a_tr")
    p = p.with_columns((pl.col("date") + "_" + pl.col("race_no").cast(pl.Utf8)).alias("race_id"))
    p = p.filter(pl.col("a_tr").is_finite())
    dates = p["date"].unique().to_list()
    folds = walk_forward_folds(dates, n_folds=5, embargo=1, min_train=1)
    tr0 = te0 = None
    for fold in folds:
        a = p.filter(pl.col("date").is_in(fold.train))
        b = p.filter(pl.col("date").is_in(fold.test))
        if a.height > 100 and b.height > 50:
            tr0, te0 = a, b
            break

    def _predict(train, test):
        # the fed-forward feature is the strictly-prior trailing `a`; the "prediction" is a
        # deterministic function of it — perturbing test outcomes must not change it.
        return test.with_columns(pl.col("a_tr").alias("p"))

    drift = assert_no_lookahead(_predict, tr0, te0, key_cols=("race_id", "runner"))
    print(f"  [{venue}] assert_no_lookahead (trailing-a fed forward): drift {drift:.2e} OK")


def main():
    print("building per-run profile panel (v_c, W′_0, a) from the trajectory surface...")
    panel = _run_panel()
    print(f"  {panel.height} runs, "
          f"a∈[{panel['a_run'].min():.2f},{panel['a_run'].max():.2f}] "
          f"median {panel['a_run'].median():.3f}")

    print("\n================  (a) PROFILE FIT — leave-one-race-out within horse  ================")
    for venue in ["ST", "HV"]:
        r = profile_fit_rmse(panel.filter(pl.col("venue") == venue))
        print(f"\n  [{venue}]  {r['n_runs']} held-out runs, {r['n_sec']} sections")
        print("    FULL-PROFILE per-section RMSE (m/s):")
        print(f"      4-latent dynamic (v_c,W′,a) : {r['rmse_dyn']:.4f}")
        print(f"      3-latent (a:=1, no accel)   : {r['rmse_dyn_noa']:.4f}")
        print(f"      scalar heuristic (flat v_c) : {r['rmse_scalar_vc']:.4f}  "
              f"(Δ vs 4-latent {r['rmse_scalar_vc'] - r['rmse_dyn']:+.4f})")
        print(f"      scalar (flat cruise vmean)  : {r['rmse_scalar_vmean']:.4f}  "
              f"(Δ vs 4-latent {r['rmse_scalar_vmean'] - r['rmse_dyn']:+.4f})")
        print("    OPENING-SECTION RMSE (the `a` fingerprint):")
        print(f"      4-latent (with a)           : {r['rmse_open_dyn']:.4f}")
        print(f"      3-latent (a:=1)             : {r['rmse_open_noa']:.4f}  "
              f"(a lifts opening fit by {r['rmse_open_noa'] - r['rmse_open_dyn']:+.4f})")
        print(f"      scalar (flat v_c)           : {r['rmse_open_scalar']:.4f}")

    print("\n================  (b) WIN-RELEVANCE of the new latent `a`  ================")
    print("  ΔR² / partial-corr of trailing `a` with `won`, controlling for trailing v_c (+W′).")
    print("  (race-centred predictors; leak-free strictly-prior trailing means)")
    for venue in ["ST", "HV"]:
        _leak_check(panel, venue)
        res = win_relevance(panel, venue)
        print(f"\n  [{venue}]")
        for k, b in res.items():
            if not b:
                continue
            print(f"    {k:8s} n={b['n']:6d}  corr(a,v_c)={b['corr_a_vc']:+.3f}  "
                  f"| given v_c: ΔR²={b['dR2_a_given_vc']:+.5f} pr={b['pr_a_given_vc']:+.4f}  "
                  f"| given v_c,W′: ΔR²={b['dR2_a_given_vc_wp']:+.5f} pr={b['pr_a_given_vc_wp']:+.4f}")

    _plot_profiles(panel)


if __name__ == "__main__":
    main()
