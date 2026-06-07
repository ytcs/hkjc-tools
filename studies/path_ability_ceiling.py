"""PATH ABILITY-CEILING ORACLE (task S1b · #68) — a DECISION-GATING diagnostic.

S1 (journal 0052) found the path-physics ability read is WEAK: the precision-weighted fused read
correlates with the leak-free form target at only ρ≈+0.083 (ST) / +0.020 (HV), barely above the
rating FLOOR (≈0) and far below the trailing-form CEILING (+0.31/+0.25 — form's own
predictability). That entry measured a SINGLE fused instrument. THIS study asks the prior question:
what is the BEST achievable path→form correlation from ALL strictly-prior path signals combined —
the ORACLE — so we can decide WHY the read is weak.

  * If the career-aggregated oracle ≈ the single fused read (~+0.10): the longitudinal sectional
    path is FUNDAMENTALLY ability-poor → cleverer extraction won't help → PIVOT (new data: CV
    lateral paths, or reframe the edge).
  * If the career oracle approaches the form ceiling (+0.25–0.31): the signal IS there and we are
    UNDER-EXTRACTING → FIX the read (career EB aggregation / the high-weight features / richer
    surfaces).

THE ORACLE is the maximum OUT-OF-SAMPLE Pearson ρ between a regularized linear combination of the
feature matrix and the form target. CRITICAL: it is held-out — a ridge regression with a
k-fold-CV-tuned penalty, fit on TRAIN folds and scored on a held-out fold (out-of-fold prediction
pooled, then ρ(prediction, form)). An in-sample R² would be a meaningless inflated "oracle" and is
NOT reported.

FEATURE MATRIX — ALL strictly-prior path-derived signals per run + KNOWN contemporaneous boundary
conditions. Built in TWO forms (both leak-free):
  (a) PER-RUN — the horse's MOST-RECENT-PRIOR run's path features (lag-1; strictly prior — a run's
      own path never enters its own features), plus the strictly-prior Hill/de-confounded panel.
  (b) CAREER  — the expanding MEAN of every path feature over the horse's ALL PRIOR runs.
Both share the strictly-prior Hill physiology panel (``build_pure_mechanism_panel``: vc_hill,
wprime_hill, hill_n_dist, vc_decon) and the same contemporaneous boundary conditions (distance,
declared mass, carried weight, going) — pre-race public, allowed for path reconstruction (PHIL §7).

Path features per run (then lagged / expanded): opening-section velocity, closing-section velocity,
mean velocity, mid/cruise velocity, velocity range, late deceleration (mid→closing), settled cruise
(``identify.settled_cruise_velocity``), de-confounded cruise (``field.deconfounded_cruise_velocity``).

FORM TARGET: reuses ``ability_read_check.py`` — the SAME leak-free targets so numbers are directly
comparable to 0052. Primary target = the run's OWN realised relative finish ``rf_now`` (0052 TABLE B,
the target whose fused/floor/ceiling baselines +0.083/+0.020, ≈0, +0.31/+0.25 the task quotes); the
strictly-prior recency-form ``form`` (0052 TABLE A) is reported as a secondary cross-check.

    PYTHONHASHSEED=0 uv run --group bayes python studies/path_ability_ceiling.py

NO market odds, NO win optimization, NO meta-signal as an INPUT (form is a TARGET only). Read-only
on src/hkjc.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from ability_read_check import (
    CACHE,
    _norm_keys,
    _pearson,
    form_targets,
    rating_floor,
)
from hkjc.model.field import cover_fraction, deconfounded_cruise_velocity
from hkjc.model.identify import settled_cruise_velocity
from hkjc.model.pure_mechanism import (
    _run_from_row,
    _VC_HI,
    _VC_LO,
    build_pure_mechanism_panel,
)

SEED = 0
N_FOLDS = 5

# Per-run path features (computed once, then turned into strictly-prior lag-1 / career-mean forms).
PATH_FEATS = [
    "v_open", "v_close", "v_mean", "v_mid", "v_range", "v_late_decel",
    "v_settled", "v_decon",
]
# Strictly-prior panel reads (already expanding-prior by build_pure_mechanism_panel construction).
PANEL_FEATS = ["vc_hill", "wprime_hill", "hill_n_dist", "vc_decon"]
# Contemporaneous boundary conditions of the CURRENT run (pre-race public; allowed, PHIL §7).
COND_FEATS = ["distance", "bodywt_lb", "carriedwt_lb", "going_firm"]


def _per_run_path_features(traj: pl.DataFrame, results_rich: pl.DataFrame) -> pl.DataFrame:
    """One row per run with raw PATH kinematic features (NOT yet made strictly prior).

    Reuses the pure-mechanism per-run assembly (section kinematics + KNOWN mass/going + observed
    field pace, for the de-confound). The de-confounded cruise here is the SAME estimator the panel
    uses (``field.deconfounded_cruise_velocity`` on the settled cruise with the run's mean cover).
    """
    keys = ["date", "venue", "race_no", "runner"]
    gm = (results_rich.with_columns(
            pl.col("date").cast(pl.Utf8).str.replace_all("/", "-").alias("date"),
            pl.col("Venue").cast(pl.Utf8).alias("venue"),
            pl.col("RaceNumber").cast(pl.Int64, strict=False).alias("race_no"),
            pl.col("HorseNo").cast(pl.Int64, strict=False).alias("runner"),
            pl.col("Going").cast(pl.Utf8).alias("going"),
            pl.col("DeclarHorseWt").cast(pl.Float64, strict=False).alias("bodywt_lb"),
            pl.col("ActWt").cast(pl.Float64, strict=False).alias("carriedwt_lb"))
          .select(keys + ["going", "bodywt_lb", "carriedwt_lb"])
          .unique(keys))
    traj = traj.with_columns(
        pl.col("date").cast(pl.Utf8).str.replace_all("/", "-").alias("date"),
        pl.col("race_no").cast(pl.Int64, strict=False))
    # observed per-(race, section) field pace + field size (the boundary condition for de-confound).
    fld = (traj.group_by(["date", "venue", "race_no", "section"])
           .agg(pl.col("velocity").mean().alias("v_field"), pl.len().alias("fsize")))
    traj = traj.join(fld, on=["date", "venue", "race_no", "section"], how="left")
    runs = (traj.sort(keys + ["section"]).group_by(keys)
            .agg(pl.col("horse_id").first(),
                 pl.col("distance").first().alias("distance"),
                 pl.col("section_dist"), pl.col("section_time"), pl.col("velocity"),
                 pl.col("position"), pl.col("fsize"))
            .join(gm, on=keys, how="left"))

    out = {k: [] for k in keys}
    out["horse_id"], out["_d"] = [], []
    for f in PATH_FEATS + COND_FEATS:
        out[f] = []
    for r in runs.to_dicts():
        v = np.asarray(r["velocity"], float)
        v = v[np.isfinite(v)]
        if v.size < 2:
            continue
        run = _run_from_row(r)
        v_settled = settled_cruise_velocity(run)
        v_decon = np.nan
        if v_settled is not None and v.size > 2:
            pos = np.asarray(r["position"], float)
            fs = np.asarray(r["fsize"], float)
            cover = np.array([cover_fraction(p, f) for p, f in zip(pos, fs)])
            cov = float(np.mean(cover[1:-1])) if cover[1:-1].size else 0.0
            try:
                dec = deconfounded_cruise_velocity(v_settled, cov, run.res)
            except Exception:
                dec = v_settled
            if _VC_LO < dec < _VC_HI:
                v_decon = float(dec)
        interior = v[1:-1] if v.size >= 4 else v
        v_mid = float(np.mean(interior)) if interior.size else float(np.mean(v))
        v_close = float(v[-1])
        for k in keys:
            out[k].append(r[k])
        out["horse_id"].append(r["horse_id"])
        out["_d"].append(r["date"])
        out["v_open"].append(float(v[0]))
        out["v_close"].append(v_close)
        out["v_mean"].append(float(np.mean(v)))
        out["v_mid"].append(v_mid)
        out["v_range"].append(float(np.max(v) - np.min(v)))
        out["v_late_decel"].append(v_mid - v_close)  # +ve = decelerated into the line (faded)
        out["v_settled"].append(float(v_settled) if v_settled is not None else np.nan)
        out["v_decon"].append(v_decon)
        out["distance"].append(float(r["distance"]))
        out["bodywt_lb"].append(float(r.get("bodywt_lb") or np.nan))
        out["carriedwt_lb"].append(float(r.get("carriedwt_lb") or np.nan))
        out["going_firm"].append(1.0 if "FIRM" in (r.get("going") or "").upper() else 0.0)
    df = pl.DataFrame(out)
    return df.with_columns(
        pl.col("_d").str.strptime(pl.Date, "%Y-%m-%d", strict=False).alias("_d"))


def _strictly_prior_forms(per_run: pl.DataFrame) -> pl.DataFrame:
    """Two strictly-prior versions of each PATH feature, per horse, ordered by date:

      * ``{f}_lag``    — the horse's MOST-RECENT-PRIOR run's value (shift(1) — the PER-RUN form).
      * ``{f}_career`` — the expanding MEAN over ALL the horse's PRIOR runs (career-aggregated).

    Both exclude the run's own path (the shift / the prior-only cumulative). Contemporaneous
    boundary conditions (COND_FEATS) are kept as-is (pre-race public).
    """
    pr = per_run.sort(["horse_id", "_d", "race_no", "runner"])
    exprs = []
    for f in PATH_FEATS:
        exprs.append(pl.col(f).shift(1).over("horse_id").alias(f"{f}_lag"))
        # expanding mean of PRIOR runs only: (cumsum - own) / (cumcount - 1), ignoring nulls.
        csum = pl.col(f).fill_null(0.0).cum_sum().over("horse_id")
        ccnt = pl.col(f).is_not_null().cast(pl.Int64).cum_sum().over("horse_id")
        own = pl.col(f).fill_null(0.0)
        own_cnt = pl.col(f).is_not_null().cast(pl.Int64)
        prior_sum = csum - own
        prior_cnt = ccnt - own_cnt
        exprs.append(
            pl.when(prior_cnt > 0).then(prior_sum / prior_cnt).alias(f"{f}_career"))
    return pr.with_columns(exprs)


def _ridge_oos_rho(X: np.ndarray, y: np.ndarray, *, seed: int = SEED) -> tuple[float, np.ndarray]:
    """OUT-OF-SAMPLE oracle: ρ(out-of-fold ridge prediction, y) with a k-fold-CV-tuned penalty.

    For each candidate ridge penalty λ we run K-fold CV; the held-out predictions are POOLED across
    folds (every row predicted from a model that never saw it) and we pick the λ with the best pooled
    OOS ρ. That pooled-OOS ρ at the best λ is the oracle. Standardise X/y on TRAIN folds only.
    Returns (oracle ρ, the standardized ridge coefficients refit on the FULL data at the best λ —
    for feature-importance reporting only, NOT used to score).
    """
    rng = np.random.default_rng(seed)
    n, p = X.shape
    perm = rng.permutation(n)
    folds = np.array_split(perm, N_FOLDS)
    lambdas = np.array([0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0])

    best_rho, best_lam = -1.0, lambdas[0]
    for lam in lambdas:
        pred = np.full(n, np.nan)
        for k in range(N_FOLDS):
            te = folds[k]
            tr = np.concatenate([folds[j] for j in range(N_FOLDS) if j != k])
            mu, sd = X[tr].mean(0), X[tr].std(0)
            sd = np.where(sd > 0, sd, 1.0)
            ym, ysd = y[tr].mean(), y[tr].std()
            ysd = ysd if ysd > 0 else 1.0
            Xtr = (X[tr] - mu) / sd
            ytr = (y[tr] - ym) / ysd
            # ridge closed form (no penalty on the intercept — data is centred so intercept = 0).
            A = Xtr.T @ Xtr + lam * np.eye(p)
            beta = np.linalg.solve(A, Xtr.T @ ytr)
            Xte = (X[te] - mu) / sd
            pred[te] = (Xte @ beta) * ysd + ym
        rho = _pearson(pred, y)
        if np.isfinite(rho) and rho > best_rho:
            best_rho, best_lam = rho, lam

    # refit on the full standardized data at the best λ for coefficient importance (reporting only).
    mu, sd = X.mean(0), X.std(0)
    sd = np.where(sd > 0, sd, 1.0)
    ym, ysd = y.mean(), y.std() or 1.0
    Xz = (X - mu) / sd
    yz = (y - ym) / ysd
    beta = np.linalg.solve(Xz.T @ Xz + best_lam * np.eye(p), Xz.T @ yz)
    return best_rho, beta


def _design(df: pl.DataFrame, feat_cols: list[str]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Build (X, y, kept_cols) for the rows where the target and ≥1 feature are defined.

    Missing feature values are mean-imputed (column mean over the available rows) + a per-feature
    missingness indicator, so a runner with a thin path history still contributes (the oracle uses
    everything it can). Constant / all-missing columns are dropped.
    """
    y = df["_target"].to_numpy()
    cols, mats = [], []
    for c in feat_cols:
        x = df[c].to_numpy().astype(float)
        finite = np.isfinite(x)
        if finite.sum() < 10 or np.nanstd(x[finite]) == 0:
            continue
        mean = float(np.nanmean(x[finite]))
        miss = (~finite).astype(float)
        xi = np.where(finite, x, mean)
        mats.append(xi)
        cols.append(c)
        if 0 < miss.sum() < len(miss):  # informative missingness indicator
            mats.append(miss)
            cols.append(f"{c}__miss")
    X = np.column_stack(mats)
    return X, y, cols


def run_venue(venue: str) -> dict:
    traj = (pl.read_parquet(CACHE / "trajectory_dataset_full.parquet")
            .filter(pl.col("venue") == venue))
    rr = pl.read_parquet(CACHE / "results_rich.parquet")

    panel = _norm_keys(build_pure_mechanism_panel(traj, rr))
    per_run = _per_run_path_features(traj, rr)
    feats = _strictly_prior_forms(per_run)

    keys = ["date", "venue", "race_no", "runner"]
    # attach panel (Hill/decon strictly-prior reads) + the form targets + the rating floor.
    df = feats.join(
        panel.select(keys + PANEL_FEATS).with_columns(
            pl.col("race_no").cast(pl.Int64, strict=False)),
        on=keys, how="left")
    forms = _norm_keys(form_targets(rr, venue))
    df = df.join(
        forms.select("date", "venue", "race_no", "runner", "form", "trailing_form", "rf_now"),
        on=keys, how="left")
    rtg = rating_floor(venue)
    df = df.sort(["horse_id", "_d"]).join_asof(
        rtg.rename({"_rdate": "_d"}), on="_d", by="horse_id", strategy="backward")

    # the same alignment as 0052: a run with a path-based ability read (Hill OR de-confounded) AND a
    # defined form target — so the oracle is scored on the SAME population the fused read was.
    base = df.filter(
        (pl.col("vc_hill").is_finite() | pl.col("vc_decon").is_finite()))

    def oracle(target: str, feat_set: list[str]) -> dict:
        sub = base.filter(pl.col(target).is_finite()).with_columns(
            pl.col(target).alias("_target"))
        X, y, cols = _design(sub, feat_set)
        rho, beta = _ridge_oos_rho(X, y)
        imp = sorted(zip(cols, beta), key=lambda kv: -abs(kv[1]))
        return {"n": sub.height, "rho": rho, "top": imp[:8]}

    perrun_feats = [f"{f}_lag" for f in PATH_FEATS] + PANEL_FEATS + COND_FEATS
    career_feats = [f"{f}_career" for f in PATH_FEATS] + PANEL_FEATS + COND_FEATS

    aligned = base.filter(pl.col("rf_now").is_finite())
    out = {
        "venue": venue,
        "n": aligned.height,
        "rf_now": {
            "perrun": oracle("rf_now", perrun_feats),
            "career": oracle("rf_now", career_feats),
            "floor": _pearson(aligned["rtg"].to_numpy(), aligned["rf_now"].to_numpy()),
            "ceiling": _pearson(aligned["trailing_form"].to_numpy(),
                                aligned["rf_now"].to_numpy()),
        },
        "form": {
            "perrun": oracle("form", perrun_feats),
            "career": oracle("form", career_feats),
            "floor": _pearson(base.filter(pl.col("form").is_finite())["rtg"].to_numpy(),
                              base.filter(pl.col("form").is_finite())["form"].to_numpy()),
            "ceiling": _pearson(
                base.filter(pl.col("form").is_finite())["trailing_form"].to_numpy(),
                base.filter(pl.col("form").is_finite())["form"].to_numpy()),
        },
    }
    return out


# 0052 single fused-read baselines (TABLE B target = rf_now; TABLE A target = recency-form).
FUSED_RFNOW = {"ST": 0.083, "HV": 0.020}
FUSED_FORM = {"ST": 0.188, "HV": 0.148}


def _print_block(rows: list[dict], key: str, fused: dict[str, float]) -> None:
    hdr = (f"  {'venue':5s} {'N':>7s} {'oracle_perrun':>14s} {'oracle_career':>14s} "
           f"{'fused(0052)':>12s} {'floor':>8s} {'ceiling':>9s}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for r in rows:
        b = r[key]
        print(f"  {r['venue']:5s} {r['n']:7d} {b['perrun']['rho']:+14.3f} "
              f"{b['career']['rho']:+14.3f} {fused[r['venue']]:+12.3f} "
              f"{b['floor']:+8.3f} {b['ceiling']:+9.3f}")


def main() -> None:
    rows = [run_venue(v) for v in ("ST", "HV")]
    print(f"\n{'=' * 90}")
    print("PATH ABILITY-CEILING ORACLE (S1b) — max OUT-OF-SAMPLE ρ(ridge[ALL path features], form)")
    print(f"{'=' * 90}")
    print(f"  ridge: {N_FOLDS}-fold CV-tuned penalty, pooled out-of-fold prediction, seed={SEED}")
    print("  features: per-run = lag-1 path kinematics; career = expanding-prior-mean; both + the")
    print("            strictly-prior Hill/de-confounded panel + contemporaneous mass/dist/going.")

    print("\n  TABLE B — TARGET = the run's OWN realised finish rf_now (0052-comparable baselines)")
    _print_block(rows, "rf_now", FUSED_RFNOW)
    print("  (0052 fused rf_now: ST +0.083 / HV +0.020; rating floor ≈0; form ceiling +0.31/+0.25)")

    print("\n  TABLE A — TARGET = strictly-prior recency-form (half-life 2, shift 1)")
    _print_block(rows, "form", FUSED_FORM)
    print("  (0052 fused recency-form: ST +0.188 / HV +0.148; ceiling = form autocorrelation)")

    print("\n  TOP-WEIGHTED FEATURES (career oracle, standardized |ridge coef|, rf_now target):")
    for r in rows:
        tops = ", ".join(f"{c} {w:+.3f}" for c, w in r["rf_now"]["career"]["top"][:6])
        print(f"    [{r['venue']}] {tops}")

    print("\n  VERDICT (career oracle vs single fused read vs form ceiling):")
    for r in rows:
        career = r["rf_now"]["career"]["rho"]
        fused = FUSED_RFNOW[r["venue"]]
        ceil = r["rf_now"]["ceiling"]
        lift = career - fused
        near_fused = abs(lift) <= 0.03
        near_ceil = career >= 0.7 * ceil
        verdict = ("ABILITY-POOR (PIVOT)" if near_fused else
                   "UNDER-EXTRACTED (FIX)" if near_ceil else "PARTIAL")
        print(f"    [{r['venue']}] career oracle {career:+.3f} vs fused {fused:+.3f} "
              f"(lift {lift:+.3f}) vs ceiling {ceil:+.3f}  →  {verdict}")


if __name__ == "__main__":
    main()
