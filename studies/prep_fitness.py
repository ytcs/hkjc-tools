"""PREP / FITNESS causal dimension (Wave-2, task #72) — DECISION-GATING diagnostic.

THE CAUSE (philosophy-pure, causal — NOT the deleted acausal ``gallop-fusion`` regression that
regressed gallop speed-figures onto ``v_c``). The mechanism here is an UPSTREAM physical cause:

    training impulses  →  an on-the-day FITNESS STATE (fitness − fatigue)  →  a MODULATION of the
    horse's OWN physiological constants (its CP / W' on race day shift with fitness).

The training load is an exogenous cause of the physiology the path then reveals. We model the
fitness state with a Banister-style impulse-response (two exponentials: a slow positive FITNESS
trace and a fast negative FATIGUE trace), strictly-prior from the trackwork gallops, and ask the
causal-identification question:

  does higher prior fitness predict a RESIDUAL in the path-read physiology — i.e. controlling for
  each horse's OWN career-mean baseline CP/W' (the ``build_pure_mechanism_panel`` reads), does
  higher prior fitness → higher EFFECTIVE CP/W' on the day?

Then the §4b CEILING CHECK (the non-negotiable mis-spec-vs-true-null discipline): measure the
fitness→physiology-residual signal as ``CEIL − a no-prep CONDITIONS ceiling`` (a flexible OOS ridge
as the DIAGNOSTIC ceiling — NEVER the model), sweeping observables (CP level, W', and the fitness
state's components fitness / fatigue separately) to separate a TRUE NULL from a MIS-SPECIFIED
fitness model. And the RE-EMERGENCE angle: does the prep latent predict held-out trainer-recent-
strike / horse recent-improvement (the deleted ``TRN_HOT_GAIN`` axis) WITHOUT feeding it as input?

THE BANISTER TIME-CONSTANTS (physiological priors, NOT outcome-fitted — fixed before any read):
  * tau_fitness = 42 d   (the classic Banister positive-influence decay; weeks-scale aerobic
                          adaptation — fitness accrues slowly and decays slowly),
  * tau_fatigue = 7 d    (the fast negative-influence decay; the days-scale recovery trace),
  * k_fatigue   = 2.0    (the negative trace is sharper than the positive — the canonical Banister
                          gain ratio k2/k1 ≈ 2, so a hard gallop hurts before it helps).
These are LITERATURE constants of the impulse-response model, not knobs tuned to CP or to the win
outcome. The per-session training impulse is ``load = gallop_velocity · worked_distance_m`` (m²/s,
the kinematic work proxy of one gallop effort), summed over the day's gallops.

    PYTHONHASHSEED=0 uv run --group bayes python studies/prep_fitness.py

DISCIPLINE. Causal mechanism only (prep→physiology); NO gain is fitted on the win/pool/market
outcome; the ridge ceiling is a DIAGNOSTIC-only probe (like the S1b oracle / 0062's field ceiling),
NEVER promoted into the model, NEVER baked. Leak-free: the fitness state uses PRIOR gallops only
(session strictly before the race date); the physiology panel is strictly-prior by construction.
READ-ONLY on src/hkjc (imports ``build_pure_mechanism_panel`` and the field/identify helpers
untouched). Study-only; no production constant changed.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from hkjc.model.field import cover_fraction, deconfounded_cruise_velocity
from hkjc.model.identify import settled_cruise_velocity
from hkjc.model.pure_mechanism import (
    _going_mass_frame,
    _run_from_row,
    _VC_HI,
    _VC_LO,
    build_pure_mechanism_panel,
)

CACHE = "data_cache"
SEED = 0
N_FOLDS = 5

# ---- Banister impulse-response time-constants (physiological priors; see module docstring) ----
TAU_FIT = 42.0   # days — slow positive (fitness) trace
TAU_FAT = 7.0    # days — fast negative (fatigue) trace
K_FAT = 2.0      # negative-trace gain ratio (k2/k1)


# ============================================================================ data loaders
def _norm_keys(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        pl.col("date").cast(pl.Utf8).str.replace_all("/", "-").alias("date"),
        pl.col("race_no").cast(pl.Int64, strict=False).alias("race_no"),
        pl.col("runner").cast(pl.Int64, strict=False).alias("runner"))


def _race_days(rr: pl.DataFrame) -> pl.DataFrame:
    """Per (horse, race) row with the bare horse code + the race date + conditions, for the join."""
    return rr.with_columns(
        pl.col("Date").cast(pl.Utf8).str.replace_all("/", "-").alias("date"),
        pl.col("Venue").cast(pl.Utf8).alias("venue"),
        pl.col("RaceNumber").cast(pl.Int64, strict=False).alias("race_no"),
        pl.col("HorseNo").cast(pl.Int64, strict=False).alias("runner"),
        pl.col("HorseID").cast(pl.Utf8).alias("code"),
        pl.col("Trainer").cast(pl.Utf8).alias("trainer"),
        pl.col("Pla").cast(pl.Float64, strict=False).alias("finish_pos"),
        pl.col("DeclarHorseWt").cast(pl.Float64, strict=False).alias("bodywt"),
        pl.col("RaceClass").cast(pl.Utf8).alias("klass"),
    ).select("date", "venue", "race_no", "runner", "code", "trainer",
             "finish_pos", "bodywt", "klass")


def _trackwork() -> pl.DataFrame:
    """One row per gallop split with a training-load impulse, keyed by the bare horse code + date."""
    tw = pl.read_parquet(f"{CACHE}/trackwork.parquet")
    return tw.with_columns(
        pl.col("horse_id").str.extract(r"_([A-Z]\d+)$").alias("code"),
        pl.col("session_date").str.strptime(pl.Date, "%Y-%m-%d", strict=False).alias("sdate"),
        # the per-gallop kinematic work proxy (m^2/s): how fast x how far it galloped.
        (pl.col("gallop_velocity") * pl.col("worked_distance_m")).alias("load"),
    ).filter(pl.col("code").is_not_null() & pl.col("load").is_finite()).select(
        "code", "sdate", "load", "gallop_velocity", "worked_distance_m")


# ============================================================================ Banister state
def _banister_states(track: pl.DataFrame, races: pl.DataFrame) -> pl.DataFrame:
    """Strictly-prior Banister fitness/fatigue at each race date, per (code, race date).

    For race day D and horse h, over the horse's gallop sessions s with sdate < D:
        fitness = sum_s load_s · exp(-(D - sdate_s)/TAU_FIT)
        fatigue = K_FAT · sum_s load_s · exp(-(D - sdate_s)/TAU_FAT)
        state   = fitness - fatigue
    Plus auxiliary strictly-prior summaries (recent 14-d load, days since last gallop, prior
    gallop count) for the ceiling sweep. STRICTLY-PRIOR: only sessions before the race date.
    """
    # daily load per horse (aggregate the day's gallops into one impulse).
    daily = (track.group_by(["code", "sdate"])
             .agg(pl.col("load").sum().alias("load"),
                  pl.col("gallop_velocity").mean().alias("gv"),
                  pl.col("worked_distance_m").sum().alias("dist")))
    sess = {c: [] for c in daily["code"].unique().to_list()}
    for row in daily.sort(["code", "sdate"]).iter_rows(named=True):
        sess[row["code"]].append((row["sdate"], row["load"]))

    out = []
    rrows = races.with_columns(
        pl.col("date").str.strptime(pl.Date, "%Y-%m-%d", strict=False).alias("_rd"))
    for r in rrows.iter_rows(named=True):
        c, rd = r["code"], r["_rd"]
        fit = fat = recent14 = recent60 = total = 0.0
        n_prior = 0
        last_dt = None
        if rd is not None and c in sess:
            for sdate, load in sess[c]:
                if sdate is None or sdate >= rd:
                    continue
                dt = (rd - sdate).days
                if dt <= 0:
                    continue
                fit += load * np.exp(-dt / TAU_FIT)
                fat += K_FAT * load * np.exp(-dt / TAU_FAT)
                total += load
                n_prior += 1
                if dt <= 14:
                    recent14 += load
                if dt <= 60:
                    recent60 += load
                if last_dt is None or sdate > last_dt:
                    last_dt = sdate
        days_since = float((rd - last_dt).days) if (last_dt is not None and rd is not None) else None
        out.append({
            "date": r["date"], "venue": r["venue"], "race_no": r["race_no"],
            "runner": r["runner"], "code": c,
            "fitness": fit, "fatigue": fat, "state": fit - fat,
            "recent14": recent14, "recent60": recent60, "load_total": total,
            "n_prior_gallops": float(n_prior),
            "days_since_gallop": days_since,
        })
    return pl.DataFrame(out)


# ============================================================================ ridge OOS ceiling
def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3 or np.std(a[ok]) == 0 or np.std(b[ok]) == 0:
        return float("nan")
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


def _design(df: pl.DataFrame, cols: list[str], target: str) -> tuple[np.ndarray, np.ndarray]:
    """(X, y) on rows with a finite target; mean-impute missing features + missingness flags."""
    y = df[target].to_numpy().astype(float)
    mask = np.isfinite(y)
    mats = []
    for c in cols:
        x = df[c].to_numpy().astype(float)
        finite = np.isfinite(x)
        if finite.sum() < 10 or np.nanstd(x[finite]) == 0:
            continue
        mu = float(np.nanmean(x[finite]))
        xi = np.where(finite, x, mu)
        mats.append(xi)
        if finite.sum() < len(x):
            mats.append(finite.astype(float))
    if not mats:
        return np.empty((0, 0)), np.empty(0)
    X = np.column_stack(mats)[mask]
    return X, y[mask]


def _ridge_oos(X: np.ndarray, y: np.ndarray, *, seed: int = SEED) -> float:
    """Pooled out-of-fold ridge ρ(prediction, y) with a CV-tuned penalty. The DIAGNOSTIC ceiling."""
    if X.shape[0] < 50 or X.shape[1] == 0:
        return float("nan")
    rng = np.random.default_rng(seed)
    n, p = X.shape
    folds = np.array_split(rng.permutation(n), N_FOLDS)
    lambdas = np.array([0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0])
    best = -1.0
    for lam in lambdas:
        pred = np.full(n, np.nan)
        for k in range(N_FOLDS):
            te = folds[k]
            tr = np.concatenate([folds[j] for j in range(N_FOLDS) if j != k])
            mu, sd = X[tr].mean(0), X[tr].std(0)
            sd = np.where(sd > 0, sd, 1.0)
            ym, ysd = y[tr].mean(), (y[tr].std() or 1.0)
            Xtr = (X[tr] - mu) / sd
            beta = np.linalg.solve(Xtr.T @ Xtr + lam * np.eye(p), Xtr.T @ ((y[tr] - ym) / ysd))
            pred[te] = ((X[te] - mu) / sd) @ beta * ysd + ym
        rho = _pearson(pred, y)
        if np.isfinite(rho) and rho > best:
            best = rho
    return best


# ============================================================================ per-run physiology
def _per_run_cp(traj: pl.DataFrame, rr: pl.DataFrame) -> pl.DataFrame:
    """CURRENT-RUN de-confounded cruise CP read [m/s] per run — the SAME estimator the panel uses
    internally, but READ on the run's OWN path (so it is the per-run physiology, not the prior
    career mean). Leak-free for the identification test: it is the TARGET measured on the current
    run; the prep PREDICTOR is strictly-prior. Replicates pure_mechanism lines 256-274 exactly.
    """
    keys = ["date", "venue", "race_no", "runner"]
    gm = _going_mass_frame(rr)
    traj = traj.with_columns(
        pl.col("date").cast(pl.Utf8).str.replace_all("/", "-").alias("date"),
        pl.col("race_no").cast(pl.Int64, strict=False))
    fld = (traj.group_by(["date", "venue", "race_no", "section"])
           .agg(pl.col("velocity").mean().alias("v_field"), pl.len().alias("fsize")))
    traj = traj.join(fld, on=["date", "venue", "race_no", "section"], how="left")
    runs = (traj.sort(keys + ["section"]).group_by(keys)
            .agg(pl.col("horse_id").first(),
                 pl.col("distance").first().alias("distance"),
                 pl.col("section_dist"), pl.col("section_time"), pl.col("velocity"),
                 pl.col("position"), pl.col("v_field"), pl.col("fsize"))
            .join(gm, on=keys, how="left"))
    out = []
    for r in runs.iter_rows(named=True):
        run = _run_from_row(r)
        v_settled = settled_cruise_velocity(run)
        vc_run = None
        if v_settled is not None and len(run.velocity) > 2:
            pos = np.asarray(r["position"], float)
            fs = np.asarray(r["fsize"], float)
            cover = np.array([cover_fraction(p, f) for p, f in zip(pos, fs)])
            cov = float(np.mean(cover[1:-1])) if cover[1:-1].size else 0.0
            try:
                dec = deconfounded_cruise_velocity(v_settled, cov, run.res)
            except Exception:
                dec = v_settled
            if _VC_LO < dec < _VC_HI:
                vc_run = float(dec)
        out.append({**{k: r[k] for k in keys}, "vc_run": vc_run})
    return _norm_keys(pl.DataFrame(out))


# ============================================================================ build the panel
def _klass_num(s: pl.Series) -> pl.Series:
    return s.str.extract(r"(\d+)").cast(pl.Float64, strict=False)


def build(venue: str) -> pl.DataFrame:
    traj = pl.read_parquet(f"{CACHE}/trajectory_dataset_full.parquet").filter(
        pl.col("venue") == venue)
    rr = pl.read_parquet(f"{CACHE}/results_rich.parquet")

    # (1) strictly-prior path physiology panel (per-run reads + the horse's own CAREER baseline).
    panel = _norm_keys(build_pure_mechanism_panel(traj, rr))
    # current-run de-confounded cruise CP (the per-run physiology read; the residual target).
    per_run = _per_run_cp(traj, rr)
    panel = panel.join(per_run, on=["date", "venue", "race_no", "runner"], how="left")

    # (2) the per-run conditions + the bare horse code (for the trackwork join).
    races_all = _race_days(rr)
    races = races_all.filter(pl.col("venue") == venue)

    # (3) the strictly-prior Banister fitness state (prior gallops only).
    states = _banister_states(_trackwork(), races)

    # field size per race + class number (no-prep CONDITIONS, the §4b de-confound).
    fs = (races_all.group_by("date", "venue", "race_no")
          .agg(pl.len().alias("field_size")))

    df = (panel
          .join(races.select("date", "venue", "race_no", "runner", "code", "trainer",
                             "finish_pos", "bodywt", "klass"),
                on=["date", "venue", "race_no", "runner"], how="left")
          .join(states.select("date", "venue", "race_no", "runner", "fitness", "fatigue",
                              "state", "recent14", "recent60", "load_total", "n_prior_gallops",
                              "days_since_gallop"),
                on=["date", "venue", "race_no", "runner"], how="left")
          .join(fs, on=["date", "venue", "race_no"], how="left")
          .with_columns(_klass_num(pl.col("klass")).alias("klass_num")))

    # (4) the per-run physiology RESIDUAL vs the horse's OWN strictly-prior career baseline.
    #   vc_run    = this run's CURRENT-path de-confounded cruise CP read [m/s] (the per-run read);
    #   vc_career = the horse's strictly-prior career-mean CP baseline (controls for own ability).
    #   The residual is the ON-THE-DAY shift the prep cause must explain (leak-free: vc_run is the
    #   measured target, the prep predictor is strictly-prior). W' level = the Hill reserve D' [m].
    df = df.with_columns(
        (pl.col("vc_run") - pl.col("vc_career")).alias("cp_resid"),
        (pl.col("wprime_hill")).alias("wprime_level"),
    )
    return df


# ============================================================================ re-emergence targets
def _trainer_strike(races_all: pl.DataFrame) -> pl.DataFrame:
    """Strictly-prior trainer recent strike-rate (exp-weighted win indicator), shift one race back.

    The TRN_HOT axis: a trainer running hot wins recently. Built per (trainer) over its OWN prior
    runners only (every runner's own result shifted out). Leak-free. Returned per run key.
    """
    rr = races_all.with_columns(
        pl.col("date").str.strptime(pl.Date, "%Y-%m-%d", strict=False).alias("_d"),
        (pl.col("finish_pos") == 1).cast(pl.Float64).alias("won"))
    rr = rr.sort(["trainer", "_d", "race_no", "runner"])
    rr = rr.with_columns(
        pl.col("won").ewm_mean(half_life=40.0, ignore_nulls=True).over("trainer").alias("_ewm"))
    rr = rr.with_columns(pl.col("_ewm").shift(1).over("trainer").alias("trn_strike"))
    return rr.select("date", "venue", "race_no", "runner", "trn_strike")


# ============================================================================ runner
def run_venue(venue: str) -> dict:
    df = build(venue)
    rr_all = _race_days(pl.read_parquet(f"{CACHE}/results_rich.parquet"))

    # rows with a real prep history AND a path read.
    have_prep = df.filter(
        (pl.col("n_prior_gallops") > 0) & pl.col("state").is_finite())

    # ---------- (2) IDENTIFICATION: prep state -> physiology residual ----------
    # raw correlations (sign / first read).
    def corr(sub: pl.DataFrame, a: str, b: str) -> tuple[float, int]:
        s = sub.filter(pl.col(a).is_finite() & pl.col(b).is_finite())
        return _pearson(s[a].to_numpy(), s[b].to_numpy()), s.height

    ident = {}
    for tgt, lbl in [("cp_resid", "CP_resid(vc_run-career)"),
                     ("vc_run", "CP_level(vc_run)"),
                     ("wprime_level", "Wprime_level(D')")]:
        for prep, plbl in [("state", "state"), ("fitness", "fitness"), ("fatigue", "fatigue")]:
            r, n = corr(have_prep, prep, tgt)
            ident[f"{plbl}->{lbl}"] = (round(r, 4), n)

    # ---------- (3) §4b CEILING CHECK (mis-spec vs true-null), per observable ----------
    COND = ["distance", "field_size", "klass_num", "bodywt", "days_since_gallop"]
    PREP = ["fitness", "fatigue", "state", "recent14", "recent60", "load_total",
            "n_prior_gallops"]
    ceiling = {}
    for tgt in ["cp_resid", "vc_run", "wprime_level"]:
        sub = have_prep.filter(pl.col(tgt).is_finite())
        Xc, yc = _design(sub, COND, tgt)
        Xp, yp = _design(sub, COND + PREP, tgt)
        # component ceilings (separate the Banister state from raw training VOLUME).
        Xf, yf = _design(sub, COND + ["fitness"], tgt)
        Xg, yg = _design(sub, COND + ["fatigue"], tgt)
        Xs, ys = _design(sub, COND + ["fitness", "fatigue", "state"], tgt)  # Banister-only
        Xv, yv = _design(sub, COND + ["recent14", "recent60", "load_total",
                                      "n_prior_gallops"], tgt)              # volume-only
        cond = _ridge_oos(Xc, yc)
        full = _ridge_oos(Xp, yp)
        ceiling[tgt] = {
            "n": int(sub.height),
            "cond": round(cond, 4),
            "cond+prep": round(full, 4),
            "prep_signal": round(full - cond, 4),
            "cond+fitness": round(_ridge_oos(Xf, yf), 4),
            "cond+fatigue": round(_ridge_oos(Xg, yg), 4),
            "cond+banister": round(_ridge_oos(Xs, ys), 4),
            "cond+volume": round(_ridge_oos(Xv, yv), 4),
        }

    # ---------- (4) RE-EMERGENCE: prep state -> held-out trainer-hot / recent-improvement ----------
    trn = _norm_keys(_trainer_strike(rr_all))
    de = df.join(trn, on=["date", "venue", "race_no", "runner"], how="left")
    reemerge = {}
    rep = de.filter((de["n_prior_gallops"] > 0) & de["state"].is_finite())
    # trainer-hot re-emergence (raw corr + the OOS ceiling over the prep block).
    s = rep.filter(pl.col("state").is_finite() & pl.col("trn_strike").is_finite())
    reemerge["state->trn_strike_corr"] = (round(_pearson(
        s["state"].to_numpy(), s["trn_strike"].to_numpy()), 4), s.height)
    Xr, yr = _design(rep, ["fitness", "fatigue", "state", "recent14", "load_total"], "trn_strike")
    reemerge["prep->trn_strike_ceiling"] = round(_ridge_oos(Xr, yr), 4)

    return {"venue": venue, "n_rows": df.height, "n_have_prep": have_prep.height,
            "ident": ident, "ceiling": ceiling, "reemerge": reemerge}


def main() -> None:
    print("=" * 78)
    print("PREP / FITNESS causal dimension — Banister prep -> path-read physiology (study #72)")
    print(f"Banister constants: TAU_FIT={TAU_FIT}d  TAU_FAT={TAU_FAT}d  K_FAT={K_FAT}")
    print("=" * 78)
    for venue in ("ST", "HV"):
        res = run_venue(venue)
        print(f"\n#### {venue}  (runs={res['n_rows']}, with-prep-history={res['n_have_prep']})")

        print("\n  (2) IDENTIFICATION — prep -> physiology (raw corr, n):")
        for k, (r, n) in res["ident"].items():
            print(f"      {k:42s} corr={r:+.4f}  n={n}")

        print("\n  (3) §4b CEILING CHECK (OOS ridge; prep_signal = (cond+prep) - cond):")
        hdr = ("observable          n     cond  cnd+prep  PREP_SIG  cnd+fit  cnd+fat  "
               "cnd+ban  cnd+vol")
        print("      " + hdr)
        for tgt, c in res["ceiling"].items():
            print(f"      {tgt:18s} {c['n']:5d}  {c['cond']:+.3f}   {c['cond+prep']:+.3f}"
                  f"   {c['prep_signal']:+.4f}  {c['cond+fitness']:+.3f}  {c['cond+fatigue']:+.3f}"
                  f"  {c['cond+banister']:+.3f}  {c['cond+volume']:+.3f}")

        print("\n  (4) RE-EMERGENCE — prep -> held-out trainer-hot (TRN_HOT axis):")
        r, n = res["reemerge"]["state->trn_strike_corr"]
        print(f"      state->trn_strike  raw corr={r:+.4f}  n={n}")
        print(f"      prep-block -> trn_strike  OOS ceiling={res['reemerge']['prep->trn_strike_ceiling']:+.4f}")


if __name__ == "__main__":
    main()
