"""S7b FIELD-PACE CEILING (mis-specification-vs-true-null diagnostic) — a DECISION-GATING probe.

Journal 0061's mean-field EQUILIBRIUM PACE (``field.equilibrium_pace``) BEATS the naive field-mean
baseline at HV but NOT at ST. The cardinal question this study separates: is the ST non-result a
TRUE NULL (the field/deployment composition genuinely carries no pace structure above baseline on
the big galloping track) or MIS-SPECIFICATION (0061's coupling FORM and/or its single OBSERVABLE —
section-1 mean velocity — are wrong, and an effect IS there that we missed)? Conflating the two is
the error this replicates the S1b (journal 0054) ceiling discipline to avoid.

THE METHOD (exactly the S1b pattern, transposed to the FIELD dimension):
  * BASELINES (de-contaminate identification). 0061's naive baseline = field-mean of prior REALISED
    paces, themselves equilibrium outcomes ⇒ the explicit-equilibrium increment is small by
    construction. We ADD a cleaner reference: each horse's INTRINSIC / SOLO pace tendency — its pace
    in prior runs where it ran CLEAR (led early, position ≤ 2 at section 1, uncontested). Increments
    are reported against BOTH baselines.
  * THE MODEL-AGNOSTIC CEILING. A CV-tuned ridge (with nonlinear / interaction / threshold features),
    OUT-OF-SAMPLE, of the realised observable from the FULL field-composition feature set
    (front-runner count, style distribution + VARIANCE, field size, draw spread, class,
    forwardness-weighted base, threshold + nonlinear terms). This is the MAX pace structure ANY
    composition model can extract from prior-race constants. The ridge is a DIAGNOSTIC CEILING PROBE
    ONLY — like S1b's oracle, NEVER the model, NEVER baked; no production constant changes.
  * VERDICT per venue × observable:  ceiling ≈ best-baseline ⇒ TRUE NULL;  ceiling ≫ 0061's
    mean-field form ⇒ MIS-SPECIFICATION (and the gap quantifies it).

MULTIPLE OBSERVABLES (the effect may not live in section-1 mean velocity, esp. on ST):
    (1) S1-MEAN  — section-1 field-mean velocity (0061's choice).
    (2) S1-STD   — pace VARIANCE / spread across the field (early-section velocity std).
    (3) LEADER   — the LEADER's pace (the fastest early-section velocity in the field).
    (4) MID      — MID-race pace (interior-section field-mean velocity).
    (5) GRADIENT — the early pace GRADIENT (section-2 minus section-1 field-mean velocity).

ALTERNATIVE COUPLINGS (to localise mis-specification, if present):
    * CONGESTION / blocking — crowding SLOWS (a NEGATIVE coupling on front-runner crowding), the
      opposite sign to 0061's chase-the-front.
    * THRESHOLD — the composition effect only above a front-runner-count cutoff.
    * W'-BURN — a forwardness-energy-burn observable our chase-the-front form ignores.
  We test whether any carries residual structure (vs realised − best-baseline) that 0061 missed.

CAVEAT (target validity): pace is a PROXY. The ultimate test is the WIN / path effect (S7b win
wiring), not the pace fit; a true-null on pace does not preclude a deployment win effect via another
channel (e.g. CV congestion). Stated in the verdict.

    PYTHONHASHSEED=0 uv run --group bayes python studies/field_pace_ceiling.py

NO market / win / pool. NO coefficient fitted to any OUTCOME (the ridge predicts a pre-race pace
OBSERVABLE, a diagnostic ceiling, not the win). Leak-free / strictly-prior throughout. Read-only on
src/hkjc (imports ``field.equilibrium_pace`` / ``field.front_group_pace`` untouched). Reuses
``studies/equilibrium_pace.py``'s data loading conventions.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from hkjc.model.field import equilibrium_pace, front_group_pace

DATA = "data_cache"
SEED = 0
N_FOLDS = 5
MIN_FIELD = 5            # need a real field to talk about composition (matches 0061)
MIN_PRIOR_FRAC = 0.5     # ≥ half the field with a strictly-prior pace (leak-free coverage, 0061)
FRONT_STYLE = 0.66       # forwardness cutoff that counts a horse as a "front-runner" (style ∈ [0,1])

OBSERVABLES = ["s1_mean", "s1_std", "leader", "mid", "gradient"]


def _race_id(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        (pl.col("date") + "_" + pl.col("venue") + "_" + pl.col("race_no").cast(str)).alias("race_id")
    )


def load_panel() -> pl.DataFrame:
    """Per-runner panel with strictly-prior pace priors + forwardness + the realised observables.

    Strictly-prior per-horse observables (earlier runs only — leak-free):
      * ``base_i``  — own opening-pace tendency: expanding prior-mean of its section-1 velocity
                      (the 0061 ``base``; the velocity it runs to its OWN deployment).
      * ``solo_i``  — own INTRINSIC / SOLO pace: expanding prior-mean of section-1 velocity over the
                      horse's prior runs where it ran CLEAR (early position ≤ 2 — uncontested /
                      clear-running). A cleaner intrinsic reference than ``base`` (which itself is an
                      equilibrium outcome). Falls back to ``base`` when a horse has no prior clear run.
      * ``style_i`` — forwardness ∈ [0,1] = ``1 − trail_early_pos_norm`` (orientation fix 0063/0065;
                      strictly-prior by construction; 1 = front-runner, low = closer).
    Plus per-race contemporaneous boundary conditions (pre-race public, allowed for path
    reconstruction): ``distance``, ``draw``, ``class_num``. And the CURRENT-race realised observables
    used as the prediction targets (collapsed to one row per race in ``per_race``).
    """
    tj = _race_id(pl.read_parquet(f"{DATA}/trajectory_dataset_full.parquet"))

    # section-1 velocity + early position (for the solo/clear-running detector) + section-2 velocity.
    s1 = (
        tj.filter(pl.col("section") == 1)
        .select(["race_id", "horse_id", "date", "venue", "distance", "velocity", "position"])
        .rename({"velocity": "v_open_i", "position": "pos1_i"})
        .sort(["horse_id", "date"])
    )
    s2 = (
        tj.filter(pl.col("section") == 2)
        .select(["race_id", "horse_id", "velocity"])
        .rename({"velocity": "v_s2_i"})
    )
    # interior (mid-race) per-horse velocity = mean over non-first, non-last sections.
    mid = (
        tj.filter((pl.col("section") > 1) & (pl.col("section") < pl.col("n_sections")))
        .group_by(["race_id", "horse_id"])
        .agg(pl.col("velocity").mean().alias("v_mid_i"))
    )

    # strictly-prior expanding mean of own opening velocity (shift-by-one; the 0061 base).
    s1 = s1.with_columns(
        [
            (pl.col("v_open_i").cum_sum().over("horse_id") - pl.col("v_open_i")).alias("_cs"),
            (pl.col("v_open_i").cum_count().over("horse_id") - 1).alias("_cc"),
        ]
    ).with_columns(
        (pl.col("_cs") / pl.when(pl.col("_cc") > 0).then(pl.col("_cc")).otherwise(None)).alias("base_i")
    )
    # SOLO / clear-running pace: expanding prior-mean of opening velocity over prior CLEAR runs only
    # (early position ≤ 2). Strictly-prior (shift-by-one): the run's own clear pace is excluded.
    clear_v = pl.when(pl.col("pos1_i") <= 2).then(pl.col("v_open_i")).otherwise(None)
    s1 = s1.with_columns(clear_v.alias("_clearv")).with_columns(
        [
            (pl.col("_clearv").fill_null(0.0).cum_sum().over("horse_id")
             - pl.col("_clearv").fill_null(0.0)).alias("_scs"),
            (pl.col("_clearv").is_not_null().cast(pl.Int64).cum_sum().over("horse_id")
             - pl.col("_clearv").is_not_null().cast(pl.Int64)).alias("_scc"),
        ]
    ).with_columns(
        (pl.col("_scs") / pl.when(pl.col("_scc") > 0).then(pl.col("_scc")).otherwise(None)).alias("solo_i")
    )

    # ORIENTATION FIX (journal 0063/0065): ``trail_early_pos_norm`` is the trailing position RANK
    # (high = closer); forwardness = 1 − it (1 = front-runner contesting the lead).
    rp = (
        _race_id(pl.read_parquet(f"{DATA}/run_policy_dataset.parquet"))
        .select(["race_id", "horse_id", "trail_early_pos_norm"])
        .with_columns((1.0 - pl.col("trail_early_pos_norm")).clip(0.0, 1.0)
                      .alias("trail_early_pos_norm"))
    )
    rr = (
        pl.read_parquet(f"{DATA}/results_rich.parquet")
        .with_columns(
            pl.col("date").cast(pl.Utf8).str.replace_all("/", "-").alias("date"),
            pl.col("Venue").cast(pl.Utf8).alias("venue"),
            pl.col("RaceNumber").cast(pl.Int64, strict=False).cast(pl.Utf8).alias("race_no"),
            pl.col("HorseID").cast(pl.Utf8).alias("horse_id"),
            pl.col("Dr").cast(pl.Float64, strict=False).alias("draw"),
            pl.col("RaceClass").cast(pl.Utf8).alias("race_class"),
        )
    )
    rr = _race_id(rr).select(["race_id", "horse_id", "draw", "race_class"])

    panel = (
        s1.join(s2, on=["race_id", "horse_id"], how="left")
        .join(mid, on=["race_id", "horse_id"], how="left")
        .join(rp, on=["race_id", "horse_id"], how="left")
        .rename({"trail_early_pos_norm": "style_i"})
        .join(rr, on=["race_id", "horse_id"], how="left")
    )
    return panel


# Map a race-class string to an ordinal number (lower = higher class). Pre-race public.
_CLASS_ORDER = {
    "Group One": 0, "Group Two": 1, "Group Three": 2,
    "Class 1": 3, "Class 2": 4, "Class 3": 5, "Class 4": 6, "Class 5": 7,
}


def _class_num(s: str | None) -> float:
    if not s:
        return float("nan")
    for k, v in _CLASS_ORDER.items():
        if s.startswith(k):
            return float(v)
    return float("nan")


def _realised(panel_g: pl.DataFrame) -> dict[str, float]:
    """The CURRENT-race realised pace observables (the prediction targets)."""
    v1 = panel_g["v_open_i"].to_numpy()
    v2 = panel_g["v_s2_i"].to_numpy()
    vm = panel_g["v_mid_i"].to_numpy()
    v1 = v1[np.isfinite(v1)]
    v2f = v2[np.isfinite(v2)]
    vmf = vm[np.isfinite(vm)]
    s1_mean = float(np.mean(v1)) if v1.size else float("nan")
    s1_std = float(np.std(v1)) if v1.size > 1 else float("nan")
    leader = float(np.max(v1)) if v1.size else float("nan")
    mid = float(np.mean(vmf)) if vmf.size else float("nan")
    grad = (float(np.mean(v2f)) - s1_mean) if (v2f.size and np.isfinite(s1_mean)) else float("nan")
    return {"s1_mean": s1_mean, "s1_std": s1_std, "leader": leader, "mid": mid, "gradient": grad}


def per_race(panel: pl.DataFrame) -> pl.DataFrame:
    """One row per race: strictly-prior baselines / 0061-form / composition features + realised obs."""
    rows = []
    for (race_id,), g in panel.group_by(["race_id"], maintain_order=True):
        n = g.height
        base = g["base_i"].to_numpy()
        solo = g["solo_i"].to_numpy()
        style = g["style_i"].to_numpy()
        prior_ok = np.isfinite(base)
        if n < MIN_FIELD or prior_ok.sum() < MIN_PRIOR_FRAC * n:
            continue
        bm = float(np.nanmean(base[prior_ok])) if prior_ok.any() else float("nan")
        if not np.isfinite(bm):
            continue
        # leak-free fallbacks: missing base → field prior-mean; missing solo → own base; missing
        # style (debut) → field-neutral style mean.
        b = np.where(prior_ok, base, bm)
        so = np.where(np.isfinite(solo), solo, b)
        sm = float(np.nanmean(style)) if np.isfinite(style).any() else 0.5
        s = np.where(np.isfinite(style), style, sm)

        # --- BASELINES (per the directive: compare increments against BOTH) ---
        naive = float(b.mean())                       # 0061 naive: field-mean of prior realised pace
        solo_naive = float(so.mean())                 # cleaner intrinsic/solo-pace baseline
        # --- 0061 mean-field equilibrium form ---
        v_eq = equilibrium_pace(b, s)
        # --- composition features (for the ceiling ridge + alternative couplings) ---
        lead = front_group_pace(b, s)                 # forwardness-weighted front-group pace
        n_front = float((s >= FRONT_STYLE).sum())     # front-runner count
        front_max = float(np.max(b[s >= FRONT_STYLE])) if (s >= FRONT_STYLE).any() else float(np.max(b))
        draw = g["draw"].to_numpy()
        draw = draw[np.isfinite(draw)]
        draw_spread = float(np.max(draw) - np.min(draw)) if draw.size > 1 else 0.0
        cls = _class_num(g["race_class"][0])
        dist = float(g["distance"][0])

        realised = _realised(g)
        rows.append(
            {
                "race_id": race_id,
                "venue": g["venue"][0],
                "distance": dist,
                "field_size": float(n),
                # baselines + 0061 form
                "naive": naive,
                "solo_naive": solo_naive,
                "v_eq": v_eq,
                # composition feature set (the ceiling) ---------------------------------
                "front_base": lead,                   # forwardness-weighted base (0061's residual carrier)
                "mean_base": naive,
                "max_base": float(b.max()),
                "min_base": float(b.min()),
                "base_std": float(b.std()) if n > 1 else 0.0,
                "mean_style": float(s.mean()),
                "style_std": float(s.std()) if n > 1 else 0.0,
                "n_front": n_front,
                "front_frac": n_front / n,
                "front_max": front_max,
                "draw_spread": draw_spread,
                "class_num": cls,
                # threshold + nonlinear terms (let the ceiling read a cutoff / curvature) -
                "n_front_sq": n_front * n_front,
                "front_x_base": n_front * float(b.mean()),
                "thresh_front": max(0.0, n_front - 2.0),    # ReLU above a 2-front-runner cutoff
                "congest": n_front / n * float(b.std() if n > 1 else 0.0),  # crowding × pace-spread
                "fast_front": float((s * b).sum() / max(s.sum(), 1e-9)) - naive,  # how much front > mean
                "wprime_burn": float((s * np.maximum(b - naive, 0.0)).sum()),     # forwardness-energy burn
                # realised observables (targets) ----------------------------------------
                **{f"y_{k}": v for k, v in realised.items()},
            }
        )
    return pl.DataFrame(rows)


# Composition feature columns for the model-agnostic ceiling ridge (the FULL set, with nonlinear
# / interaction / threshold terms). DIAGNOSTIC ONLY — never a production read.
CEILING_FEATS = [
    "front_base", "mean_base", "max_base", "min_base", "base_std",
    "mean_style", "style_std", "n_front", "front_frac", "front_max",
    "draw_spread", "class_num", "field_size", "distance",
    "n_front_sq", "front_x_base", "thresh_front", "congest", "fast_front", "wprime_burn",
]

# Pre-race CONDITIONS only (distance / field-size / class / draw-spread) — NOT composition. The
# critical de-confound (S1b "distance-only" subset oracle): distance dominates the raw pace (0061:
# corr −0.49) and is NOT carried by the per-horse base baseline (which is a field-MEAN that averages
# scale away). A ceiling that beats the baseline only by re-adding conditions is NOT composition
# mis-spec. So we also fit a CONDITIONS-ONLY ceiling and a CONDITIONS+per-horse-PACE ceiling (no
# style/composition). Composition mis-spec is real only if the FULL ceiling beats the
# conditions+pace ceiling.
COND_FEATS = ["distance", "field_size", "class_num", "draw_spread"]
# conditions + the per-horse PACE aggregates the baseline already uses, but NO style/composition
# interaction terms — the "best a non-composition pace model can do" reference.
PACE_FEATS = COND_FEATS + ["mean_base", "max_base", "min_base", "base_std"]


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3 or np.std(a[m]) == 0 or np.std(b[m]) == 0:
        return float("nan")
    return float(np.corrcoef(a[m], b[m])[0, 1])


def _rmse(pred: np.ndarray, y: np.ndarray) -> float:
    m = np.isfinite(pred) & np.isfinite(y)
    return float(np.sqrt(np.mean((pred[m] - y[m]) ** 2)))


def _ridge_oos(X: np.ndarray, y: np.ndarray, *, seed: int = SEED) -> tuple[float, float]:
    """OUT-OF-SAMPLE ceiling: (ρ, RMSE) of the pooled out-of-fold ridge prediction at the CV-best λ.

    For each candidate λ run K-fold CV (standardise on TRAIN folds only); pool the held-out
    predictions (every row predicted by a model that never saw it); pick the λ with the best pooled
    OOS ρ. Returns that ρ and the matching RMSE. This is the S1b oracle, transposed — a diagnostic
    ceiling, NOT a build.
    """
    rng = np.random.default_rng(seed)
    n, p = X.shape
    perm = rng.permutation(n)
    folds = np.array_split(perm, N_FOLDS)
    lambdas = np.array([0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0])
    best_rho, best_pred = -2.0, None
    for lam in lambdas:
        pred = np.full(n, np.nan)
        for k in range(N_FOLDS):
            te = folds[k]
            tr = np.concatenate([folds[j] for j in range(N_FOLDS) if j != k])
            mu, sd = X[tr].mean(0), X[tr].std(0)
            sd = np.where(sd > 0, sd, 1.0)
            ym, ysd = y[tr].mean(), y[tr].std() or 1.0
            Xtr = (X[tr] - mu) / sd
            ytr = (y[tr] - ym) / ysd
            beta = np.linalg.solve(Xtr.T @ Xtr + lam * np.eye(p), Xtr.T @ ytr)
            pred[te] = ((X[te] - mu) / sd) @ beta * ysd + ym
        rho = _pearson(pred, y)
        if np.isfinite(rho) and rho > best_rho:
            best_rho, best_pred = rho, pred
    return best_rho, _rmse(best_pred, y)


def _design(df: pl.DataFrame, cols: list[str], target: str) -> tuple[np.ndarray, np.ndarray]:
    """Build (X, y) for rows with a finite target; mean-impute features + missingness indicators."""
    y = df[target].to_numpy().astype(float)
    keep = np.isfinite(y)
    df = df.filter(pl.Series(keep))
    y = y[keep]
    mats = []
    for c in cols:
        x = df[c].to_numpy().astype(float)
        finite = np.isfinite(x)
        if finite.sum() < 10 or np.nanstd(x[finite]) == 0:
            continue
        mean = float(np.nanmean(x[finite]))
        mats.append(np.where(finite, x, mean))
        if 0 < (~finite).sum() < len(x):
            mats.append((~finite).astype(float))
    return np.column_stack(mats), y


def analyse(races: pl.DataFrame, venue: str) -> dict:
    rv = races.filter(pl.col("venue") == venue)
    out: dict[str, dict] = {"venue": venue, "n": rv.height, "obs": {}}
    for obs in OBSERVABLES:
        tgt = f"y_{obs}"
        sub = rv.filter(pl.col(tgt).is_finite())
        if sub.height < 50:
            continue
        y = sub[tgt].to_numpy()
        naive = sub["naive"].to_numpy()
        solo = sub["solo_naive"].to_numpy()
        veq = sub["v_eq"].to_numpy()
        # the model-agnostic ceiling (full composition feature set, OOS ridge).
        X, yc = _design(sub, CEILING_FEATS, tgt)
        ceil_rho, ceil_rmse = _ridge_oos(X, yc)
        # de-confound: conditions-only and conditions+per-horse-pace (NO composition) ceilings.
        Xc, ycc = _design(sub, COND_FEATS, tgt)
        cond_rho, _ = _ridge_oos(Xc, ycc)
        Xp, ycp = _design(sub, PACE_FEATS, tgt)
        pace_rho, _ = _ridge_oos(Xp, ycp)
        # residual carried by each ALTERNATIVE COUPLING vs realised − best-baseline (which baseline
        # has the higher correlation to the observable).
        base_rho = {"naive": _pearson(naive, y), "solo": _pearson(solo, y)}
        best_base = naive if abs(base_rho["naive"]) >= abs(base_rho["solo"]) else solo
        resid = y - best_base
        alt = {
            "congestion": _pearson(sub["congest"].to_numpy(), resid),      # crowding (sign read below)
            "threshold": _pearson(sub["thresh_front"].to_numpy(), resid),  # front-count cutoff
            "wprime_burn": _pearson(sub["wprime_burn"].to_numpy(), resid), # forwardness-energy burn
            "front_base": _pearson(sub["front_base"].to_numpy(), resid),   # 0061's residual carrier
            "fast_front": _pearson(sub["fast_front"].to_numpy(), resid),
        }
        out["obs"][obs] = {
            "n": sub.height,
            "naive_rho": base_rho["naive"], "naive_rmse": _rmse(naive, y),
            "solo_rho": base_rho["solo"], "solo_rmse": _rmse(solo, y),
            "eq_rho": _pearson(veq, y), "eq_rmse": _rmse(veq, y),
            "ceil_rho": ceil_rho, "ceil_rmse": ceil_rmse,
            "cond_rho": cond_rho, "pace_rho": pace_rho,
            "best_base_rho": max(abs(base_rho["naive"]), abs(base_rho["solo"])),
            "alt": alt,
        }
    return out


def _verdict(o: dict) -> str:
    """TRUE NULL vs MIS-SPEC, per the S1b discipline — DE-CONFOUNDED by the conditions+pace ceiling.

    The full ceiling can beat the (field-mean) baseline simply by re-adding distance/field-size/class
    (which the per-horse base baseline averages away) — that is NOT composition mis-spec. The
    composition increment is the FULL ceiling MINUS the conditions+per-horse-pace ceiling (no
    style/composition). Mis-spec is real only if THAT increment is material AND beats 0061's form.
    """
    ceil = o["ceil_rho"]
    pace = o["pace_rho"]                # conditions + per-horse pace, NO composition
    eq = o["eq_rho"]
    comp_gap = ceil - pace             # what STYLE/COMPOSITION adds over a non-composition pace model
    if comp_gap >= 0.03 and ceil - eq >= 0.03:
        return (f"MIS-SPEC (ceil {ceil:+.3f} vs no-composition pace ceiling {pace:+.3f}; "
                f"composition adds +{comp_gap:.3f}, 0061 form {eq:+.3f})")
    if comp_gap < 0.015:
        return (f"TRUE NULL (ceil {ceil:+.3f} ≈ no-composition pace ceiling {pace:+.3f}; "
                f"composition adds {comp_gap:+.3f} — the lift over baseline is CONDITIONS, not composition)")
    return f"WEAK/PARTIAL (composition adds {comp_gap:+.3f}; ceil {ceil:+.3f}, pace {pace:+.3f}, eq {eq:+.3f})"


def main() -> None:
    panel = load_panel()
    races = per_race(panel)
    print("=" * 100)
    print("S7b — FIELD-PACE CEILING  (mis-specification-vs-true-null; the S1b discipline, field dim)")
    print("=" * 100)
    print(f"min_field={MIN_FIELD}  min_prior_frac={MIN_PRIOR_FRAC}  front_style_cutoff={FRONT_STYLE}")
    print(f"total races with leak-free coverage: {races.height}")
    print("ridge ceiling = max OUT-OF-SAMPLE ρ(ridge[full composition set], realised observable),")
    print("  CV-tuned λ, pooled out-of-fold, seed=0. DIAGNOSTIC CEILING PROBE ONLY — never the model.\n")

    results = {v: analyse(races, v) for v in ("ST", "HV")}

    for v in ("ST", "HV"):
        r = results[v]
        print(f"\n{'#' * 100}\n##  {v}   (n_races={r['n']})\n{'#' * 100}")
        hdr = (f"  {'observable':10s} {'n':>5s} | {'NAIVE ρ':>8s} {'SOLO ρ':>8s} "
               f"{'0061 ρ':>8s} {'COND ρ':>8s} {'PACE ρ':>8s} {'CEIL ρ':>8s}")
        print(hdr)
        print("  COND = conditions-only (dist/field/class/draw); PACE = COND + per-horse base aggs,")
        print("         NO style/composition; CEIL = full composition set. Composition signal = CEIL−PACE.")
        print("  " + "-" * (len(hdr) - 2))
        for obs in OBSERVABLES:
            if obs not in r["obs"]:
                continue
            o = r["obs"][obs]
            print(f"  {obs:10s} {o['n']:5d} | {o['naive_rho']:+8.3f} {o['solo_rho']:+8.3f} "
                  f"{o['eq_rho']:+8.3f} {o['cond_rho']:+8.3f} {o['pace_rho']:+8.3f} "
                  f"{o['ceil_rho']:+8.3f}")
        print("\n  VERDICT per observable (TRUE NULL vs MIS-SPEC):")
        for obs in OBSERVABLES:
            if obs not in r["obs"]:
                continue
            print(f"    [{obs:9s}] {_verdict(r['obs'][obs])}")
        print("\n  ALTERNATIVE COUPLINGS — ρ(feature, realised − best-baseline residual):")
        for obs in OBSERVABLES:
            if obs not in r["obs"]:
                continue
            a = r["obs"][obs]["alt"]
            s = ", ".join(f"{k} {val:+.3f}" for k, val in a.items())
            print(f"    [{obs:9s}] {s}")

    # Headline cross-venue summary on the section-1 mean (0061's observable) + best alternative.
    print(f"\n{'=' * 100}\nHEADLINE\n{'=' * 100}")
    for v in ("ST", "HV"):
        o = results[v]["obs"].get("s1_mean")
        if o:
            print(f"  [{v}] s1_mean (0061's choice): naive ρ {o['naive_rho']:+.3f}, 0061 ρ "
                  f"{o['eq_rho']:+.3f}, CEILING ρ {o['ceil_rho']:+.3f}  →  {_verdict(o)}")
        # which observable carries the largest COMPOSITION signal (CEIL − no-composition PACE ceiling).
        gaps = {ob: oo["ceil_rho"] - oo["pace_rho"] for ob, oo in results[v]["obs"].items()}
        if gaps:
            top = max(gaps, key=gaps.get)
            print(f"       largest COMPOSITION signal (CEIL−PACE): [{top}] {gaps[top]:+.3f}")


if __name__ == "__main__":
    main()
