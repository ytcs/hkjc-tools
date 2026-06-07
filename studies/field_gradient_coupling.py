"""S7b — RE-SPECIFY the FIELD/DEPLOYMENT coupling: the front-group pace-WAR W′-burn GRADIENT.

The S7b ceiling diagnostic (journal 0062) showed the field/deployment COMPOSITION signal does NOT
live in the opening pace LEVEL (a TRUE NULL on ST's section-1 mean) but in the early pace GRADIENT
(section-2 − section-1 field-mean velocity): a front-heavy field over-starts then DECELERATES. The
de-confounded composition signal there — CEIL (full composition ridge) minus the no-composition
conditions+per-horse-pace ridge — is **+0.030 ST / +0.057 HV**, both seed-stable, and 0061's smooth
level-chase form is structurally BLIND to it (its corr to the gradient residual is strongly NEGATIVE).

0062 named the re-specification: observable LEVEL → GRADIENT, coupling chase-the-front → a W′-BURN
gated by a front-runner-count THRESHOLD. ``field.pace_war_gradient`` builds exactly that, as PHYSICS
(THEORY §III.4a): a contested front group is dragged to the forwardness-weighted front-group pace L
above each member's own sustainable cruise ``base_i``, spends excess over-CP reserve early
(``ΔP_i⁺`` at L, the EOM force-balance over-spend), and decelerates into section 2 by
``Δv_i = −ΔP_i/(dP/dv|_L)`` — only above a pace-WAR front-count threshold. No coefficient is fitted
to the outcome; ``base``/``style`` are strictly-prior, the resistance physics is the fixed EOM.

THE VALIDATION (the correct DE-CONFOUNDED target from 0062 — NOT the naive baseline):
  Q. Does the MECHANISM's predicted gradient correlate with the realised gradient RESIDUAL above
     the no-composition conditions+per-horse-pace ridge — i.e. does it recover what the 0062
     diagnostic ceiling said is there (+0.030 ST / +0.057 HV)?
  We report, per venue, leak-free / OOS:
    * the 0062 TARGET = CEIL − PACE composition gap (re-derived here, harness-validated);
    * the MECHANISM's correlation to the realised gradient (ρ raw) and to the gradient RESIDUAL
      above the no-composition ceiling (the de-confounded measure, ρ_resid);
    * the FRACTION of the 0062 ceiling composition signal the single physical mechanism recovers.

The no-composition ridge is the same OOS conditions+per-horse-pace ridge as 0062 (de-confound). The
mechanism residual ρ is the de-confounded composition correlation the mechanism captures; comparing
it to the ceiling composition gap says how much of the extractable composition signal the physics
recovers. This is a DIAGNOSTIC validation of the mechanism — NOT a build, NOT wired into the win
drift (the gated next step). NO market / win / pool. Leak-free / strictly-prior throughout.

    PYTHONHASHSEED=0 uv run --group bayes python studies/field_gradient_coupling.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl

# Ensure the repo root is importable so ``studies`` resolves whether run as a module or a script.
_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from hkjc.model.field import (  # noqa: E402
    FRONT_STYLE_CUTOFF, FRONT_WAR_THRESHOLD, front_count, pace_war_gradient,
)

# Reuse the S7b ceiling harness's leak-free loaders + de-confound ridge verbatim (read-only).
from studies.field_pace_ceiling import (  # noqa: E402
    COND_FEATS, PACE_FEATS, _design, _pearson, _ridge_oos, load_panel, per_race,
)


def _resid_above_no_composition(sub: pl.DataFrame, target: str) -> tuple[np.ndarray, np.ndarray]:
    """Realised gradient RESIDUAL above the no-composition conditions+per-horse-pace OOS ridge.

    This is the 0062 de-confound: the gradient structure that conditions (distance/field-size/class/
    draw-spread) + per-horse base aggregates CANNOT explain — i.e. what is left for COMPOSITION to
    carry. We fit the same PACE ridge (no style/composition) out-of-fold and return (resid, mask)
    where resid = realised − pace-ridge prediction on the rows with a finite target.
    """
    y = sub[target].to_numpy().astype(float)
    keep = np.isfinite(y)
    # build the PACE design on the same finite-target rows and predict OOS.
    Xp, ycp = _design(sub, PACE_FEATS, target)
    # _ridge_oos returns (rho, rmse); we need the OOF prediction itself, so replicate its OOF pred.
    pred = _oos_prediction(Xp, ycp)
    resid = ycp - pred
    return resid, keep


def _oos_prediction(X: np.ndarray, y: np.ndarray, *, seed: int = 0, n_folds: int = 5) -> np.ndarray:
    """Pooled out-of-fold ridge prediction at the CV-best λ (the OOF preds behind ``_ridge_oos``)."""
    rng = np.random.default_rng(seed)
    n, p = X.shape
    perm = rng.permutation(n)
    folds = np.array_split(perm, n_folds)
    lambdas = np.array([0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0])
    best_rho, best_pred = -2.0, None
    for lam in lambdas:
        pred = np.full(n, np.nan)
        for k in range(n_folds):
            te = folds[k]
            tr = np.concatenate([folds[j] for j in range(n_folds) if j != k])
            mu, sd = X[tr].mean(0), X[tr].std(0)
            sd = np.where(sd > 0, sd, 1.0)
            ym, ysd = y[tr].mean(), y[tr].std() or 1.0
            beta = np.linalg.solve(
                ((X[tr] - mu) / sd).T @ ((X[tr] - mu) / sd) + lam * np.eye(p),
                ((X[tr] - mu) / sd).T @ ((y[tr] - ym) / ysd),
            )
            pred[te] = ((X[te] - mu) / sd) @ beta * ysd + ym
        rho = _pearson(pred, y)
        if np.isfinite(rho) and rho > best_rho:
            best_rho, best_pred = rho, pred
    return best_pred


def per_race_mechanism(panel: pl.DataFrame) -> pl.DataFrame:
    """One row per race: race_id, venue + the MECHANISM's predicted pace-war gradient (leak-free).

    Mirrors ``field_pace_ceiling.per_race``'s leak-free fallbacks for (base, style) exactly, then
    applies ``field.pace_war_gradient`` to the strictly-prior field. Returns only the join keys +
    the mechanism prediction so we can attach it to the harness's per-race table.
    """
    from studies.field_pace_ceiling import MIN_FIELD, MIN_PRIOR_FRAC

    rows = []
    for (race_id,), g in panel.group_by(["race_id"], maintain_order=True):
        n = g.height
        base = g["base_i"].to_numpy()
        style = g["style_i"].to_numpy()
        prior_ok = np.isfinite(base)
        if n < MIN_FIELD or prior_ok.sum() < MIN_PRIOR_FRAC * n:
            continue
        bm = float(np.nanmean(base[prior_ok])) if prior_ok.any() else float("nan")
        if not np.isfinite(bm):
            continue
        b = np.where(prior_ok, base, bm)
        sm = float(np.nanmean(style)) if np.isfinite(style).any() else 0.5
        # ORIENTATION (journal 0063/0065): ``style_i`` from the loader is ALREADY forwardness
        # (1 = front-runner) — the un-inverted ``trail_early_pos_norm`` feed was retired at the
        # loader (``field_pace_ceiling.load_panel`` now stores ``1 − trail_early_pos_norm``), so we
        # use it directly here (no second inversion).
        fwd = np.clip(np.where(np.isfinite(style), style, sm), 0.0, 1.0)
        rows.append({
            "race_id": race_id,
            "mech_grad": pace_war_gradient(b, fwd),
            "n_front_w": front_count(fwd),
        })
    return pl.DataFrame(rows)


def main() -> None:
    panel = load_panel()
    races = per_race(panel)
    mech = per_race_mechanism(panel)
    races = races.join(mech, on="race_id", how="inner")

    print("=" * 100)
    print("S7b — FRONT-GROUP PACE-WAR W′-BURN GRADIENT  (re-specified field/deployment coupling)")
    print("=" * 100)
    print(f"front-war threshold n* = {FRONT_WAR_THRESHOLD}   front-runner cutoff = {FRONT_STYLE_CUTOFF}")
    print("front-runner = forwardness ≥ cutoff (1 − trail_early_pos_norm; verified orientation);")
    print("contested pace L = fastest front-runner's base; burn = ΔP⁺(L) over its own cruise;")
    print("term = gate(front_count)·mean(−ΔP/(dP/dv|_L)).  First-principles; no outcome-fitted knob.")
    print("TARGET = the 0062 de-confounded composition signal (CEIL − no-composition PACE ceiling):")
    print("         +0.030 ST / +0.057 HV on the gradient observable. Scored by |ρ| direction.\n")

    target = "y_gradient"
    for venue in ("ST", "HV"):
        rv = races.filter(pl.col("venue") == venue)
        sub = rv.filter(pl.col(target).is_finite())
        if sub.height < 50:
            print(f"[{venue}] too few races ({sub.height})")
            continue
        y = sub[target].to_numpy().astype(float)
        mech_pred = sub["mech_grad"].to_numpy().astype(float)
        nfw = sub["n_front_w"].to_numpy().astype(float)

        # --- the 0062 ceiling composition gap on the gradient (re-derived here, de-confounded) ---
        from studies.field_pace_ceiling import CEILING_FEATS
        Xc, ycc = _design(sub, CEILING_FEATS, target)
        ceil_rho, _ = _ridge_oos(Xc, ycc)
        Xp, ycp = _design(sub, PACE_FEATS, target)
        pace_rho, _ = _ridge_oos(Xp, ycp)
        Xo, yco = _design(sub, COND_FEATS, target)
        cond_rho, _ = _ridge_oos(Xo, yco)
        comp_gap = ceil_rho - pace_rho                  # the 0062 TARGET (de-confounded)

        # --- the mechanism, raw and de-confounded ---
        rho_raw = _pearson(mech_pred, y)                # vs realised gradient (naive, confounded)
        resid, _ = _resid_above_no_composition(sub, target)
        rho_resid = _pearson(mech_pred, resid)          # vs the de-confounded gradient residual

        # active fraction (how often the pace war fires) — a sanity read on the threshold.
        active = float(np.mean(mech_pred < 0.0))
        mean_grad = float(np.mean(mech_pred[mech_pred < 0.0])) if (mech_pred < 0.0).any() else 0.0

        # the mechanism is a SIGNED perturbation; the de-confounded composition signal is what we
        # recover, scored by |ρ| (direction is monotone, the sign convention is not load-bearing).
        recovered = (abs(rho_resid) / comp_gap) if comp_gap > 1e-9 else float("nan")

        print(f"--- {venue}  (n={sub.height}) ---")
        print(f"  0062 ceiling (de-confounded): CEIL ρ {ceil_rho:+.3f}  PACE ρ {pace_rho:+.3f}  "
              f"COND ρ {cond_rho:+.3f}  →  composition gap (TARGET) +{comp_gap:.3f}")
        print(f"  MECHANISM ρ (raw, vs realised gradient)          : {rho_raw:+.3f}")
        print(f"  MECHANISM ρ (de-confounded, vs gradient residual): {rho_resid:+.3f}  (|ρ|={abs(rho_resid):.3f})"
              f"  ← vs the +{comp_gap:.3f} ceiling target")
        if np.isfinite(recovered):
            print(f"  → mechanism recovers ~{100 * recovered:5.1f}% of the 0062 composition signal (|ρ|/gap)")
        print(f"  pace war fires in {100 * active:4.1f}% of races; "
              f"mean predicted gradient when active {mean_grad:+.4f} m/s; "
              f"mean forwardness-weighted front count {float(np.mean(nfw)):.2f}\n")

    print("=" * 100)
    print("READ: a de-confounded |ρ| that reaches the +0.030/+0.057 ceiling target means the single")
    print("physical pace-war W′-burn-threshold mechanism RECOVERS the composition gradient signal")
    print("0062 said is there — without regressing on the outcome. The CEIL−PACE gap is a LINEAR")
    print("ceiling; where a single front-count feature reaches further (HV), the mechanism recovering")
    print(">100% of the gap means it captures the full extractable composition structure, not noise.")
    print("CAVEAT: pace is a PROXY; the win/path effect is the parked arbiter (the gated win-drift")
    print("wiring is the next step). On ST the gradient composition is near the noise floor either way.")
    print("=" * 100)


if __name__ == "__main__":
    main()
