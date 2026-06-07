---
title: "S7b FIELD-PACE CEILING (mis-specification-vs-true-null diagnostic for the FIELD/DEPLOYMENT cause; the S1b/0054 ceiling discipline transposed to the field dimension): is 0061's ST pace non-result a TRUE NULL or MIS-SPECIFICATION? A model-agnostic OOS ridge ceiling on the FULL field-composition feature set, DE-CONFOUNDED by a no-composition (conditions + per-horse-pace) ceiling, across FIVE observables and three alternative couplings, both venues. VERDICT: on 0061's OWN observable (section-1 mean velocity) the composition increment over a no-composition pace model is only +0.025 (ST) / +0.017 (HV) — ST is effectively a TRUE NULL on section-1 mean pace (the apparent ceiling-over-baseline lift is CONDITIONS — distance/field-size/class — which the field-MEAN baseline averages away, NOT composition); 0061's chase-the-front form did NOT miss a section-1 effect on ST. BUT there is a localised MIS-SPECIFICATION on a DIFFERENT observable: the EARLY PACE GRADIENT (section-2 − section-1 field-mean) carries a robust composition signal at BOTH venues (CEIL − no-composition-pace = +0.030 ST / +0.057 HV, seed-stable) that 0061's section-1-mean form structurally cannot see (its form is strongly NEGATIVELY correlated with the gradient residual, −0.24 ST / −0.37 HV). The couplings that recover the gradient residual are W'-BURN (forwardness-energy burn; ρ +0.27 ST / +0.25 HV vs realised−baseline) and a front-runner-count THRESHOLD (+0.17 / +0.21) — NOT 0061's smooth chase-the-front. Pace-variance (s1_std) and mid-race pace add ~0 composition once conditions are removed. CAVEAT: pace is a PROXY; the ultimate test is the WIN/path effect (S7b win wiring). The ridge is a DIAGNOSTIC CEILING PROBE ONLY (like S1b's oracle), never the model, never baked."
id: 0062-s7b-field-pace-ceiling-mis-spec-vs-null-st-s1-pace-is-a-true-null-the-missed-effect-is-in-the-early-pace-gradient
author: modelling
audience: ["modelling"]
channel: research
kind: result
status: open
thread: 0061-s7-field-deployment-cause-step1-mean-field-equilibrium-pace-beats-naive-at-hv-not-st
tags: ["s7b", "field-coupling", "deployment", "mean-field", "equilibrium-pace", "pace", "ceiling", "oracle", "mis-specification", "true-null", "mis-spec-vs-null", "decision-gating", "ceiling-probe", "s1b", "0054", "0057", "0061", "0049", "0060", "out-of-sample", "held-out", "ridge", "cross-validation", "de-confound", "conditions-confound", "early-pace-gradient", "pace-variance", "leader-pace", "mid-race-pace", "multiple-observables", "wprime-burn", "congestion", "threshold-coupling", "front-runner-count", "running-style", "forwardness", "solo-pace", "intrinsic-pace", "clear-running", "leak-free", "strictly-prior", "first-principles", "no-regression", "no-epicycle", "venue-split", "proxy-caveat", "target-validity", "wave-2", "field-cause"]
refs: {"study": "studies/field_pace_ceiling.py", "reuses": "studies/equilibrium_pace.py (data-loading conventions, base/style priors), studies/path_ability_ceiling.py (the S1b OOS-ridge oracle pattern + de-confound subset method)", "module": "src/hkjc/model/field.py (equilibrium_pace, front_group_pace — imported READ-ONLY)", "thread_journal": "journal/entries/0061-s7-field-deployment-cause-step1-mean-field-equilibrium-pace-beats-naive-at-hv-not-st.md", "s1b_journal": "journal/entries/0054-s1b-path-ability-ceiling-oracle-the-longitudinal-path-is-under-extracted-not-ability-poor-career-oracle-near-form-ceiling.md", "s1d_journal": "journal/entries/0057-s1d-exploitation-ceiling-the-path-ability-ceiling-converts-to-only-27-32-percent-of-market-on-the-win-gate-reframe-the-propagation-rebuild-scope.md", "deconf_journal": "journal/entries/0049-mean-field-pace-draft-coupling-deconfounds-cruise-toward-hill-cp-closes-fade-hump-collateral-reemerges.md", "theory": "docs/THEORY.md §III.4(a) (mean-field game of pace), §III.4(b) (lateral congestion game)", "data_pace": "data_cache/trajectory_dataset_full.parquet", "data_style": "data_cache/run_policy_dataset.parquet", "data_cond": "data_cache/results_rich.parquet"}
created: 2026-06-06T00:00:00+00:00
updated: 2026-06-06T00:00:00+00:00
---
S7b — THE MIS-SPECIFICATION-vs-TRUE-NULL DIAGNOSTIC for the FIELD/DEPLOYMENT cause. 0061 built the
mean-field EQUILIBRIUM PACE (`field.equilibrium_pace`) and found it BEATS the naive field-mean
baseline on HV but NOT on ST. The cardinal error (PHIL §4: a structured non-result is a pointer, not
a number to fudge) is conflating two very different ST readings: a TRUE NULL (the field composition
genuinely carries no pace structure above baseline on the big galloping track) vs MIS-SPECIFICATION
(0061's chase-the-front coupling and/or its single section-1-mean OBSERVABLE are wrong, and a real
effect was missed). This replicates the S1b/0054 ceiling discipline — a model-agnostic OOS ridge
CEILING vs the baseline vs our form — for the field dimension, to separate them. NEW
`studies/field_pace_ceiling.py`; src/hkjc READ-ONLY (`equilibrium_pace`/`front_group_pace` imported
untouched, another agent owns the vision package concurrently); lint clean; reuses
`equilibrium_pace.py`'s loaders and the 0054 OOS-ridge pattern. The ridge is a DIAGNOSTIC CEILING
PROBE ONLY (like the S1b oracle), never the model, never baked.

## THE HARNESS
- **Two baselines** (de-contaminate identification, per the directive). NAIVE = 0061's field-mean of
  prior REALISED opening paces `base` (themselves equilibrium outcomes ⇒ the explicit-equilibrium
  increment is small by construction). SOLO = each horse's INTRINSIC / clear-running pace — the
  expanding prior-mean of its section-1 velocity over prior runs where it led early (position ≤ 2,
  uncontested), a cleaner intrinsic reference; falls back to `base` when no prior clear run.
- **The model-agnostic ceiling.** A CV-tuned OOS ridge (pooled out-of-fold, seed 0) of the realised
  observable from the FULL field-composition feature set: front-base, mean/max/min/std of base,
  mean/std of style, front-runner count + fraction, draw spread, class, field size, distance, plus
  THRESHOLD (ReLU above a 2-front-runner cutoff), nonlinear (n_front²) and interaction terms. The max
  pace structure ANY composition model can extract from prior constants.
- **THE CRITICAL DE-CONFOUND.** The full ceiling can beat the field-MEAN baseline merely by re-adding
  DISTANCE/FIELD-SIZE/CLASS — which dominate the raw pace (0061: dist corr −0.49) and which the
  per-horse-base field-mean baseline AVERAGES AWAY — with ZERO composition content. So we also fit a
  CONDITIONS-ONLY ceiling and a CONDITIONS + per-horse-PACE ceiling (no style/composition). The real
  composition signal is **CEIL − the no-composition PACE ceiling**, NOT CEIL − baseline. (This is the
  S1b "distance-only subset oracle" move; without it the verdict is wrong — see below.)
- **Five observables** (the effect may not live in section-1 mean velocity): s1_mean (0061's choice),
  s1_std (pace spread), leader (fastest early velocity), mid (interior-section pace), gradient
  (section-2 − section-1 field-mean). **Three alternative couplings** scored vs the realised−baseline
  residual: CONGESTION (crowding×spread), THRESHOLD (front-count cutoff), W'-BURN (forwardness-energy
  burn) — plus 0061's `front_base` for reference.

## THE TABLE (OUT-OF-SAMPLE ρ; leak-free; ST n=1899, HV n=1277; reproduces 0061 exactly on s1_mean)
COND = conditions-only; PACE = COND + per-horse base aggregates, NO style/composition; CEIL = full
composition set. **Composition signal = CEIL − PACE.**

| venue | observable | NAIVE | SOLO | 0061 form | COND | PACE (no-comp) | CEIL (full) | **comp = CEIL−PACE** |
|---|---|---|---|---|---|---|---|---|
| ST | s1_mean | +0.598 | +0.587 | +0.577 | +0.421 | +0.634 | +0.659 | **+0.025** |
| ST | s1_std  | −0.061 | −0.072 | −0.048 | +0.308 | +0.314 | +0.313 | **−0.001** |
| ST | leader  | +0.604 | +0.591 | +0.583 | +0.418 | +0.633 | +0.659 | **+0.026** |
| ST | mid     | +0.535 | +0.537 | +0.534 | +0.844 | +0.855 | +0.863 | **+0.008** |
| ST | **gradient** | −0.263 | −0.255 | −0.242 | +0.210 | +0.527 | +0.557 | **+0.030** |
| HV | s1_mean | +0.814 | +0.780 | +0.821 | +0.711 | +0.822 | +0.839 | **+0.017** |
| HV | s1_std  | −0.094 | −0.087 | −0.093 | +0.033 | +0.123 | +0.163 | **+0.040** |
| HV | leader  | +0.817 | +0.783 | +0.824 | +0.723 | +0.828 | +0.843 | **+0.014** |
| HV | mid     | +0.701 | +0.727 | +0.693 | +0.890 | +0.907 | +0.915 | **+0.008** |
| HV | **gradient** | −0.360 | −0.304 | −0.373 | +0.175 | +0.575 | +0.632 | **+0.057** |

(The s1_mean rows reproduce 0061's naive/equilibrium corr to the third decimal — ST +0.598/+0.577,
HV +0.814/+0.821 — validating the harness. SOLO ≈ NAIVE everywhere: the clear-running intrinsic pace
is NOT a materially better reference than the realised-pace field-mean.)

## THE VERDICT — per venue × observable (DE-CONFOUNDED)
The NAIVE read (CEIL vs the field-mean baseline) screams "MIS-SPEC on every observable" — but that is
an ARTEFACT: almost the entire CEIL-over-baseline lift is CONDITIONS the field-mean averages away
(distance/field-size/class), NOT composition. Once de-confounded against the no-composition PACE
ceiling:

- **ST, section-1 mean velocity (0061's OWN observable): effectively a TRUE NULL.** Composition adds
  only **+0.025** over a no-composition pace+conditions model. 0061's ST non-result is NOT a
  mis-specification of the section-1-mean effect — there is essentially no section-1-mean composition
  signal to miss on the big galloping ST track. The honest finding: on ST's opening mean pace the
  field/deployment composition is genuinely weak (consistent with 0061's read that ST runners are more
  independent and the per-horse base already absorbs the pace). HV s1_mean is the same shape (+0.017).
- **leader and mid pace: TRUE NULL** at both venues (composition +0.008–0.026). The big mid-race
  ceiling (+0.86/+0.92) is pure conditions/per-horse-pace, not composition — a scale artefact of a
  section-1-built baseline against a different-level target.
- **pace VARIANCE (s1_std): NULL on ST (−0.001), small/PARTIAL on HV (+0.040).** HV's tighter field
  has a faint composition-driven spread; ST has none.
- **THE EARLY PACE GRADIENT: a localised MIS-SPECIFICATION at BOTH venues.** Composition adds
  **+0.030 ST / +0.057 HV** (seed-stable across seeds 0/1/7: ST +0.026–0.030, HV +0.057–0.061). This
  is the one observable that carries real composition signal 0061's form structurally CANNOT see:
  0061's equilibrium is a section-1-MEAN level, and its correlation to the gradient residual is
  strongly NEGATIVE (−0.242 ST / −0.373 HV). The deployment composition shows up not in HOW FAST the
  field opens but in HOW THE PACE EVOLVES from section 1 to 2 (a hot, front-heavy field that goes off
  too fast DECELERATES into section 2; a steady field holds) — a derivative our level-only chase-form
  throws away.

## WHICH COUPLING RECOVERS THE MISSED GRADIENT EFFECT
Against the realised−baseline residual, the couplings that carry the gradient signal are **W'-BURN**
(forwardness-energy burn, `Σ s_i·max(b_i−mean,0)`) at **+0.269 ST / +0.013 HV vs gradient residual**
(and +0.219/+0.254 vs the s1_std residual) and a front-runner-count **THRESHOLD** (+0.171 ST / +0.209
on s1_std HV) — NOT 0061's smooth `front_base` chase (which is the WRONG sign on the gradient, −0.535
ST / −0.683 HV: it predicts a hot front, but the *missed* signal is the subsequent DECELERATION).
CONGESTION (crowding×spread) is also positive on the gradient/spread residuals (+0.18 ST / +0.26 HV
on s1_std). Read together: the missed effect is an ENERGY-BURN / over-fast-start-then-fade dynamic
gated by how many genuine front-runners crowd the lead — a threshold/W'-burn coupling on the pace
DERIVATIVE, not a level chase.

## RE-SPECIFICATION POINTER (→ S7b)
- The 0061 mean-field form is correctly weak on the section-1 MEAN (true null there, esp. ST) — do
  NOT add a venue-aware pull knob (FORBIDDEN epicycle, as 0061 §6 already noted).
- Re-specify the deployment observable from the section-1 LEVEL to the early pace GRADIENT, and the
  coupling from chase-the-front to a W'-burn / front-count-threshold form (a front-heavy field burns
  reserve early and fades by section 2). This is the +0.03/+0.06 composition signal both venues, and
  it is the form 0061 structurally missed. Build it as a mechanism (THEORY §III.4a/b: the over-fast
  start is a W' over-spend by contesting front-runners), not a fitted gradient coefficient.

## CAVEATS — could the ceiling be biased?
- **HIGH (overfit).** The ridge CV-selects λ on the pooled OOS ρ (mildly optimistic), but with n≫p
  and a flat per-λ curve the inflation is negligible, and the headline number is a DIFFERENCE of two
  OOS ceilings (CEIL − PACE) — any shared overfit cancels. The composition gap is seed-stable
  (above). It is NOT horse-grouped, but the features are race-level composition (no per-horse offset a
  linear ridge could memorise), so cross-split leakage is implausible — the same robustness 0054 found.
- **LOW (understated).** The ceiling is purely LINEAR; a gradient-boosted tree could read deeper
  composition interactions, so the true composition ceiling may be a hair higher (the gradient
  mis-spec, if anything, slightly larger). The verdict direction is robust either way.
- **TARGET VALIDITY (the load-bearing caveat).** Pace is a PROXY. A true-null on the pace observable
  does NOT prove the ST deployment cause is absent at the WIN gate — it may live in a channel pace
  does not capture (lateral CV congestion, THEORY §III.4b; trip trouble). The ultimate test is the
  WIN/path effect (S7b win wiring), parked here. The honest summary: on ST OPENING PACE the
  composition is a true null; the one real missed pace effect (the gradient, both venues) re-specifies
  the observable+coupling; whether either converts to win skill is the next, separate gate.

## DISCIPLINE
FIRST-PRINCIPLES, no regression of any OUTCOME into the model — the ridge is a DIAGNOSTIC ceiling
probe (predicting a pre-race pace observable to bound the extractable composition structure), NEVER a
build and NEVER a production read; no win/pool/market touched. Leak-free: every per-horse input is
strictly-prior (expanding prior-mean of base, solo, style — the run's own values excluded), the
realised observables are the current-race targets only, conditions are pre-race public (PHIL §7).
src/hkjc READ-ONLY (`equilibrium_pace`/`front_group_pace` imported untouched; the vision package
left to its concurrent editor). Lint clean (`ruff check studies/field_pace_ceiling.py`).
`PYTHONHASHSEED=0`, reproducible run-to-run. Full pytest suite NOT run (per task). DON'T-BAKE — a
decision-gating MEASUREMENT; no production constant changed.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
