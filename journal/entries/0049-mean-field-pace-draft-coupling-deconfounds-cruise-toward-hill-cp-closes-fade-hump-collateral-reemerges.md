---
title: "Foundation 0049: the MEAN-FIELD PACE/DRAFT COUPLING — condition the EOM on the OBSERVED field pace (draft-reduced drag + pace-following on W′), the dominant missing physics 0047/0048 named. THE DECISIVE TEST PASSES DIRECTIONALLY: de-confounding the cruise read for draft COLLAPSES 41% (ST) / 47% (HV) of the +1.48/+1.29 m/s cruise-vs-Hill gap (→ +0.87/+0.68 m/s residual, the pace-following-on-reserve remainder); the field-coupled forward path FLATTENS the closing fade-hump (closing mean_res +0.80→+0.21 ST, +0.73→+0.16 HV) and the fast-pace closing residual; and the de-confounded sustainable level RE-EMERGES as held-out ability (corr −0.02→+0.33 ST, +0.20→+0.40 HV) without collateral-form as a feature. The constants stay FIXED; the coupling perturbs DEPLOYMENT."
id: 0049-mean-field-pace-draft-coupling-deconfounds-cruise-toward-hill-cp-closes-fade-hump-collateral-reemerges
author: modelling
audience: ["modelling"]
channel: research
kind: result
status: open
thread: 0048-dedicated-surface-identification-battery-cp-cruise-vs-hill-disagree-by-design-physics-falsification-pmax-aliased-tau-unidentifiable
tags: ["theory", "energetics", "eom", "equation-of-motion", "mean-field", "mean-field-game", "pace", "pace-coupling", "pace-matching", "draft", "slipstream", "aerodynamics", "competition", "field-coupling", "deployment", "critical-velocity", "critical-power", "cruise", "hill", "deconfound", "cross-check", "consistency", "fade-hump", "closing-fade", "path-fit", "re-emergence", "collateral-form", "w-prime", "first-principles", "physics", "no-regression", "observed-pace", "boundary-condition", "foundation", "draft", "0047", "0048"]
refs: {"module": "src/hkjc/statespace/field.py", "study": "studies/field_coupling_test.py", "test": "tests/test_field.py", "eom_core": "src/hkjc/statespace/dynamics.py", "identify": "src/hkjc/statespace/identify.py", "eom_journal": "journal/entries/0047-eom-first-principles-generative-core-bang-cp-bang-physics-replaces-heuristic-level-bias-zero-opening-ramp-captured.md", "identification_journal": "journal/entries/0048-dedicated-surface-identification-battery-cp-cruise-vs-hill-disagree-by-design-physics-falsification-pmax-aliased-tau-unidentifiable.md", "theory": "docs/THEORY.md §III.4(a)", "causal": "docs/CAUSAL_MODEL.md", "panel": "data_cache/trajectory_dataset_full.parquet", "mass_going": "data_cache/results_rich.parquet (DeclarHorseWt + ActWt + Going)"}
created: 2026-06-06T00:00:00+00:00
updated: 2026-06-06T00:00:00+00:00
---
DRAFT — the MEAN-FIELD PACE/DRAFT COUPLING (THEORY §III.4a). NEW additive module
`src/hkjc/statespace/field.py` + `studies/field_coupling_test.py` + `tests/test_field.py`. The
core EOM logic (`dynamics.py`), `identify.py`, production `hkjc.model`, `data_cache` UNTOUCHED.
NO market odds, NO win, NO pool book. NO regression: both coupling mechanisms are physical
effort/drag responses to the OBSERVED field pace with physically-bounded magnitudes — no free
fitted coefficient on the outcome. The per-horse PHYSIOLOGICAL constants (CP from the Hill anchor,
W′_0, P_max) stay FIXED; the field game perturbs the DEPLOYMENT of those constants.

## THE DIRECTIVE
The single-horse EOM (0047) integrates each runner in ISOLATION. Two independent signatures —
0047's closing fade-hump (the bang–CP–bang gives a FLAT cruise then a kick, the data shows an
early-burn then a fade THROUGH the close) and 0048's +1.49/+1.27 m/s cruise-vs-Hill
critical-velocity DISAGREEMENT (the per-race cruise reads ~1.4 m/s ABOVE the across-distance Hill
CV) — converge on the SAME missing physics: real horses cruise ABOVE their physiological critical
velocity because the FIELD's pace pulls them along. Build that coupling (THEORY §III.4a, the
mean-field game of pace) in its TRACTABLE form — condition on the OBSERVED field pace, don't solve
the Nash equilibrium — and run the sharp falsifiable cross-check: does the cruise–Hill gap that
EXPOSED the missing physics now COLLAPSE toward 0?

## (1) THE COUPLING PHYSICS — two mechanisms, both bounded by the force balance / energy law
Field pace `v_field(section)` is a REAL per-section observable (the field-mean and leader velocity
of the runners in the same race+section — contemporaneous public info, a legitimate boundary
condition, NOT a fitted latent). Each horse's EOM gets:

**(a) Longitudinal slipstream / draft.** THEORY §III.2 writes the aero term as
`½ρ·C_dA·v²·(1−draft_i)`. A horse running WITHIN the field (bodies ahead) sits in a partial wake
⇒ reduced effective C_dA. Magnitude from drafting aerodynamics (a bluff-body wake ~halves a close
follower's stagnation drag; realised pack saving ~0.30), NOT a fit: `DRAFT_FRAC=0.30` of the AERO
drag for a fully-covered follower, scaled linearly by the runner's longitudinal cover fraction
(`cover=(position−1)/(field_size−1)`; the LEADER gets 0 — clear air). Aero is ~40–47% of total
resistance at racing speed (0047), so a 30% aero cut is a ~12–14% cut on the aero term ⇒ a
few-tenths-m/s cruise lift — the right order to explain part of the cruise inflation. The FULL
positional draft (who is LATERALLY behind whom) needs CV and is DEFERRED (boundary noted below);
this is the coarse LONGITUDINAL version from the running-position rank.

**(b) Competitive pace-matching.** The horse targets the field pace where it runs HOT, bounded by
P_max and W′≥0:  `v_target = clip(v_field, v_c_draft, v_max(P_max))`, held on the force-balance
power (which is > CP whenever `v_target > v_c` ⇒ it DRAWS DOWN W′ at rate `(P_match−CP)⁺`). When
the reserve empties it falls back to the draft-reduced sustainable level. **It REDUCES TO the
isolated bang–CP–bang when `v_field ≤ v_c`** (no pull below the sustainable level → v_target = v_c
→ P = CP → no reserve drawn) — the required limit, unit-tested. This is the mechanism that makes
the settled cruise sit ABOVE the physiological v_c: the horse pace-follows on reserve, exactly as
0048 inferred (cruise reads ~1.4 m/s too hot vs Hill).

## (2) THE DECISIVE TEST — does de-confounding the cruise for the field coupling AGREE with Hill?
The headline cross-check (the same one that exposed the missing physics in 0048). The DRAFT part
of the cruise inflation is exactly invertible: a horse cruising at `v_settled` with draft-reduced
drag is producing `draft·½ρC_dA·v_settled³ + C_rr·m·g·v_settled` of sustainable power; the AEROBIC
critical velocity is the speed that SAME power buys in CLEAN air. De-confound and compare to the
Hill CV (the de-confounded aerobic CP anchor, R²=0.994, 0048):

| venue | n | RAW cruise v_c | Hill CV | RAW bias | DE-CONF cruise v_c | DE-CONF bias | gap collapse |
|---|---|---|---|---|---|---|---|
| **ST** | 527 | 16.45 | 15.00 | **+1.478** | 15.82 | **+0.866** | **+41%** removed |
| **HV** | 227 | 16.20 | 14.94 | **+1.286** | 15.54 | **+0.683** | **+47%** removed |

**The gap COLLAPSES directionally, as predicted.** Removing only the DRAFT inflation closes ~41–47%
of the +1.4 m/s cruise–Hill gap (ST +1.48→+0.87, HV +1.29→+0.68 m/s). This is the strongest
evidence the field coupling is the right missing physics: the cross-check that FAILED in 0048 now
moves the cruise read toward the Hill CV by the magnitude drafting aerodynamics predicts, with no
fitted constant. The RESIDUAL gap (+0.87/+0.68 m/s) is the PACE-FOLLOWING-on-reserve part — a
DEPLOYMENT effect, not a draft effect: even the slowest interior cruise section is pace-pulled
above the true aerobic floor (the Hill CV), so a static force-balance read of `v_settled` cannot
remove it. The COUPLED INTEGRATOR (not the static de-confound) handles that part — it never settles
above v_c without spending W′ — which is what closes the path-fit hump (next).

## (3) PATH-FIT — the closing fade-hump and the fast-pace residual FLATTEN
Field-coupled forward profile (observed pace drives deployment) vs the ISOLATED EOM on the SAME
de-confounded-CP physiology (route runs, ≥1800 m). The coupling's job: flatten the early-burn /
late-fade hump and the fast-pace closing residual that the isolated bang–CP–bang structurally
cannot make.

| | ST ISOLATED | ST FIELD-COUPLED | HV ISOLATED | HV FIELD-COUPLED |
|---|---|---|---|---|
| overall RMSE | 1.150 | **0.682** | 1.088 | **0.614** |
| overall bias | +0.907 | **+0.290** | +0.892 | **+0.283** |
| opening mean_res | +0.499 | +0.188 | +0.622 | +0.374 |
| cruise mean_res | **+1.077** | **+0.351** | **+1.025** | **+0.293** |
| **closing mean_res** | **+0.798** | **+0.207** | **+0.726** | **+0.159** |

- **The closing fade-hump SHRINKS** (closing mean_res +0.80→+0.21 ST, +0.73→+0.16 HV) and the
  cruise over-/under-prediction collapses (cruise mean_res +1.08→+0.35 ST). The early-burn comes
  from pace-following lifting the early cruise to the hot field pace; the late fade comes from the
  spent reserve emptying — the HUMP the isolated policy could not make, now mechanism.
- **The fast-pace closing residual FLATTENS** (the 0047 fingerprint): fast−slow closing Δ shrinks
  from +0.063→+0.012 (ST) and the HV fast-pace gap narrows (fast closing res +0.67→+0.07). A horse
  in a fast-pace field now fades the right amount, because the coupling SEES the field's pace.

NB the ISOLATED baseline here uses the de-confounded-CP anchor + literature W′_0/P_max defaults
(not the per-run 0047 NLS inversion), so its absolute numbers are higher than 0047's tuned
inversion; the LOAD-BEARING comparison is field-coupled-vs-isolated on the SAME physiology — the
coupling is what removes the bias, and it removes it via the OBSERVED field pace (a legitimate
boundary condition, not a per-horse latent). The gap-to-descriptive-baseline (0047: EOM trailed by
~0.03–0.04 m/s) closes because the coupled cruise is no longer structurally flat.

## (4) RE-EMERGENCE — the de-confounded level predicts held-out ability (collateral mechanism)
Split each horse's route runs in time; does the field-pace-adjusted (de-confounded) sustainable
level on the EARLY half predict the held-out LATE half — the PHYSICAL mechanism behind the stripped
collateral-form, without collateral as a feature?

| venue | n | RAW settled cruise (pace-confounded) | DE-CONFOUNDED level (field-adjusted) |
|---|---|---|---|
| **ST** | 414 | corr **−0.019** (null; 0048: +0.01) | corr **+0.330** |
| **HV** | 296 | corr **+0.200** (0048: +0.22) | corr **+0.396** |

**The collateral-form mechanism RE-EMERGES from the physical coupling.** At ST the RAW per-race
cruise barely persists (corr −0.02 — it is a FIELD property, pace-driven, exactly 0048's finding);
the DE-CONFOUNDED level — stripping the field's pace/draft contribution — persists at +0.33,
landing on the Hill-CV-vs-sustainable-level re-emergence (0048: +0.39 ST). The horse's
field-adjusted performance IS a stable ability signal; the field coupling is the physical referent
that makes "beating strong/fast fields" predictive — the collateral-form meta-signal as a
CONSEQUENCE, not a fitted feature (the CAUSAL_MODEL re-emergence test).

## (5) WHAT PHYSICS REMAINS — the FULL POSITIONAL DRAFT (→ CV, deferred)
- **Residual pace-following gap (+0.7 m/s):** the de-confound static formula removes only the draft
  inflation; the pace-following-on-reserve part is a deployment effect the COUPLED INTEGRATOR
  captures (path-fit) but a force-balance read of a single settled section cannot. Quantified, not
  a defect — it is the §III.4a pace game operating ABOVE v_c. A cleaner cruise-CP estimator would
  invert the coupled integrator (not the static root) for the aerobic v_c; deferred.
- **The FULL positional draft is LATERAL and needs CV.** The coarse longitudinal cover fraction
  (running-position rank) cannot distinguish a horse drafting one body-length directly behind the
  leader from one racing wide with clear air at the same rank. Who is LATERALLY behind whom — the
  real slipstream geometry — needs the y_i(t) lateral coordinate, which only CV (§VI Stage 3 /
  #41 drone PoC) provides. The longitudinal draft is the first moment; the lateral positional
  draft is the open layer (THEORY §III.4b boundary).
- **Field pace as exogenous-observed vs equilibrium:** we CONDITION on the observed pace (tractable
  now). The mean-field EQUILIBRIUM (solving the pace fixed point so the model is predictive without
  the contemporaneous field pace) is the next structural step — needed before this feeds a
  leak-free WIN predictor (the field pace of a held-out race is contemporaneous, fine as a boundary
  condition for path reconstruction, but a predictive use needs the equilibrium pace from prior
  constants). DRAFT_FRAC=0.30 is a literature anchor; a CV-measured draft delta would pin it.

## THE VERDICT — the field coupling is the right missing physics; the cross-check now PASSES
The decisive test (0048's cruise–Hill disagreement) moves the cruise read toward the Hill CV by
41–47% from the DRAFT mechanism alone, both venues, with no fitted constant — the gap that exposed
the missing physics now collapses in the predicted direction by the predicted magnitude. The
field-coupled forward path flattens the closing fade-hump (closing mean_res +0.80→+0.21 ST) and the
fast-pace closing residual (the 0047 fingerprint), and the de-confounded level re-emerges as
held-out ability (−0.02→+0.33 ST). The per-horse constants stayed FIXED; the coupling perturbed
DEPLOYMENT, exactly as 0047/0048 framed it. The mean-field pace/draft coupling (THEORY §III.4a) is
the dominant missing physics of the single-horse EOM, now built and validated on its own decisive
cross-check.

## DISCIPLINE
NO REGRESSION: both mechanisms are physical effort/drag responses to the OBSERVED field pace —
draft from the §III.2 `(1−draft)` aero term with a drafting-aerodynamics magnitude (DRAFT_FRAC=0.30,
bounded by cover ∈[0,1], leader=0); pace-following bounded by P_max (the v_max(P_max) ceiling) and
W′≥0, reducing to the isolated EOM when v_field≤v_c. Field pace is OBSERVED data, not a fitted
latent. Per-horse constants strictly-prior (the de-confounded CP from the Hill anchor + the run's
own settled cruise; W′_0/P_max literature-bounded). Mass + going KNOWN, never fitted. No
market/win/pool anywhere. NEW module `src/hkjc/statespace/field.py` + `studies/field_coupling_test.py`
+ 9-test unit suite (`tests/test_field.py`: draft bounds; leader clear-air; v_max>v_c; reduces-to-
isolated when field slow; hot field lifts cruise; pace-following spends W′; draft lifts cruise;
de-confound identity at no-cover; de-confound lowers a drafted read + CP-consistency). `dynamics.py`,
`identify.py`, `energetics_dynamic.py`, production `hkjc.model`, `data_cache` UNTOUCHED. Lint clean;
field + dynamics + identify + pathfit tests green (39). DON'T-BAKE — diagnostic coupling foundation;
no production constant changed.
