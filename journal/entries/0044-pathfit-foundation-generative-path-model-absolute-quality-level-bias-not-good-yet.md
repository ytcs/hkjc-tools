---
title: "Foundation 0044: does the generative path model FIT observed race paths well? — held-out reconstruction is OK-not-good (R²≈0.5, +56–60% over the within-horse noise floor), dominated by a +0.42 m/s SYSTEMATIC LEVEL UNDER-PREDICTION; a flexible descriptive baseline beats it ⇒ the generative FORM is too rigid and must be fixed before latent identification is trustworthy"
id: 0044-pathfit-foundation-generative-path-model-absolute-quality-level-bias-not-good-yet
author: modelling
audience: ["modelling"]
channel: research
kind: result
status: open
thread:
tags: ["theory", "energetics", "w-prime", "v_c", "acceleration", "phase-structure", "profile-likelihood", "sectional", "path-fit", "reconstruction", "calibration", "noise-floor", "generative", "identification", "foundation", "draft"]
refs: {"module": "src/hkjc/statespace/pathfit.py", "study": "studies/pathfit_foundation.py", "energetics_dynamic": "src/hkjc/model/energetics_dynamic.py", "theory": "THEORY.md §III/§IV (generative path model + phase structure)", "panel": "data_cache/trajectory_dataset_full.parquet", "plots": "studies/out/pathfit_profiles.png, studies/out/pathfit_residual_calibration.png, studies/out/pathfit_generative_vs_descriptive.png"}
created: 2026-06-06T00:00:00+00:00
updated: 2026-06-06T00:00:00+00:00
---
DRAFT — PATH-FIT QUALITY ONLY. No market odds, no win gate, no pool book anywhere. Production
`hkjc.model` UNTOUCHED (import/reuse). New additive module `hkjc.statespace.pathfit`. The
orchestrator serializes any MODEL_STATE change / generative-form fix.

## THE QUESTION (the foundation before ANY gate)
The reformulation (THEORY §III/§IV) rests on the claim that the generative W′-balance + first-passage
path model REPRODUCES real K-section velocity profiles, from which θ=(v_c, W′_0, a, ρ) are identified.
Before trusting any downstream identification we must establish the foundation: **does
`expected_profile(θ, conditions)` actually fit observed race paths WELL, in ABSOLUTE terms?** This is a
pure held-out PREDICTIVE path-fit. The pilot 0040 only showed the 4-latent profile beats a scalar
heuristic by 22–30%; that is a RELATIVE win. This entry asks the ABSOLUTE + structural question.

## CONSTRUCTION (leak-free, additive)
`hkjc.statespace.pathfit` + `studies/pathfit_foundation.py`. 39,256 runs (ST 24,099 / HV 14,416 with
≥1 other run), n_sec≥3.
- **Held-out θ:** leave-one-RACE-out per horse — the held-out run's predicted (v_c, W′_0, a) is the mean
  of that horse's OTHER runs only (own run subtracted out). `assert_path_no_lookahead` drift **0.00**
  both venues (the prediction is algebraically independent of the held-out run's own θ/outcome).
- **Noise floor:** the within-(horse, n_sec, section) race-to-race velocity std (ddof=1, ≥2 runs,
  pooled) — the BEST any model conditioned on horse identity could do; the part no θ can remove.
- **Descriptive baseline (QUESTION 4):** a flexible physics-FREE per-(band, going, n_sec, section) mean
  velocity-ratio shape × the horse's strictly-prior LOO cruise level. Shape fit TRAIN-only (date split
  at the 70th pct); held-out level strictly-prior. The yardstick for "is the physics too rigid".

## RESULTS (held-out, m/s + normalized)

### (1) Absolute reconstruction — OK, NOT good; far from the floor
| venue | gen RMSE | gen R² | a:=1 RMSE | descriptive RMSE | noise floor | gen excess over floor |
|-------|----------|--------|-----------|------------------|-------------|-----------------------|
| ST    | 0.841    | +0.49  | 1.204     | 0.681            | 0.539       | **+56%**              |
| HV    | 0.652    | +0.46  | 0.846     | 0.553            | 0.408       | **+60%**              |
The generative model explains ~half the section-velocity variance and beats the no-accel ablation
clearly (a is real and load-bearing — confirms 0040). But it sits **56–60% above the irreducible
floor**, and **a flexible descriptive baseline is markedly closer** (ST excess +26%, HV +36%). The
physics is leaving large, structured, REMOVABLE error on the table.

### (2) Calibration — OVER-dispersed (predicted bands too WIDE)
PIT mean **0.62** (ideal 0.50 — the positive shift IS the level bias below), PIT std **0.23 ST / 0.19
HV** (ideal 0.289) ⇒ **OVER-dispersed**: the σ=0.06 log-velocity band is too wide, 68% band covers
0.78 ST / 0.89 HV (>0.68), 90% band covers 0.94/0.98. The predictive distribution is not calibrated —
but this is secondary to the mean bias (a mis-centred, too-wide band).

### (3) Residual structure — a SYSTEMATIC LEVEL UNDER-PREDICTION + an opening miss
Mean residual `v_obs − v_gen` is **positive EVERYWHERE** (+0.39 to +0.49 m/s) — the generator
systematically runs the whole path TOO SLOW. This is **structural, not LOO shrinkage**: with the run's
OWN in-sample θ the bias is still +0.36 m/s (gen mean profile 14.8/16.9/16.7/16.5 vs observed
15.2/17.5/17.0/16.7). Decomposition (ST): of the 0.841 RMSE, removing the global +0.42 bias drops it to
**0.73** — the level bias alone is ~⅓ of the squared error.
- **By phase** (ST mean_res / floor): opening **+0.49** (floor 0.64, de-biased rmse 0.94 — STILL the
  worst), cruise **+0.47** (floor 0.44, de-biased 0.59 — near floor once centred), closing **+0.19**
  (floor 0.58). The CRUISE shape is actually good after de-biasing; the OPENING is genuinely mis-fit
  even after `a`; the whole curve is shifted down.
- **By distance:** sprint is the failure — ST sprint RMSE **1.03, R²=+0.06, mean_res +0.74**; middle/
  staying fit well (R² 0.73/0.33). HV sprint mean_res +0.51. The start transient is a large fraction of
  a sprint and `a` (a single opening ratio) under-captures it.
- **By going:** flat (RMSE 0.83–0.86 across GOOD/FIRM/SOFT) — going is NOT a path-fit driver here.
- **By run-style:** flat (front/mid/closer within ~0.01 RMSE) — the single-horse model misses the
  pace-game equally across styles (no style-specific signal because there is no rival coupling at all).
- **By pace:** ST closing residual is +0.10 m/s WORSE in fast-pace races (fast +0.23 vs slow +0.13) —
  a real pace-dependent miss the single-horse generator cannot see (the §III.4 pace game); weak at HV.
- **By field size:** flat.

### (4) Generative adequacy — the physics PAYS a real price (it is too rigid)
The flexible descriptive baseline beats the physics generator at BOTH venues by a wide margin
(ST 0.681 vs 0.841; HV 0.553 vs 0.652) — closing ~half the gap to the floor that the physics leaves
open. A small gap would say the generative structure is adequate; this gap is LARGE. The generative FORM
is over-constraining the profile shape (the linear-ramp W′-spend that pins the closing section to v_c
and the single opening ratio) in a way a memorised mean shape does not.

## DIAGNOSIS → what must be FIXED in the generative form (before identification is trustworthy)
1. **The level under-prediction is the dominant defect and a FORM bug, not a fit bug.** `expected_profile`
   anchors the non-opening profile at `v_c·(1 + W′/L·w)` with a linear ramp `w` that forces the FINAL
   section to exactly v_c (w→0). But observed late velocity sits ABOVE the recovered v_c, and the cruise
   sections carry more boost than the linear ramp allocates ⇒ the mean profile is shifted down ~0.4 m/s.
   The W′-boost distribution (the §III.2 spend kinetics) and/or the v_c anchor (the exhaustion criterion
   pinning terminal reserve to 0) is mis-specified. **This must be corrected before any latent recovered
   through this generator is trusted** — a biased generator biases the identified θ.
2. **The opening phase is genuinely under-modelled** (worst de-biased RMSE, R²≈0 in sprints). `a` as a
   single scalar ratio captures the level of the ramp but not its shape; the start transient implies a
   richer acceleration latent (a ramp rate / time-constant), per THEORY §III.3 Phase 1.
3. **The pace-dependent closing miss is the fingerprint of the MISSING interaction mechanism** (§III.4):
   a single-horse generator has no representation of the field's pace, so it cannot fit a fast-pace
   closing fade — this is real, measured here (+0.10 m/s), and is exactly the interacting-PL / mean-field
   pace coupling the theory defers. Closing mis-fit ⇒ reserve/recovery (ρ) + pace coupling.
4. **Calibration (σ, the per-horse D_i) is secondary** — fix the mean first; the band is too wide and
   mis-centred, re-fit σ AND make it phase-dependent (opening band ≠ cruise band) once the mean is right.

## VERDICT
**The generative model is NOT a good path model YET.** It is directionally right (R²≈0.5, beats the
no-`a` ablation, cruise shape near-floor after de-biasing) but it (a) carries a **+0.42 m/s systematic
level under-prediction** that is a generative-FORM bug, (b) is **over-dispersed**, and (c) **loses to a
flexible descriptive baseline** — the physics is too rigid. Latent identification through this generator
is NOT yet trustworthy because a biased generator biases θ. **Fix order: (1) the W′-spend / v_c-anchor
level bias [must], (2) the opening-ramp shape [sprint failure], (3) phase-dependent σ, (4) then the
pace interaction.** Only after the level bias is removed and the gap to the descriptive baseline closes
should profile-identified latents be promoted into the §V coupling/gate program.

DON'T-BAKE. Diagnostic foundation only; no production change. Plots in studies/out/pathfit_*.png.
