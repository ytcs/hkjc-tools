---
title: "Foundation 0050: THE CAUSAL PAYOFF TEST — build the win model on PATH-INFERRED PHYSIOLOGY ONLY (Hill critical-velocity anchor + Hill reserve + field/draft deployment + physical mass), EVERY meta-signal stripped, and test whether FORM re-emerges. VERDICT: FORM does NOT re-emerge at the win-prob level — the pure-physics model does NOT reproduce the form→result gradient (recovers only 7% ST / 4% HV of the empirical form→win slope; non-monotone), and path-inferred ability predicts held-out form only weakly (Hill v_c→held-out form ρ≈+0.07 ST / +0.03 HV, well above the rating floor ≈+0.01/−0.02 but far below the trailing-form ceiling +0.38/+0.30). The GoF FALLS to the floor: %market 65→−8 (ST), 60→+5 (HV), paired fixed-seed — the honest baseline we rebuild from. PATH-FIT: the Hill-anchored field-coupled forward path under-resolves the tight per-cell descriptive baseline (RMSE 1.38 vs 0.45 ST). The directional WIN re-emergence of 0049 (collateral, ρ −0.02→+0.33) does NOT carry to the win-probability gradient — the path-inferred ability is real but too WEAK a discriminator to recover form WITHOUT the win-calibrated gain (the 0039 under-discrimination, re-confirmed on the pure-mechanism model)."
id: 0050-causal-payoff-test-pure-mechanism-hill-physiology-only-form-does-not-reemerge-gradient-not-recovered-gof-falls-to-floor
author: modelling
audience: ["modelling"]
channel: research
kind: result
status: draft
thread: 0049-mean-field-pace-draft-coupling-deconfounds-cruise-toward-hill-cp-closes-fade-hump-collateral-reemerges
tags: ["causal", "causal-rebuild", "payoff-test", "pure-mechanism", "form", "re-emergence", "reemergence", "form-reemergence", "hill", "critical-velocity", "deconfound", "field-coupling", "physiology", "strip", "meta-signal", "cause-not-effect", "gof-fall", "percent-of-market", "win-gradient", "form-gradient", "decisive-test", "path-fit", "leak-free", "strictly-prior", "first-principles", "no-regression", "stage-1", "0037", "0039", "0048", "0049", "theta-builder", "pure-mechanism-theta-builder"]
refs: {"module": "src/hkjc/model/pure_mechanism.py", "study": "studies/causal_reemergence.py", "test": "tests/test_pure_mechanism.py", "contrast": "src/hkjc/model/assembly.py (production_theta_builder)", "generative": "src/hkjc/model/generative.py", "identify": "src/hkjc/model/identify.py (hill_fit)", "field": "src/hkjc/model/field.py (deconfounded_cruise_velocity)", "design": "docs/CAUSAL_MODEL.md", "thread_journal": "journal/entries/0049-mean-field-pace-draft-coupling-deconfounds-cruise-toward-hill-cp-closes-fade-hump-collateral-reemerges.md", "panel": "data_cache/trajectory_dataset_full.parquet", "mass_going": "data_cache/results_rich.parquet (DeclarHorseWt + ActWt + Going)"}
created: 2026-06-06T00:00:00+00:00
updated: 2026-06-06T00:00:00+00:00
---
DRAFT — THE CAUSAL PAYOFF TEST (CAUSAL_MODEL.md §6, the FORM headline of §4.2/§4.3). NEW additive
module `src/hkjc/model/pure_mechanism.py` (the `PureMechanismThetaBuilder`) + `studies/causal_reemergence.py`
+ `tests/test_pure_mechanism.py`. The EOM core (`dynamics.py`), `identify.py` (Hill), `field.py`
(coupling), the production ability fit, and `data_cache` are IMPORTED untouched. NO odds-gating; win%
is MEASURED, never optimized. Leak-free strictly-prior (assert_no_lookahead drift = 0, both venues).
(NB the causal-physics package was folded `hkjc.statespace`→`hkjc.model` 2026-06; paths reflect that.)

## THE BUILD — PATH-INFERRED PHYSIOLOGY ONLY, ZERO META-SIGNALS
The `PureMechanismThetaBuilder` is the cause-not-effect CONTRAST to `assembly.production_theta_builder`
(which stamps the meta-signal-fused production `v_c_eb`). Each runner's drift is assembled from, in
strictly-prior source priority:
1. **Hill critical velocity** (`identify.hill_fit` slope over the horse's PRIOR distance–time points,
   ≥3 distinct distances) — the DE-CONFOUNDED aerobic CP anchor (0048/0049), NOT the pace-inflated
   per-race cruise. W′ = the **Hill intercept** D′ [m] (anaerobic distance capacity, on the same
   distance-unit axis the generative `w′/L` boost reads).
2. else the horse's prior **de-confounded settled cruise** (`field.deconfounded_cruise_velocity`, the
   field/draft deployment read of 0049) — the coarser ability proxy for a horse with too few distinct
   distances for a Hill line.
3. else the **rating-conditioned population mean** (fit TRAIN-ONLY per fold) — the debut/thin-prior
   concession (the flagged T1; rating enters HERE ONLY, never as a per-runner ability channel).
Mass = declared weight (physical). ENV kept: draw, going→C_rr/surface, geometry, field size.
θ feeds `generative.py` (the byte-exact Wald first-passage core) → win probs.

**Confirmation ZERO meta-signals are used.** The module imports NONE of the meta-signal tables — no
form, recency-form, speed-figure, collateral, peak-gap, finishing-sectional, fade, barrier-trial,
gallop, jockey, trainer, trainer-hot, early-pos, gate-break. The builder DELIBERATELY leaves
`gate_break_shrunk`/`fig_disp` (stripped meta-signal geometry) OFF the conditions. A unit test
(`test_no_meta_signal_imports`) asserts no banned table name appears in the module's import lines; the
drift is Hill-CP + field coupling + mass, full stop. Coverage (of the path panel): ST Hill 8,395 /
de-confounded 15,663 of 24,433 runners; HV Hill 1,665 / de-confounded 6,304 of 14,823.

## THE HEADLINE — does FORM re-emerge? VERDICT: NO (at the win-prob level)

### (a) ability → HELD-OUT FORM (the consequence arrow)
Does path-inferred ability predict a horse's held-out realized relative-finish (the form SIGNAL),
vs the population FLOOR (rating) and the trailing-form CEILING (form's own autocorrelation)?

| venue | path-ability v_c → held-out form | rating FLOOR → held-out | trailing-form CEILING → held-out | v_c → trailing-form |
|---|---|---|---|---|
| **ST** (n=15,663) | pearson **+0.066** (spearman +0.062) | +0.015 (+0.001) | **+0.382** (+0.373) | +0.123 |
| **HV** (n=6,304)  | pearson **+0.033** (spearman +0.029) | −0.022 (−0.022) | **+0.302** (+0.289) | +0.070 |

The path-inferred ability DOES carry held-out-form signal **above the rating floor** (+0.066 vs +0.015
ST; +0.033 vs −0.022 HV) — the de-confounded Hill ability is a real, leak-free ability read, not noise.
But it sits **far below the trailing-form ceiling** (+0.38/+0.30): a direct past-form measurement
predicts future form ~6–9× better than the path-inferred physiology does. The ability→form consequence
arrow is PRESENT but WEAK.

### (b) THE DECISIVE JOINT TEST — reproduce the form→result gradient WITHOUT form
Bin runners by their (held-out) recency-form; does the PURE-PHYSICS win-prob track the realized
win-rate gradient across form bins as the form-FED production model does?

| | ST EMPIRICAL win | ST PURE-MECH p | ST PRODUCTION p | HV EMPIRICAL | HV PURE-MECH | HV PRODUCTION |
|---|---|---|---|---|---|---|
| form-bin 0 (low) | 0.019 | 0.077 | 0.016 | 0.032 | 0.084 | 0.028 |
| form-bin 1 | 0.050 | 0.079 | 0.043 | 0.065 | 0.088 | 0.059 |
| form-bin 2 | 0.066 | 0.073 | 0.064 | 0.078 | 0.087 | 0.080 |
| form-bin 3 | 0.089 | 0.083 | 0.099 | 0.102 | 0.088 | 0.110 |
| form-bin 4 (high) | **0.175** | **0.088** | **0.177** | **0.159** | **0.090** | **0.159** |
| **form→win SLOPE** | **+0.247** | **+0.017** (7%) | +0.262 (106%) | **+0.216** | **+0.009** (4%) | +0.231 (107%) |
| monotone | yes | **NO** | yes | yes | **NO** | yes |
| bin-level ρ(p,emp) | — | +0.78 | +0.996 | — | +0.86 | +0.996 |

**The pure-physics model does NOT reproduce the form→result gradient.** It recovers only **7% (ST) /
4% (HV)** of the empirical form→win slope and is NON-MONOTONE; the form-fed production model recovers
~106%. The pure-mechanism win-prob is nearly FLAT across the form bins (ST 0.077→0.088 vs the realized
0.019→0.175) — it cannot tell the top form-quintile from the bottom at the win-probability level, even
though its ability latent has the right SIGN (bin-level ρ +0.78/+0.86). **FORM does not re-emerge.**

## THE GoF-FALL (MEASUREMENT, not a gate) — the honest baseline
Paired fixed-seed win gate, pure-mechanism vs production, both venues:

| venue | production %market (ECE) | PURE-MECH %market (ECE) | FALL |
|---|---|---|---|
| **ST** | **65.4%** (0.0068) | **−7.7%** (0.0227) | **−73.1pp** of market |
| **HV** | **60.0%** (0.0059) | **+4.7%** (0.0112) | **−55.3pp** of market |

The fall is to the FLOOR (ST −8% is below uniform; HV +5% is barely above). This is the expected /
accepted Stage-1 fall (CAUSAL_MODEL.md §5.3, T2 — recency-form alone is +7.8pp and the whole ability
stack is most of production's edge), MEASURED honestly. ECE roughly doubles–triples (the pure model is
less calibrated, as a flat-gradient model must be).

**Decomposition — debut/thin vs the stripped ability stack** (mean −log p of the realized winner,
split by the WINNER's strictly-prior ability source):

| source | ST n_wins | ST pure-mech / production (Δ) | HV n_wins | HV pure-mech / production (Δ) |
|---|---|---|---|---|
| Hill-anchored winner | 676 | 2.808 / 2.221 (**+0.587**) | 123 | 2.689 / 2.163 (**+0.526**) |
| de-confounded-cruise winner | 608 | 2.027 / 2.116 (**−0.089**) | 437 | 2.193 / 2.231 (**−0.038**) |
| debut/population-fallback winner | 646 | 2.798 / 2.141 (**+0.657**) | 717 | 2.511 / 2.172 (**+0.339**) |

The fall is carried by BOTH the debut concession (Δ+0.66 ST / +0.34 HV — no path to infer from, the
population mean is a poor discriminator) AND the stripped ABILITY stack (the Hill-winner Δ+0.59 ST /
+0.53 HV — even where the physics has a clean Hill anchor, it loses heavily to the meta-signal-fused
production ability). Notably the de-confounded-cruise bin is ~neutral/slightly BETTER than production
(Δ−0.09 ST) — the field-coupled cruise read (0049's re-emergence channel) is the one place the pure
physics is competitive, consistent with 0049's directional collateral re-emergence. But it is a minority
of winners and does not lift the aggregate.

## PATH-FIT — the Stage-0 bar (descriptive baseline)
Field-coupled forward path from the Hill anchor (literature-bounded W′_0/P_max + observed field pace)
vs a PHYSICS-FREE descriptive baseline (population mean velocity per (distance, section-index, n_sec)
cell — it never sees the individual horse). Route runs ≥1400 m, Hill-anchored, ~6k sections/venue:

| venue | field-coupled(Hill) RMSE (bias) | descriptive cell-mean RMSE | bar |
|---|---|---|---|
| **ST** | 1.38 m/s (+1.10) | **0.45** m/s | **FAIL** |
| **HV** | 1.10 m/s (+0.88) | **0.41** m/s | **FAIL** |

The Hill-anchored forward path FAILS this descriptive bar — but read this honestly: (1) the descriptive
baseline here is the TIGHT per-cell mean (the strongest physics-free predictor — 0044's "flexible
descriptive baseline" that the generator also trailed); (2) the forward path carries a +0.9–1.1 m/s
LEVEL OVER-prediction because it uses literature W′_0/P_max defaults (not the per-run NLS inversion) on
the de-confounded LOWER Hill anchor — exactly the 0049 isolated-baseline bias note (0049's isolated EOM
also sat at +0.9 bias; its field coupling on the SAME physiology removed it). The pure-mechanism win
DRIFT does not integrate the forward path (it reads the Hill v_c directly into the Wald law), so this
path-fit FAIL is a diagnostic on the forward GENERATOR's level bias at the Hill anchor, not on the win
build's ability read. STILL: it does not clear the Stage-0 adequacy bar as posed, and that is recorded.

## THE VERDICT — FORM does not re-emerge; the path ability is real but too WEAK a discriminator
The decisive test FAILS: the pure-mechanism model, built on path-inferred physiology with every
meta-signal stripped, does NOT reproduce the form→result gradient (7%/4% of the slope, non-monotone)
and predicts held-out form only weakly (ρ≈+0.07/+0.03, above the rating floor but ~6–9× below the
trailing-form ceiling). The win% falls to the floor (65→−8 ST, 60→+5 HV). This is the 0039 finding
re-confirmed on the CAUSAL model: the path identifies a REAL de-confounded ability (it beats the rating
floor; the de-confounded-cruise winners are ~neutral vs production), but the honestly-inferred Hill
ability UNDER-DISCRIMINATES for win — it cannot do the order-statistic TAIL work the win-calibrated form
gain does. 0049's directional WIN-level re-emergence (collateral ρ −0.02→+0.33) is real but does NOT
carry to the win-PROBABILITY gradient: a +0.33 held-out-ability correlation is not a steep-enough
discriminator to sort the form quintiles at the win-prob level.

This is a §4.4 POINTER, not a defeat: FORM failing to re-emerge ⇒ **the path is not yet identifying the
FULL, tail-relevant ability** (needs the margins surface + the closing precision + the pace interaction
layer, CAUSAL_MODEL.md Stage 2). The fix is NEVER to re-introduce the form gain — it is to enrich the
causal structure until the path-inferred ability is sharp enough to re-derive the gradient on its own.

## WHAT RE-EMERGES NEXT / WHAT'S MISSING
- **The de-confounded-cruise channel is the live re-emergence** (Δ−0.09/−0.04 vs production on its
  winners; 0049's +0.33 held-out persistence). It is the field-coupling deployment read — the COLLATERAL
  mechanism. FORM (the self-read) is weaker than COLLATERAL (the relational read) exactly as
  CAUSAL_MODEL.md §4.4 predicted ("collateral re-emerges but form does not ⇒ the field-coupling is the
  missing piece"). Build the field/pace interaction into the WIN drift (not just the path de-confound).
- **The Hill anchor needs the W′/margins surface.** The Hill intercept D′ is a coarse reserve; the
  forward path's +1.0 m/s level bias (literature W′_0/P_max on the de-confounded anchor) says the
  reserve/deployment is mis-scaled. Stage 2's margins term (`likelihood.margin_term`) should sharpen
  W′ and lift the held-out-form correlation toward the ceiling.
- **Debut/thin is a large, separable chunk of the fall** (Δ+0.66 ST / +0.34 HV) — a structural limit
  of physiology-only (no path to infer from), not a model error; the population-prior concession (T1).
- **PATH-FIT generator level bias** at the de-confounded Hill anchor (the +0.9–1.1 m/s) must close
  (the 0049 field-coupling removed it on the per-run physiology; wire that into the forward path used
  for the Stage-0 bar) before the path-fit adequacy claim can be made.

## DISCIPLINE
FIRST-PRINCIPLES, no regression, no meta-signal features — the drift is Hill-CP + field coupling +
mass. NO odds-gating (win% measured, never optimized). Leak-free: per-horse Hill/de-confounded
physiology strictly-prior (a run's own path never enters its anchor — unit-tested); the population
prior fit TRAIN-ONLY per fold; `assert_no_lookahead` drift = 0 both venues. Production `hkjc.model` win
fit, `dynamics.py`/`identify.py`/`field.py` core, `data_cache` UNTOUCHED (imported). NEW:
`pure_mechanism.py` + `studies/causal_reemergence.py` + 5-test unit suite (no-meta-signal-import guard;
source priority Hill→decon→pop; stripped geometry off; drift clamp; strictly-prior Hill leak-free).
Lint clean; pure-mechanism + field + skeleton tests green (24). DON'T-BAKE — a research-model
measurement; no production constant changed. The GoF fall is the honest baseline we rebuild from by
ADDING causal structure (the field/pace interaction + margins), NOT by re-feeding a stripped signal.
