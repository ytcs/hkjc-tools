---
title: "Causal rebuild DESIGN (owner-directed): re-ground the win model cause-not-effect — STRIP all meta-signals (form/jockey/trainer/collateral/peak-gap/speed-fig are EFFECTS of the causes), drive the model from physiology/control/prep CAUSES inferred from the PATH, validate by the meta-signals RE-EMERGING as consequences. Primary metric = path-fit + re-emergence; win% checked LAST, expected to fall then recover fundamentally. Authoritative design: docs/CAUSAL_MODEL.md."
id: 0046-causal-rebuild-design-cause-not-effect-strip-metasignals-reemergence-validation
author: modelling
audience: ["all"]
channel: research
kind: design
status: draft
thread:
tags: ["causal", "causal-rebuild", "cause-not-effect", "confound", "meta-signals", "strip", "re-emergence", "form", "jockey", "trainer", "collateral", "peak-gap", "speed-figure", "ability-latent", "v_c", "control", "prep", "crossed-effects", "cross-classified", "random-effects", "path-likelihood", "identifiability", "lateral", "cv", "statespace", "generative", "first-passage", "pure-mechanism", "first-build", "gof-fall", "validation", "staging", "design", "draft", "0037", "0039", "0040", "0044", "queue-50"]
refs: {"design": "docs/CAUSAL_MODEL.md", "theory": "docs/THEORY.md", "statespace_design": "docs/STATESPACE_DESIGN.md", "program": "docs/RESEARCH_PROGRAM.md", "model_state": "journal/state/MODEL_STATE.md", "causes_stub": "src/hkjc/statespace/causes.py", "likelihood": "src/hkjc/statespace/likelihood.py", "energetics_dynamic": "src/hkjc/model/energetics_dynamic.py", "features": "src/hkjc/model/features.py", "calibration": "src/hkjc/model/calibration.py", "factor_audit": "journal/entries/0037-vc-latent-factor-audit-catch-all-is-one-ability-factor-collapse.md", "fusion_collapse": "journal/entries/0039-queue-39-ability-fusion-collapse-dont-bake-separate-gains-load-bearing.md", "profile_pilot": "journal/entries/0040-profile-likelihood-pilot-acceleration-latent-validated-fit-to-surface-direction.md", "pathfit_foundation": "journal/entries/0044-pathfit-foundation-generative-path-model-absolute-quality-level-bias-not-good-yet.md"}
created: 2026-06-06T00:00:00+00:00
updated: 2026-06-06T00:00:00+00:00
---
DESIGN + DOCS ONLY. No inference built. Production `hkjc.model` (65/60% of market) UNTOUCHED and
stays the deployable fallback. `energetics_dynamic.py` NOT modified (#50 edits it concurrently). NO
market odds in the near-term stages. The deliverable is the authoritative design `docs/CAUSAL_MODEL.md`
plus the reframed THEORY/STATESPACE_DESIGN and design-surface stubs.

## THE PRINCIPLE (owner's directive)
The current model CONFOUNDS CAUSE AND EFFECT. FORM/JOCKEY/TRAINER/COLLATERAL/PEAK-GAP/SPEED-FIG are
not mechanisms — they are downstream CONSEQUENCES of the true causes. FORM is high BECAUSE ability is
high (a noisy past-measurement of `v_c`); the JOCKEY signal exists BECAUSE the rider exerts better
CONTROL; COLLATERAL = ability via opponents; PEAK-GAP = ability's career trajectory. Feeding these as
features makes the model circular ("winners win again") — it fits well because it is tautological.
THE REBUILD: decouple ALL meta-signals, build on the mechanistic CAUSES only, infer them from the
PATH, and validate by the meta-signals RE-EMERGING as consequences. Short-term GoF falls sharply; that
is ACCEPTED (research model only; production stays live). The empirical proof the meta-signals ARE
effects of one cause is already in hand: `0037` (form/figure/collateral/sectional = ONE over-measured
ability latent, PC1≈25%, 5/5 retrodiction) + `0039` (the additive gains are gate-calibrated couplings
~7× the honest slope, not honest measurements).

## THE CAUSAL DAG (docs/CAUSAL_MODEL.md §1)
Causes → BOTH the path/outcome AND the meta-signals (two arrows out of every cause):
  - LATENT CAUSES: physiology (v_c, W'_0, a, rho, D_i, mass), jockey CONTROL (c_lat lateral / c_pol
    policy), trainer PREP (fitness/trajectory), environment (geometry, going, draw, field pace).
  - cause → THE PATH (longitudinal v(t) now; + lateral y(t) via CV) → FINISHING ORDER (argmin first-
    passage — the VALIDATED §III physics, preserved as the generative core).
  - cause → PAST PATHS → the META-SIGNALS (ability → past paths → form/figure/collateral/peak-gap;
    control → positioning → jockey-SR; prep → readiness → trainer-hot). The meta-signals are LEAVES.
Identifiability annotation (NOW vs needs-CV): physiology identifiable from the longitudinal profile
NOW; trainer-prep PARTIAL now; jockey-control fundamentally LATERAL → WEAK now, clean split needs CV
(#34); pace/MFG coupling not identifiable from a single-horse generator (deferred).

## THE STRIP LIST + RE-DERIVATION (docs/CAUSAL_MODEL.md §2)
- ABILITY-EFFECTS → STRIP, re-emerge from ability `v_c`/`W'_0`: recency_form/form, speed_figure,
  collateral_form[_v2], rating_trajectory peak-gap, finishing_sectional_vc, trajectory fade (→W'),
  barrier_trial/trial_vc, gallop_vc. These are noisy past-measurements of ONE latent (0037 PC1) —
  the model infers `v_c`/`W'` from the path; form/figure/collateral become PREDICTED consequences.
- CONTROL-EFFECTS → RE-DERIVE causally: jockey_table → a crossed JOCKEY random effect on the path
  (coarse now, fully causal with CV `c_lat` at Stage 3); early_pos/style → split (pace shadow stays on
  W', positioning → control latent); gate_break → DROP (0037: it IS early_pos, R²=0.998; the kinematic
  `a` capacity cause replaces it).
- PREP-EFFECTS → RE-DERIVE: trainer/trainer_hot → a crossed TRAINER×time random effect (coarse now,
  no new data).
- NOT stripped (genuine causes/env): rating (ONLY as debut/thin-horse population prior — tension T1),
  draw, going, geometry, allometric mass (v_c ∝ m^(b-1)), field size.

## THE INFERENCE — separate ability / control / prep (docs/CAUSAL_MODEL.md §3)
All three shape the same path. Decouple via a CROSS-CLASSIFIED (crossed random-effects) hierarchy over
the path likelihood: log v_obs = θ_horse·B_phys + u_jockey·B_ctrl + w_trainer·B_prep + env + ε, with
the (horse, jockey, trainer) crossings in the data separating the effects. GENERALIZES the existing EB
hierarchy (calibration.fit_energetics_population, band×rating + horse RE) to add jockey + trainer
groupings. The meta-signals NEVER appear; the latent EFFECTS are inferred from the path + crossings.
HONEST IDENTIFIABILITY MAP:
  - NOW (longitudinal): horse physiology STRONG (0040/0044); trainer-prep PARTIAL (dense crossings).
  - WEAK now / needs CV: jockey-control is fundamentally LATERAL (positioning/ground-saved invisible to
    v(t)), confounded longitudinally with v_c → coarse upper bound only; clean c_lat/c_pol split is
    CV-only (#34, Stage 3). State "control only partially identified until CV" in every Stage-1/2 result.
  - NOT identifiable: pace/MFG (single-horse generator has no field; 0044's +0.10 m/s fast-pace closing
    residual is the fingerprint).
  - THIN-CROSSING RISK: where horse×jockey×trainer is thin the three effects are aliased — shrink hard /
    report combined; quantify crossing density as an entry diagnostic.
Why the path (not win) makes this honest: 0039 proved win can't identify the coupling honestly; the
path is a denser likelihood, so the wager is it recovers the tail-relevant ability shape the win-bit
fusion discarded. If path-inferred ability under-predicts winning, that is the MEASURED GoF fall we
accept and close by adding causal structure — NEVER by re-feeding the meta-signal.

## RE-EMERGENCE VALIDATION SUITE — the falsifiable test (docs/CAUSAL_MODEL.md §4)
For each stripped signal: infer the latent from the PATH only → predict the held-out meta-signal from
the latent's consequence arrow → score vs the population baseline (floor) and the signal's own trailing
mean (ceiling). Per-signal metrics: FORM (ability → rank corr of held-out rel-finish), SPEED-FIG (RMSE/
R²), COLLATERAL (reproduce Massey from inferred field abilities), PEAK-GAP (ability trajectory → corr +
sign), JOCKEY-SR (jockey RE → corr/rank), TRAINER-HOT (prep trajectory → corr/lead-lag). THE DECISIVE
JOINT TEST (§4.3): reproduce the form→result relationship WITHOUT form as input — bin runners by observed
form, does the causal model's win prob (form stripped) track the empirical win-rate-by-form curve? A
signal that re-emerges ⇒ mechanism captured; one that FAILS ⇒ a precise pointer to a missing mechanism
(FORM fails → path not identifying full ability; JOCKEY-SR fails → confirms the CV dependency; COLLATERAL
re-emerges but FORM not → the relational/field-coupling piece is missing). The re-emergence map is the
engine of progress — it names which mechanism to build next.

## THE METRIC + STAGING (docs/CAUSAL_MODEL.md §5)
PRIMARY = path-fit quality (0044 harness: held-out RMSE vs noise floor + descriptive baseline, PIT,
residual structure) + re-emergence score. LAST = win% (expected to fall sharply on first strip; checked
only to MEASURE the fall and confirm fundamental recovery). This INVERTS the production %-of-market-first
gate during the rebuild.
  - Stage 1 — strip ability-proxies; build physiological causal latents from the longitudinal path;
    test FORM re-emergence. Entry: #50's CORRECTED generator (0044 level bias removed). Exit: path-fit ≥
    descriptive baseline + FORM (and figure/collateral) re-emerge + the win% fall MEASURED+bounded.
  - Stage 2 — trainer-prep crossed effect + margins surface + the pace/field interaction. Exit: TRAINER-
    HOT re-emerges + margins identify W'/rho + pace coupling reduces the 0044 fast-pace residual + win%
    recovers materially.
  - Stage 3 — jockey-control with CV lateral (c_lat/c_pol). Exit: JOCKEY-SR re-emerges (the Stage-1
    fail-by-design test now passes) + control supersedes the draw shadows + win% recovers to/past
    production, now causal.
GoF-FALL CONTRACT: fall expected (Stage 1 large — recency-form alone is +7.8pp; ability stack is most
of production's edge over the ~48% start), recover by stage; NEVER closed by re-feeding a meta-signal.

## THE FIRST PURE-MECHANISM BUILD (docs/CAUSAL_MODEL.md §6 — dispatch this)
A win model whose drift is set by the PATH-INFERRED physiology latents ONLY (v_c, W'_0, a, rho, D, m
from the horse's earlier paths via #50's corrected estimate_run_profile + the calibration EB prior,
horse RE only), with EVERY meta-signal feature STRIPPED; rating only as the debut/thin prior. Generator
= the production wald first-passage law on #50's bias-corrected form. MEASURED on: (1) path-fit ≥ the
0044 descriptive baseline on #50's generator; (2) FORM re-emergence (§4.2 + the §4.3 form→result-without-
form curve) — the headline pass/fail; (3) the GoF-fall magnitude (paired fixed-seed win gate vs
production, decomposed by which stripped gain carried how much win%) — a MEASUREMENT not a gate. Lives
additively in statespace (a `PureMechanismThetaBuilder`, contrast assembly.production_theta_builder which
stamps the meta-signal-laden v_c_eb); re-emergence suite = studies/causal_reemergence.py. HARD PREREQ:
#50's corrected generator (a biased generator biases every inferred cause — 0044).

## WHAT THIS ENTRY SHIPPED (design surface only; no inference)
- docs/CAUSAL_MODEL.md — NEW, authoritative (the DAG, strip list, crossed inference + identifiability
  map, re-emergence suite, metric+staging, first build, 5 open tensions).
- docs/THEORY.md — reframed cause-not-effect (banner + §0 + §II/§III intros): the meta-signals are
  EFFECTS to re-derive; §III physics preserved as the cause→path→order generator (now on #50's
  corrected form).
- docs/STATESPACE_DESIGN.md — re-pointed to the causal rebuild (banner + the FIRST BUILD work items
  superseding the gate-coupling Stage 1, kept as proven infra); causes.py + pathfit.py added to layout.
- src/hkjc/statespace/causes.py — NEW design stub: LatentCauses (crossed PhysiologyCause / ControlCause
  / PrepCause + env) + infer_crossed_causes (NotImplementedError, Stage 2/3). likelihood.crossed_term
  stub added. Wired into __init__; ruff clean; 10 skeleton tests pass; imports + stubs verified.

## OPEN DESIGN TENSIONS (for the owner — docs/CAUSAL_MODEL.md §7)
- T1 rating's dual nature: keep as debut/thin prior only, or rule fully exogenous? (both cause + effect).
- T2 the 0039 GoF-fall magnitude: the win-calibrated ability gains carried order-statistic tail work; a
  Stage-1 model could sit well below production on win% while passing every causal test. Accepted per the
  directive — flagging the likely magnitude.
- T3 thin crossings: shrink-to-population+combined, or restrict crossed inference to the dense sub-graph?
  (recommend the latter, separability-diagnostic-gated).
- T4 pace interaction timing: coarse Stage-2 field-pace term vs full Stage-3 MFG (recommend coarse now).
- T5 #50 dependency: the entire causal inference is BLOCKED on #50's corrected generator (hard prereq).

DON'T-BAKE. Design only; no production change; no gate run. The first build is dispatched against this
design + #50's fix.
