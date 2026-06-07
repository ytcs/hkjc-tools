---
title: "S7b RE-SPECIFICATION of the FIELD/DEPLOYMENT coupling (per 0062): a FIRST-PRINCIPLES front-group pace-WAR W′-burn × front-runner-count THRESHOLD mechanism for the early pace GRADIENT (section-1→2). Built ``field.pace_war_gradient`` (+ ``front_war_gate``, ``front_count``, ``_dp_dv``): genuine front-runners (forwardness ≥ cutoff) contest a front pace L (the fastest front-runner sets it) above each member's own sustainable opening cruise base_i, spending excess over-CP work ΔP⁺=[P(L)−P(base_i)]⁺ early (EOM eq.3 resistance physics, reference ~500 kg horse), converted to an early over-pace δv=ΔP/(dP/dv|_L), GATED on the front-runner COUNT exceeding a pace-war threshold n*=2 via Michaelis–Menten g(n)=(n−n*)⁺/((n−n*)⁺+1). Validated vs the 0062 DE-CONFOUNDED gradient composition signal (CEIL − no-composition conditions+pace ridge, re-derived exactly: +0.030 ST / +0.057 HV). VERDICT: the mechanism RECOVERS the gradient composition signal at HV (de-confounded |ρ|=0.213 ≫ the +0.057 linear-ceiling gap — it captures the full extractable front-count structure; a single front-count feature alone hits −0.24 vs the ridge residual) and matches the ST signal which is itself a NOISE FLOOR (|ρ|=0.026 vs +0.030; every single composition feature ≤0.07 on ST's ridge residual). Robust across forwardness cutoff 0.30–0.40 and ridge seeds 0/1/7 (±0.002). KEY CORRECTION: ``trail_early_pos_norm`` is VERIFIED positively correlated with section-1 position (1=leader) ⇒ HIGH=closer; forwardness = 1−trail_early_pos_norm (0061/0062 used it un-inverted). CAVEAT: pace is a PROXY; the win/path effect is the parked arbiter and the effect may be too small to move the win gate. NOT wired into the win drift (the gated next step). No regression, no outcome-fitted constant; strictly-prior / leak-free."
id: 0063-s7b-respec-field-deployment-coupling-pace-war-wprime-burn-threshold-gradient-mechanism-recovers-hv-composition-signal-st-noise-floor
author: modelling
audience: ["modelling"]
channel: research
kind: result
status: open
thread: 0061-s7-field-deployment-cause-step1-mean-field-equilibrium-pace-beats-naive-at-hv-not-st
tags: ["s7b", "field-coupling", "deployment", "mean-field", "mean-field-game", "pace", "pace-gradient", "early-pace-gradient", "gradient", "pace-war", "wprime-burn", "w-prime", "critical-power", "over-spend", "front-runner", "front-count", "front-runner-count", "threshold", "threshold-coupling", "michaelis-menten", "saturation", "contested-pace", "forwardness", "running-style", "trail-early-pos-norm", "orientation-fix", "re-specification", "respec", "level-vs-gradient", "ceiling", "de-confound", "ridge", "out-of-sample", "leak-free", "strictly-prior", "first-principles", "no-regression", "no-epicycle", "no-fitted-constant", "venue-split", "hv-recovers", "st-noise-floor", "proxy-caveat", "target-validity", "dont-wire", "win-drift-gated", "wave-2", "field-cause", "0049", "0060", "0061", "0062"]
refs: {"module": "src/hkjc/model/field.py (pace_war_gradient, front_war_gate, front_count, _dp_dv — added; sole editor)", "study": "studies/field_gradient_coupling.py", "test": "tests/test_field.py (+5 focused tests)", "reuses": "studies/equilibrium_pace.py + studies/field_pace_ceiling.py (loaders, de-confound OOS-ridge, CEIL/PACE/COND feature sets — imported read-only)", "thread_journal": "journal/entries/0061-s7-field-deployment-cause-step1-mean-field-equilibrium-pace-beats-naive-at-hv-not-st.md", "diagnostic_journal": "journal/entries/0062-s7b-field-pace-ceiling-mis-spec-vs-null-st-s1-pace-is-a-true-null-the-missed-effect-is-in-the-early-pace-gradient.md", "deconf_journal": "journal/entries/0049-mean-field-pace-draft-coupling-deconfounds-cruise-toward-hill-cp-closes-fade-hump-collateral-reemerges.md", "theory": "docs/THEORY.md §III.4(a) (mean-field game of pace), §III.2 (critical-power / W′ energy law, eq.3)", "data_pace": "data_cache/trajectory_dataset_full.parquet", "data_style": "data_cache/run_policy_dataset.parquet", "data_cond": "data_cache/results_rich.parquet"}
created: 2026-06-06T00:00:00+00:00
updated: 2026-06-06T00:00:00+00:00
---
S7b — RE-SPECIFYING the FIELD/DEPLOYMENT coupling per the 0062 pointer. 0061 built the mean-field
EQUILIBRIUM PACE (a section-1 LEVEL) and beat the naive baseline on HV not ST. 0062 (the
mis-spec-vs-null ceiling diagnostic) localised the missed effect: NOT in the opening level (ST s1_mean
is a TRUE NULL) but in the early pace GRADIENT (section-2 − section-1 field-mean), where a
DE-CONFOUNDED composition signal sits at **+0.030 ST / +0.057 HV** (CEIL full-composition OOS ridge
minus the no-composition conditions+per-horse-pace ridge), and named the form: a W′-BURN +
front-runner-count THRESHOLD coupling on the pace DERIVATIVE, NOT 0061's smooth level chase. This
entry BUILDS that mechanism as physics and validates it CAPTURES the 0062 gradient signal. It does
NOT wire into the win drift (the deliberate gated next step). No market / win / pool; no regression.

## THE MECHANISM (``field.pace_war_gradient`` — first-principles, no outcome-fitted knob)
THEORY §III.4a + the §III.2 critical-power law (eq. 3), built on strictly-prior per-horse observables:
- **Genuine front-runners** = horses with forwardness ``style ≥ FRONT_STYLE_CUTOFF`` (race in the
  front third). The contested **front pace ``L``** = the fastest front-runner's own opening cruise
  ``base`` (it sets the pace the others must match to hold position).
- **The W′-burn (over-CP over-spend).** Each front-runner forced to hold ``L`` above its OWN
  sustainable opening cruise ``base_i`` (its CP-level opening pace) runs ABOVE CP and spends reserve.
  The excess work-rate is the EOM force-balance power difference (pure resistance physics, no fit):
  ``ΔP_i = [resist_power(L) − resist_power(base_i)]⁺``, converted to the early over-pace it
  represents via the local pace sensitivity ``δv_i = ΔP_i/(dP/dv|_L)`` [m/s]. The resistance model
  is the EOM's own reference (~500 kg horse, good turf, ``dynamics.M_REF``); the burn is a power
  RATIO so it is robust to the exact mass.
- **The THRESHOLD (a pace war, not a lone leader).** A single uncontested leader sets its own pace —
  no over-spend. The burn FIRES only above a front-runner COUNT threshold ``n* = 2`` via a smooth
  Michaelis–Menten gate ``g(n) = (n−n*)⁺/((n−n*)⁺+1) ∈ [0,1)`` (a duel escalates above the threshold
  and SATURATES). At/below n* the gate is 0 ⇒ no induced gradient (the isolated-deployment null).
- **The gradient term** = ``−g(n_front)·mean_i δv_i`` — a signed early-over-pace perturbation,
  validated by CORRELATION to the de-confounded realised gradient residual (so its monotone
  direction, not the sign convention, is load-bearing). Null when no front-runner is held above its
  base, or below the count threshold (both unit-tested).

## KEY CORRECTION (load-bearing): the forwardness ORIENTATION
``run_policy_dataset.trail_early_pos_norm`` is VERIFIED **positively** correlated with the section-1
running position (1 = leader): corr(trail_early_pos_norm, position) = **+0.368** (and the cleaner
``early_pos_norm`` = +0.673). So HIGH ``trail_early_pos_norm`` = a CLOSER, LOW = a front-runner — the
INVERSE of 0061's docstring ("1 = front-runner"). 0061/0062 fed it un-inverted; this study inverts
``forwardness = 1 − trail_early_pos_norm`` so the mechanism identifies genuine front-runners. The
forwardness distribution is compressed (median 0.19, an expanding prior-mean of a normalised rank), so
the front-runner cutoff is **0.34** (front third), giving a mean ~2.5 front-runners/race and the pace
war firing in ~44–49% of races — a physical reading of position rank, not an outcome fit.

## VALIDATION (``studies/field_gradient_coupling.py``; OOS / leak-free; ST n=1899, HV n=1277)
Target = the realised early pace gradient RESIDUAL above the no-composition conditions+per-horse-pace
OOS ridge (the 0062 de-confound — NOT the naive baseline). The CEIL−PACE composition gap is
re-derived here and reproduces 0062 EXACTLY (+0.030 ST / +0.057 HV), validating the harness.

| venue | n | 0062 CEIL−PACE gap (TARGET) | MECHANISM ρ (raw) | MECHANISM \|ρ\| (de-confounded) | recovers | pace-war fires |
|---|---|---|---|---|---|---|
| **ST** | 1899 | +0.030 | −0.063 | **0.026** | ~85% of a noise-floor signal | 44.4% |
| **HV** | 1277 | +0.057 | −0.176 | **0.213** | full extractable structure | 49.3% |

Seed-/cutoff-stable: HV \|ρ\| 0.20–0.23, ST \|ρ\| 0.02–0.04 across forwardness cutoffs 0.30/0.34/0.40
and ridge seeds 0/1/7 (±0.002). The mechanism's sign is consistently negative at both venues (the
perturbation tracks the residual monotonically).

## THE READ — honest, venue-split (matches 0062's shape)
- **HV: the mechanism RECOVERS the gradient composition signal.** De-confounded \|ρ\|=0.213 ≫ the
  +0.057 LINEAR-ceiling gap. The >100%-of-gap is NOT overfit: the CEIL−PACE gap is a linear ceiling
  that here UNDERSTATES the single-feature reach — against the same ridge residual a single
  front-COUNT feature alone hits ρ ≈ −0.24 (front_frac −0.240, mean_style −0.238, n_front/thresh_front
  −0.215). The mechanism captures the full extractable FRONT-COUNT/threshold structure. HV is the
  tight, congesting, tactical track where a crowded front genuinely imposes a contested pace — exactly
  where the physics says the pace war bites.
- **ST: the mechanism matches the signal, which is itself a NOISE FLOOR.** De-confounded \|ρ\|=0.026
  vs the +0.030 ceiling gap. Crucially, on ST's ridge residual EVERY single composition feature is
  ≤ ±0.07 (fast_front +0.060, wprime_burn +0.040) — the +0.030 ST ceiling gap is the linear ridge
  squeezing a marginal multi-feature combination near the noise floor, not a single extractable
  effect. The mechanism is correctly weak where the composition is correctly weak (ST runners are
  more independent; 0061/0062's read).

## A HONEST WRINKLE — the W′-burn alone does NOT survive the proper de-confound; the THRESHOLD does
0062 reported the W′-burn coupling at ρ +0.269 ST vs the ``realised − naive`` residual. That residual
is essentially the RAW gradient (the field-mean ``naive`` is a ~16 m/s LEVEL, near-constant vs the
gradient — its subtraction barely changes the correlation): ρ(wprime_burn, raw gradient) = +0.249 ST,
reproduced exactly. But against the PROPER no-composition ridge residual (the de-confounded target),
the bare energy-burn term collapses to ρ ≈ +0.02 ST / −0.01 HV — most of its raw correlation is the
conditions/level confound the ridge already explains. What survives the de-confound is the front-COUNT
/ THRESHOLD density (HV especially). So the mechanism's load-bearing physics is the pace-war
COUNT-THRESHOLD gate; the W′-burn is the physical MAGNITUDE/reason it fires, and the two coupled
(gate × burn) recover the de-confounded signal. The realised gradient is POSITIVE (mean +2.39 ST /
+1.57 HV — section 1 is the from-gate slow start; section 2 is the faster settle); the composition
effect is the residual perturbation to that evolution, not an absolute deceleration.

## DISCIPLINE / ARTEFACTS
ADDED to ``src/hkjc/model/field.py`` (its sole editor): ``pace_war_gradient``, ``front_war_gate``,
``front_count``, ``_dp_dv`` + the ``FRONT_WAR_THRESHOLD`` / ``FRONT_STYLE_CUTOFF`` constants and the
``_REF_RES`` reference resistance. NEW ``studies/field_gradient_coupling.py`` (reuses the
``equilibrium_pace.py`` + ``field_pace_ceiling.py`` loaders + de-confound OOS-ridge, imported
read-only). +5 focused tests in ``tests/test_field.py`` (gate threshold-null + saturation; front-count
counts only genuine front-runners; gradient null below threshold; null when the front group is
single-speed; fires negative + steepens with a hotter front). ``generative.py``, the pricer,
``gate.py``, ``pure_mechanism.py``, ``dynamics.py``, ``identify.py`` UNTOUCHED;
``src/hkjc/vision/`` left to its concurrent editor. ``ruff check`` clean on field + study + test;
``PYTHONHASHSEED=0 pytest -q`` green (181 passed, +5). No regression, no win/pool/market, no
outcome-fitted constant — ``base``/``style`` are strictly-prior, the resistance physics is the fixed
EOM, the threshold + contested-pace are physical structure. Ran SYNCHRONOUSLY in the foreground.

## CAVEATS
- **TARGET VALIDITY (load-bearing, per 0062).** Pace is a PROXY. The mechanism captures the
  de-confounded gradient PACE signal, but whether that converts to WIN skill is the separate, parked
  arbiter — the win-drift wiring (feed the pace-war perturbation into the generative drift) is the
  DELIBERATE next step, gated on this validation. The effect may be too small to move the win gate,
  especially on ST where the gradient composition is a noise floor.
- **HV carries it, ST barely.** As with 0061, the field/deployment composition earns its keep on the
  congesting HV track and is near-null on the galloping ST track. This is the expected shape of a
  one-dimension causal probe, not a bug — do NOT add a venue-aware knob (FORBIDDEN epicycle; the right
  deeper mechanism is the §III.4b lateral congestion game, CV/Stage 3).
- **Cutoff choice.** The 0.34 forwardness cutoff (front third) and n*=2 threshold are physical
  readings, and the result is stable across 0.30–0.40, but the compressed forwardness prior means the
  front-runner identifier is coarse; a CV running-position read (§III.4b) would sharpen it.

## NEXT
The gated win-drift wiring (NOT done here): feed ``pace_war_gradient`` as a leak-free deployment
perturbation into the generative section-1→2 drift (the front group's early over-spend → its
late-race W′ deficit → its win prob), then run the parked paired fixed-seed win gate to test whether
the recovered HV gradient signal converts to win skill. That is the separate arbiter S7b's win step
was always going to be.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
