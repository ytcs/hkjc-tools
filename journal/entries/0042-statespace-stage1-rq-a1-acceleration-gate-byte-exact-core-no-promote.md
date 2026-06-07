---
title: "Stage-1 / RQ-a1 (statespace): the generative core is byte-exact faithful, but acceleration `a` does NOT convert on the market gate — promote NO (ECE always worse; %market lift loads on sprints as predicted, +0.5% sprint / +0.1% overall, but tiny)"
id: 0042-statespace-stage1-rq-a1-acceleration-gate-byte-exact-core-no-promote
author: modelling
audience: ["modelling"]
channel: research
kind: result
status: open
thread:
tags: ["statespace", "generative", "state-space", "acceleration", "start-phase", "gate-calibration", "coupling", "phase-structure", "wald", "first-passage", "market-gate", "pct-of-market", "ece", "leak-free", "stage-1", "rq-a1", "draft", "#40", "#41"]
refs: {"design": "docs/STATESPACE_DESIGN.md", "gate_study": "studies/statespace_a_gate.py", "assembly": "src/hkjc/statespace/assembly.py", "generative": "src/hkjc/statespace/generative.py", "coupling": "src/hkjc/statespace/coupling.py", "gate": "src/hkjc/statespace/gate.py", "pipeline": "src/hkjc/statespace/pipeline.py", "test": "tests/test_statespace_skeleton.py", "production": "src/hkjc/model/fit.py", "pilot_0040": "journal/entries/0040-profile-likelihood-pilot-acceleration-latent-validated-fit-to-surface-direction.md", "panel": "data_cache/trajectory_dataset_full.parquet"}
created: 2026-06-06T00:00:00+00:00
updated: 2026-06-06T00:00:00+00:00
---
DRAFT — RQ-a1 result. Production `hkjc.model` UNTOUCHED (imported/reused only). All work is in
`src/hkjc/statespace/` + `studies/` + a test. The GATE decides, not the pilot's t-stat. Orchestrator
serializes any MODEL_STATE update; `a` is NOT promoted into the production win model.

## WHAT WAS BUILT (Stage-1 items 1–3, the architecture's first end-to-end loop)
The first end-to-end pass of the canonical generative state-space: IDENTIFY the #41 opening-ramp
latent `a` from the sectional-profile surface → COUPLE it into the start-phase drift with
GATE-CALIBRATED strength → GATE on the pool-book market skill%. Concretely:

1. **theta_builder — real strictly-prior θ over the production panels** (`statespace/assembly.py`).
   `make_production_pre_fit` runs the PRODUCTION `fit_predict_traffic` once per fold (population fit
   train-only, every ability channel strictly-prior) and stamps its `v_c_eb`/`w_prime_eb` onto the
   test frame; `production_theta_builder` reads those as the runner's v_c/W′. The acceleration latent
   `a` is the horse's STRICTLY-PRIOR trailing mean of `estimate_run_profile`'s opening ramp
   (`a=v_1/v_cruise`), expanding/shifted-by-one (the #41/0040 recipe; `attach_trailing_a`). 90.9% ST /
   95.4% HV runner coverage; median trailing `a` ≈ 0.88/0.90.
2. **`a` → start-phase drift coupling** (`statespace/coupling.py`). `μ_i *= (1 − a_start_gain ·
   start_frac · ā_centred)`, `start_frac = min(1, L_ref/L)` sourced from production `GATE_BREAK_LREF`
   (1200m — sprint-biased per §3e Phase 1), `ā_centred` within-race-centred. A faster opener (higher
   `a`) ⇒ reaches cruise sooner ⇒ shorter effective time. `a_start_gain` is the SINGLE new knob.
3. **The generative core now reproduces the production wald law byte-for-byte** (`generative.py`):
   `_drift_times`/`_effective_length`/`_diffusion_shape` port the production μ/λ assembly (bend
   ground-loss draw, gate-break→effective-L, draw→D / figure-spread→D / gate-break→D variance knobs,
   surface diffusion) sourced from `hkjc.model.constants`. Same race-id seed, same n_samples, same
   Laplace smoothing.
4. **Gate harness** (`statespace/gate.py` `calibrate_coupling` + `studies/statespace_a_gate.py`):
   exact %-of-market (`(uniform−model)/(uniform−market)`) + ECE, paired fixed-seed
   (PYTHONHASHSEED=0) full walk-forward, BOTH venues, ST broken out by distance band.

## (1) ACCEPTANCE GATE A — the generative core is FAITHFUL (byte-exact)
With `a_start_gain=0`, the state-space pipeline reproduces production `fit_predict_traffic`:

  ST: max|Δp| = 0.00e+00 over 4704 test runners   → FAITHFUL (byte-exact)
  HV: max|Δp| = 0.00e+00 over 2871 test runners   → FAITHFUL (byte-exact)

(One diagnosis on the way: the HV 1650m surface heuristic — production treats ALL 1650m as AWT by
distance, independent of venue. Mirroring `_surface` exactly closed an initial HV max|Δp|=0.16 on
1650m races and lifted the HV a=OFF baseline to the expected ~60%.) The generative core IS the
production win model before `a` is added — the architecture's first acceptance is met on BOTH venues.

## (2) THE GATE — a=OFF vs best `a_start_gain` (FULL paired walk-forward, %market | ECE)
                a=OFF        best non-zero gain         verdict
  ST      65.22% | 0.0036    65.36% | 0.0041 (g=.075)  %↑ +0.14%, ECE WORSE +0.0004
  HV      60.08% | 0.0016    60.08% | 0.0026 (g=.05)   %↑ +0.00%, ECE WORSE +0.0010

  ST grid: g=.025 +0.08%, g=.05 +0.11%, g=.075 +0.14%, g=.1 +0.11%, g=.2 −0.17%, g=.4 −1.95%
           — every nonzero gain RAISES ECE; %market peaks ~+0.14% then decays.
  HV grid: g=.05 +0.00%, g=.1 −0.16%, g=.2 −0.63%, g=.4 −2.69% — no %market lift at any gain.

  ST distance-band breakout (a=OFF vs g=.075), the #41/0040 SPRINT prediction:
    sprint   69.35% → 69.83%   Δ%market +0.48%   (ECE 0.0040 → 0.0054)
    mile     65.38% → 65.07%   Δ%market −0.31%
    middle   54.53% → 54.89%   Δ%market +0.35%

## (3) VERDICT — does acceleration CONVERT on the market gate?
**The lift loads on SPRINTS exactly as #41/0040 predicted** (+0.48–0.58% sprint vs −0.31% mile), and
the directional win-ORDER signal is real and monotone in the small-gain regime — the IDENTIFY→COUPLE
chain transmits the right mechanism. BUT the magnitude is tiny (+0.14% overall %market at best) and
the cost is CALIBRATION: **ECE is strictly worse at every nonzero gain, both venues.** The start-phase
order-statistic nudge from `a` does not survive the pool-book gate as a net win.

This is the #40 lesson landing for real: the path-honest latent (profile-fit +22–30% held-out,
t=+9.98 ST-sprint win-relevance) does NOT convert at the win order-statistic — and unlike a t-stat,
the paired %market+ECE gate is unforgiving. The pilot's ΔR² over-stated the win value (as warned).

## (4) PROMOTE / DON'T — and what Stage-1.5 inherits
**DON'T PROMOTE `a` via the start-phase DRIFT channel.** Decision rule (best non-zero gain must beat
a=OFF on BOTH %market AND ECE) FAILS on both venues — %market gain is sub-noise and ECE always worse.
The generative core, the leak-free seam, and the gate harness are KEPT (proven, reusable).

ALSO TESTED — the **`a` → D VARIANCE channel** (`a_d_gain`, the parallel coupling the drift-vs-
calibration symptom pointed to: a troubled opener injects start-phase variance → larger D). It ALSO
fails: ST grid g=.1 −0.01%, g=.2 −0.04%, g=.4 −0.08%, g=.8 −0.25% %market, no ECE benefit at any gain.
So `a` does NOT convert via the drift OR the variance channel.

Stage-1.5 candidates, in priority order from this evidence:
  - **SPRINT-ONLY drift coupling** — the only positive signal is the sprint band (+0.48–0.58%); a
    band-gated `a_start_gain` (zero outside sprints) avoids the mile regression and is the last
    plausible net-convert for `a`. If even that doesn't beat a=OFF on %market AND ECE within sprints,
    SHELVE `a` for the win pool (it remains validated for PROFILE FIT / margins — the Stage-2 surface).
  - **Move on to a richer GATE (Stage 2 — margins + pool book).** The win order-statistic is, as #40
    warned, too coarse to reward a start-phase latent; `a` may convert against the MARGIN/pool-book
    objective (where the opening split actually shows up) rather than the win bit. This is the more
    promising direction than further tuning `a`→win.
  - **Fold `a` into the EB hierarchy proper** (band×rating random effect; Stage-1 item 4) — the clean
    home if `a` is revisited, but unlikely to change the win-gate verdict on its own.
  - **ρ stays population-fixed** (Stage-1 scope); per-horse ρ remains a Stage-3 CV target.

## (5) LEAK-FREE CONFIRMATION
`assert_no_lookahead` on `build_fit_predict` (a_start_gain=0.2, real folds, both venues): drift
**0.00e+00**. θ is strictly-prior (trailing `a` shifted-by-one; v_c/W′ from the production strictly-
prior EB); population fit train-only per fold (inside the reused `fit_predict_traffic`); the sampler
is a pure function of (θ, conditions) + race-id seed. The numbers are paired fixed-seed (PYTHONHASHSEED=0),
full-scale walk-forward — not the ~4×-overstating quick_eval. `tests/test_statespace_skeleton.py` 7/7
green (incl. the new a-OFF==production-wald acceptance unit test); `ruff` clean; full suite 155 passed.
