---
title: "Pilot 0040 (Fork 2): fit-to-SURFACE validated — the sectional-profile likelihood beats the scalar heuristic AND surfaces a NEW win-relevant latent (acceleration `a`) that win alone cannot identify"
id: 0040-profile-likelihood-pilot-acceleration-latent-validated-fit-to-surface-direction
author: modelling
audience: ["modelling"]
channel: research
kind: result
status: open
thread:
tags: ["theory", "energetics", "w-prime", "v_c", "acceleration", "start-phase", "phase-structure", "profile-likelihood", "sectional", "fit-to-surface", "identification", "pilot", "fork-2", "queue-31", "draft"]
refs: {"study": "studies/profile_likelihood_pilot.py", "energetics_dynamic": "src/hkjc/model/energetics_dynamic.py", "energetics": "src/hkjc/model/energetics.py", "theory": "THEORY.md §3e (Phase 1 start transient)", "panel": "data_cache/trajectory_dataset_full.parquet", "plot": "studies/out/profile_likelihood_pilot.png"}
created: 2026-06-05T00:00:00+00:00
updated: 2026-06-05T00:00:00+00:00
---
DRAFT — IDENTIFICATION + win-relevance proxy ONLY. fit.py UNTOUCHED; `a` NOT wired into the
production win model. Orchestrator serializes any wire-in + MODEL_STATE update.

## THE DIRECTION TESTED (batch strategic finding)
The WIN outcome is too coarse a likelihood to identify rich mechanism (three studies confirmed
it can't resolve energy-unit/body-shape/recovery). DIRECTION: anchor the generative W′-balance
model to a BROAD SURFACE OF MEASUREMENTS (here the K-section velocity profile) for FIT/DISCOVERY,
keep win→%market(+ECE) as the FINAL eval. This pilot (a) validates the direction on the sectional
profile and (b) tests whether the rich likelihood surfaces a NEW latent — **acceleration `a`**, the
opening-split ramp (THEORY §3e Phase 1, the start transient the homogeneous Wald law cannot
represent) — that win alone cannot identify.

## CONSTRUCTION (leak-free, additive, fit.py untouched)
- EXTENDED `energetics_dynamic.py` additively: `estimate_run_profile` recovers (v_c, W′_0, **a**)
  where `a = v_1 / v_cruise` (opening-section velocity over the settled cruise mean); `expected_profile`
  GENERATES the K-section velocity profile from (v_c, W′_0, a) — W′-balance sets the cruise+closing
  shape, `a` sets the opening; `profile_loglik` is the log-velocity Gaussian observation model (≡
  log-normal section times, the production noise family). NO existing function changed.
- Per-run latent recovery is a DETERMINISTIC transform of the run's OWN public sectionals (no
  outcome, no other horse) — same leak status as the scalar/dynamic recoveries. Win-relevance uses
  STRICTLY-PRIOR trailing (expanding) per-horse means. Profile-fit uses LEAVE-ONE-RACE-OUT within a
  horse. `assert_no_lookahead` on the trailing-`a` fed-forward frame: drift **0.00e+00** both venues.
- The recovered `a` is a clean, well-supported latent: median 0.91, range [0.64, 2.04] across 39,256
  runs — i.e. the opening split is typically ~90% of cruise (the standing-start ramp), with real
  per-run spread. The mean ST 4-section profile is unmistakably ramp-shaped (sec1 15.0 → sec2 17.8
  peak → fade), the `a` fingerprint (plot: studies/out/profile_likelihood_pilot.png).

## (a) PROFILE FIT — 4-latent model CRUSHES the scalar heuristic on held-out profile RMSE
Per-section velocity RMSE (m/s), leave-one-race-out within a horse, full panel:

           4-latent(v_c,W′,a)   3-lat(a:=1)   scalar(flat v_c)   scalar(flat cruise)
  ST           0.8435            1.1933         1.1818 (+0.34)      1.4461 (+0.60)
  HV           0.6732            0.8622         0.8673 (+0.19)      1.0462 (+0.37)

  OPENING-section-only RMSE (where `a` lives — the fingerprint):
  ST   4-lat 1.0685 | a:=1 1.9409 (a lifts the opening fit by 0.87 m/s) | flat-v_c 1.6554
  HV   4-lat 0.7771 | a:=1 1.2817 (a lifts the opening fit by 0.50 m/s) | flat-v_c 1.0899

⇒ The dynamic 4-latent generative profile beats the min-of-last-two scalar heuristic by a LARGE
  margin (ST −0.34, HV −0.19 m/s per-section RMSE; ~30%/22% lower error). The denser surface IS more
  fittable than the win bit. The acceleration latent does the heavy lifting on the opening split
  (the a:=1 ablation nearly DOUBLES opening RMSE) — the W′-balance dynamics alone (a:=1) barely beat
  the flat-v_c null on the opening, because they have no start-transient term. This is the
  identification claim made concrete: the profile pins down a start-phase shape the scalar cannot.

## (b) WIN-RELEVANCE — `a` carries STRONG INDEPENDENT win signal given v_c (and W′)
ΔR² / partial-corr of leak-free trailing `a` with the winner indicator (race-centred predictors):

  [ST] full     n=21609  ΔR²(a|v_c)=+0.0016  pr=+0.041  | given v_c,W′: ΔR²=+0.0055 pr=+0.074
       sprint   n= 8543  ΔR²(a|v_c)=+0.0050  pr=+0.071  | given v_c,W′: ΔR²=+0.0114 pr=+0.107
       mile     n=10644  ΔR²(a|v_c)=+0.0001  pr=+0.010  | given v_c,W′: ΔR²=+0.0021 pr=+0.046
       middle   n= 2333  ΔR²(a|v_c)=+0.0017  pr=+0.041  | given v_c,W′: ΔR²=+0.0030 pr=+0.055
  [HV] full     n=12689  ΔR²(a|v_c)=+0.0022  pr=+0.047  | given v_c,W′: ΔR²=+0.0025 pr=+0.050

Multivariate OLS slope significance (won ~ vc_c + wp_c + a_c, race-centred, HC-naive t):
  ST/sprint  a-coef +1.11  t(a)=+9.98   (vc t=+11.8 — `a` is nearly AS strong as v_c in sprints)
  ST/mile    a-coef +0.52  t(a)=+4.77
  HV         a-coef +0.64  t(a)=+5.64

⇒ `a` is IDENTIFIABLE and win-relevant, INDEPENDENT of v_c and W′. Sign is POSITIVE: at equal
  ability, a horse whose opening split runs faster relative to its own cruise (higher `a`) wins more.
  The ΔR² is modest in absolute terms (win is ~8-9% base rate, so R² is tiny for everything) but the
  t-stats are decisive and the partial corr is real. The DISTANCE-BAND GRADIENT is exactly what
  THEORY §3e predicts: the signal is STRONGEST in ST sprints (t=+10, ΔR²|v_c,W′=+0.011 — biggest cell),
  weaker in milers (t=+4.8), intermediate at HV — because the start phase is a larger fraction of a
  short trip. `corr(a,v_c)≈−0.2 to −0.26`: a fast opener tends to a LOWER settled v_c (it spent early),
  so `a` is genuinely orthogonal new information, not a v_c restatement.

## VERDICT: the fit-to-SURFACE direction is VALIDATED (both legs pass)
1. The rich (sectional-profile) likelihood FITS BETTER than the scalar heuristic — ST −0.34 / HV
   −0.19 m/s per-section RMSE, ~30%/22% lower; the a-ablation shows the gain is the start-phase shape.
2. The rich likelihood SURFACES A NEW WIN-RELEVANT LATENT (`a`) that win alone could not identify
   (it is invisible to the settled-level order statistic but the opening-split fingerprint pins it
   down): independent win signal, t(a)=+10 in ST sprints, distance-band gradient as theory predicts.
   This is NOT the §3e gate-break trait that DON'T-BAKED earlier (0030/queue-31): that was first-call
   RANK-minus-draw (a positional/policy read); `a` is the kinematic opening-velocity ramp recovered
   from the profile — a capacity/acceleration latent, and it converts as a partial-corr where the
   rank-based gate trait did not.

## RECOMMENDED NEXT STEP (the wire-in gate)
Wire `a` into the START-PHASE drift and run the paired fixed-seed full-scale win gate. Concretely:
the production homogeneous Wald has one drift over the whole trip; add a leak-free strictly-prior
trailing `a` per horse loading a start-phase drift bump (THEORY §3e Phase 1: `T_i = T_0,i + T_main,i`,
`a` sets `T_0,i` — the time gained/lost off the gate). Expect it to pay MOST on ST sprints (the t-stat
gradient + the §3e "start is a big fraction of a short trip" argument; weight expectations by distance
band). Then, per the batch direction, ADD MARGINS to the fitting surface (the profile-likelihood here is
the velocity surface; later finishing margins) to identify further latents. CAVEATS: ΔR² is small in
absolute terms (the proxy over-states relative to the win order-statistic, like every t-stat — the
paired %market+ECE gate is the real test, NOT this proxy); `v_cruise`/`a` definition uses the
mid-trip mean, robust for K≥4 but K=3 uses the single middle section (noisier). The win-gate, not this
proxy, decides the bake.

## ARTIFACTS (fit.py + production untouched)
- src/hkjc/model/energetics_dynamic.py — ADDITIVE: `ProfileRunEnergetics`, `estimate_run_profile`
  (recovers `a`), `expected_profile` (generative K-section profile), `profile_loglik` (log-velocity
  Gaussian). No existing function changed; ruff + the 4 energetics tests pass.
- studies/profile_likelihood_pilot.py — the pilot: LOO profile-fit RMSE (4-latent vs a-ablation vs two
  scalar nulls), win-relevance ΔR²/partial-corr with ST distance-band breakout, assert_no_lookahead,
  the `a`-fingerprint plot (studies/out/profile_likelihood_pilot.png).
