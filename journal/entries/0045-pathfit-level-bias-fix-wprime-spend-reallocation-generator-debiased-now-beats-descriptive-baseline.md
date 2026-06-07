---
title: "Foundation 0045: the +0.42 m/s generative LEVEL BIAS is FIXED — root cause was the W′-spend allocation in `expected_profile` (full-trip denominator + linear ramp pinning the close to bare v_c); re-allocating the reserve over the NON-OPENING trip with a geometric (~halving) spend share de-biases the path to ~0 and the physics generator now BEATS the flexible descriptive baseline at both venues"
id: 0045-pathfit-level-bias-fix-wprime-spend-reallocation-generator-debiased-now-beats-descriptive-baseline
author: modelling
audience: ["modelling"]
channel: research
kind: result
status: open
thread: 0044-pathfit-foundation-generative-path-model-absolute-quality-level-bias-not-good-yet
tags: ["theory", "energetics", "w-prime", "v_c", "generative", "path-fit", "reconstruction", "level-bias", "spend-kinetics", "calibration", "noise-floor", "identification", "foundation", "draft"]
refs: {"module": "src/hkjc/model/energetics_dynamic.py", "study": "studies/pathfit_foundation.py", "pathfit": "src/hkjc/statespace/pathfit.py", "theory": "THEORY.md §III.2 (Hill distance-reserve law)", "panel": "data_cache/trajectory_dataset_full.parquet", "prior": "journal/entries/0044-*.md", "plots": "studies/out/pathfit_*.png"}
created: 2026-06-06T00:00:00+00:00
updated: 2026-06-06T00:00:00+00:00
---
DRAFT — PATH-FIT FOUNDATION FIX. No market odds, no win gate, no pool book anywhere. Production
`hkjc.model/fit.py`, data_cache, bake UNTOUCHED. The fix is in the generative forward model
(`energetics_dynamic.expected_profile`) only; full test suite (158) + the byte-exact statespace
a-OFF==production acceptance gate still green.

## THE DEFECT (from 0044)
`expected_profile` ran the whole velocity path +0.42 m/s TOO SLOW everywhere — a structural LEVEL
bias present even with the run's OWN in-sample θ (so it is the FORWARD model, not θ recovery). The
generator lost to a physics-free flexible descriptive baseline (ST 0.841 vs 0.681; HV 0.652 vs
0.553). A biased generator biases every latent identified through it → BLOCKS the program.

## DIAGNOSIS (confirmed: root cause (a) AND (b), one unified mechanism)
Instrumented `expected_profile` on 16,092 ST/HV 4-section runs with each run's OWN θ. Per-section
mean (in-sample, isolates the forward model):

| sec | obs | gen(old) | obs−gen | gen−v_c |
|-----|-----|----------|---------|---------|
| 1 (open)  | 15.18 | 14.82 | +0.355 | −1.673 |
| 2 (cruise)| 17.52 | 16.95 | +0.571 | +0.457 |
| 3 (cruise)| 16.99 | 16.72 | +0.264 | +0.229 |
| 4 (close) | 16.75 | 16.49 | +0.257 | **+0.000** |

Two compounding bugs, both flavours of the 0044 hypothesis:
- **(b) the close pinned to bare v_c.** The linear ramp `w=(k−1−idx)` forces the final weight to
  EXACTLY 0 ⇒ gen closing = v_c precisely. But observed runs the close +0.257 ABOVE v_c (the horse
  fades TOWARD v_c, it does not arrive AT it). A clean v_c-as-close mis-centering.
- **(a) the boost MAGNITUDE was scale-collapsed.** The old form used
  `boost_frac = W′_0/dist_total` (FULL trip, incl. the opening) and the linear ramp halved the
  realized lift. Net: the generated time-weighted mean fell BELOW v_c (16.33 < v_c 16.49,
  n_sec=4), while the observed non-opening mean sits +0.55 m/s (≈+3.5% of v_c) ABOVE v_c across ALL
  n_sec (3/4/5/6: +0.561/+0.576/+0.527/+0.468).

The reconciliation that fixes it: `W′_0 = Σ(v−v_c)^+·t` is reserve-DISTANCE (the extra metres run
above the ceiling), spent over the NON-OPENING trip (the opening is the `a` start transient, not
the reserve). Then the time-weighted non-opening lift is exactly `v_c·(W′_0/L_nonopening)` —
measured `W′_0/L_nonop = 0.0354` vs observed boost frac `0.0354` (1e-3 agreement). The observed
per-section excess over v_c decays geometrically (~halving: 0.063 → 0.031 → 0.015 of v_c) — the
reserve depletes MULTIPLICATIVELY and the close stays above v_c.

## THE FIX (faithful mechanism, not a fudge constant)
Re-specified the non-opening spend in `expected_profile`:
1. **Budget over the non-opening trip:** `boost_frac = W′_0 / L_nonopening` (was `/dist_total`).
   This is the conserved reserve-distance per non-opening metre — the THEORY §III.2 Hill lift
   `v_eff = v_c·(1+W′/L)` applied to the phase the reserve actually governs.
2. **Geometric spend share** `g_j = q^j` (q=0.5: each non-opening section ~half the previous
   boost), normalised to time-weighted mean 1 over the non-opening sections so the realized lift
   equals the conserved budget. The close carries `g_last > 0` — it is NEVER pinned to bare v_c.
3. Opening override `v_1 = a·cruise` unchanged. k=1 fallback: lift over its own distance.
Reserve stays non-negative; the budget is conserved (the time-weighted lift integral = W′_0/L).
`q=0.5` matches the observed geometric decay; verified de-biasing holds for n_sec=3/4/5/6
(in-sample overall bias +0.058/+0.016/+0.003/−0.022, vs +0.36 before).

## RE-MEASURED PATH-FIT (held-out, leak-free; LOO θ, leak drift 0.00 both venues)

| metric | ST before | ST after | HV before | HV after |
|--------|-----------|----------|-----------|----------|
| gen RMSE (m/s)        | 0.841 | **0.673** | 0.652 | **0.521** |
| gen R²                | +0.49 | **+0.68** | +0.46 | **+0.65** |
| excess over floor     | +56%  | **+25%**  | +60%  | **+28%**  |
| descriptive RMSE      | 0.681 | 0.681     | 0.553 | 0.553     |
| **gen vs descr**      | LOSES | **WINS**  | LOSES | **WINS**  |
| PIT mean (ideal .500) | 0.625 | **0.498** | 0.608 | **0.494** |
| PIT std (ideal .289)  | 0.226 | 0.218     | 0.189 | 0.177     |
| mean residual (m/s)   | +0.42 | **+0.02** | +0.31 | **−0.02** |

Per-phase mean residual collapsed to ~0 (ST opening +0.49→+0.07, cruise +0.47→+0.04, close
+0.19→−0.15; HV similarly). The a:=1 ablation got WORSE (ST 1.20→1.33) — `a` is even more
load-bearing once the level is right (confirms #41). Noise floor unchanged (0.539 ST / 0.408 HV).

## DID IT CLEAR THE BAR? — YES.
Stage-0 exit = **bias ≈ 0 AND the gap to the descriptive baseline closed**. Both met:
- Bias: +0.42 → +0.02 (ST), +0.31 → −0.02 (HV) — essentially eliminated.
- Gap: the physics generator now **beats** the flexible physics-free baseline at BOTH venues
  (ST 0.673<0.681; HV 0.521<0.553), and sits closer to the floor than descriptive (ST gen excess
  +0.134 < descr +0.142; HV +0.113 < descr +0.145). The physics no longer pays a price for being
  physics — it is now the better path model. PIT re-centred to 0.50.

## RESIDUAL DEFECTS NOW DOMINANT (clean targets for the NEXT agent — NOT touched here)
With the level fixed, the structured residual is exactly what 0044 predicted would remain:
1. **OPENING ramp shape** — still the worst phase (ST opening de-biased RMSE 0.84 vs floor 0.64;
   sprint R² +0.57 now but opening still carries the largest excess). `a` as a single scalar
   captures the ramp LEVEL not its SHAPE → the §III.3 Phase-1 acceleration latent (ramp rate /
   time-constant). Sprint mean_res is now the only band materially off (ST +0.16).
2. **PACE-dependent closing miss** — now MORE visible (the level bias was masking it): closing
   residual is −0.11 m/s worse in fast-pace races (ST), −0.20 (HV). This is the §III.4 mean-field
   pace coupling a single-horse generator structurally cannot see. The closing now slightly
   OVER-predicts (mean_res −0.15 ST) — the geometric q may want to be phase/pace-conditioned.
3. **σ over-dispersion** — band still too wide (PIT std 0.218/0.177 vs ideal 0.289; 68% coverage
   0.86/0.94). Now that the mean is centred, re-fit σ (and make it phase-dependent: opening band ≠
   cruise band) — secondary, mechanical.
4. **Staying** (n_sec≥6, thin n) shows a residual per-section wave (cruise over-, close under-boost)
   — the single global q=0.5 is slightly off for long races; a mild n_sec-dependence or the
   true balance-path spend would refine it. Low priority (228/804 runs).

Fix order for the program is now: (1) [DONE] level bias; (2) opening-ramp shape; (3) phase-dep σ;
(4) the pace interaction. Only after these should profile-identified latents promote into the §V
coupling/gate program.

## DISCIPLINE
Leak-free held-out throughout (LOO θ, drift 0.00). The +0.42 diagnosis used in-sample θ to isolate
the forward model (correct); the final numbers above are the held-out ones. Full pytest suite
158/158 green, incl. `test_a_off_reproduces_production_wald` (byte-exact statespace acceptance) and
`test_coupling_a_off_matches_bare_law` — the fix is in the profile-likelihood forward model only and
does not perturb the production wald win law. `estimate_run_profile`/`profile_loglik` θ-recovery
unchanged (they call the fixed `expected_profile`; the recovery of v_c/W′_0/a is untouched). Lint
clean. DON'T-BAKE — diagnostic foundation; no production constant changed. Plots in
studies/out/pathfit_*.png.
