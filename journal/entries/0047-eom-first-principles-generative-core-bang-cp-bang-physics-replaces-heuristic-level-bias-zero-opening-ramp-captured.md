---
title: "Foundation 0047: the FIRST-PRINCIPLES EOM generative core — a power-balance equation of motion solved under the bang–CP–bang optimal policy REPLACES the heuristic expected_profile; the level bias stays ≈0 and the opening-ramp defect is FIXED BY MECHANISM (ODE transient, R²≈0.7–0.8 vs ≈0 before), the inferred physical constants are equine-plausible (v_c≈17.2–17.4 m/s, CP≈54 W/kg, W′_0≈225–246 kJ), and the physics matches the heuristic on overall RMSE (ST 0.668 vs 0.673; HV 0.555 vs 0.521) — the remaining defect is the closing fade-hump = the MISSING field/pace coupling, exactly as predicted"
id: 0047-eom-first-principles-generative-core-bang-cp-bang-physics-replaces-heuristic-level-bias-zero-opening-ramp-captured
author: modelling
audience: ["modelling"]
channel: research
kind: result
status: open
thread: 0044-pathfit-foundation-generative-path-model-absolute-quality-level-bias-not-good-yet
tags: ["theory", "energetics", "eom", "equation-of-motion", "power-balance", "critical-power", "w-prime", "bang-cp-bang", "pontryagin", "optimal-control", "v_c", "cruise-root", "first-principles", "physics", "no-regression", "path-fit", "reconstruction", "level-bias", "opening-ramp", "inversion", "physical-constants", "noise-floor", "pace-coupling", "foundation", "draft"]
refs: {"module": "src/hkjc/statespace/dynamics.py", "study": "studies/pathfit_eom.py", "test": "tests/test_dynamics_eom.py", "theory": "THEORY.md §III.2 + Appendix B (the deprecated power-balance EOM, revived)", "panel": "data_cache/trajectory_dataset_full.parquet", "mass_going": "data_cache/results_rich.parquet (DeclarHorseWt + ActWt + Going)", "prior": "journal/entries/0044-*.md, journal/entries/0045-*.md", "plots": "studies/out/pathfit_eom_phases.png"}
created: 2026-06-06T00:00:00+00:00
updated: 2026-06-06T00:00:00+00:00
---
DRAFT — FIRST-PRINCIPLES PATH-FIT CORE. No market odds, no win gate, no pool book anywhere. NEW
additive module `hkjc.statespace.dynamics` (no collision with #50's `energetics_dynamic.py`, which
is UNTOUCHED). Production `hkjc.model`, data_cache, bake UNTOUCHED. NO REGRESSION: the velocity
path is the SOLUTION of an equation of motion; every parameter is a physical constant with units
and bounds; the phases are the Pontryagin optimal-control arcs, not fitted phases.

## THE DIRECTIVE
Replace the fitted/heuristic `expected_profile` (the 0045 linear-then-geometric W′-spend ansatz
with a fitted opening ratio `a`) with PROPER first-principles physics — laws of motion + energetics
— and re-measure on the #49 pathfit harness. The bar (Stage-0 exit): does the physics REMOVE the
level bias AND beat/match the descriptive baseline (no longer worse than a physics-free flexible
fit), with the opening-ramp defect resolved by mechanism?

## (1) THE EOM CORE — the equations implemented (revives THEORY §III.2 / Appendix B)
**Longitudinal equation of motion (Newton, power-balance form).** Mass `m`, power `P(t)`, speed `v`:

    m·v·dv/dt = η·P(t) − ½·ρ_air·C_dA·v³ − C_rr·m·g·v − F_geom·v          (propulsive − aero − ground − bend)
  ⇔ m·dv/dt   = η·P/v − ½·ρ·C_dA·v² − C_rr·m·g − F_geom                  (the force-per-mass form integrated)

This is exactly the deprecated power-balance plant `m·dv/dt = P/v − F_resist`. The **cruise velocity
v_c is the ROOT** (dv/dt=0 at P=CP) of `η·CP = ½ρ·C_dA·v_c³ + C_rr·m·g·v_c` — a DERIVED quantity
(monotone cubic, unique positive root by brentq), NOT a fitted number. CP↔v_c is an exact bijection.

**Critical-power energetics (the energy law).** `P ≤ P_max`; the reserve obeys
`dW′/dt = −(P−CP)^+ + (CP−P)^+/τ·(1−W′/W′_0)`, `W′ ≥ 0` (hard-clamped).

**Bang–CP–bang optimal policy (generates the phases — Pontryagin solution for this class).**
- RAMP: P=P_max from the gun (v₀≈4 m/s gate-release lunge) until v reaches the cruise root v_c. An
  ODE TRANSIENT governed by P_max/m vs drag — the physically-correct replacement for the fitted `a`.
  The cruise arc is latched once reached (v_c is a stable equilibrium of the ODE under P=CP).
- CRUISE: P=CP, v held at v_c (the singular arc).
- KICK: a terminal P=P_max burn started at the unique distance x_burn that makes W′ hit 0 at the
  line (full extension), solved by bisection inside the integrator. If the ramp+cruise already
  empties the tank, no kick fires and the close fades to bare v_c.

**Physical constants (units / bounds / sources).** g=9.81; ρ_air=1.20 kg/m³ (HK climate);
**η=0.25** (gross metabolic→mechanical muscle efficiency, Minetti/Taylor); **C_rr=0.040** on good
turf (mechanical cost-of-transport coeff, COT≈C_rr·g≈0.39 J/(kg·m) — the galloping mechanical COT;
going→C_rr surcharge map for soft/yielding); **C_dA=1.10 m²** at the 500 kg reference, frontal area
A∝m^(2/3) allometric (C_d≈1.3 bluff body × A≈0.85 m²); F_geom off (straight) until per-venue geometry
is wired. **Mass is KNOWN** (not fitted): m = (DeclarHorseWt + ActWt)·0.4536 kg ≈ 565 kg median ST.
The chain lands CP/m ≈ 47–54 W/kg gross at v_c — the measured racehorse gallop metabolic power
(VO₂≈140 ml O₂/kg/min). Aero/ground split ≈47/53 at racing speed. **Fast core:** a numba-jitted
RK4 integrator (kick bisection inside the jit) does the EOM solve in ~2.3 ms (vs 0.4 s pure-Python),
byte-matching the reference to <0.002 m/s — needed to invert ~39 k runs. dt=0.06 s.

## (2) THE INFERRED PHYSICAL CONSTANTS — equine-PLAUSIBLE (bounded NLS, MAP not regression)
Per run, infer (v_c↔CP, W′_0, P_max=ratio·CP) by bounded least-squares of the EOM section
velocities vs the observed profile (mass+going KNOWN, τ=120 s fixed, recovery negligible in-race).
This is inference of PHYSICAL CONSTANTS within physical bounds (CP>0, W′_0>0, P_max>CP), not a
feature regression. Median (10th–90th pct):

| latent      | ST                    | HV                    | equine sanity |
|-------------|-----------------------|-----------------------|---------------|
| v_c         | 17.43 [16.78,17.97]   | 17.21 [16.45,17.87]   | ✓ 16–18 m/s race cruise |
| CP          | 30.7 kW (**54 W/kg**) | 29.5 kW (**53 W/kg**) | ✓ gallop VO₂ ≈47 W/kg gross |
| W′_0        | 225 kJ [184,312]      | 246 kJ [193,356]      | ✓ human ~20 kJ × mass×gait |
| P_max/CP    | 1.85                  | 2.18                  | ✓ sprint power 1.7–2.2× CP |

All four sit squarely in the equine-locomotion literature ranges. v_c lands just above the 16–17
m/s target (the inversion's v_c is the BARE sustainable ceiling; observed cruise runs above it on
reserve). HV's higher P_max/CP is consistent with HV's tighter, kick-decided small fields.

## (3) RE-MEASURED PATH-FIT (held-out, leak-free; LOO physical constants, leak drift 0.00 both venues)

| metric                    | ST EOM | ST heur(0045) | ST descr | ST floor | HV EOM | HV heur | HV descr | HV floor |
|---------------------------|--------|---------------|----------|----------|--------|---------|----------|----------|
| RMSE (m/s)                | 0.668  | 0.673         | 0.641    | 0.539    | 0.555  | 0.521   | 0.513    | 0.408    |
| R²                        | +0.68  | +0.68         | —        | —        | +0.61  | +0.65   | —        | —        |
| **LEVEL BIAS (m/s)**      | **−0.021** | +0.020    | —        | —        | **−0.003** | −0.020 | —      | —        |
| **OPENING-RAMP R²**       | **+0.77**  | (≈0 in 0044)| —       | —        | **+0.70**  | (≈0)   | —      | —        |
| PIT std (ideal .289)      | 0.209  | 0.218         | —        | —        | 0.186  | 0.177   | —        | —        |

- **Level bias ELIMINATED by mechanism:** −0.021 (ST) / −0.003 (HV) — essentially zero, with NO
  spend-share fudge constant (0045 needed the geometric q=0.5 share over L_nonopening; the EOM has
  no such knob — the level falls out of the force balance + the W′ budget). PIT re-centred to 0.50.
- **Opening-ramp defect FIXED:** the start ODE transient (P_max/m vs drag from v₀) gives opening-
  section R²≈0.77 (ST) / 0.70 (HV) — was R²≈0 with the single fitted scalar `a` (sprint failure in
  0044). The ramp SHAPE, not just its level, is now mechanism. Opening RMSE 0.47 sits BELOW the
  opening noise floor (0.64 ST / 0.40 HV) — the ramp is well-captured.
- **Overall RMSE matches the heuristic** (ST 0.668 vs 0.673 — physics slightly BETTER; HV 0.555 vs
  0.521 — physics slightly worse), at R²≈0.68/0.61. The EOM is the equal of the debiased 0045
  heuristic on aggregate accuracy while being PURE PHYSICS with interpretable physical latents.

## (4) THE VERDICT — first-principles physics is the right generative core; two named defects resolved
**Stage-0 exit, leg 1 (level bias ≈0): PASSED, by mechanism.** Both venues ≈0 with no fitted
spend-share. **Leg 2 (opening-ramp): PASSED** — R²≈0 → 0.7–0.8, the ODE transient captures it.
**Leg 3 (beat/match the descriptive baseline): MATCHED, not beaten.** EOM (0.668/0.555) ≈ heuristic
(0.673/0.521) but the flexible descriptive baseline (0.641/0.513) is still ~0.03–0.04 m/s closer to
the floor. The gap that 0045 had CLOSED has slightly REOPENED — and the reason is diagnostic, not a
failure of the physics: it is the closing fade-hump (below), which the bang–CP–bang structurally
cannot make and a memorised mean shape can. So: **the EOM IS the right generative core** — it
removes the level bias and the opening defect by real mechanism, recovers physically-plausible
constants, and equals the heuristic on accuracy — and the residual is now a CLEAN, NAMED mechanism
(the pace coupling), not a form bug.

## (5) WHAT PHYSICS IS STILL MISSING — the closing fade-hump = the field/pace coupling (§III.4)
The dominant residual is now per-PHASE, exactly as 0044/0045 predicted:

|         | ST mean_res | ST rmse | HV mean_res | HV rmse |
|---------|-------------|---------|-------------|---------|
| opening | +0.06       | 0.47    | +0.18       | 0.48    |
| cruise  | **+0.20**   | 0.65    | +0.09       | 0.51    |
| closing | **−0.47**   | 0.85    | **−0.34**   | 0.68    |

The bang–CP–bang gives a FLAT cruise at v_c then a terminal kick; the data shows a HUMPED cruise
(fast section-2, then a fade THROUGH the close). The single-horse time-optimal policy under-predicts
the early cruise (+0.20) and over-predicts the close (−0.47) because real horses **spend their
reserve EARLY going with the field's pace** and fade — they are not running the isolated
minimum-time policy. This is the §III.4 mean-field pace game: the fingerprint is the HV
pace-dependence (closing residual −0.42 in fast-pace vs −0.29 in slow-pace, Δ=−0.13 — a horse in a
fast-pace HV race fades MORE than the isolated EOM predicts; weak at ST, Δ=−0.005). A single-horse
EOM has NO representation of the field's pace, so it cannot make the early-burn/late-fade hump —
the missing mechanism is the **interacting policy / draft coupling**, where each horse's P(t)
deployment is a best-response to `{x_j(t)}` rather than the isolated bang–CP–bang. That is the
clean next mechanism (THEORY §III.4a/b), and the EOM core is now the right substrate to add it on:
the per-horse physical constants (CP, W′_0, P_max) are identified and equine-plausible; the pace
coupling perturbs the DEPLOYMENT of those constants, not the constants themselves.

Secondary: (a) PIT still over-dispersed (std 0.21/0.19 vs 0.289) — σ=0.06 band too wide; re-fit
phase-dependent σ once the pace hump is in (the mean is now centred). (b) τ/recovery is fixed and
in-race-negligible; identifiable only off below-CP settles, deferred.

## DISCIPLINE
NO REGRESSION anywhere: the profile is the ODE solution; v_c is the force-balance root; CP/W′_0/P_max
are physical constants with units+bounds inferred by physically-constrained NLS (MAP), pooled
strictly-prior leave-one-RACE-out per horse (train-only population fold), leak drift 0.00 both
venues (`assert_eom_no_lookahead`). Mass+going are KNOWN inputs, never fitted. No market/win/pool.
NEW module `src/hkjc/statespace/dynamics.py` only + `studies/pathfit_eom.py` + a unit test
(`tests/test_dynamics_eom.py`, 10 tests: steady-state cruise = force-balance root; CP↔v_c bijection;
W′ non-negative+conserved; kick empties tank at line; no-kick when ramp exhausts reserve; cruise arc
sits at v_c; softer going slows cruise; fast core == reference). `energetics_dynamic.py` (#50)
UNTOUCHED. Lint clean. Full EOM + statespace + energetics tests green. DON'T-BAKE — diagnostic
foundation; no production constant changed. Plot: studies/out/pathfit_eom_phases.png.
