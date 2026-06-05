---
id: 0064-re-diagnosis-at-fg-4-ability-fixed-draw-over-wei
title: Re-diagnosis at FG=4: ability fixed; DRAW over-weight (t=-17) is now the #1 miss, then W-prime under-weight (t=+13), then bodywt/mass
kind: result
status: open
tags: ["goodness-of-fit", "market-residual", "re-diagnosis", "draw", "bend-curvature", "w-prime", "mass", "diagnose-recalibrate-loop", "standing-order"]
refs: {}
related: []
created: 2026-06-05T02:17:12+00:00
updated: 2026-06-05T02:17:12+00:00
---
RE-DIAGNOSIS at FORM_GAIN=4 (market_residual_diagnosis re-run, same 24187 OOS ST runners). mean |resid|
0.0457 -> 0.0430 (fit improved, consistent with +4.9pp). The residual structure RESHUFFLED as expected:

ABILITY now CORRECTLY weighted (the fix landed, slightly past optimum): form +0.0133 t=+32 -> -0.0039 t=-9.8
(now mildly OVER-weighted); v_c_eb +0.0104 -> -0.0036 t=-8.2. The sign-flip confirms FORM_GAIN=4 sits just
past the optimum (true peak ~3.5), but full-scale % of market still strongly favours 4.0 over 2.0, so keep 4.0;
a 3.5 fine-tune is marginal. rtg now t=-1.0 (neutral).

NEW residual queue (priority by |t|):
1. DRAW OVER-weighted t=-17.4 (was -17.1; UNCHANGED by the form fix -> a genuine SEPARATE miscalibration). The
   ABM gives too much win-prob advantage from barrier draw vs market+outcomes. This is the bend geometry coupling
   (bend_curvature=0.0037 ST + off0 lateral start). Like FORM_GAIN, CALIBRATED_BEND_CURVATURE was likely set on
   the old 3-month subset and is now too strong on full history. CLEAN next calibration: sweep bend_curvature
   down, measure % of market. Biggest single remaining miss.
2. W_PRIME UNDER-weighted t=+12.9 (was +5.6, now the top ability-axis miss). Raising FORM_GAIN sharpened v_c but
   the anaerobic RESERVE channel is now relatively under-used -- the market values stamina/reserve more than the
   model. Candidate: a W'_GAIN (reserve analog of FORM_GAIN) or relax the W' EB shrinkage. But note W' is the
   noisier sectional estimate, so amplify with care (the SPEED_NOISE lesson).
3. BODYWT UNDER-weighted t=+9.0 (persistent): the market prices body MASS, which traffic_jax ignores entirely
   (no mass term). The mass mechanism. Confound caveat: bodywt tangles with condition + class, so needs a
   careful mechanistic coupling, not a flat add.

(style OVER t=-4.0, actwt/burden OVER t=-4.3/-9.7 -- secondary; burden over-weight likely a handicap-confound
artifact. GOING/CLASS still uninformative here -- race-level constants vanish under race-normalised residual;
they need a calibration-by-condition test, TODO.)

Recommended next: calibrate bend_curvature (draw) -- the #1 remaining miss and a clean single-constant fix like
FORM_GAIN was. PAUSED for user instruction per request.
