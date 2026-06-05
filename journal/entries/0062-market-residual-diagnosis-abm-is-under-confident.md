---
id: 0062-market-residual-diagnosis-abm-is-under-confident
title: Market-residual diagnosis: ABM is under-confident on ability (form t=32); mass is the next missing mechanism
kind: result
status: open
tags: ["goodness-of-fit", "market-residual", "calibration", "form-gain", "speed-noise", "under-confident", "mass", "diagnosis", "standing-order"]
refs: {}
related: []
created: 2026-06-05T00:05:54+00:00
updated: 2026-06-05T00:05:54+00:00
---
MARKET-RESIDUAL DIAGNOSIS (market_residual_diagnosis.py): the principled goodness-of-fit detector.
Ran the leak-free ST ABM walk-forward (24187 OOS runners, 1932 races, N=1500), took resid = market_prob -
model_prob (both de-vigged + renormalised per race), regressed on observable mechanisms (cluster-robust by race).

mean |resid| = 0.0457 (per-runner prob gap to market).

HEADLINE: the ABM is UNDER-CONFIDENT ON ABILITY. Every ability axis has POSITIVE residual slope (market puts
MORE prob on strong horses than the model does): form t=+32.0, v_c_eb t=+20.8, w_prime_eb t=+5.6, rtg t=+2.8.
The model win-prob distribution is too FLAT vs the market. Likely cause: FORM_GAIN=2.0 too small and/or
SPEED_NOISE=0.12 too large (the backtest deliberately set noise high; the residual says it over-flattens).
This is the #1 fit gap and it is a CALIBRATION of a real mechanism parameter, not a new covariate.

Secondary, real: bodywt UNDER-weighted t=+11.6, actwt t=+6.0 -> the market prices MASS, which the ABM ignores
entirely (traffic_jax has NO mass term; DynamicsRaceModel mass unused). The mass mechanism is the next lever
after ability spread. draw OVER-weighted t=-17.1 and style OVER-weighted t=-5.4 and burden OVER t=-5.6 -> the
draw/geometry + style couplings are mis-set (too strong / mis-signed); revisit after fixing the ability spread.

METHOD CAVEAT (recorded so we do not repeat it): the GOING and CLASS categorical means were ALL ~0 because
they are RACE-LEVEL constants and the residual is race-normalised (sums to zero within race) -> per-level mean
is mechanically zero. Race-level effects (going x K_LOCO, class) need a CALIBRATION-BY-CONDITION test (e.g.
log-loss or favourite-hit-rate by going), NOT a per-runner residual mean. TODO.

NEXT: sweep FORM_GAIN x SPEED_NOISE, measure OOS % of market (race log-loss vs OUTCOMES, not vs market), confirm
the under-confidence and re-calibrate. Then wire the mass term (bodywt) into the sim. Objective is FIT not edge.
