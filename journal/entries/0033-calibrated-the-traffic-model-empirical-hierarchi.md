---
id: 0033-calibrated-the-traffic-model-empirical-hierarchi
title: Calibrated the traffic model: empirical/hierarchical Bayes energetics + per-venue bend_curvature
kind: finding
status: open
tags: ["calibration", "traffic-model", "empirical-bayes", "hierarchical-bayes", "numpyro", "jax", "draw-bias", "rating"]
refs: {}
related: ["0032-consolidated-geometry-canonical-race-domain-obje"]
created: 2026-06-04T09:22:53+00:00
updated: 2026-06-04T09:22:53+00:00
---
Two-layer Bayesian calibration of the interacting traffic model on the 2026 cached data (HKJC_tools). Module hkjc.calibration (numpy, core) + hkjc.calibration_bayes (NumPyro) + hkjc.model.traffic_jax (JAX sim). Experiment: experiments/calibrate_traffic_model.py.

LAYER 1 - horse energetics, hierarchical/empirical Bayes, RATING-SEGMENTED.
One-way Normal-Normal random-effects on per-run (v_c, w_prime) with a fixed effect of distance-band + a rating slope (population is rating-conditioned, not one mono-pool). Fetched per-run official ratings for all energetics horses (experiments/build_rating_data.py -> data_cache/rating_dataset.parquet: 16,706 (horse,date) ratings, 1083/1116 horses), joined as-of nearest.
- v_c band means: sprint 17.13 > mile 16.65 > middle 16.43 > staying 16.07 m/s (faster v_c at shorter trips, as expected).
- v_c rating slope +0.0093 m/s per rating pt (100-pt class gap ~ +0.93 m/s). w_prime rating slope negative (an estimator artifact: higher v_c leaves less velocity above v_c).
- Variance components: sigma_within 0.569 > tau_between 0.212 for v_c => per-run v_c is NOISIER than the between-horse spread after conditioning on rating+distance. Shrinkage at n=3 = 0.29 -> a 3-run horse is pulled 71% to its rating-predicted mean. Strong justification for partial pooling. Rating segmentation provably reduces error for one-run horses (test).
- Two implementations agree: closed-form EB (calibration.py, fast, leakage-safe LOO) and full hierarchical Bayes via NumPyro/NUTS (calibration_bayes.py, proper posteriors). 

LAYER 2 - per-venue bend_curvature (the draw-bias magnitude), two objectives.
Held LOO-EB abilities fixed; fit bend_curvature per venue.
(a) Bayesian winner-likelihood: P(actual winner | real abilities, draws, geometry) over a grid, lognormal prior, venue-level EB pooling. ST -> 0.0040 (1/247). HV -> rails at grid floor 0.0010 (underpowered at 98 races; conditional-on-ability winner objective too strict for HV).
(b) Draw-slope moment match (directly fits empirical mean-finish-vs-draw): HV emp slope +0.154 -> bend 0.0023 (1/429); ST emp slope +0.166 -> bend 0.0037 (1/269).
For ST the two objectives CONVERGE (0.0040 vs 0.0037) - strong validation.

RESULTS / baked: CALIBRATED_BEND_CURVATURE = {HV: 0.0023, ST: 0.0037} (moment-match; Race.traffic_model uses per-venue automatically).
1. Both ~half the uncalibrated default 1/120=0.0083 => the model WAS over-amplified; now corrected.
2. HV curvature (0.0023) < ST (0.0037) despite HV being more bend-dominated - geometrically self-consistent: HV bend_fraction 0.62 vs ST 0.42 amplifies the same curvature more, so matching similar empirical draw-slopes needs LESS curvature at HV.
3. Open: HV winner-likelihood under-identification (more races via the flywheel; Tier-B aerial-video lateral trajectories; or a joint winner+draw-curve objective). AWT uses the ST value (no AWT-specific data yet). ST straight 1000m chute correctly has no bend term.

SPEED: JAX port of the traffic sim (traffic_jax, lax.scan + vmap over the bend grid, traced bend/speed params) matches numpy to <0.005 in P(win|draw); the JAX likelihood backend is ~5.4x faster than numpy for the grid sweep. NumPyro/JAX are an opt-in dep group (uv sync --group bayes); core stays light.
