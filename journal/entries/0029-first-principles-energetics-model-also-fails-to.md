---
id: 0029-first-principles-energetics-model-also-fails-to
title: First-principles energetics model also fails to beat the market (5th negative)
kind: result
status: open
tags: ["energetics", "first-principles", "market-efficiency", "model-edge-closed"]
refs: {"experiment": "experiments/validate_energetics_model.py"}
related: []
created: 2026-06-04T06:16:58+00:00
updated: 2026-06-04T06:16:58+00:00
---
Generative critical-velocity model (269 races, 27 meetings). Trailing pace-neutral (v_c, W') -> EnergeticsRaceModel simulates finishing times -> win probs. Walk-forward race_win_log_loss: market 2.049 baseline; physics-only 2.610 (-27%, worse than uniform -> raw physics poorly calibrated); clogit[mkt] 2.061 (-0.6%); clogit[mkt,physics] 2.066 (-0.8%, physics ADDS NOTHING, marginally hurts). The mechanistic model carries no independent signal beyond the closing line. FIFTH consecutive negative across BOTH paradigms: statistical (crude 0011, rating/form 0023, sectional 0025, trouble 0026) AND first-principles dynamical (energetics, this). CONCLUSION: the HK win market is efficient w.r.t. everything we can extract -- unsurprising, as it's one of the most heavily-modelled markets in the world (Benter et al. already run exactly these speed/stamina/pace models with decades of data). MODEL-EDGE LINE IS CLOSED. Caveats: 3-month samples; trailing (v_c,W') noisy (1-2 prior runs/horse); time_noise uncalibrated -- but the combined logit optimally reweights the physics feature and still fails, and the 5x consistency + strong prior make this robust for resource allocation. Only remaining model-edge lever is full-season data (low prior, big cost) -- NOT recommended over the carryover structural line, which is the project's real edge and is compounding live data.
