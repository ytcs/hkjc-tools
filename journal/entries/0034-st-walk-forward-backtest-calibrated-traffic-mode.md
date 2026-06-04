---
id: 0034-st-walk-forward-backtest-calibrated-traffic-mode
title: ST walk-forward backtest: calibrated traffic model is well-calibrated but ~0.40 nats/race short of the market
kind: finding
status: open
tags: ["backtest", "walk-forward", "traffic-model", "sha-tin", "market", "calibration", "leakage-audit"]
refs: {}
related: []
created: 2026-06-04T09:39:56+00:00
updated: 2026-06-04T09:39:56+00:00
---
Leak-free walk-forward backtest of the calibrated interacting traffic model at Sha Tin (experiments/backtest_traffic_st.py), scored through the eval harness vs the de-vigged market and a uniform baseline. 161 ST races, 2044 runner-rows, 15 meeting-days, 5 expanding folds (embargo=1). N=6000 JAX sims/race, bend_curvature=0.0037 (calibrated structural constant).

Leakage controls (the point of the exercise): inside every fold the energetics population is fit on TRAIN ONLY, and each test horse ability is the empirical-Bayes estimate from its strictly-prior (train) runs, adapted to the test distance band + the horse rating at race time (rating joined as-of BACKWARD). Debut/no-sectional horses fall back to the rating-conditioned population mean. The model never consumes odds (structural). Trailing-only property unit-tested (horse_energetics_trailing).

RESULTS (race_win_log_loss, lower better; nats/race):
  uniform 2.5338 | traffic 2.4882 | market 2.0922
  skill vs uniform: traffic +0.0455 | market +0.4416
  gap to market: traffic - market = +0.3961 (traffic worse)
  Brier: uniform .0718 | traffic .0711 | market .0650
  ECE (calibration): traffic .0207 | market .0247 | uniform .0005

INTERPRETATION:
1. The structural physics model has MODEST but REAL skill - it beats uniform by +0.046 nats/race (draw geometry + rating-informed trailing energetics carry genuine signal).
2. It captures only ~10% of the market skill over uniform (0.046/0.442); the market is ~10x more informative and the traffic model sits ~0.40 nats/race behind it. Expected - the crowd prices jockey/trainer/intent/gear/condition/insider info the physics doesn't see.
3. It is WELL-CALIBRATED out-of-sample: ECE 0.021, on par with (slightly better than) the market's 0.025. So no leakage inflated earlier work, and the model is an honest probabilistic forecaster - just not sharp. Not overconfident.

TAKEAWAY: The model is a sound, calibrated, structural prior, far from market-beating on raw win prediction (as expected for an efficient pool). Its potential value is residual/relative, not standalone: where it disagrees with the market on specific draw/pace/geometry configurations. Next: (a) decompose the gap by config (does the model add anything on extreme draws / unusual pace where the crowd rule-of-thumbs?), (b) blend traffic prior + market via the conditional-logit and test for incremental edge through the harness, (c) more data via the flywheel; Tier-B aerial-video lateral trajectories for the under-identified lateral dynamics.
