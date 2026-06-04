---
id: 0056-unblocking-the-data-cap-lifted-the-abm-from-38-t
title: Unblocking the data cap lifted the ABM from ~38%% to 43.7%% of market (20x more races)
kind: finding
status: open
tags: ["ABM", "data", "energetics", "market-share", "standing-order"]
refs: {}
related: []
created: 2026-06-04T20:49:39+00:00
updated: 2026-06-04T20:49:39+00:00
---
Built energetics_dataset_full (3248 races, 2911 horses, 5 seasons, 0 nulls; build_energetics_full.py,
sectionals available across the whole span) and wpq_dataset_full (pure results_rich transform, matches
original exactly). Re-ran the ST ABM on the full panel via ENERGETICS_FILE/WPQ_FILE env override:
  Sha Tin 1952 races (was 161): traffic LL 2.2793, market 1.9729, uniform 2.5171
  -> traffic skill +0.2378 / market +0.5442 = 43.7%% of market (was ~38%% on 3 months).
Pure data structure, zero new modeling: 20x more training data sharpens the empirical-Bayes ability
estimates (and the form table, also computed on the full panel). Confirms the cap was the limiter, not
the model. Next: flip the defaults to the full panels so the ABM runs at full power by default.
