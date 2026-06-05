---
title: Calibration-by-condition LOCATES the next levers: distance-noise gradient + AWT broken
id: 0013-calibration-by-condition-locates-the-next-levers
author: modelling
audience: ["all"]
channel: research
kind: finding
status: open
thread: 
tags: ["calibration-by-condition", "noise-model", "distance", "awt", "diagnostic"]
refs: {}
created: 2026-06-05T16:56:07+00:00
updated: 2026-06-05T16:56:07+00:00
---
Built the instrument the residual regression lacks (race-level effects). Full-scale probit OOS, confidence ratio = mean(model top prob)/(model's-pick win rate); >1 over-confident, <1 under. BY DISTANCE: 1000m conf 1.36 (over), 1200 0.98, 1400 0.90, 1600 0.89 (best, 57% market), 1800 0.84, 2000 0.71 (under, 31% market). Clean monotone gradient => speed_noise should DECREASE with distance (sprints ~57s are luck/break-dominated -> more noise; routes ~2min are ability-dominated -> less noise). First-principles, the #1 noise-structure lever. BY FIELD SIZE: 12+ conf 1.03 (ok), <=8 conf 0.81 (under, smaller n). BY GOING: GOOD/GTF ~48% ok; soft/wet too few to bucket. SEPARATE SEVERE MISS: 1650m (AWT) = 13% market, conf 1.92 -- the turf-calibrated ability model does not transfer to the all-weather surface (119 races). Two targets: distance-dependent noise (affects all races) and AWT-specific calibration.
