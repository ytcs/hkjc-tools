---
title: CONFIRMED: probit matches the traffic sim for WIN (48.1% full-scale), 35x faster
id: 0012-confirmed-probit-matches-the-traffic-sim-for-win
author: modelling
audience: ["all"]
channel: research
kind: result
status: confirmed
thread: 
tags: ["probit", "architecture", "parsimony", "speed", "win-model", "calibration"]
refs: {}
created: 2026-06-05T16:53:04+00:00
updated: 2026-06-05T16:53:04+00:00
---
Fair matched-sample head-to-head (both n_samples=6000): kinematic vs probit tie within subsample noise (seed7 48.7/44.2, seed11 45.6/50.2; means ~47.2 each) -- the earlier +6pp was the sample-count confound. FULL-SCALE noise-free probit (1932 races, 28s/config): current gains 48.1% = the sim's 47.7-49%. So for the WIN target the traffic Monte-Carlo is over-engineering: the model IS a probit on log effective-velocity, winner=argmax(v_target*(1+eps)), eps~N(0,speed_noise). The traffic machinery (blocking/bend/draw) is confirmed-inert weight for win -- consistent with draw/mass/speed-figure all testing low-leverage. The baked gains transfer cleanly (FG4, WPG2, JG5, sn0.12 all interior optima under the probit). ACTION: re-base the calibration loop on the probit (sub-second -> full-scale in 28s, no geometry recompile); reserve the traffic sim for generative exotic pricing (place/trifecta, where ordering+traffic matter). Unlocks fast testing of the remaining audit items; next high-leverage lever is the NOISE-MODEL STRUCTURE (single global speed_noise controls all calibration; test distance/field/going dependence + calibration-by-condition).
