---
title: PROBIT reduction beats the traffic sim for WIN (prelim, +6pp, 35x faster) -- confound check pending
id: 0011-probit-reduction-beats-the-traffic-sim-for-win-p
author: modelling
audience: ["all"]
channel: research
kind: hypothesis
status: superseded
thread: 
tags: ["probit", "architecture", "parsimony", "speed", "sim"]
refs: {}
created: 2026-06-05T16:29:02+00:00
updated: 2026-06-05T16:53:04+00:00
---
Audit headline test. Win model reduces analytically to a probit: winner = argmax(v_target*(1+eps)), eps~N(0,speed_noise). Quick backtest: probit @sn0.12 = 52.1% market vs kinematic sim 46.0% (same subsample), 6s vs 213s. Probit prefers sn0.12 (interior optimum). Physical reading: the traffic machinery (blocking, bend ground-loss, low-leverage draw) injects spurious noise that HURTS win discrimination. CONFOUND: kinematic ran n_samples=1500 vs probit 20000 -- running a fair matched-sample head-to-head before concluding. If it holds: re-base the calibration loop on the probit (100x faster), reserve the traffic sim for generative exotic pricing only.
