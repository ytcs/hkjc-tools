---
id: 0065-residual-queue-persisted-to-model-state-md-carry
title: Residual queue persisted to MODEL_STATE.md (carry-forward for refactor)
kind: decision
status: open
tags: ["model", "calibration", "residual-queue", "refactor", "carry-forward"]
refs: {}
related: []
created: 2026-06-05T04:56:12+00:00
updated: 2026-06-05T04:56:12+00:00
---
Persisted the open market-residual queue + current calibration constants to /MODEL_STATE.md so it survives the planned journal wipe. Queue at FORM_GAIN=4: (1) draw OVER-weighted t=-17.4 [recommended next, single-constant bend_curvature recalibration]; (2) w_prime UNDER-weighted t=+12.9; (3) bodywt/mass UNDER-weighted t=+9.0 [traffic_jax lacks mass term -> single-source the EOM]. Constants baked: FORM_GAIN=4.0, SPEED_NOISE=0.12, bend_curvature(ST)=0.0037 (suspected stale). Fit: ST ~48%, HV ~43% of market. See MODEL_STATE.md for full methodology + framing.
