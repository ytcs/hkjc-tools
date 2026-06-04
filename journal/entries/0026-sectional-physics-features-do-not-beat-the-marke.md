---
id: 0026-sectional-physics-features-do-not-beat-the-marke
title: Sectional physics features do NOT beat the market (3rd consecutive negative)
kind: result
status: open
tags: ["trajectory", "sectional", "physics", "market-efficiency", "negative-result"]
refs: {"experiment": "experiments/validate_sectional_model.py"}
related: []
created: 2026-06-04T04:13:32+00:00
updated: 2026-06-04T04:13:32+00:00
---
Trajectory-line kill/confirm (269 races, 27 meetings, 2026-03..05). Trailing (prior-run) field-relative sectional energetics — finishing speed, early speed, fade=last/best section. Walk-forward race_win_log_loss: market 2.054 baseline; clogit[mkt] 2.064 (-0.5%); clogit[mkt,finish,fade] 2.108 (-2.6%); clogit[mkt,all-sectional] 2.115 (-3.0%). Sectional pace/fade features HURT, same as rating/form. THIRD consecutive negative (journal 0011 crude, 0023 rating/form, now sectionals): the win market is efficient w.r.t. odds/draw/weight/rating/form/sectional-pace. KEY INSIGHT for next step: sectionals capture PACE & FADE (which the market prices via the chart) but NOT GROUND-LOST (actual meters travelled / wide trips) -- that signal is invisible to section TIMES and only recoverable from the 2D video trajectory. So ground-lost via video is the one plausibly-unpriced signal left. CAVEATS: small sample, trailing features noisy (few prior runs/horse in a 3-month window), crude features (field-relative section times, not a full CP-W' fit with real distances) -- but the consistent 3x direction makes more sectional engineering low-value. Decision per plan: if sectionals yield nothing, move to video for finer granularity (ground-lost).
