---
id: 0023-rating-form-do-not-beat-the-win-market-efficient
title: Rating + form do NOT beat the win market (efficient w.r.t. fundamentals)
kind: result
status: open
tags: ["model", "market-efficiency", "rating", "form", "negative-result"]
refs: {"experiment": "experiments/validate_form_model.py"}
related: []
created: 2026-06-04T02:17:49+00:00
updated: 2026-06-04T02:17:49+00:00
---
Richer-features experiment (140 races, 14 meetings, 2026-04-15..05-31, horse-form-derived rating + trailing form). Walk-forward race_win_log_loss: market 2.048 baseline; clogit[market] 2.049 (-0.0%, reproduces market, coef~1, no info loss); clogit[market,rating] 2.064 (-0.8%); clogit[market,rating,form] 2.167 (-5.8%). CONCLUSION: official rating and recent form add NOTHING over the closing win line and overfit on small data -> the market already prices standard fundamentals efficiently. Strengthens journal 0011: a WIN model edge will NOT come from standard fundamentals; it needs information the market lacks (trip comments/corunning, sectionals/trajectory, alt-data). Reinforces the two right lines: carryover STRUCTURAL edge (main) + dense-data/trajectory (information) for any model edge. CAVEATS: 140 races underpowered; generate_historical_data truncates history so trailing-form features are noisy (part of why form hurt most) -- a cleaner full-history form build might do less badly but is unlikely to beat market given the rating result.
