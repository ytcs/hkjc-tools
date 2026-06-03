---
id: 0011-first-real-backtest-crude-model-does-not-beat-th
title: First real backtest: crude model does NOT beat the HKJC win market (efficient)
kind: result
status: open
tags: ["backtest", "market-efficiency", "win", "negative-result", "benter"]
refs: {"experiment": "experiments/validate_market_model.py"}
related: []
created: 2026-06-03T22:32:18+00:00
updated: 2026-06-03T22:32:18+00:00
---
Built a real W/P/Q dataset from result pages: 3216 runner-rows, 261 races, 26 meetings (2026-03-01..05-31). Walk-forward (5 folds) race_win_log_loss: uniform 2.506; MARKET baseline 2.070; clogit[log_market] 2.077 (-0.3% vs market); clogit[+draw,wt] 2.090 (-1.0%); clogit[+draw,wt,declwt] 2.099 (-1.4%). Flat-stake ROI proxy (payout at closing odds) negative at every edge threshold (-0.187 @ edge>0, -0.043 @ edge>10% on 60 bets). CONCLUSION: with crude features (draw + weight + the market SP) on a small sample, NO edge over the closing market — the win market is highly efficient and the harness correctly refuses to manufacture a fake edge (this VALIDATES the objective function). clogit[log_market] ~= market confirms no info loss (coef~1; favourite-longshot correction negligible here); extra crude features overfit tiny first folds and hurt. PATH FORWARD: edge needs richer features (official rating, recent form, speed/pace, SpeedPro, the Chinese alt-data + video axes) and more data (full seasons), NOT this feature set. Result page has no rating; must join the racecard/GraphQL or SpeedPro for fundamentals.
