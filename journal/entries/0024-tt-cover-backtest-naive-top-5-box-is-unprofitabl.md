---
id: 0024-tt-cover-backtest-naive-top-5-box-is-unprofitabl
title: TT cover backtest: naive top-5 box is unprofitable & hit-rate-bound (and dark)
kind: result
status: open
tags: ["jackpot-carryover", "backtest", "cover", "hit-rate", "statistically-dark"]
refs: {"experiment": "experiments/backtest_carryover_cover.py"}
related: []
created: 2026-06-04T02:38:10+00:00
updated: 2026-06-04T02:38:10+00:00
---
experiments/backtest_carryover_cover.py over 2026-03..05: 18 won Triple Trios (14 settleable). A fixed top-5 box cover (1000 combos/event, win-odds leg models, settled at REAL dividends, dilution ignored=optimistic): hit rate 7% (1/14), total PnL -1,064 on 14,000 staked, ROI -7.6%. The one hit was a LOW dividend (12,936); the big carryovers (e.g. 810,610 on 03/11) were MISSED. READ: (1) the COVER hit-rate is the binding constraint -- top-5 rarely contains the exact trio in all 3 legs; (2) this bet ALL TT events, not just +EV carryover windows (we can't identify those historically without C), so it's not the real strategy; (3) 14 events is statistically DARK (one hit dominates) -- exactly the quant-team warning. IMPLICATION: the carryover structural edge is real in aggregate but capturing it needs (a) filtering to high-C/G windows [needs live C from the flywheel], (b) cover-size optimization + a better leg model, (c) far more events. Historical backtest can't validate the strategy; FORWARD live capture (Modal) is the critical path.
