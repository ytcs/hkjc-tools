---
id: 0038-exotic-pool-backtest-at-st-no-edge-market-prices
title: Exotic-pool backtest at ST: no edge - market prices quinella/trio well too; trio ROI +58% is a 5-hit small-sample artifact
kind: finding
status: open
tags: ["backtest", "exotics", "quinella", "trio", "sha-tin", "market-efficiency", "dividends"]
refs: {}
related: []
created: 2026-06-04T10:29:51+00:00
updated: 2026-06-04T10:29:51+00:00
---
Exotic-pool backtest (experiments/backtest_exotics_st.py) on 139 OOS ST races, leak-free walk-forward; model exotic probs from the traffic finishing-order distribution, market = Plackett-Luce expansion of de-vigged win probs; settled on actual declared dividends (cached experiments/build_dividends_st.py -> dividends_st.parquet, 160 races).
PREDICTION (mean -log P of realised combo, lower better): QUINELLA traffic 4.170 | market 3.469 | uniform 4.287 -> traffic 14% of market skill. TRIO traffic 5.381 | market 4.545 | uniform 5.544 -> 16%.
=> On exotic PREDICTION the structural model captures ~14-16% of market skill - same ballpark as place (17%), modestly above win (12%), but the market predicts exotics much better. No prediction edge.
VALUE (ROI betting top-1 combo on real dividends): QUINELLA traffic 4/139 hits ROI -71.8% (market 19/139, -16.2%); TRIO traffic 5/139 hits ROI +58.2% (market 8/139, -30.9%). The +58% trio ROI is a SMALL-SAMPLE ARTIFACT (5 hits, one big dividend) - not significant. 139 races is far too thin for exotic ROI (need the flywheel). VERDICT: crowd-weakness-on-exotics hypothesis NOT supported; exotic markets are efficient too. The win->place->exotic robustness gain (12%->~16%) is real but small.
