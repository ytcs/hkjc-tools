---
title: Refactor acceptance: ST 47.7% of market reproduced on the new library
id: 0003-refactor-acceptance-st-47-7-of-market-reproduced
author: modelling
audience: ["all"]
channel: research
kind: result
status: confirmed
thread: 
tags: ["acceptance", "refactor", "goodness-of-fit"]
refs: {}
created: 2026-06-05T06:46:44+00:00
updated: 2026-06-05T06:46:44+00:00
---
studies/backtest_st.py (N=3000, 5-fold walk-forward) through hkjc.model.fit + hkjc.backtesting.harness: traffic LL 2.2576 vs market 1.9729 vs uniform 2.5171 -> 47.7% of market, ECE 0.0074 (market 0.0069). Matches pre-refactor 48.2% (N=6000) within sample noise. The 7-package refactor preserved the model end-to-end.
