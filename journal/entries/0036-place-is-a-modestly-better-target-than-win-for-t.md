---
id: 0036-place-is-a-modestly-better-target-than-win-for-t
title: Place is a modestly better target than win for the structural model (17% vs 12% of market skill); still far behind
kind: finding
status: open
tags: ["backtest", "place", "exotics", "traffic-model", "sha-tin", "robustness"]
refs: {}
related: []
created: 2026-06-04T10:11:58+00:00
updated: 2026-06-04T10:11:58+00:00
---
Win-vs-place walk-forward backtest at Sha Tin (experiments/backtest_place_st.py), same leak-free setup; market place benchmark = Plackett-Luce expansion of de-vigged win probs. 139 races.
WIN race-win log-loss: traffic 2.4796 | market 2.0940 | uniform 2.5331 -> traffic recovers 12% of market skill over uniform; gap +0.386.
PLACE (top-k binary log-loss): traffic 0.5250 | market 0.4461 | uniform 0.5408 -> traffic recovers 17% of market skill; gap +0.079.
Gap as fraction of uniform->market skill: WIN 0.88, PLACE 0.83.
=> Robustness thesis (win is a split-second knife-edge; place averages it out, leaning on whole-field ordering the structural model produces) CONFIRMED but MODEST: ~1.4x more market skill captured on place than win (17% vs 12%). Still ~83% short of market on place. By trip the place gap is flat (sprint .075, mile .083, route .076) - the middle-distance WIN-vs-uniform advantage washes out vs the strong market on place.
CAVEAT/next: benchmark was the WIN-market-implied place (very efficient), NOT actual PLACE/exotic pool prices. Real crowd-weakness test on exotics needs actual QPL/QIN/TRI dividends (get_race_dividends, unused here).
