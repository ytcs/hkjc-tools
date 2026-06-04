---
id: 0051-jockey-confirmed-at-happy-valley-9pp-replicates
title: Jockey confirmed at Happy Valley (+9pp, replicates Sha Tin)
kind: finding
status: open
tags: ["jockey", "happy-valley", "backtest", "signal", "confirmed"]
refs: {}
related: []
created: 2026-06-04T19:33:17+00:00
updated: 2026-06-04T19:33:17+00:00
---
Extended jockey coupling (effective-v_c measurement, JOCKEY_GAIN) to the HV backtest.
Paired bootstrap, jockey on/off, N=4000, 99 HV races / 4 folds:
  jockey-OFF: LL 2.3419  (21% of market)
  gain 3.0:  +0.0410 nats/race  P(helps)=98.7%
  gain 5.0:  +0.0493 nats/race  P=95.0%  -> 30% of market
  gain 10.0: over-couples (P=64.8%)
Jockey lifts HV MORE than ST (+9pp vs +7pp), interior optimum 3-5 (same as ST 3-6),
over-couples by 10 at both. Confirms jockey is a genuine cross-regime ability signal,
not a Sha-Tin galloping-track artifact. Baked JOCKEY_GAIN=5.0 is well-placed at both venues.
experiments/jockey_backtest_hv.py
