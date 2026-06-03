---
id: 0002-clv-is-the-north-star-leading-indicator
title: CLV is the north-star leading indicator
kind: decision
status: open
tags: ["clv", "evaluation", "odds-ticks"]
refs: {"modules": ["hkjc.data.odds_ticks", "hkjc.eval.clv"]}
related: []
created: 2026-06-03T21:25:01+00:00
updated: 2026-06-03T21:25:01+00:00
---
In an efficient pari-mutuel market, beating the CLOSING line is the strongest available signal of true edge, and it exists for every race whether or not we bet — partially rescuing the ~800-races/year sparse-feedback problem. Built the odds tick store (hkjc.data.odds_ticks) + hkjc.eval.clv to make it measurable. Negative open->close drift = informed money (steam).
