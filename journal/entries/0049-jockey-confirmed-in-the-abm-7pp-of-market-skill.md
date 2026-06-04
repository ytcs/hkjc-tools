---
id: 0049-jockey-confirmed-in-the-abm-7pp-of-market-skill
title: Jockey CONFIRMED in the ABM: +7pp of market skill, baked (29%->38%)
kind: result
status: open
tags: ["jockey", "backtest", "form", "breakthrough"]
refs: {}
related: []
created: 2026-06-04T18:56:14+00:00
updated: 2026-06-04T18:56:14+00:00
---
Coupled trailing jockey strike rate into the ST traffic backtest as an effective-v_c measurement (parallel to form), keyed (date, HorseID->horse_id); leak-free, all 3307 energetics rows match a jockey. Paired bootstrap vs jockey-off: gain 3 -> +0.0296 nats/race (95% CI [+0.005,+0.055], P=99.2%); gain 6 -> +0.0397 (best estimate); interior optimum ~3-6; gain 15 over-couples and HURTS (-0.044). Lifts model from ~29% to ~38% of market skill; gap to market +0.310->+0.271. Baked JOCKEY_GAIN=5. This replicates the regression scan in the mechanistic model — jockey is the first added signal since form to genuinely help (pace failed, freshness marginal).
