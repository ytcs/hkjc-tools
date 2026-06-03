---
id: 0012-quant-team-verdict-build-the-jackpot-carryover-h
title: Quant team verdict: build the Jackpot Carryover Harvester; model-edge exotics are look-ahead traps
kind: decision
status: open
tags: ["strategy", "jackpot-carryover", "exotics", "look-ahead", "workflow"]
refs: {"workflow": "quant-arb-strategy", "modules": ["hkjc.data.live", "hkjc.data.historical", "hkjc.bets"]}
related: []
created: 2026-06-03T22:54:04+00:00
updated: 2026-06-03T22:54:04+00:00
---
A 6-analyst + adversarial-verify + synthesis workflow ranked HKJC arb/exotic strategies. WINNER: Jackpot Carryover Harvester (Triple Trio/Double Trio/Six Up) as a DETECTOR + research instrument (not a deployment). It is the ONLY proposal with a STRUCTURAL edge (un-won jackpot carries ~85% net forward, so new money on a carryover meeting faces NEGATIVE effective takeout on the aggregate pool: eff_takeout = (1-net) - C/G; flag when C/G > 1-net ~ 0.15) and the ONLY one backtestable on data we can assemble now (live jpEsts/estimatedDividend/poolInvs already in live.py; realized settlement already in get_race_dividends which maps DT/TT/6UP). KEY WARNING the team surfaced: every model-edge exotic play (QIN/QPL coherence, place arb, steam) is NOT backtestable on our data — result pages publish ONLY the winning combo's dividend, never the pre-close crowd matrix of losing combos you must price to detect an overlay; and selecting bets on the realized settlement dividend (as bets.price_dividends edge does) is LOOK-AHEAD the walk-forward embargo forbids. The tau place-bias play (proposal 6) is empirically falsified (optimum at tau=1.0). Carryover weaknesses to respect: endogenous dilution (settle at final crowd; publicized carryover invites other harvesters), partial Flexi cover still leans on a barely-calibrated win-odds Harville top_set(3), and lottery variance means it may be +EV yet statistically dark (few events/season). Build plan: jackpot ledger from dividend remarks -> reconstruct C,G -> detector (EV two ways: naive vs estimatedDividend + stressed late pool) -> Flexi cover via simulator+pareto -> score through walkforward/FDR/lockbox -> forward capture. Model-edge exotic lines need a forward live odds-tick capture season first.
