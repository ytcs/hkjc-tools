---
id: 0007-deflated-sharpe-needs-an-honest-trial-count
title: Deflated Sharpe needs an HONEST trial count
kind: gotcha
status: open
tags: ["selection", "overfitting", "fdr", "trial-count"]
refs: {"module": "hkjc.eval.selection"}
related: []
created: 2026-06-03T21:32:46+00:00
updated: 2026-06-03T21:32:46+00:00
---
DSR / expected-max-Sharpe haircuts a result by how many strategies were tried (best-of-N is guaranteed to look good on ~800 races/yr). The math is only as honest as n_trials. The orchestrator MUST track the true number of hypotheses tested (candidate the ledger as the source of truth) and feed it in — otherwise selection silently under-corrects and we bank noise. Random splits are banned for the same reason: horses/jockeys/trainers/going persist across meetings.
