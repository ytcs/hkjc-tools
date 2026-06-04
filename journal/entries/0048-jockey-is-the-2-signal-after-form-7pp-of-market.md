---
id: 0048-jockey-is-the-2-signal-after-form-7pp-of-market
title: Jockey is the #2 signal after form (+7pp of market skill at scale)
kind: finding
status: open
tags: ["jockey", "signals", "form", "walk-forward"]
refs: {}
related: []
created: 2026-06-04T18:39:59+00:00
updated: 2026-06-04T18:39:59+00:00
---
At-scale walk-forward conditional-logit on results_rich (3197 races / 5 seasons / 86 jockeys, ST+HV).
Form baseline = 27% of market skill. Adding leak-free trailing JOCKEY strike rate -> 34% (+7pp, LL 2.355->2.321).
Everything else adds ~0 marginal beyond form: trainer (~0), carried weight (handicap-equalized, ~0), body weight (-1pp), body-weight change (~0), days-since (~0), running-style-as-covariate (+0.3pp, confirms emergent not additive). form+ALL collapses (logit instability, not signal).
Verdict: couple JOCKEY mechanistically (as trip/policy quality, not a flat v_c bump); skip the other condition signals. Jockey confounds with horse quality + the market already prices it, so it closes OUR gap (model lacks jockey) more than it beats the market.
