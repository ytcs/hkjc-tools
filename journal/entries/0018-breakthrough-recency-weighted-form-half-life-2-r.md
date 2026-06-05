---
title: BREAKTHROUGH: recency-weighted form (half-life ~2 races) lifts ST 48% -> 56% of market (+7.8pp)
id: 0018-breakthrough-recency-weighted-form-half-life-2-r
author: modelling
audience: ["all"]
channel: research
kind: result
status: confirmed
thread: 
tags: ["recency", "form", "breakthrough", "ability", "shrinkage", "knob-scan"]
refs: {}
created: 2026-06-05T17:53:17+00:00
updated: 2026-06-05T17:53:17+00:00
---
Knob scan found the biggest lever of the session. Replacing the career expanding-mean form with an EXP-WEIGHTED (recency) form, half-life ~2 races: career 48.1% -> recency 55.9%. Half-life sweep is a clean smooth interior optimum (0.7->49.4, 1.0->53.4, 1.5->55.6, 2.0->55.9, 3.0->55.2, 5.0->53.6, 10->51.4, career->48.1) -- the signature of real signal. LEAK-VERIFIED: recency_form_table uses ewm_mean().shift(1).over(horse_id), strictly prior (first run null); manually confirmed prior-only. This OVERTURNS the old 'recency doesn't transfer to the ABM' finding (that was the stale FORM_GAIN=2 ABM). Mechanism: career-average form is stale over HK's long careers (horses improve/decline, change class); recency tracks CURRENT ability and aligns with how the market weights recent runs -> big %market jump. ACTION: make recency form (hl~2) the production form; re-calibrate FORM_GAIN and re-confirm W_PRIME_GAIN/JOCKEY_GAIN/shrink_mult under it (the form spread changed). Also found this scan: EB shrink_mult on the sectional v_c residual is over-trusted -- 0.5 (48.8%) and even 0.0 (48.5%) beat 1.0 (48.1%); the sectional energetics barely earns its keep for win. Trainer parked (+0.2pp, redundant with jockey+rating).
