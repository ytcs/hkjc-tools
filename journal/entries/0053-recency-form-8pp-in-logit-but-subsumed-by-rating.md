---
id: 0053-recency-form-8pp-in-logit-but-subsumed-by-rating
title: Recency form: +8pp in logit, but SUBSUMED by rating in the ABM (nonlinear channel)
kind: finding
status: open
tags: ["form", "recency", "rating", "ABM", "featurization"]
refs: {}
related: []
created: 2026-06-04T20:07:51+00:00
updated: 2026-06-04T20:07:51+00:00
---
Exp-weighted (half-life ~2 race) form beats the career mean by +8pp of market in a bare-form
conditional logit (conditional_form.py: career 28%% -> EW 36%%; plateau HL 1.5-2). Distance-
conditional form was a bust (23%%, too sparse). BUT it does not transfer to the mechanistic ABM:
at each form'''s own best FORM_GAIN (both optimal at 2.0, identical std 0.223), EW is +0.0025 nats/
race WORSE than career (form_gain_recency_sweep.py, P(EW better)=26%%). Reconciliation: the recency
the market prices is already carried in the ABM by the official rating, which enters NONLINEARLY via
v_c.fixed(band,rtg) -- a channel the linear logit lacks (a crude linear z_rtg logit even goes -22%%).
Career mean kept in the ABM. NOT defeatism: the signal is real, it just already lives in the rating layer.
