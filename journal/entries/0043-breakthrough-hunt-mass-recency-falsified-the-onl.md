---
id: 0043-breakthrough-hunt-mass-recency-falsified-the-onl
title: Breakthrough hunt: mass + recency falsified; the only over-market signal is a course-B inner-draw mispricing (~+11pp), now validating on multi-season data
kind: finding
status: open
tags: ["breakthrough", "draw", "rail-position", "course-b", "market-mispricing", "mass", "form", "edge-hunt"]
refs: {}
related: []
created: 2026-06-04T11:00:47+00:00
updated: 2026-06-04T11:00:47+00:00
---
Data-guided drive through model stages. Cheap tests FALSIFIED two hypotheses (good, fast):
1. MASS: within-horse d(log v_c)/d(log mass) = -0.11, not the -1 naive F=ma predicts. Carrying weight barely slows a horse at the ~5% body-mass variation seen; mass*v_c is NOT more stable than v_c (CV 0.0259 vs 0.0252). => mass is a weak lever; the EOM's K_LOCO*m*g term over-states it ~9x; do NOT prioritise weight; recalibrate the EOM mass term to ~-0.11.
2. RECENCY: recency-weighting v_c does not beat the flat prior mean (won~flat+recency: flat +0.087, recency -0.045). And corr(v_c, won)=0.06 - our sectional v_c is a WEAK ability measure; prior finish positions beat it. The recent-form residual is non-energetic and, crucially, ~0 over the MARKET (priced) - closing it makes us realistic, not profitable.
REFRAME: an EDGE can only live in OVER-MARKET residuals. Triage had only two: freshness and draw. Freshness checks out weak/noisy. DRAW is the lead:
MARKET residual (won - p_market) by draw x rail-position (course), 2026 panel: market prices draw correctly on A/C/C+3 but on COURSE B massively UNDER-prices inner draws: inner(1-4) market resid +0.082, outer(9+) -0.030, inner-outer +0.111 (vs A +0.038, C -0.004, C+3 -0.002). Mechanistically exactly the draw diagnostic's prediction: the crowd uses a generic draw heuristic and misses how strong the inner-rail bias is at the B rail position. Implied ROI of backing inner draws on B ~ +40-50% IF real.
CAVEAT: only ~22 B-course races - far too thin. Fetching multi-season ST history (experiments/build_st_history.py, 2022-09..2026-06) to test draw x rail x market with real power + per-season robustness + race-clustered ROI CI. If it holds out-of-sample across seasons, that is the breakthrough (a structural, rail-position-specific draw edge the market misprices); if it collapses, it was noise.
