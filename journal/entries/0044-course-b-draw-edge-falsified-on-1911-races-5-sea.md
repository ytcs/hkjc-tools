---
id: 0044-course-b-draw-edge-falsified-on-1911-races-5-sea
title: Course-B draw edge FALSIFIED on 1911 races / 5 seasons: it was a 22-race 2026 fluke; market prices draw correctly across ALL rail positions
kind: finding
status: closed
tags: ["draw", "rail-position", "course", "market-efficiency", "falsified", "validation", "overfitting"]
refs: {}
related: []
created: 2026-06-04T11:15:56+00:00
updated: 2026-06-04T11:15:56+00:00
---
Scraped 5 seasons of Sha Tin results (experiments/build_st_history.py -> st_history.parquet, 23978 runner-rows, 1911 races, 2022-2026) to validate the 2026-panel hint that the market under-prices inner draws at the B rail position (+11pp inner-outer market residual on ~22 races). VALIDATION (experiments/validate_course_draw.py):
- Empirical inner-draw edge by rail position is SMALL and roughly UNIFORM: A +0.010, A+3 +0.031, B +0.027, B+2 +0.025, C +0.029, C+3 +0.032. B is NOT special; the +0.154 from the 2026 panel was noise.
- Market residual (won - p_market) inner-outer ~0 on every course (-0.011..+0.007) => the market prices the draw/rail bias CORRECTLY everywhere.
- ROI of backing inner draws (1-4) is strongly NEGATIVE on every rail position (-17% to -40%, race-clustered 90% CIs below 0). No +EV anywhere.
- SMOKING GUN (per-season): course-B inner-draw ROI 2022 -70%, 2023 -24%, 2024 -23%, 2025 -48%, 2026 +94%. The entire 'edge' was ONE season (2026) - the data we found it in. Overfitting to a 22-race fluke.
VERDICT: the HK ST win market is efficient w.r.t. draw across all rail positions and seasons; no inner-draw betting edge. The breakthrough hypothesis is dead. Also informs modelling: rail position does NOT materially change the draw-bias magnitude on the big sample (all ~+0.01-0.03), so no large per-course bias term is warranted - the per-venue bend_curvature is roughly right; the course correctness fix (real width per race) is still correct but modest. Cumulative: mass, recency, form-edge, jockey-edge, freshness, draw/course-edge, exotics-edge all falsified -> the win/place/exotic markets are efficient. Edge, if any, is structural (carryover/jackpot), not predictive.
