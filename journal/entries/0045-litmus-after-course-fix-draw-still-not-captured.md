---
id: 0045-litmus-after-course-fix-draw-still-not-captured
title: Litmus after course fix: draw STILL not captured (dLL 1.04->1.13m). Geometry was never the issue - draw residual is pace-tactics + noise. Predictive-edge door is closed across all avenues.
kind: finding
status: open
tags: ["litmus", "draw", "course", "market-efficiency", "synthesis", "policy", "conclusion"]
refs: {}
related: []
created: 2026-06-04T11:20:16+00:00
updated: 2026-06-04T11:20:16+00:00
---
Re-ran the backtest with REAL per-race course (rail position) wired into load_st, regenerated the panel, and re-ran the factor-triage litmus. RESULT: draw dLL/model 1.04m -> 1.13m (unchanged); model draw curve still flat (trf_win 0.097/0.095/0.102/0.104 across draws 1-4). Fixing the geometry did NOT capture draw - confirmed: width barely moves the per-venue bend_curvature, and the 5-season validation showed draw bias is modest/uniform across rail positions (no per-course geometric signal to capture). The draw residual is the pace/positioning TACTICAL component (step 2 territory) plus sampling noise (the 2026 draw-2/3 +0.05 bumps are noise; the well-powered market residual is ~0). market+traffic still HURTS (-0.028); market+v_c still +0.0023; nothing changed. Sanity intact (v_c/rating absorbed ->0).
SYNTHESIS / honest state of the predictive-edge question: across the whole drive we have now falsified - with increasing rigor - every predictive avenue: model statistical signals (odds/draw/weight/rating/form/sectional/trouble/energetics), win vs market, place vs market, exotics vs market, mass coupling, recency/form-state, jockey, freshness, and draw/rail-position (the last on 1911 races / 5 seasons). The HK Sha Tin win/place/exotic markets are EFFICIENT w.r.t. every extractable signal including rail-specific draw bias. The structural model is a sound, well-calibrated prior (~12-17% of market skill, ECE on par with market) but adds nothing the market lacks. Over-market residuals are ~0 everywhere.
IMPLICATION: building step 2 (pace policy) and step 3 (state coupling) will make the model MORE REALISTIC and would finally capture draw mechanistically (draw acts through pace/positioning), which is scientifically valuable - but the over-market residual is ~0, so it will NOT produce predictive edge. The genuine breakthrough is the rigorous, well-powered CONCLUSION that the win market is efficient -> stop hunting predictive edge; the only +EV play is STRUCTURAL (carryover/jackpot dilution math, the original main line). Decision point for the user: keep building the dynamical model for UNDERSTANDING (no edge) vs pivot to the structural/carryover edge.
