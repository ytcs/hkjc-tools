---
title: Lean re-architecture (hkjc.data + hkjc.recon); Bayesian multi-feed fusion; finish-line camera; odometer-tick metric
id: 0134-lean-rearchitecture-bayesian-fusion-finish-camera
author: recon
kind: finding
tags: [cv, recon, fusion, finish-camera, odometer, architecture]
created: 2026-11-13
---
## Re-architecture (owner request)
Library reduced to two packages: `hkjc.data` (all HKJC interfaces + replay videos + silks + track
geometry; `race_info()` bundles one race) and `hkjc.recon` (`RaceRun`, cached stages, CLI
`python -m hkjc.recon DATE VENUE NO`). Removed (in git history): core, model, betting, backtesting,
agents, reconstruction, vision, studies, old tests. New tests (12) + lint clean.

## Fusion: feeds are proposals, the path is a posterior
`recon/fusion.py`: per-horse MAP on a 0.1 s grid (s, lat), acceleration priors (1.2 / 0.25 m/s²),
lateral velocity prior, box bounds (outside racing rail, inside course width), every observation at
its own timestamp/noise. Exact sparse solve + active set (iterative LS — lsq_linear/lsmr — does NOT
converge on this stiff system: 1.5 m error; REJECTED). An along-track velocity prior decelerated paths
across end-of-race gaps (REJECTED). Association: chi² of anonymous tracklets vs posteriors; orphan
chains (stitched anonymous tracklets) assigned to path-less horses by other-feed anchors.

## Finish-line camera (RF/drone shared finish replay)
Fixed overhead camera on the line, slow motion with VARIABLE playback (interpolated near the winner):
unique-frame index is NOT linear in time (k varied 0.0045..0.011 s/frame) -> temporal gaps REJECTED.
Saddlecloth numbers (yellow on dark) OCR'd per unique frame; gaps are SPATIAL at an instant
(gap_h = gap_r + x_h - x_r over shared frames), playback-independent. Orders match official on R08
(10/11), HVB (9/12), R02A (12/12, one 0.0x s swap). px/s constant ~1650-1750 on HV (diagnostic only).
In-pipeline: C fitted jointly with the clock anchor against render-fused arrivals + a physical prior
(fused speed x px/m from isolated horse boxes, nose-to-tail 2.6 m).

## Distance metric: odometer ticks, markers only when consistent
The painted-marker grid mis-assigned markers on R02A/HVB (ppm 22.2/16.1 vs ~18; finish 40 m off).
Distance now from odometer TICKS (round-up display, convention c=0 measured on R08 -0.55 m, R03
+0.42 m): robust global line + smooth spline; the painted FINISH pins d=0 only if the marker grid is
consistent with the ticks (|c_m|<3 m, ppm within 3%) — cross-feed check.

## Other fixes
Full-panel detection (legend icon band rejected; turf strip below the legend kept) +11% sectional
coverage on R08; two-sided ridge test (the mean-of-sides test let the turf/apron STEP edge capture
the R02A rail); HUD-band static-overlay gate (a disclaimer glyph got numbered as #1 on R02A).

## State (fixed gates: leader 0.25 s, sectional 0.25 s, finish rho 0.9, winner, behind 0.2 s)
R08 HV C 1800: ALL PASS (leader 0.172, sectional 0.178 @78%, rho 0.93 n=11, winner, behind 0.027 s).
HVB HV B 1650: ALL PASS. R02A HV A 1650 / R03 HV C 1200: render identity/fragmentation failures.
ST races + HV 2200 in progress. Open: drone registration (TVCalib-style plan), non-overlap QP.
