---
title: TASK #95 PARTIAL — held-out R08 (1800m route) per-feed + identity built with the HARDENED pipeline, NO tuning: RF legend backbone +0.96/+0.98/+0.93/+0.98 (the corpus-best body order), identity consensus (FIXED weights) patrol gap +12.74m CORRECT, but AERIAL render FORESHORTENS on the route (within/between 2.55 vs sprint 1.59, autocorr +0.86 globally yet NEGATIVE -0.16 per bend, rail-vs-wide -3.2m WRONG) — the route foreshortening stress confirmed
id: 0091-task-95-partial-held-out-r08-1800m-route-per-fee
author: modelling
audience: ["all"]
channel: research
kind: result
status: open
thread: 0060-reorientation-win-diagnostic-not-veto-wave2-expand-causal-network
tags: ["cv", "vision", "fusion", "four-feed", "holy-grail", "generalize", "held-out", "scale-slow", "task-95", "r08", "hv", "happy-valley", "1800m", "route", "aerial", "drone", "patrol", "rf", "identity", "consensus", "draw-anchor", "foreshortening", "metric-gap", "render-inversion", "legend-backbone", "per-pole", "spearman", "rail-vs-wide", "stewards", "no-tuning", "0088", "0089", "0090", "wave-2"]
refs: {"common": "studies/cv_r08_common.py", "aerial": "studies/cv_r08_aerial.py", "rf": "studies/cv_r08_rf.py", "patrol": "studies/cv_r08_patrol.py", "drone": "studies/cv_r08_drone.py", "identity": "studies/cv_r08_identity.py", "anchor_cache": "data_cache/cv_identity_R08.npz", "panels": "data_cache/cv_r08_{aerial_render,rf_longitudinal,drone_numbered,patrol_numbered}.parquet", "official": "data_cache/race_official/20260603R8_results.json (VALIDATION ONLY)"}
created: 2026-06-07T09:43:54+00:00
updated: 2026-06-07T09:43:54+00:00
---
TASK #95 PARTIAL — per-feed extractions + the unified identity on the HELD-OUT R08 (HV 2026-06-03,
Class 2 1800M turf "C", winner #6 SILVERY BREEZE, RP "6 6 7 7 1" came from behind, won by NK; field
of 11; 5 sectionals 13.78/23.56/24.56/23.62/23.10=108.62). The pipeline was hardened on R01(1650)+
R03(1200) with R08 DELIBERATELY held out. REUSE: every R08 driver imports the R1/R03 geometry +
the #94 identity consensus UNCHANGED; only race-PARAMETERS re-pointed (DIST 1800, sections 4->5,
field 12->11, video paths, clock anchors, auto-detect scan range). NO identity weight / feed
threshold tuned. src/hkjc untouched; R01/R03 parquets unmutated; ruff clean; PYTHONHASHSEED=0;
official VALIDATION-only (draw = pre-race initial condition).

== PER-FEED GENERALIZATION TO THE ROUTE ==
RF / race-film — CLEAN, the star. Legend NAME->number running-ORDER backbone validated
rho(legend,RP) = +0.96/+0.98/+0.93/+0.98 at the route c1/c2/c3/c4 calls (late +0.36, the bunched
come-from-behind finish) — the STRONGEST body-order of the whole corpus (R1 +0.98/+1.00/+1.00, R03
+1.00/+1.00). Winner #6's mid-pack-then-surge captured. Margin ORDER rho +1.00, winner #6 clear;
MAE 2.12len (magnitude over-reads, same finish-cam-blur class as R03 1.70). RF_OFFSET auto-resolved
1.08s. 5-section clock + name LUT rebuilt from the R08 official parse, all race-agnostic.

AERIAL render — the FORESHORTENING STRESS CASE bites, exactly as predicted. Render-inversion ported
unchanged (split-screen auto-detect FOUND 321 aerial-active frames, rail valid 321/321; odometer
35 frames; legend bound 7/11). BUT the 4b faithfulness DEGRADES: within/between RATIO 2.55 (R1
1.6-1.8, R03 1.59 — the WORST of the corpus), lag-1 autocorr +0.86 GLOBALLY but NEGATIVE (-0.16
median) when split per arclen BAND (ratio 2.61 on the 1000-1200m bend), and the rail-vs-wide GAP
flipped to -3.2m WRONG SIGN (only 4 horses bound). The route's continuous bends break the oblique
render-inversion faithfulness that held on the sprint/mile straights — the metric-gap thesis,
quantified.

PATROL — DEGRADED GEOMETRY (route-specific). The route patrol REPLAY pans/follows the field down
the home straight from an oblique elevated angle rather than holding the FIXED head-on shot R01/R03
had. The rail-slope-stability auto-detect still found its best window (t=95-111s, rail CV 7.0% —
between R1's 3.0% and R03's 9.7%), but the head-on geometry is weaker. Identity consensus (the FIXED
#94 weights) assigned 7 tracks; the patrol-CONSENSUS rail-vs-wide gap = +12.74m CORRECT SIGN (#6
rail 4.9m vs wide #1/5/8/11 17.6m) — the consensus + draw-order vote held the sign on the route.

DRONE — supplementary as designed. Time-anchored on the shared clock (gate-jump vid 3.0 -> finish
vid 120.5, auto-cut scan vid 127); 5/11 RF-finish-cam silk anchors, 4 numbered, 14.5k coverage rows.
Metric-null ordering role, as on R01/R03.

== IDENTITY (the hardened #94 consensus, FIXED weights, NO tuning) ==
RF anchor silk LUT 7/11 [1,2,5,6,8,9,11] (strong votes #11:32/#2:24/#6:20) + patrol-augment own LUT
6/11; rank backbone 12-tp validated +0.96/+0.98/+0.93/+0.98. consensus_assign (own 1.0 / RF 0.45 /
reliability-gated order 0.6 / draw 0.7) assigned 7 patrol tracks -> patrol gap +12.74m CORRECT. The
OCR bottleneck stays cached-gone (whole anchor < a few min, cached npz). draw->y_lat(0): stall_width
1.52m, span 0.8-16.0m, cross-checked vs patrol span 4.0-21.8m (consistent scale, reported not tuned).

NEXT (this entry's continuation): the 4-feed FUSION + the full foreshortening recon + the PASS/LIMIT
verdict (full journal to follow).
