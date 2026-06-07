---
title: S0d AERIAL #82 REVAMP — PLAN + render-inversion approach + the stewards Comments-on-Running ground-truth (logged pre-run)
id: 0072-s0d-aerial-82-revamp-plan-render-inversion-appro
author: modelling
audience: ["all"]
channel: research
kind: note
status: open
thread: 0060-reorientation-win-diagnostic-not-veto-wave2-expand-causal-network
tags: ["s0d", "cv", "vision", "mode-b", "aerial-virtual-view", "render-inversion", "deterministic-graphics", "not-a-camera", "homography", "rail-spline", "temporal-consistency", "ground-contact", "shadow", "distance-markers", "absolute-anchors", "y-lat", "lateral", "control", "faithfulness", "stylized-vs-true", "stewards-comments", "running-commentary", "ground-truth", "4b", "plan", "task-82", "0071", "0069", "wave-2"]
refs: {"render_crops": "/tmp/s0d_scrutiny/render_030s.png,050s,070s,089s", "aerial_video": "data_cache/video/aerial_20260603_01.mp4", "official": "data_cache/race_official/20260603R1_results.html", "prior": "journal/entries/0071-s0d-aerial-78-per-0069-extended-recon-through-th.md"}
created: 2026-06-07T01:18:18+00:00
updated: 2026-06-07T01:18:18+00:00
---
PLAN + GROUND-TRUTH for task #82 (REVAMP aerial render geometry around the 'render = deterministic 3D graphics, not a camera' insight). Logged BEFORE the heavy CV run per the incremental-journaling guardrail.

THE INSIGHT (user-confirmed): the aerial replay RIGHT panel (Aerial Virtual View) is NOT a camera; it is deterministic 3D GRAPHICS — broadcaster places each horse at its TRACKED (long, lat) position on a KNOWN 3D track model and renders through a virtual camera. No parallax/blur/lighting noise. The paywalled feed is recovered by INVERTING THE RENDER. 0071's y_lat NULL (within-runner 3.8m > between 3.3m) happened because the OLD pipeline treated noise-free graphics like a noisy camera: per-frame rail RE-FIT on pixels + BODY-centroid offset injected measurement noise that swamped the trip.

REVAMP deliverables: (1) STABLE rail spline w/ temporal consistency -> ground-plane homography -> (arclen, y_lat in m); (2) ABSOLUTE longitudinal anchors from painted distance markers (200M/100M/etc, rendered crisp at t=89) + odometer + minimap; (3) GROUND-CONTACT (shadow base) points NOT body centroid (body is up-in-air at oblique angle -> camera-angle lateral bias); (4) project ground-contact -> (arclen, y_lat) + legend/gate identity. REUSE: _read_legend_robust + start-gate silk anchor (identity solved, ~10-12/12).

FRAME SCRUTINY confirms feasibility: render crops /tmp/s0d_scrutiny/render_*s.png show small 3D horse figures WITH SHADOWS on green turf; a crisp white inner-rail curve along the bottom; painted distance markers '200M'/'100M' on the ground at t=89 (absolute arclen anchors); odometer + minimap insets; field fans out around the bend (t=30/70).

EXTERNAL GROUND-TRUTH (official stewards Comments-on-Running, found in results HTML = the qualitative lateral truth for the faithfulness gate):
 #12 THE WAY WE WIN rp 1-1-1-2 WIRE-TO-WIRE LEADER (rail/inside).
 #3 FAMILY FORTUNE rp 7-7-6-1 WINNER: near 150M SHIFTED to the OUTSIDE of #12 to secure clear running (went wide to pass) -> EXPECT a late outward y_lat move.
 #6 TELECOM POWER rp 11-11-10-3: CAUGHT WIDE early, settled back for cover -> EXPECT wide early.
 #2 SETANTA rp 9-9-9-4: from a WIDE BARRIER shifted across behind runners early.
 #7 ZETTA FORCE rp 12-12-11-9: from a WIDE BARRIER shifted across behind runners early.
 #8 MANYTHANKS FOREVER rp 6-2-2-6: improved to the INSIDE of #10, raced tight.
 #9 rp 4-4-3-5 bumped 150M; #4 rp 5-6-5-8 forward move from 700M.
Finish: 3,12,6,2,9,8,11,4,7,10,5,1. Sections official [27.65,23.30,23.75,24.15].

FAITHFULNESS GATE (the deliverable that matters, 4b-honest): is the render lateral TRUE (GPS rail-offset) or STYLIZED (artist-smoothed TV lanes)? Tests: (a) dynamics — smooth autocorrelated drift vs fixed-lane vs noise; within/between ratio should drop <<1 if faithful+clean; (b) rp-consistency — improvers swung wide / ground-savers on rail; (c) the stewards commentary above. VERDICT honestly: faithful -> render is THE lateral source; stylized -> 4b NULL it. Do NOT force positive.

OWN: src/hkjc/vision/mode_b/* + new studies/cv_*aerial* files. Do NOT touch cv_bakeoff_s0d.py / cv_s0d_drone.py. Next: build studies/cv_s0d_aerial_render.py (render-inversion geometry) + run.
