---
title: S0d AERIAL #82 PARTIAL — render-inversion geometry WORKS+STABLE; y_lat is SMOOTH (ac1 +0.90) but does NOT recover the rail truth -> faithful-mis-spec vs stylized to disambiguate next
id: 0074-s0d-aerial-82-partial-render-inversion-geometry
author: modelling
audience: ["all"]
channel: research
kind: note
status: open
thread: 0060-reorientation-win-diagnostic-not-veto-wave2-expand-causal-network
tags: ["s0d", "cv", "vision", "mode-b", "aerial-virtual-view", "render-inversion", "rail-spline", "stable-rail", "ground-contact", "shadow", "autocorrelation", "smooth", "within-between", "perspective", "depth-confound", "homography", "faithfulness", "stylized-vs-true", "partial", "4b", "task-82", "0071", "wave-2"]
refs: {"study": "studies/cv_s0d_aerial_render.py", "panel": "data_cache/cv_s0d_aerial_render.parquet", "fig_geom": "data_cache/figs/cv_s0d_render_geom.png", "fig_trip": "data_cache/figs/cv_s0d_render_trip.png", "prior": "journal/entries/0072-s0d-aerial-82-revamp-plan-render-inversion-appro.md"}
created: 2026-06-07T01:35:51+00:00
updated: 2026-06-07T01:35:51+00:00
---
PARTIAL (coarse stride=15 validation run; logged before the full pass + the faithful-vs-stylized disambiguation). The render-inversion GEOMETRY now WORKS and is STABLE; but the first y_lat read is SMOOTH-yet-does-NOT-recover-the-rail-truth, which reframes the faithfulness question.

GEOMETRY (the 0071 root-cause fixes, all landed):
 * STABLE RAIL SPLINE: cubic fit on the crisp white inner rail, restricted to a robust track band (text strip + legend EXCLUDED via a non-green-saturated legend-top detector — find_layout is unreliable on the green render, 0071). Valid in 151/151 frames; frame-to-frame rail-midpoint shift median 5.9px. QC overlay (cv_s0d_render_geom.png) confirms the cyan spline sits ON the inner rail on BOTH the straight (t=89, rail horizontal at top) and the bend (t=30/70, rail curved at bottom). The geometry plumbing is SOUND — not the noisy per-frame refit of 0071.
 * GROUND-CONTACT (shadow base): 827/831 dets got a shadow ground-contact (dark turf-hued blob beneath the figure) instead of the body centroid. Effect on the 0071 within/between ratio: 1.01 (body) -> 0.97 (ground-contact); within-runner dispersion 3.00m -> 2.82m. REAL but SMALL — ground-contact helps, but does not by itself clean the trip.
 * ABS ARCLEN: odometer in 66 frames (the absolute longitudinal anchor). PAINTED DISTANCE MARKERS ARE NOT OCR-ABLE: the on-turf 200M/100M numerals render as a stylized outline RapidOCR reads as CJK chars (父父) — so the odometer, not the painted markers, carries absolute arclen. (Honest: markers tried, failed; odometer suffices, sections already 0.125s in 0071.)
 * IDENTITY REUSED: legend + assign_by_legend bound 11/12 ([1,2,3,4,5,6,8,9,10,11,12]; missing #7, the tail 12-12-11-9).

THE RESHAPED FAITHFULNESS PICTURE (vs 0071):
 * within/between RATIO 0.97 (~1, NOT <<1) — the revamp did NOT collapse the within-runner dispersion as the insight predicted IF faithful+clean.
 * BUT lag-1 AUTOCORRELATION = +0.90 (median; 100% of runners >0.3). The y_lat is SMOOTH, NOT the high-frequency noise 0071 diagnosed. So the dispersion is REAL smooth MOVEMENT, not jitter. This is a DIFFERENT regime than 0071.
 * RP-consistency corr(median y_lat, mean rp) = +0.19 (weak). Commentary cross-check FAILS the sharp tests: #12 WIRE-TO-WIRE RAIL LEADER reads y_lat 7.3m (NOT tightest-to-rail); #8 raced-INSIDE reads HIGHEST 13.2m (opposite). Weakly consistent: #3 winner early 4.1 -> late 6.4m (a small late outward move, matches 'shifted outside near 150M'); #2 wide-barrier reads wide 11.5m.

SO THE OPEN QUESTION IS NOW SHARP: the geometry is correct and the signal is SMOOTH (deterministic-graphics-like), yet a flat rail-perpendicular offset does NOT recover the known rail truth (the leader is not on the rail). TWO hypotheses to disambiguate next: (A) FAITHFUL-but-MIS-SPECIFIED — the OBLIQUE projection means a flat rail-offset carries a longitudinal/DEPTH perspective confound (the leader #12 is always furthest-along => deepest => its perpendicular-px offset is foreshortened differently); the task asked for a full HOMOGRAPHY precisely to remove this, and my single global m/px does not. (B) STYLIZED — the render draws TV lanes, the leader genuinely mid-track. Next: (1) test the depth confound (does y_lat correlate with longitudinal screen-depth within a frame?); (2) EYEBALL whether the render itself draws #12 nearest the inner rail (if the pixels put the leader mid-track, it is stylization, not a measurement bug); (3) if depth-confound, fit the ground-plane HOMOGRAPHY proper and re-read. Then the honest verdict. Full pass pending.

Files: studies/cv_s0d_aerial_render.py (NEW, owned); data_cache/cv_s0d_aerial_render.parquet; data_cache/figs/cv_s0d_render_geom.png + cv_s0d_render_trip.png. ruff clean. src/hkjc/model|core UNTOUCHED; mode_b imported READ-ONLY (no lib edit yet).
