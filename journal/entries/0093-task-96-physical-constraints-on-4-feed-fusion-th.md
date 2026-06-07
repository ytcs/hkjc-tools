---
title: TASK #96 — physical constraints on 4-feed fusion: the BOX bounds the route bend sd 52->3m (real, no-regression) but does NOT close the bend METRE gap; joint ordering near-inert on routes (sparse co-presence + foreshortened order) -> bends still need the rail-arc homography
id: 0093-task-96-physical-constraints-on-4-feed-fusion-th
author: modelling
audience: ["all"]
channel: research
kind: result
status: open
thread: 0060-reorientation-win-diagnostic-not-veto-wave2-expand-causal-network
tags: ["cv", "vision", "fusion", "four-feed", "constraints", "box", "truncated-gaussian", "ordering", "isotonic", "pava", "joint-projection", "min-separation", "horse-width", "track-geometry", "happy-valley", "c-course", "width-22.5", "bend-radius", "arclen-bend-map", "foreshortening", "metric-gap", "s7", "s9", "drag", "route", "r08", "r01", "r03", "rail-vs-wide", "classification", "per-pole", "winner-recovery", "no-regression", "draw-anchor", "initial-condition", "study-only", "task-96", "0087", "0090", "0092", "wave-2", "control"]
refs: {}
created: 2026-06-07T13:55:39+00:00
updated: 2026-06-07T13:55:39+00:00
---
TASK #96 — physical constraints on the 4-feed fusion smoother (box + joint ordering-within-box + track-geometry prior); paired OFF/ON on R01/R03/R08. VERDICT: the HARD BOX 0<=y_lat<=W is a real, defensible win that ELIMINATES the unphysical lateral states (negative y_lat + the >W route bands) and caps the ballooned route sd, but it does NOT close the bend METRE gap (S7) — that still needs the rail-arc homography. The JOINT ORDERING adds little on the bends and I gate it honestly so it does not manufacture false certainty.

GEOMETRY (HKJC racing_course page, authoritative): HV turf running WIDTHS A 30.5 / A+2 28.5 / B 26.5 / B+2 24.5 / B+3 23.5 / C 22.5 / C+3 19.5 m. So the C-course running width is 22.5m (the studies baked 21.3m as a banked-rail approximation — close, ~5% low; I used 22.5m for the box and report it). Home straight C=334m, circuit ~1450m -> two ~semicircular bends, bend radius ~130m (recorded for the future rail-arc homography). Built an a-priori arclen->{straight,bend} map FROM this geometry (not empirics) to down-weight the aerial lateral metric on bends — replaced the empirical >run-in heuristic.

METHOD: copy/extend the RTS smoother (studies/cv_fusion_constrained.py). Box = per-node TRUNCATED-GAUSSIAN moment-match onto [0,W] (principled; vs a naive clip that leaves sd untouched). Ordering = precision-weighted ISOTONIC/PAVA projection of the per-node lateral posteriors onto {ordered, in [0,W], >=1.1m min-sep}, with the constrained sd set by the feasible half-width but GATED by per-node order-reliability (fraction of co-present horses with a genuine lateral OBS this frame) so bend order-noise does not collapse the sd. Draw->y_lat(0) anchor added to ALL runs (fair OFF/ON; draw = allowed initial condition). Longitudinal channel never touched.

RESULTS (per-band lateral sd m, OFF -> naive-CLIP -> BOX-only -> ON=box+order+geom):
  R01 (mile, long straights, dense coverage): bands 2.25/1.73/1.08/1.44 -> CLIP identical -> BOX 2.97/1.86/1.45/0.96 -> ON 2.47/1.67/1.13/0.78. gap +4.30(8/8) -> +6.66(8/8). winner #3 YES, per-pole +0.95/+0.82/+0.88/+0.71 UNCHANGED. R01 was already metred (patrol+draw cover the whole mile).
  R03 (sprint): 4.89/9.37/11.65/14.22 -> BOX 2.38/3.22/2.30/1.80 -> ON 2.32/2.56/2.19/1.67. gap +5.81(8/10) -> +6.57(8/10). winner #7 YES, per-pole +0.99/+0.99/+0.10 UNCHANGED.
  R08 (1800m ROUTE, the bend stress case): 12.33/28.38/39.06/52.03 -> CLIP identical (52.03!) -> BOX 4.06/3.48/3.58/3.14 -> ON 4.01/3.38/3.29/3.10. gap +3.97(3/6) -> +3.49(3/6) STILL CORRECT SIGN. winner #6 YES, per-pole +0.96/+0.95/+0.98/+0.94/+0.60 UNCHANGED. field y_lat span OFF [4.2,21.3] -> ON [4.1,20.8] (now inside [0,22.5]).

HONEST VERDICT (user wanted the truth either way):
(1) The BOX genuinely helps and is the right thing to bake: it removes the physically-impossible states (the naive CLIP leaves the 52m sd untouched; the truncated-Gaussian box caps it to ~W/sqrt12 ~3-4m on the route). R08 route bend sd drops 52->3m — a 16x sd reduction from ONE hard physical fact.
(2) BUT 3m sd on a 22.5m track is 'which seventh of the track', NOT the 1-2m absolute metres the patrol anchor gives on straights. The box BOUNDS the nonsense; it does not RECONSTRUCT the bend metres. The rail-vs-wide CLASSIFICATION is UNCHANGED by the constraints on R08 (3/6 both) — the box caps sd but the posterior MEAN (which drives classification) is unmoved where it was already in-box. So the S9 drag-energetics bend-metre gap is NOT closed.
(3) The JOINT ORDERING (the hoped-for gap-closer) FAILS to pin bend metres, for a concrete diagnosed reason: on the route the field is longitudinally STRUNG OUT (gaps to leader up to ~30m), so at any instant only ~2-3 horses are genuinely co-located with lateral reads (median 3 present, 78% of nodes have >=2 lat-obs but the bend nodes specifically have 0-1), and the unanchored horses (#3/#10 had ZERO lateral obs in the 600-1100m band -> raw sd 120-150m) are too far from the anchored ones for the ordering to pin them. '11 ordered bodies across 22.5m' is the premise; the reality on bends is '2-3 bodies, order read from a foreshortened render'. I refuse to collapse the sd to the 0.55m feasible-width floor (an early un-gated version did, falsely) — gating by order-reliability, ordering adds only BOX 3.48 -> ON 3.38 on the R08 mid bend. Where order IS reliable (R01/R03 straights, patrol+drone reads) ON<BOX (R01 run-in 0.96->0.78).

CONCLUSION: bake the BOX (truncated-Gaussian, W=22.5m C-course) — it is a free, correct, no-regression tightening that kills the unphysical tail. Keep the geometry-based bend down-weight. The joint ordering is sound but near-inert on routes (sparse co-presence + foreshortened order) — it is NOT a substitute for the bend rail-arc homography. The bend ABSOLUTE-METRE gap (CV_RECON S7, the S9 drag level) really does still need the per-frame bend homography; the physical constraints prettify and bound the bands and remove impossible states, but do not add the missing metric information on bends.

ARTIFACTS: study studies/cv_fusion_constrained.py; parquets data_cache/cv_fusion_trajectories_{R01,R03,R08}_constrained.parquet; figure data_cache/figs/cv_fusion_constrained_qc.png (OFF top vs ON bottom, all 3 races, box drawn). Banked R01/R03/R08 fusion parquets UNMUTATED; ruff clean; PYTHONHASHSEED=0. Official=validation-only; draw=initial condition.
