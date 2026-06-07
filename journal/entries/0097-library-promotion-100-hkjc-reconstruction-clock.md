---
title: library promotion #100: hkjc.reconstruction (clock + fusion v2 + orchestrator) built, ruff+pytest green
id: 0097-library-promotion-100-hkjc-reconstruction-clock
author: modelling
audience: ["all"]
channel: research
kind: note
status: open
thread: 0060
tags: ["cv", "reconstruction", "library-promotion", "fusion", "clock"]
refs: {}
created: 2026-06-07T14:55:31+00:00
updated: 2026-06-07T14:55:31+00:00
---
LIBRARY PROMOTION (#100, the RECONSTRUCTION half). Built the additive hkjc.reconstruction
package on top of the perception-port (0095): clock.py + fusion.py + reconstruct.py + a synthetic
unit test. Pure consolidation of the validated study code into clean library form — same
math/thresholds, no redesign. ruff (src/hkjc tests studies) clean; full pytest green (193 passed).

WHAT WAS BUILT
- reconstruction/clock.py — the shared race_t clock unified in one place. ClockAnchors (frozen):
  per-race section table + per-feed OFFSETS/anchors as INPUTS (not hard-coded). arclen<->race_t via
  the official (cum-time, cum-arclen) curve (section_curve helper); patrol/RF race_t = vid - offset;
  drone AFFINE vid->race_t from the gate-jump + finish-cut anchors. anchors_from_recipe() defaults
  section_bounds to an even arclen split when the official call marks are unknown.
- reconstruction/fusion.py — ONE smoother v2 merging the three layers. Obs dataclass ported
  field-for-field from cv_fusion_4feed (runner, race_t, arclen|None, ylat|None, feed, sd_arclen,
  sd_ylat; + a derived .kind). smooth_horse = per-horse KF-forward + RTS-backward on z=[s,sdot,l,ldot]
  at 5Hz with: LONGITUDINAL = the #97 phase-gated EOM physics kernel (template_velocity ramp->cruise->
  finish; per-horse levels v_c,d_fin from fit_levels on that horse's own obs; STRONG OU pull in the
  ~7.5s gate-ramp, WEAK in cruise/finish to preserve order-bearing velocity differences); LATERAL =
  bounds-only white-accel (no shape prior — it is the measured control signal). Then the #96
  constraints per node: truncated-Gaussian hard box 0<=y_lat<=W (W from RaceGeom.width), joint
  cross-horse PAVA isotonic ordering gated by per-node order-reliability, geometry bend down-weight.
  smooth_field(obs_by_runner, grid, geom, *, do_box, do_order, race_t_to_arclen) is the public entry;
  bend_downweight_obs() applies RaceGeom.bend_weight() as an input sd re-tag of aerial/drone laterals.
  Bend classification is consumed from RaceGeom (sources core); core's bend map NOT touched (the
  1650/1800 start-chute approximation stays a separate deferred follow-up).
- reconstruction/reconstruct.py — RaceRecipe (frozen): 4 video paths, section (time,arclen) table,
  per-feed offsets/anchors, windows, name2num LUT. Reconstructor(race, recipe).run() -> RaceTrajectory.
  build_obs adapter implements the CV_RECON-6 contract: PATROL anchors the lateral METRIC (sd~1.2/conf),
  AERIAL+DRONE carry lateral ORDERING (inflated sd) + arclen, RF carries longitudinal ORDER (legend
  rank-offset from the field backbone) + finish-MARGIN metric (RANK_GAP_M = THOROUGHBRED_LENGTH_M),
  DRAW anchors a low-sd lateral obs at race_t=0 via RaceGeom.draw_to_metres(). .run() = extract_feeds()
  (the heavy CV IO seam) -> from_feeds() (the pure, testable tail: build_obs -> grid -> smooth_field ->
  assemble). assemble() emits TrajPoints with d = distance - arclen, off = y_lat, speed = sdot,
  conf from joint sd, coverage from node_has_obs.
- tests/test_reconstruction.py — imports the 3 modules with no heavy CV deps; clock round-trip +
  per-feed maps; a SYNTHETIC smooth_field test (3-runner 1200m HV C + hand-built Obs) asserting
  monotone-ish arclen, y_lat in [0,W], finite sds, correct shapes, preserved rail<mid<wide order +
  surviving rail-vs-wide gap; a box-caps-unanchored-lateral test. Full module paths only (no
  hkjc.vision package-level re-exports, per the in-flight vision/__init__ restructure).

CONSOLIDATION DECISIONS where the three studies differed:
- The smoother is ONE function (smooth_horse) carrying the #97 longitudinal kernel + #96 lateral block,
  NOT the #97 study's two paired smoothers (white vs physics). The physics kernel is the keeper (it was
  the validated v2); the white-accel base is subsumed (it is the physics kernel with the pull off).
- The #96 study applied constraints as a separate apply_constraints() over a dict of per-horse results;
  kept that structure (smooth_field calls smooth_horse per runner then apply_constraints jointly).
- RF obs assembly: ported the visible-rank + absent-back-marker rank-offset logic from load_rf, but the
  finish-ORDER carry-forward (the rf_finish_order soft anchors for legend-seen horses with no margin)
  was study-specific bookkeeping that depended on hard-coded winner refs / LATE_T fractions — left it
  OUT of the library build_obs (the finish margins + the rank backbone already carry order to the line);
  a caller can add finish-order anchors via the rf['finish'] list. This is the one behaviour I did not
  consolidate verbatim — flagged for follow-up if a race needs the legend->line carry.
- Track width / draw / bend ALL sourced from RaceGeom (which wraps core.Race); the studies' baked
  TRACK_WIDTH=21.3 / 22.5 and the empirical run-in bend threshold are replaced by core's 22.5 (HV C)
  and RaceGeom.bend_weight.

COULD NOT CONSOLIDATE CLEANLY: extract_feeds() is left as a NotImplementedError seam — the per-feed
vision.feeds.* extractor wiring + identity solve is feed-specific IO that the studies validated against
cached panels; the pure fusion tail (from_feeds) is what the unit test locks. A deployment supplies
cached numbered feed rows and calls from_feeds() directly, or implements the extractor wiring for a
live run. The finish-order legend->line carry (above) is the other deferred item.

DISCIPLINE: additive only (new files under src/hkjc/reconstruction/ + the test); core/data/vision/
studies untouched; official results VALIDATION-only (not consumed in .run()).
