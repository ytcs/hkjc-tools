---
id: 0032-consolidated-geometry-canonical-race-domain-obje
title: Consolidated geometry + canonical Race domain object for agent-based modelling
kind: infrastructure
status: open
tags: ["geometry", "traffic-model", "refactor", "sha-tin", "happy-valley", "abm"]
refs: {}
related: []
created: 2026-06-04T08:08:57+00:00
updated: 2026-06-04T08:08:57+00:00
---
Consolidated the codebase around the agent-based-modelling approach and integrated race-course geometry as first-class reference data.

**Geometry (hkjc.track, promoted out of hkjc.model.track).** Now covers all HK regimes, not just Happy Valley:
- HAPPY_VALLEY: turf 1450m, bend-dominated (bend map from sectional sampling points).
- SHA_TIN_TURF: 1900m galloping oval, 430m home straight, long straights => weaker draw bias.
- SHA_TIN_STRAIGHT: the separate 1000m sprint chute with NO bends => no bend-geometry draw bias (only watering gradient, not modelled). track_for("ST",1000) returns this, not the oval.
- SHA_TIN_AWT: dirt 1560m, fixed single rail, uniform 22.8m.
Resolver track_for(venue, distance, surface) picks the right geometry. Per-course widths/home-straights are official HKJC figures; ST oval/AWT bend *boundaries* are two-turn reconstructions from circumference+home-straight (approximate, documented).

**Canonical Race/Runner domain object (hkjc.race).** Bundles runners + geometry + conditions; resolves .geometry/.width/.bend_fraction and builds any model in one call (.traffic_model(), .energetics_model(), .plackett_luce()). Constructors from_results(df) / from_racecard(df) / from_runners(). Surface inferred from the ALL WEATHER TRACK label (going codes alone are ambiguous: both turf and AWT have GOOD). This is the ABM seam — no more passing loose parallel arrays.

**Validation.** Geometry now produces the correct venue ordering purely from physics: at 1200m, HV (62% on turns) yields a 29x inner/outer win ratio vs Sha Tin turf (42% on turns) at only 3.1x — HV draw bias >> ST, as in reality. Still over-amplified in absolute terms (calibration pending), but the relative structure is right and emergent.

**Cleanup.** Removed the four falsified-model experiment scripts (form/market/sectional/trouble). Kept energetics (basis of the traffic model) + the live carryover/capture runners. 103 tests pass (was 91), ruff clean. version 1.6.0.

Next: calibrate bend_curvature/noise to the empirical per-venue draw curves (DrawStatistics / results), then test the calibrated interacting model vs market on unusual draw/pace/course configs.
