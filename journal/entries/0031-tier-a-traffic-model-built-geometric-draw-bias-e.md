---
id: 0031-tier-a-traffic-model-built-geometric-draw-bias-e
title: Tier-A traffic model built; geometric draw bias EMERGES (needs calibration)
kind: result
status: open
tags: ["traffic-model", "tier-a", "draw-bias", "emergent", "needs-calibration"]
refs: {"module": "hkjc.model.traffic"}
related: []
created: 2026-06-04T07:55:31+00:00
updated: 2026-06-04T07:55:31+00:00
---
hkjc.model.traffic.TrafficRaceModel: vectorized agent-based sim. Each horse runs its energetic even-pace target v_c*(1+W'/D) (reduces to the energetics model with empty track) but is subject to traffic: car-following (capped behind a same-lane horse within follow_gap), lane-change/gap-acceptance when blocked (stochastic, moving out), drift back to rail when clear, and BEND GROUND-LOSS (ground_factor = 1 + off*curvature on turns; draw -> starting lateral offset, width from the day's Course). Uses hkjc.model.track HV geometry. VALIDATION (key): with EQUAL-ability horses, the inner-draw advantage EMERGES purely from geometry -- HV 1200m A course: draw1 win 32% monotonically down to draw12 0.6%. Mechanism works (draw bias produced, not fitted); ability can also overcome a wide draw (tested). BUT currently OVER-amplified vs reality -> needs CALIBRATION of (bend_curvature, speed noise, lane params) to the empirical HV draw-bias curve (HKJC DrawStatistics or our results), then the edge test vs market (esp. unusual draw/pace configs). Tier-B (true lateral lanes) still needs aerial-video trajectories. ~17s/8k-sample sim (per-step O(n^2) vectorized over samples).
