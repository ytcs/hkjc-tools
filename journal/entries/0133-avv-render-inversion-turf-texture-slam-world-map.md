---
title: AVV render inversion: turf-texture SLAM -> world map -> rails/metric once -> horse paths (R08 end-to-end)
id: 0133-avv-render-inversion-turf-texture-slam-world-map
author: modelling
audience: ["all"]
channel: handoff
kind: finding
status: open
thread: 
tags: ["cv", "aerial", "avv", "slam", "pose-graph", "mosaic", "rail", "odometer", "identity"]
refs: {"package": "src/hkjc/vision/avv", "studies": "studies/cv_avv_*.py", "supersedes_approach": "0115-0129 per-frame rail tracking"}
created: 2026-11-13T09:16:03+00:00
updated: 2026-11-13T09:16:03+00:00
---

## Reframe: the AVV render is a static textured world seen by a smooth near-nadir camera -> SLAM it

All prior aerial work (journals 0103-0129) tracked rails per frame in IMAGE space. This session
replaces that with world registration (new package `hkjc.vision.avv`, studies `cv_avv_*.py`):

1. **Camera ego-motion from turf texture** (ORB on HUD/horse-masked turf, RANSAC similarity).
   Chained steps trace the Happy Valley oval and close the 1.24-lap loop of an 1800 m trip.
2. **Pose graph** (sequential + long-baseline d2/d5/d12/d24 + loop-closure edges, robust LS,
   mid-race gauge, outlier pruning). R08: edge residual median 0.2-1.2 px, loop 0.86 px.
3. **World map = mosaic** of ~970 registered frames: horses average out; rails + painted markers
   are crisp. Pass-1 vs pass-2 maps of the doubly-covered home straight coincide (rails overlay).
4. **Racing rail** extracted ONCE on the static map: stations along the field's inner edge,
   thin-bright-ridge profiles (median-filtered along s to kill painted markers), Viterbi DP,
   sub-px refinement along the rail's own normals.
5. **Metric**: painted FINISH/50/100/150/200/300 markers -> 18.58 world px/m + finish anchor.
   **Odometer (OCR) vs reconstructed leader distance-to-go: median |err| ~2-4 m** over the race.
6. Horses (joint YOLO, 11/frame median) -> world -> (d to finish, lateral off racing rail, m);
   world-frame Hungarian tracker; identity from legend running order + elimination + silk.

## Hard facts learned (do not re-learn)
- The render updates at **10 Hz** (25 fps MP4, pattern new,dup,new,dup,dup) -> de-dup frames.
- Camera is **near-nadir, no tilt** (no keystone term vs motion); it **does zoom slowly** (a few %),
  so poses are similarities (rigid model fits 5x worse on long-baseline edges).
- **LMEDS refit is biased** (FAST keypoints are integer-located; median snaps steps to whole px,
  speed-dependent sign) -> use least-squares (Umeyama) on RANSAC inliers.
- **ECC refinement REJECTED**: mowing-stripe aperture problem -> scale drifted 0.95..1.11.
- The painted distance markers are **dynamic** (drawn only near the finish, absent in pass 1).
- The first render frame after the live->render dissolve is unreliable (exclude / never gauge on it).
- Loop-closure candidates must be matched **unguided** (chain drift 100-200 px exceeds any gate).
- Projection to the rail must be **time-consistent** on multi-pass trips (nearest vertex in space
  can be the other lap).
- Field direction check: in every phase the racing rail is on the field's RIGHT (clockwise HV).
- `npz[...]` slicing inside a loop re-decompresses the whole array each time (caused an OOM crash):
  load once. Heavy jobs now run under `studies/capped.sh` (systemd MemoryMax).

## Open (next)
Identity-free validation vs official splits; odometer-transition s calibration; drop detections in
the odometer box; camera-relative-s stitching through bend dropouts; official-silk prior self-test;
then scale-out to other venues / tracks / weather with one driver.
