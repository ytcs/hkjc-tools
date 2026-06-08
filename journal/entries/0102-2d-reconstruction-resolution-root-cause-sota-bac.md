---
title: 2D reconstruction: resolution root cause + SOTA backbone blueprint (BHITK camera-pose tracker)
id: 0102-2d-reconstruction-resolution-root-cause-sota-bac
author: modelling
audience: ["modelling", "all"]
channel: handoff
kind: decision
status: open
thread: 
tags: ["cv", "aerial", "reconstruction", "homography", "handoff"]
refs: {"updates": "0101", "task": 101}
created: 2026-06-08T02:00:00+00:00
updated: 2026-06-08T02:00:00+00:00
---
## 2D reconstruction: resolution diagnosis + the backbone revamp blueprint (SOTA-informed)

### Where we are (committed thru 8bc8bfc + WIP)
detect_avatars (rail-bounded, count-aware) -> per-frame rail (select_running_rail) -> project to s,perp
-> ODOMETER-anchored arclen (odo = HEAD m-to-finish, CONFIRMED at the finish: f2735 shows "20M" + the
FINISH line, field ~20m back; OCR verified: precision 100%, recall ~88%, interp-covered) -> metric-
space tracklet association with the ODOMETER as the motion prior (longest track 1216 frames, top
304/290/237; 538->202 fragments). y_lat is POSITIVE distance from the rail.

### Resolution root cause (studies/cv_aerial_track_id.py PASS4 cv_recon_res.png)
WITHIN stable straights resolution is GOOD (smooth perp/s). Damage is at: (1) STEEP-OBLIQUE bends /
the home turn f2196-2316 (f2256 = 340M: low oblique on a sharp bend, rail barely visible, heavy
foreshortening) -> per-frame rail fit unstable + single GLOBAL px->m scale wrong -> tens-of-px jumps;
(2) HEAD-REFERENCE jitter (anchoring to max(s), a single jittery lead detection -> offset baseline
snaps to 0); (3) isolated outlier spikes.

### The revamp (SOTA literature, all CPU) — BHITK-style camera-pose tracker over the rigid track plane
Per-frame map is NOT a free 8-DOF homography (degenerate on straights, our 0099 finding) but a
CONSTRAINED camera pose+focal (cv2.solvePnP, Z=0 track plane). Layers:
  (A) per-frame constrained pose from rail LINES (inner + outer at the known 22.5 m offset) + odometer
      + distance poles;
  (B) DENSE-RELATIVE init: LK optical flow on background -> estimateAffinePartial2D similarity
      (pan/rotate/zoom) carries the previous pose as the init -> the temporal init 0099 said was
      missing (fixes the oblique-bend local minima);
  (C) temporal smoothing of the CAMERA TRAJECTORY via EKF/RTS -> REUSE reconstruction/fusion.py
      Kalman+RTS (same forward/backward pattern, on the camera not the horses). GTSAM factor graph +
      oval loop-closure (RailGeometry closure gap) only if it drifts over a lap.
Why it fixes everything: positions read off the FIXED metric track frame (no per-frame origin -> no
max(s) jitter, no bend level-shifts); homography gives a PERSPECTIVE-CORRECT local px->m scale with
the 22.5 m rail separation as the metric ruler (resolves CV_RECON §7 under-determination); pose-graph
is globally consistent; dynamics carry the pose through holes. Render is TRUE PERSPECTIVE (confirmed:
near horses larger than far on f2256), so PnP applies. AVV is synthetic (no lens distortion) -> pose
model even cleaner.

### Plan (proceeding with all)
- QUICK WINS on the current pipeline (interim, low-risk): robust head reference (smoothed/percentile,
  not max), per-track outlier rejection + light smoothing. [local two-rail scale deferred -> subsumed
  by the backbone.]
- BACKBONE: new studies/cv_pose_backbone.py: solvePnP pose + optical-flow init + EKF/RTS over the
  camera, validated f1080 (straight) then f2256 (home turn), then promote to hkjc.reconstruction.

### Key SOTA refs
BHITK arXiv:2311.10361 (sequential Bayesian homography, sparse-absolute + dense-relative Kalman) — the
direct analogue. PnLCalib arXiv:2404.08401 (points+lines, 3D model). TVCalib arXiv:2207.11709
(camera-model vs free homography). GTSAM (factor graph, CPU). Line-vs-point DLT (Dubrofsky/Woodham).
