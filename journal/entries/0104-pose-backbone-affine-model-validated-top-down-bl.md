---
title: Pose backbone: AFFINE model validated (top-down); blocker is unreliable outer rail -> EKF + outlier-reject next
id: 0104-pose-backbone-affine-model-validated-top-down-bl
author: modelling
audience: ["modelling", "all"]
channel: handoff
kind: finding
status: open
thread: 
tags: ["cv", "aerial", "reconstruction", "homography", "handoff"]
refs: {"updates": "0103", "task": 101}
created: 2026-06-08T05:30:00+00:00
updated: 2026-06-08T05:30:00+00:00
---
## Pose backbone: AFFINE is the right model (user's top-down insight); blocker = unreliable OUTER rail

### KEY FINDING — the render is a tilted top-down view, fit AFFINE not homography (user's insight)
Compared similarity/affine/homography ICP fits of the two-rail template:
  f2440 straight:  sim 75.7  AFFINE 3.4  homog 3.4   (affine == homography)
  f2256 home turn: sim 14.4  AFFINE 4.1  homog 10.3  (AFFINE BEATS homography)
So: similarity (4-DOF, uniform scale) too rigid; the view is anisotropic/tilted top-down -> 6-DOF
AFFINE. Affine matches/beats the free homography AND its params live in a FLAT SMOOTHABLE space (the
homography did not). cv_pose_backbone.py now fits estimateAffine2D. This is the right per-frame model.

### But the across-race LATERAL is STILL noisy in regions (f750-1000, f1700-2150) — and NOT a
### smoothing problem
Tried anchor-index median, then FRAME-windowed robust median + outlier-resistant: same systematic
errors (median y_lat ramps 22->1 m, jumps to 25 m) in those regions. A robust median CANNOT fix a
region where the ANCHORS are SYSTEMATICALLY wrong.

### ROOT CAUSE: unreliable OUTER-rail detection -> wrong metric ruler -> wrong scale anchors
The INNER (running) rail is reliable (field-median). The OUTER rail = "the other main rail" is NOT:
in some stretches it's a shadow / marker line / mis-detection, not the true 22.5 m outer. A wrong
outer -> wrong 22.5 m ruler -> wrong affine scale -> systematically wrong y_lat for that whole
stretch. This is the same inner/outer ambiguity from the rail work, now biting the metric scale.

### NEXT (focused build): reliable outer + EKF pose propagation with outlier rejection
1. Validate the OUTER rail: its implied scale (sep_px) must be consistent with the slowly-varying
   camera zoom and the known geometry; reject anchors whose scale is inconsistent (wrong outer).
2. EKF/RTS over the AFFINE pose (6 params, now a proper state) with INNOVATION-GATED measurement
   updates (rejects bad-fit anchors) + a smooth dynamics model -> propagates the pose through
   unreliable/one-rail frames instead of trusting every per-frame fit. (Reuse reconstruction/fusion.py
   Kalman+RTS.) This is the piece batch-smoothing can't do (it can't reject systematic regional bias).

### State
Affine model committed. The best clean 2D recon remains partial; the EKF + reliable-outer build is the
remaining work to make it production-good. Recommend a focused session for it.
