---
title: Pose backbone: core validated; free-homography smoothing fails -> solvePnP pose + EKF is the path
id: 0103-pose-backbone-core-validated-free-homography-smo
author: modelling
audience: ["modelling", "all"]
channel: handoff
kind: finding
status: open
thread: 
tags: ["cv", "aerial", "reconstruction", "homography", "handoff"]
refs: {"updates": "0102", "task": 101, "file": "studies/cv_pose_backbone.py"}
created: 2026-06-08T04:00:00+00:00
updated: 2026-06-08T04:00:00+00:00
---
## Pose backbone (task #101): core validated, but FREE-homography smoothing fails -> need solvePnP

Built studies/cv_pose_backbone.py: per-frame two-rail homography (track-plane inner b=0 + outer
b=22.5 m -> image), read each horse's metric lateral b via H^-1. The KNOWN 22.5 m rail separation is
the metric ruler; positions read off the FIXED track plane (no per-frame s-origin, no s_head jitter,
perspective-correct scale).

### What WORKS (core validated)
Per-frame fit is clean where both rails are well-detected: f2440 resid 3.4 px (y_lat 3-14 m), f2200
15.8 px. Across the race the LATERAL band is clean + physical in the dense-anchor regions
(f1800-2780: median ~4-9 m, tight). 1312 two-rail anchors over 2162 frames.

### What FAILS (the wall)
1. TEMPORAL CARRY freezes: on long one-rail stretches (outer rail undetected, ~845 frames) carrying
   prev_H froze -> a constant 22.5 m plateau (f1500-1620).
2. Smoothing the raw 8 homography params + interpolating across anchors does NOT clean it: a
   homography is NOT a flat vector space (perspective terms h20/h21 tiny+sensitive, h02 ~hundreds), so
   median-smoothing breaks the geometry and interpolation across sparse anchors gives ramps. f600-1500
   stays jittery (median swinging 3-25 m).
3. ROOT CAUSE: the free 8-DOF homography is too unstructured + the along-axis is degenerate on
   straights (two parallel rails don't fix along-position), so the ICP correspondence slides and
   perturbs even the (over-determined) lateral.

### Conclusion / NEXT (per the SOTA, BHITK/PnLCalib/TVCalib — journal 0102)
Must parametrize and smooth the CAMERA POSE, not the free homography: cv2.solvePnP for
rotation+translation+focal against the Z=0 track plane (well-conditioned on straights because the
camera model removes the unconstrained DOFs), then EKF/RTS smoothing of the POSE state (reuse
reconstruction/fusion.py Kalman+RTS), with optical-flow dense-relative as the pose dynamics. That is
the focused next build.

### Meanwhile
The best current 2D recon is the main tracker (studies/cv_aerial_track_id.py): detect_avatars +
field-median running rail + odometer-anchored arclen + two-rail per-frame sep-scale y_lat + the
spike-audit validity gate. Decent (bulk physical) but not production-clean; the solvePnP pose backbone
is what makes it good.
