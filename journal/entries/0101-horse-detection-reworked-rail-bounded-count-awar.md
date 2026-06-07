---
title: Horse detection reworked: rail-bounded count-aware watershed (k-means dropped); longitudinal next
id: 0101-horse-detection-reworked-rail-bounded-count-awar
author: modelling
audience: ["modelling", "all"]
channel: handoff
kind: finding
status: open
thread: 
tags: ["cv", "aerial", "detection", "reconstruction", "handoff"]
refs: {"promoted": "hkjc/vision/feeds/aerial_detect.py::detect_avatars", "updates": "0100"}
created: 2026-06-08T00:30:00+00:00
updated: 2026-06-08T00:30:00+00:00
---
## Horse detection reworked — rail-bounded, count-aware, k-means dropped (promoted)
Promoted `detect_avatars` to hkjc.vision.feeds.aerial_detect (replaces the `_kmeans_field_both`
forces-the-count approach). Pipeline: turf/bright suppress -> Canny+fill ∩ colour -> CLOSE to
reconnect each avatar's split body/silk pieces -> RAIL-BOUND the ROI (track region between the two
main rails, from rail candidates = field-INDEPENDENT, drops grandstand/apron false-positives) ->
rail-parallel close -> split only OVER-LARGE blobs (genuine front-to-back merges) into
round(area/single) centres via distance-transform NMS, with the single-horse area SOLVED from the
known field/legend count (the user's "iterate until we get the number of horses", as a global
constraint). Count-AWARE, not count-FORCED.

Tried + judged visually (studies/cv_avatar_detect2.py): iterative rail-parallel blur alone (good on
spread, fails on bunched=under-segmented); naive distance-peaks (over-counts, one peak per fragment);
the field-anchored watershed split = WINNER (f1080/f2200/f2440 -> exactly 11 with dots on the horses;
f1760 -> 14, a few avatars stay split). Bidirectional blur-down-then-split regressed; reverted.

Whole-race: median 11 in-lane (honest variance 5-20 vs the old fake-forced 11); 4 clean long
tracklets. Good enough to drive longitudinal reconstruction (user: losing 1-2 frames is fine).

## NN research (background subagent, orthogonal)
Keep the rail classical (SAM2/segmentation are weak on thin structures; the rail problem is geometric
disambiguation). The only NN worth a small experiment is a fine-tuned YOLO11-n DETECTOR (CPU-feasible
~16fps; auto-labelable from the classical detector + legend; ~1-2 days) for the residual bunched/faint
cases. Zero-shot/SAM2-video re-create the merge problem and need GPU. Deferred.

## NEXT (in progress): longitudinal / metric reconstruction
Goal: per-horse (arclen, y_lat, t) 2D paths. Use the on-screen ODOMETER widget (M TO FINISH;
aerial_ocr.read_odometer; pre-extracted in cv_joint_aerial_R08_obs.npz, ~148 finite frames) as the
key-frame absolute-arclen calibration to convert per-frame pixel s along the rail -> metric arclen,
and to set the px->m scale (its rate of change). Then assemble high-quality 2D paths.

## OPEN
- (c) A-config single-rail fragment coalescing (still pending).
- f1760-type detection over-segment residual (a few frames); per-frame count variance.
