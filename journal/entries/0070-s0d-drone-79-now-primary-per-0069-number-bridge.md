---
title: S0d DRONE (#79, NOW PRIMARY per 0069): NUMBER-BRIDGE UNLOCKED (0068 FALSIFIED) but the LATERAL y_lat TRIP is NOT YET VALIDATED — an honest §4b partial. (1) The finish-zoom "Photo Finish System" replay (t≈118-140s) OCRs saddlecloth numbers NON-CIRCULARLY: RapidOCR read 11/12 (#1,2,3,5,6,7,8,9,10,11,12; only #4 missed), silk-LAB sampled at each number bridges to back-straight colour clusters, 10/12 tracks bound — so 0068's "numbers ILLEGIBLE from altitude, no honest cluster→number bind" is DEFINITIVELY FALSE, exactly as 0069 predicted. (2) SCALE is DIMENSIONALLY FIXED: per-frame m/px (~0.0213) from a distance-transform perpendicular turf-width == C-width 21.3m; y_lat now CLAMPED in-bounds (max 20.35m ≤ 21.3, was overspilling 29m). (3) BUT two hard limits the prior agents glossed: (3a) ARC-LENGTH CANNOT reach 1650m — the drone PANS/ZOOMS/CUTS and TRACKS the field, so in-frame rail-projected travel ≠ ground travel; field-centroid arc totals only ~379m for 1650m. The 0067/0068 "3-4x inflation" was not a constant scale bug but a footage property: a moving-camera drone gives NO clean full-race longitudinal axis. (3b) The y_lat TRIP CONTRADICTS the known race: the wire-to-wire RAIL leader #12 reads WIDE (5.7m, rank 8/10), not tight — because the per-frame inner-rail polyline (turf-mask boundary, PCA-ordered) is an UNSTABLE reference (y_lat median swings 0.8→6.3m across clean frames) and the low-contrast silk clusters BLEND multiple horses. So the novel lateral/control signal is MIS-SPECIFIED, not yet a validated per-runner trip. TRUSTWORTHY: the number bridge + the dimensional scale + that the field IS laterally resolvable on the top-down back-straight (t≈35-95s). NOT trustworthy: per-runner y_lat magnitude/ordering, full-race arc-length, the gate-load (t<10s, crowd clutter) and run-in. Study-only; src/hkjc/model+core+vision/mode_b untouched (ocr.py imported READ-ONLY); ruff clean.
id: 0070-s0d-drone-79-now-primary-per-0069-number-bridge
author: modelling
audience: ["all"]
channel: research
kind: result
status: open
thread: 0060-reorientation-win-diagnostic-not-veto-wave2-expand-causal-network
tags: ["s0d", "cv", "vision", "drone", "oblique-camera", "primary", "lateral", "y-lat", "control", "trip", "ground-saved", "number-bridge", "finish-zoom", "saddlecloth", "rapidocr", "ocr", "reid", "silk-colour", "backward-propagation", "non-circular", "scale-fix", "px-to-m", "distance-transform", "turf-width", "c-course", "21.3m", "arc-length", "odometer", "rail-relative", "homography", "moving-camera", "panning", "in-frame-vs-ground", "mis-specification", "ceiling-check", "4b", "unvalidated", "honest", "partial", "0067", "0068", "0069", "wave-2", "external-validation", "official-data", "queue-79", "real-horse"]
refs: {"drone_pipeline": "studies/cv_s0d_drone.py", "validate": "studies/cv_s0d_drone_validate.py", "qc": "studies/cv_s0d_drone_qc.py", "figure": "data_cache/figs/cv_s0d_drone_qc.png", "drone_panel": "data_cache/cv_s0d_drone.parquet", "drone_video": "data_cache/video/drone_20260603_01.mp4", "official": "data_cache/race_official/20260603R1_results.html", "ocr_wrapper": "src/hkjc/vision/mode_b/ocr.py (READ-ONLY)", "supersedes_claim": "journal/entries/0068 (drone-weak-anonymous) + 0069 plan #79", "prior": "journal/entries/0069-s0d-synthesis-correction-reconciles-0067-first-p.md"}
created: 2026-06-07T00:41:52+00:00
updated: 2026-06-07T00:41:52+00:00
---
S0d task #79 — the DRONE made PRIMARY by 0069 for the novel lateral/control signal. I own
`studies/cv_s0d_drone.py` (+ new `cv_s0d_drone_validate.py`, `cv_s0d_drone_qc.py`). Run on ST
2026-06-03 R1 (Class 5, 1650M, turf "C"); validated against the OFFICIAL result + the in-video
finish-zoom (numbers are video-only, non-circular).

## WHAT 0069 GOT RIGHT — the number bridge is real (0068 falsified)
The finish-zoom "Photo Finish System" slow replay (t≈118-140s) pans the whole field across a close
oblique shot where saddlecloth numbers ARE legible. Single decode pass collects those frames;
`hkjc.vision.mode_b.ocr._rapidocr` (RapidOCR, imported READ-ONLY) localises digit tokens; for each
high-conf in-field number I sample the jockey-silk LAB just above the number box and bridge to the
back-straight silk-colour clusters by min-LAB assignment, propagating the label backward along the
track. Result: **11/12 numbers OCR'd** (#1,2,3,5,6,7,8,9,10,11,12; only #4 missed), **10/12 tracks
bound** (the colour-stitch produced only 10 clusters). This is the genuine, non-circular
cluster→number bind 0068 declared impossible. The unlock is confirmed.

## SCALE — dimensionally fixed (y_lat in-bounds), but the longitudinal axis is unreachable
- m/px per frame from a **distance-transform perpendicular turf-strip width == C-width 21.3 m**
  (~0.0213 m/px, stable across the back-straight). y_lat is now the rail-perpendicular offset
  CLAMPED to [0, 21.3]: **max 20.35 m, median 3.5 m — in-bounds** (was overspilling to 29 m in
  0068).
- BUT total **arc-length CANNOT reach 1650 m**. The drone is a MOVING camera that pans, zooms and
  cuts while TRACKING the field, so in-frame rail-projected displacement ≠ ground displacement: the
  field-centroid rail-projected arc totals only **~379 m** for the 1650 m race, and per-track ~50 m.
  The 0067/0068 "arc inflated 3-4x" framing was wrong in kind: this is not a constant scale bug to
  divide out — a tracking/cutting drone yields NO clean full-race longitudinal axis. (Same class of
  caveat as the aerial odometer being field-shared, 0068 §6, but worse: here even the per-frame
  geometry is non-stationary.)

## THE LATERAL TRIP — NOT YET VALIDATED (§4b: mis-specification, not a true null)
The whole point of the drone (0069) is the rail-relative y_lat trip — ground saved/lost, invisible
to chart/form. It does NOT yet survive the known-race check:
- **#12 wire-to-wire RAIL leader reads WIDE** (y_lat 5.7 m, rank 8/10 rail→wide) — the opposite of
  the expected near-rail trip. #8 (mid-pack 6-2-2-6) reads TIGHTEST (1.0 m). The ordering does not
  track the official running position.
- Root causes (localised, per §4b): (a) the per-frame inner-rail reference (turf-mask boundary,
  PCA-ordered polyline) is UNSTABLE — its position/tangent jump as the mask changes (shadows,
  field occluding the boundary, the boundary leaving frame on bends), so per-frame y_lat of the
  same horse swings 0.8→6.3 m across clean back-straight frames; (b) the silk clusters are
  LOW-CONTRAST from altitude (most cluster LABs collapse to a* 114-128 / b* 130-144), so a cluster
  BLENDS several horses and y_lat is a lateral smear, and the colour bind beyond the OCR-anchored
  numbers is weak (several bind LAB-distances > 30).

## HONEST SEGMENT RELIABILITY
- TRUSTWORTHY: the finish-zoom number bridge; the dimensional scale (m/px, y_lat ∈ [0,21.3]); that
  the field IS laterally resolvable by eye on the high near-top-down back-straight (t≈35-95 s).
- NOT trustworthy: per-runner y_lat magnitude/ordering (unstable rail reference + blended clusters);
  any full-race longitudinal arc-length (moving camera); the gate-load (t<10 s, crowd/ground-crew
  clutter) and the motion-blurred run-in.

## VERDICT / NEXT (does NOT unlock CONTROL yet)
Net for the program: the drone gives a NON-CIRCULAR identity bridge and a dimensionally-sane lateral
coordinate, but the per-runner lateral TRIP — the actual control-cause surface — is not yet
extractable from this single clip with this rail reference. Before S8/CONTROL can consume a drone
y_lat, the mis-specification must be closed: a STABLE rail reference (track a single global rail
spline across frames / register frames to a fixed ground plane rather than re-fitting the turf-mask
boundary), per-HORSE track coherence (don't merge by low-contrast colour — keep the motion tracklets
and ride OCR-anchored identity onto them), and ideally a second top-down race (this clip's camera
work limits it). This is a §4b mis-specification verdict (the gap localises the error), NOT a true
null — the lateral signal is plausibly there on the back-straight, our extraction of it is not yet
faithful. Order/finish remains available cheaply from the legend/chart (0069); the drone's value is
strictly the lateral cause, which stays GATED on closing the above.

Artifacts: `data_cache/cv_s0d_drone.parquet` (2293 rows, 10 numbered tracks),
`data_cache/figs/cv_s0d_drone_qc.png` (A arc-length(t) / B y_lat trip vs official RP / C silk→number
bridge). `uv run ruff check` clean on the three study files; src/hkjc/model+core untouched,
vision/mode_b/ocr.py imported READ-ONLY.
