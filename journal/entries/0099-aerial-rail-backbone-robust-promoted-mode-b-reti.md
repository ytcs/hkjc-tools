---
title: Aerial rail backbone robust + promoted; mode_b retired (field-median running-rail, sign-stable)
id: 0099-aerial-rail-backbone-robust-promoted-mode-b-reti
author: modelling
audience: ["modelling", "all"]
channel: handoff
kind: decision
status: open
thread: 
tags: ["cv", "aerial", "rail", "reconstruction", "refactor", "handoff"]
refs: {"promoted": ["hkjc/vision/feeds/aerial_rail.py", "hkjc/vision/aerial_layout.py", "hkjc/vision/aerial_ocr.py"], "deleted": "src/hkjc/vision/mode_b/", "tasks": [101, 104, 105]}
created: 2026-06-07T21:00:00+00:00
updated: 2026-06-07T21:00:00+00:00
---
## Headline
Aerial RAIL BACKBONE is now robust + promoted to the library, and the legacy `mode_b` package is
retired. The whole-race anonymous (s, perp) path is sign-consistent and spike-free (scale not yet
normalised). Built collaboratively with the user; supersedes the rail-fragility status in 0098.

## What changed (rail detection — the SOTA)
The old per-frame `geometry_cv.fit_rail_curve` latched the wrong line and flipped its perp sign at
bend/rotation transitions. Diagnosis (visually scrutinised, many frames): NO occlusion — the field
runs BESIDE the rail. The real nuisances are camera-angle ROTATION (rail goes diagonal/near-vertical),
BACKGROUND clutter entering on pans, and a near-parallel TWIN (the rail's shadow / a marker line).

New pipeline, promoted to `hkjc.vision.feeds.aerial_rail`:
- `line_candidates`: bright components with shape stats; THIN (rail piece) vs fat blob via
  elongation + mean width (= area/extent). Rejects field cluster, apron, painted text by SHAPE.
- `fit_curve_pca`: orientation-free robust fit (parametrise along PCA major axis, minor = poly(major)).
  Kills the x-vs-y switch that caused perp SIGN flips; handles diagonal/near-vertical rail.
- `enumerate_lines`: grows every distinct line by gathering collinear THIN pixels (dashes re-joined),
  with `coverage` (support px / length) separating real rails (~[1.5,5]) from sparse fit-artifacts
  AND the fat field blob (coverage >> 5).
- `select_running_rail`: of the two main rails (top-2 coverage), pick the one the FIELD HUGS by MEDIAN
  distance to the raw detect_edgefill blobs. NON-CIRCULAR — those blobs come from the green-turf
  `track_band`, not the rail; the median resists the scattered background false-positives the detector
  emits (the lane-FILTERED field would be circular; the raw field is not). TEMPORALLY SMOOTH: carry
  the previous pick, let field distance override only when confident (min_dist < 0.7*max_dist); skip a
  frame whose nearest rail is >180px (a bad fit). Returns a tangent-oriented curve so the perp SIGN is
  stable across the race.

## Dead ends (recorded so they are not retried)
- A per-frame rigid two-rail HOMOGRAPHY/ICP template (w=22.5m baked in) was prototyped. Concept valid
  (f1080 1.4px, clean inner/outer) but per-frame-from-scratch is NOT robust: straights are degenerate
  for an 8-DOF DLT (collinear), oblique bends fall into ICP local minima. Needs temporal init — parked.
- CONCAVITY ("inner = concave-side member") is NOT reliable: the apparent curvature sign FLIPS as the
  camera pans through a bend, so it cannot anchor the choice. (It looked good on single frames; the
  whole-bend trace refuted it.) Field-median + temporal smoothing replaced it.
- The clockwise-invariant carried in IMAGE space fails — the camera views the track from either side,
  mirroring image left/right, so a world-space hand is not a fixed image side.

## Library refactor (mode_b retired)
- `mode_b/layout.py` -> `hkjc/vision/aerial_layout.py` (Layout, find_layout).
- `mode_b/ocr.py` (+ `_read_legend_robust` & helpers from pipeline_b) -> `hkjc/vision/aerial_ocr.py`.
- `mode_b/detect.py` (detect_blobs, BlobDet) -> appended to `hkjc/vision/feeds/aerial_detect.py`.
- DELETED the legacy lateral-path pipeline (extract_lateral_paths, rail, rail_transform, reid).
- Consumers rewired: `feeds/aerial.py`, `vision/ocr.py`. `grep mode_b src/hkjc` = empty.
- Verify: vision/recon tests pass (8); ruff clean on src/hkjc/vision + the two active rail studies;
  the whole-race tracker (`studies/cv_aerial_track_id.py`) runs through the library unchanged-result
  (530 frames, median 11 in-lane).

## Diagnostics (data_cache/diag, VISUALLY SCRUTINISED)
- `cv_track_id_tracklets.png`: coherent single-sided band across the race; the bend/home-straight
  mirror flips and level-shifts are GONE; residual = scale steps (deferred) only.
- The remaining non-smoothness is PX SCALE (camera zoom) — deliberately deferred per the user.

## Validations run
- A-config (subagent, HV 2026-05-20 R02): course "A" has ONE physical rail (no moved-rail pair, no
  shadow twin). The two-main-rails logic has nothing to choose; worse, the SOLID rail's coverage
  exceeds the 5.0 ceiling and is rejected (basis="none" on a clearly-visible rail). A needs a
  single-rail branch + the ceiling raised/removed + fragment coalescing.
- The same coverage-ceiling rejection causes the mid-race f2028-2156 nmain=0 block on R08 (the solid
  rail there is rejected) — NOT live footage (f2050 is a clean render).

## OPEN ROADMAP (next, in order)
(a) RENDER-ACTIVE detector to TRIM the initial LIVE footage. The video opens full-width with the live
    TV broadcast (f0-~350: gate + early run), then switches to the split-screen render. find_layout is
    UNRELIABLE (True through the live gate). Recommended signal: the render's top-right mini-map /
    odometer widget (fixed-position, render-only); pixel-style (saturated green) did NOT separate
    cleanly. Per-frame gate (mostly a clean prefix in practice).
(b) COVERAGE-CEILING fix (raise/remove the 5.0 cap; coalesce near-collinear fragments). Recovers the
    f2028-2156 block AND is a prerequisite for A-config.
(c) A-CONFIG single-rail path (branch when only one main rail; carry temporally).
- Also: validate on a LEFT-handed track (the selector is handedness-agnostic — field-median, no
  hardcoded side — but unvalidated). Metric SCALE normalisation (px->m via known geometry) for y_lat.
  Absolute arclen still deferred.

## INVARIANTS honored
Rail selection is field-position-aware but NON-CIRCULAR (raw track_band detection, not rail-gated).
Nothing hand-tuned per race; thresholds data-driven/stated. Diagnostics scrutinised before claims.
Banked parquet untouched; ruff clean on touched code.

## KEY FILES
- hkjc/vision/feeds/aerial_rail.py     promoted rail detection + running-rail selection (SOTA)
- hkjc/vision/aerial_layout.py         find_layout (ex mode_b) — UNRELIABLE, replace in (a)
- hkjc/vision/aerial_ocr.py            odometer/legend OCR (ex mode_b)
- hkjc/vision/feeds/aerial_detect.py   detect_edgefill (+ legacy detect_blobs)
- studies/cv_aerial_track_id.py        whole-race anonymous-path tracker (uses the library)
- studies/cv_rail_robust.py            candidate-visualisation diagnostic
