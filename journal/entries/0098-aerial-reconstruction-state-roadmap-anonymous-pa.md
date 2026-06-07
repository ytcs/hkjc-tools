---
title: Aerial reconstruction — state & roadmap (anonymous paths; rigid-rail backbone next)
id: 0098-aerial-reconstruction-state-roadmap-anonymous-pa
author: modelling
audience: ["modelling", "all"]
channel: handoff
kind: decision
status: superseded
thread: 
tags: ["cv", "aerial", "rail", "reconstruction", "handoff"]
refs: {"tasks": [101, 103, 104, 105, 102], "deleted": "studies/cv_rail_geompredict.py", "promoted": ["hkjc/vision/feeds/aerial_detect.py", "hkjc/vision/geometry_cv.py"]}
created: 2026-06-07T19:42:27+00:00
updated: 2026-06-07T21:00:00+00:00
---
## Goal
From the AVV aerial render, recover ONE continuous ANONYMOUS per-horse trajectory (arclen, y_lat, t)
along the track — real front-to-back order + lateral trip. Identity is NOT done from aerial (it is
recovered later from the DRONE feed). Aerial deliverable = anonymous paths.

## Feed facts
- Split-screen; right half is the 3D render: panel = img[:, 960:] (HALF=960).
- MP4: https://streaminghkjc-a.akamaihd.net/hdflash/replay-aerial/{YYYY}/{YYYYMMDD}/{NN}/eng/replay-aerial_{YYYYMMDD}_{NN}_eng_2500kbps.mp4
  Local: data_cache/video/aerial_20260603_08.mp4 (HV 2026-06-03 R08, 1800m "C", 11 runners).
- Top-right "M TO FINISH" odometer. Pre-extracted obs: data_cache/cv_joint_aerial_R08_obs.npz
  (keys frames/odo/fcx/fcy/rails/markers/horse_gc; odo finite ~148 frames, arclen 160-1780m).
  arclen = np.interp(frame, FR[fin], ODO[fin]).
- Track geom: HV "C", circ 1450m, bends-from-finish 300-770 & 920-1450, R~159m; R08 DIST=1800m.
  studies/cv_rigid_rail.py::build_rail_geometry("HV","C") + local_radius(geom, d).
- Decode: PyAV, stride 4, keep find_layout(img).aerial_active.

## DONE & DURABLE (promoted to hkjc.vision; tested test_vision_recon_imports.py; ruff clean)
- hkjc/vision/feeds/aerial_detect.py: detect_edgefill(panel,ty0,ty1,field)->(dets_a,dets_b,debug)
  + AerialBlob(cx,cy,angle,area). Dialed-in detector (turf+rail suppress -> Canny -> fill -> ∩
  non-turf colour -> field-ROI -> PCA rail-rejection -> k-means-to-field). dets_a = pixel centroid
  (validated default). Clean one-blob-per-horse on SPREAD fields.
- hkjc/vision/geometry_cv.py: candidate_rail_curves (every white line -> its curve), fit_rail_curve
  (field-selected rigid straight/arc), project_to_polyline (-> s along-rail, signed perp).
- studies/cv_aerial_track_id.py (anonymous paths): detect -> rail -> project -> RAIL-BOUND lane
  filter (data-driven half-width 1.35*p98 field|perp|; rejects off-track bg, counts never >field)
  -> motion-gated tracklets -> per-frame s-rank front-to-back order. Diagnostics -> data_cache/diag/.

## WORKS
- Detection clean when field SPREAD (median exactly 11 in-lane, never >11; legend strip excluded).
- s-rank = genuine per-horse longitudinal order (shoulder-to-shoulder defect FIXED).
- Anonymous (s,perp) paths clean WITHIN a stable rail segment.

## BROKEN / OPEN (next agent's work)
1. RAIL BACKBONE FRAGILE (root cause of fragmented global paths). Per-frame fit_rail_curve (a)
   returns None on some frames -> HOLES, (b) re-anchors/flips image-side at bend transitions ->
   perp LEVEL-SHIFTS; so (s,perp) not in a consistent frame across the race. Inner running rail is
   OCCLUDED by the field running on it; outer rail/hedge is crisp -> "nearest line" LATCHES THE
   OUTER (wrong) rail at occluded frames (f772, f2200). APPEARANCE CANNOT disambiguate inner/outer
   (turf both sides, colour boundary ~symmetric) -> must be GEOMETRIC.
   NOTE: a 1D "predict ry@field-x from arclen" prototype (cv_rail_geompredict.py) was tried and
   DELETED. The user will RE-INITIATE the geometric-prediction piece WITH HIS GUIDANCE — coordinate
   before building. Do NOT resurrect a 1D-height shortcut: it cannot render the curved rail (drew a
   straight line, wrong on bends) and the ry≈fy calibration biases low on bunched bends.
2. DETECTION FORCES FIELD COUNT (_kmeans_field_both uses k=field) -> on a BUNCHED BEND (f2440) a
   merged 2-horse blob splits (DOUBLE-COUNT) while a faint masked-out horse VANISHES (MISS).
   Frame-local k-means can't resolve genuine overlap.
3. AWT blocked (#102): brown dirt background blends with avatars.

## ROADMAP (corrected — in order)
1. RIGID-CENTRELINE RAIL BACKBONE (task #101, the real fix). Rail = rigid body of KNOWN shape.
   Per-frame HOMOGRAPHY track-plane->image pinned to odometer arclen + rigid centreline curvature
   (local_radius) + rail-pixel/marker anchors -> draws the actual CURVED rail, survives occlusion
   (rail predicted from track position, not hidden pixels), removes side-flips, yields METRIC
   (arclen, y_lat). USER GUIDES re-init of the geometric-prediction piece — coordinate first.
2. COUPLE DETECTION TO TRACKER (task #104). Stop forcing k=field; use nearby frames jointly — split
   a merged region two tracklets enter; interpolate a momentarily-missing horse. Fixes f2440.
3. FEED backbone into cv_aerial_track_id.py (task #105) -> one hole-free consistent-frame anonymous
   (arclen, y_lat, t) trajectory per horse across the whole race.
4. Identity parked for the DRONE feed (do not attempt from aerial).

## INVARIANTS & DISCIPLINE (HARD)
- Nothing hand-tuned per race. Input = {4 video URLs + official metadata}. Only DRAW is an allowed
  pre-race initial condition; all other official data is VALIDATION-ONLY. Physical constants OK.
  Thresholds must be data-driven and said so.
- VISUALLY SCRUTINIZE raw + diagnostic images before trusting a number / claiming a fix
  (CV_RECON.md §0). This was violated once this session (over-claimed geompred "validation") — do
  not repeat.
- Fusion smoother OFF until raw backbone sound + measurements in physical ballpark (finish
  margins/order vs official; lateral in [0, track width]).
- Never mutate banked parquet. Keep ruff clean. Run PYTHONHASHSEED=0 uv run --group bayes.
  Diagnostics -> data_cache/diag/ (clear when confusing).

## KEY FILES
- hkjc/vision/feeds/aerial_detect.py        promoted detector
- hkjc/vision/geometry_cv.py                promoted rail primitives
- studies/cv_aerial_track_id.py             anonymous-path tracker
- studies/cv_rigid_rail.py                  rigid rail geometry (backbone source)
- studies/cv_s0d_aerial_render.py           track_band + render helpers
- data_cache/cv_joint_aerial_R08_obs.npz    pre-extracted odometer/field/rail obs (R08)
- studies/cv_r08_common.py                  R08 facts + official parse (draw/sections/clock)
- Tasks: #101 (backbone), #104 (detection-tracker coupling), #105 (integrate), #103 done, #102 AWT.
