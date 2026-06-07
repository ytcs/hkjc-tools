---
title: hkjc.vision PERCEPTION promotion — feeds + shared primitives + identity ported (additive)
id: 0095-hkjc-vision-perception-promotion-feeds-shared-pr
author: modelling
audience: ["all"]
channel: research
kind: note
status: open
thread: 0060
tags: ["cv", "vision", "promotion", "reconstruction"]
refs: {}
created: 2026-06-07T14:43:32+00:00
updated: 2026-06-07T14:43:32+00:00
---
Promoted the PERCEPTION half of the validated 4-feed reconstruction from studies/ scratch into
hkjc.vision as clean, de-duplicated, race-agnostic library code. ADDITIVE ONLY — new files only; no
existing file edited. The fusion smoother / orchestrator / old-skeleton deletion / __init__ rewrite are
left to the user.

NEW FILE TREE (src/hkjc/vision/):
  geometry_cv.py      — rail primitives (shared core + 3 distinct fit strategies)
  ground_contact.py   — shadow_base + plumb_to_hoof (shared signature)
  ocr.py              — side-on/finish-cam saddlecloth OCR (dual-scale RapidOCR)
  identity.py         — Identity (consensus_assign/propagate/save/load), build_rf_anchor,
                        augment_silk_lut_from_patrol, the fixed consensus weights
  feeds/__init__.py   — FeedContext (frozen per-race scalars) + package doc
  feeds/aerial.py     — render inversion (cv_s0d_aerial_render)
  feeds/patrol.py     — fixed head-on lateral + silk identity (cv_patrol_lateral + cv_patrol_identity)
  feeds/drone.py      — panning-window lateral + clock (cv_drone_revisit)
  feeds/rf.py         — longitudinal order + finish margins (cv_rf_longitudinal)
  + tests/test_vision_recon_imports.py

DEDUP MAP (extracted helper -> feeds now importing it):
  silk descriptor (vision.silk): silk_signature/silk_distance/silk_distinctness — identity.py + patrol.py
    (killed the 3 private copies: cv_identity.silk_hist, cv_patrol_identity._hue_hist/_hist_dist).
  rail core (geometry_cv): robust_polyfit + white_rail_pixels shared by both white-rail fits.
    - fit_fixed_rail / fit_white_rail_line / fit_homography_from_rail / lateral_from_homography
      <- patrol.py (was cv_patrol_lateral._fit_white_rail/fit_fixed_rail/build_homography/project).
    - RailSpline / fit_rail_spline <- aerial.py (was cv_s0d_aerial_render.RailSpline/fit_rail_spline).
    - fit_stable_rail / rail_perp <- drone.py (was cv_drone_revisit._fit_stable_rail/_rail_perp).
  ground_contact: shadow_base <- aerial.py; plumb_to_hoof <- patrol.py (were the inline plumb in
    cv_patrol_lateral.detect_jockeys + the shadow crop in cv_s0d_aerial_render.ground_contact /
    cv_drone_revisit ground-contact); drone keeps its by+bh feet-base inline (different substrate).
  ocr (vision.ocr): ocr_numbers + best_per_number <- rf.py (merged cv_rf_longitudinal.ocr_numbers +
    cv_patrol_identity OCR; identity.build_rf_anchor uses rf.ocr_saddlecloths). The AVV legend OCR
    stays in mode_b (reused via mode_b.ocr._rapidocr — single cached engine, no second instance).
  THOROUGHBRED_LENGTH_M from hkjc.core.track (replaced rf LEN_M=2.4); track_width threaded as a param
    (replaced COURSE_WIDTH_M / C_WIDTH_M / TRACK_WIDTH_M = 21.3 baked constants).

PER-RACE CONSTANTS THREADED AS PARAMETERS (the FeedContext fields / call args, killing the F1.DIST=
module-re-point pattern):
  FeedContext: video, distance, track_width_m, field_size, sections, section_bounds, clock_offset,
    fps, windows.
  aerial: video, distance (odometer max_m + arclen scale), track_width_m, field_size; stride/ocr_every/
    marker_every now call args (were S0D_* env vars).
  patrol: clock offset + section_times/section_bounds (were PATROL_OFFSET=18.65, SEC_T/SEC_S);
    track_width_m + field_span_m for the homography/mpp (were COURSE_WIDTH_M=21.3, /15.0); T0/T1 head-on
    window -> the caller passes the frames (window auto-detect stays a study concern).
  drone: gate_vid_t / finishcut_vid_t / winner_finish_race_t (were DRONE_START_VID_T=3.0 /
    DRONE_FINISHCUT_VID_T=113.68 / WINNER_FINISH_RACE_T=98.85), sections, track_width_m, fps; windows
    are call args.
  rf: clock offset (was RF_OFFSET=1.34), sections/section_bounds, field_size, winner (margins ref),
    name2num LUT (was the baked FIELD/NAME2NUM/OFFICIAL_* tables), finish/legend windows as frame
    iterables.
  identity: build_rf_anchor takes frame iterables + name2num + rf_offset + field_size (was the cv_r03
    driver baking + the RF.* module re-point). draw passed in (consensus DRAW vote). N_STALLS / draw->
    metres NOT re-added: that START anchor lives in reconstruction.geometry.draw_to_start_lateral; the
    draw VOTE here only needs the draw order, which the caller supplies.

NOT CLEANLY SEPARABLE (left in studies for the fusion/orchestrator owner):
  - the official-HTML parser (parse_official) + all OFFICIAL_* / STEW_* / COMMENTARY validation tables
    and the faithfulness/dispersion/cross-feed-agreement batteries — validation-only, correctly DROPPED
    from the library.
  - per-race window AUTO-DETECT (patrol rail-slope-stability scan, drone straightness census) — drone
    census_straightness is ported as a reusable classifier but CHOOSING the windows + mapping them to
    arclen bands stays a study/orchestration step.
  - the decode/sampling loops are partly threaded as frame-iterable inputs ((vid_t, BGR)); aerial keeps
    its own av decode (it is one fused pass over legend+odometer+markers+dets+rail+ground-contact).
  - the affine drone clock + the per-feed clock OFFSETS are still per-race scalars the caller supplies
    (frame-verified anchors); the auto-scan that finds them is orchestration, not perception.

GREEN: ruff (src/hkjc tests studies) clean; PYTHONHASHSEED=0 pytest -q -> 189 passed (incl. the new
import/contract test). The new modules pull NO CV deps (cv2/av/rapidocr) at import — lazy inside
functions, matching the mode_b/schema lazy-load contract. (scipy/polars are pulled by hkjc.__init__
itself, same as the pre-existing test_vision.py, not by these modules.)
