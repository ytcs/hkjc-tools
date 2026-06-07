---
title: task #99 diagnostic walkthrough video — 4-feed fusion reconstruction made legible (R01 + R08, bend metre-gap exhibit)
id: 0096-task-99-diagnostic-walkthrough-video-4-feed-fusi
author: modelling
audience: ["all"]
channel: research
kind: note
status: open
thread: 0060
tags: ["cv", "fusion", "recon", "walkthrough", "video", "visualization", "bend", "lateral", "task99"]
refs: {}
created: 2026-06-07T14:52:07+00:00
updated: 2026-06-07T14:52:07+00:00
---
TASK #99 — a DIAGNOSTIC WALKTHROUGH VIDEO that makes the 4-feed fusion reconstruction legible, especially its key difficulty (the bend absolute-lateral-metre gap, CV_RECON s7). Study module studies/cv_walkthrough_video.py (ruff-clean). Two mp4s shipped.

ARTIFACTS:
  data_cache/figs/cv_recon_walkthrough_R01.mp4  (248 frames, 12 fps, 20.7s, 1920x1080) — the 12-horse mile, most instrumented.
  data_cache/figs/cv_recon_walkthrough_R08.mp4  (281 frames, 12 fps, 23.4s, 1920x1080) — the 1800m route, the bend metre-gap exhibit.
  Per-feed overlay caches (decode-once detector passes, re-render is cheap): data_cache/cv_walkthrough_cache_{R01,R08}_{aerial,patrol,drone}.npz.

COMPOSITE LAYOUT (each frame = one shared-clock race_t). CENTER (largest): the top-down RECONSTRUCTION — track box from the rail + the 22.5m C-course width (#96 hard box), every horse plotted at its fused (arclen->distance-along-track, y_lat) with a number, an uncertainty ELLIPSE (sd_arclen x sd_ylat straight from the physkernel parquet #96+#97), and a fading tail. HUD: race_t, field arclen, race PHASE (gate/straight/BEND/finish from the #96 bend_fraction_map geometry), and WHICH INSTRUMENTS ARE LIVE this instant (e.g. "aerial:ORDERING-only(bend) patrol:OUT(panning) drone:live RF:live draw-anchor"). On a bend the HUD spells out WHY the ellipses balloon (render foreshortening -> homography under-determined -> absolute lateral metre unconstrained, box-capped at ~W/sqrt12). FOUR side panels = the synced SOURCE frames, each pulled at ITS OWN video time via the shared-clock offset (CV_RECON s3), overlays on the actual pixels: AERIAL render (rail spline + ground-contacts + odometer arclen readout; homography lane grid SOLID on straights, drawn BROKEN/red + "HOMOGRAPHY UNDER-DETERMINED" on bends — the key-difficulty panel); PATROL head-on (fixed rail + metric lane grid + silk ground-contacts; greyed "NO COVERAGE — camera panning" outside the head-on window); RF/race-film (cached legend ORDER + finish-MARGIN gauge near the line); DRONE (rail reference + cached numbered ordering cross-check).

CLOCK OFFSETS (all REUSED, not re-tuned): R01 RF race_t=vid-1.34, patrol vid-18.65 (117.5 winner-cross), drone (vid-3.0)*0.8931. R08 RF vid-1.08 (109.7 cross), patrol vid-12.08 (120.7 cross), drone (vid-3.0)*108.62/117.5. Geometry/phase + W=22.5 from cv_fusion_constrained; aerial detectors from cv_s0d_aerial_render; patrol rail+silk from cv_patrol_lateral; RF/drone numbered from their cached parquets.

HONESTY LABELS: aerial + patrol overlays are FRESHLY DETECTED at the exact frame (labelled "live detect"); RF order/margins + drone numbered ordering are read from the cached per-feed panels at the nearest race_t (labelled "cached") because re-running their OCR/saddlecloth detectors across the clip is not cheap and would not change the pedagogy. The bend homography is drawn BROKEN exactly where the algorithm genuinely has none — never a confident grid on a bend.

GREYED PANELS / HONEST GAPS:
  - PATROL is greyed outside its head-on window on BOTH races (it only covers a short head-on band: R01 race_t ~64-77s, R08 ~86-100s) — this absence TEACHES the partial-sensor structure, kept deliberately. R08 patrol rail slope_cv 0.39 (vs R01 0.03): the route head-on window is messier (panning bleed), noted on-panel.
  - AERIAL: every grid point resolved a render frame via the odometer->arclen->nearest-frame map (R01 248/248, R08 281/281). The R08 odometer is sparse (cached x_arclen ~11% of frames, 7/11 runners legend-bound) so some R08 aerial frames are the NEAREST odometer-tagged frame rather than the exact race_t frame — the rail/contacts are still freshly detected, the arclen readout is the odometer's own.
  - DRONE rail is a crude reference line only (the drone is metric-null, CV_RECON s2) — labelled "rail=reference", never used for metres.

BEND DIFFICULTY — IS IT VISIBLY CONVEYED? YES, in R08. The route's bend arclen zones are maroon-shaded in the center panel; the fused lateral ellipses are visibly WIDER there and the aerial panel flips to the broken-grid "UNDER-DETERMINED" banner while the HUD names the cause. Quantified from the physkernel parquet: median sd_ylat BEND vs STRAIGHT = R01 1.7m vs 0.9m, R08 3.1m vs 2.2m (the box caps the raw route bend sd ~52m at ~W/sqrt12; the clip shows the capped-but-still-wide bend ellipse, honestly). The clip doubles as the CV_RECON s7 exhibit.

REGENERATE: PYTHONHASHSEED=0 uv run python studies/cv_walkthrough_video.py {R01,R08}. First run does the decode-once detector passes (~minutes, the slow part is the aerial render detect across the clip); caches persist so a re-render (overlay/HUD tweaks) is ~seconds-to-a-couple-minutes of compositing only. No banked parquet mutated; official data VALIDATION-ONLY; draw is the allowed t=0 initial condition.
