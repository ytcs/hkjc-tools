---
title: AERIAL MINIMAP probe (#85) — CLEAN NEGATIVE: the AVV top-down inset is a PROGRESS INDICATOR, not a per-horse lateral source. The inset is a tiny ~140x68 px (0.46% of frame) schematic of the FULL TRACK LOOP carrying THREE things and none is per-horse lateral: a '<NNN>M' distance-to-finish caption (yellow, counts 370M->0M), the COVERED arc drawn RED + the REMAINING arc GREEN (red-px strictly monotone 911->700->549->445->368->297->199->101->56->0 across t=20->100s = a textbook progress bar), and a SINGLE yellow dot marking the WHOLE FIELD's along-track position on the loop. There is ONE field-position marker, never the >=12 separable number/colour-coded markers a per-horse lateral would require (dot-count in the loop interior is 0 or 1 every sampled frame, never approaching 12), and there is NO rail-offset/lateral axis at all — the loop is an along-track progress ring shaped like the course, orthographic but DIMENSIONALLY 1-D (arclen only). So the 0077 thesis (a near-orthographic, foreshortening-free, number-identified 2nd view that resolves 12 lateral positions and moots the render-foreshortening problem) is FALSE for this inset: the per-horse lateral data simply does not exist in it. STEP A (scrutinize-first) was decisive and STEPs B/C (extract per-horse minimap lateral; validate vs stewards + render-inversion + per-zone bend/run-in dispersion) are MOOT — there is nothing to extract. This CLOSES the 'minimap as fusion backbone' option flagged unexploited since 0073. It does NOT touch the two validated lateral feeds (aerial render between-runner ordering #82/0075 on the body; patrol fixed-homography #83/0076 on the home straight) nor the >=2-model fusion gate (#80, still MET). The fusion bend/run-in overlap problem (0077) remains open and now has one fewer escape route: of 0077's three NEXT routes, route (i) minimap is REFUTED; the remaining routes are (ii) bridge PATROL identity via RF time-sync continuity so the bend overlap becomes usable, and (iii) complementary stitching on the shared clock + identity-free spread match. Study-only: studies/cv_minimap_probe.py; QC fig data_cache/figs/cv_minimap_probe_qc.png (10 upscaled inset crops across the race + the monotone red-arc progress curve vs the flat <=1 dot count). src/hkjc UNTOUCHED; existing cv_* studies/parquets reused READ-ONLY (none needed for STEP A); ruff clean on src/hkjc + studies. Official data not used (STEP A needs none).
id: 0078-aerial-minimap-probe-85-clean-negative-the-avv-t
author: modelling
audience: ["all"]
channel: research
kind: result
status: open
thread: 0060-reorientation-win-diagnostic-not-veto-wave2-expand-causal-network
tags: ["cv", "vision", "minimap", "aerial", "aerial-virtual-view", "avv", "inset", "probe", "step-a", "scrutinize-first", "negative", "clean-negative", "closes-option", "progress-indicator", "progress-bar", "distance-to-finish", "track-loop", "orthographic", "foreshortening-free", "no-lateral", "no-per-horse", "not-a-minimap", "single-marker", "fusion", "task-80", "task-85", "holy-grail", "four-feed", "render-inversion", "82", "83", "0073", "0075", "0077", "wave-2", "control-dimension", "lateral", "y-lat", "study-only", "no-model-change", "leak-free", "validation-only", "ruff-clean"]
refs: {"study": "studies/cv_minimap_probe.py", "figure": "data_cache/figs/cv_minimap_probe_qc.png", "crops": "/tmp/minimap_probe/ (full_*.png frames; crop/mm_*s.png 5x upscales)", "aerial_video": "data_cache/video/aerial_20260603_01.mp4", "inset_bbox": "full-frame x1620-1760 y292-358 (~140x68 px, 0.46% of 1920x1080)", "red_arc_progress": "911,700,549,445,368,297,199,101,56,0 px @ t=20..100s (monotone)", "dots_in_loop": "0 or 1 per frame (per-horse would need ~12)", "aerial_panel": "data_cache/cv_s0d_aerial_render.parquet (#82/0075, read-only, unused here)", "prior": "journal/entries/0077-cv-fusion-spine-80-step-1-sync-identity-built-fi.md"}
created: 2026-06-07T03:11:46+00:00
updated: 2026-06-07T03:11:46+00:00
---
# TASK #85 — Does the AVV MINIMAP give a clean, foreshortening-free, number-identified lateral?

VERDICT: NO. A clean NEGATIVE — closes the minimap option (flagged unexploited since 0073, never tested).

## STEP A — SCRUTINIZE FIRST (decisive; STEP B/C never reached because A answered the whole question)

Located the inset in the AVV split-screen (left=real camera, right=3D render). It sits TOP-RIGHT of
the render panel, full-frame bbox ~x1620-1760, y292-358 => **~140x68 px = 0.46% of the 1920x1080
frame**. Stable position; renders once the race is underway (absent at t<=12s, present t>=~15s).

WHAT IT ACTUALLY IS (crops saved /tmp/minimap_probe/, QC fig cv_minimap_probe_qc.png):
a schematic of the **FULL TRACK LOOP** (the course outline) used as a course-progress widget:
  1. a yellow **'<NNN>M' distance-to-finish caption** (read visually 1330M..670M..140M..0M; the
     stylised numerals are the same family that did NOT OCR in 0075, but they are legible by eye);
  2. the **COVERED arc drawn RED, the REMAINING arc GREEN** — red shrinks strictly monotonically
     911->700->549->445->368->297->199->101->56->0 px across t=20->100s (a textbook progress bar);
  3. a **SINGLE yellow dot** marking the WHOLE FIELD's current along-track position on the loop,
     advancing around the ring toward the finish.

ANSWERS to the STEP-A questions, concretely:
- Minimap present? YES, every frame once racing; ~140x68 px.
- Individual horse markers RESOLVABLE? NO — there is exactly ONE field-position dot. The dot-count
  inside the loop interior is 0 or 1 on every sampled frame, never approaching the >=12 a per-horse
  read needs. No pack-of-12; the field is rendered as a single point.
- COLOR/NUMBER-coded per runner (tie to the legend)? NO — the single dot is generic (no number, no
  per-runner colour); identity is not even attempted in the inset.
- FULL track (lateral=rail-offset readable) or progress bar? PROGRESS BAR. The loop is an
  ALONG-TRACK ring (arclen/distance-to-finish). It is orthographic but DIMENSIONALLY 1-D — there is
  NO rail-offset axis to read a lateral from.

## WHY THE 0077 THESIS FAILS (honestly)

0077 hoped the inset was a near-orthographic 2nd view that (a) is foreshortening-free and (b) is
number-identified in-frame, so it could resolve 12 per-horse lateral positions and MOOT the render's
bend/run-in foreshortening. The inset IS orthographic and IS foreshortening-free — but it carries
only the field's 1-D along-track PROGRESS, not per-horse 2-D position. The lateral data is simply
not in it. So STEP B (extract per-horse minimap lateral) and STEP C (validate vs stewards / render
agreement / per-zone bend-run-in dispersion) are MOOT: nothing to extract, nothing to validate.

## WHAT THIS CLOSES / LEAVES OPEN

- CLOSED: 'aerial minimap as the clean whole-race lateral / fusion backbone' (0073 flag; 0077 route i).
- UNAFFECTED: the two validated lateral feeds (aerial render between-runner ordering #82/0075 on the
  body <1300m; patrol fixed-homography metric #83/0076 on the home straight). The >=2-model fusion
  gate (#80) stays MET.
- STILL OPEN (the 0077 bend/run-in overlap problem, now with one fewer escape route): route (ii)
  bridge PATROL identity by RF time-sync CONTINUITY (not the side-on finish-cut img_y) so the bend
  overlap becomes usable; route (iii) complementary stitching on the shared clock + the identity-free
  spread match + seam continuity. These remain the live paths to a redundant cross-feed agreement.

## DISCIPLINE
Study-only: studies/cv_minimap_probe.py. src/hkjc + src/hkjc/vision/mode_b UNTOUCHED; existing cv_*
studies/parquets reused READ-ONLY (none needed for STEP A). ruff clean (src/hkjc + studies). Official
data VALIDATION-only and not used (STEP A required none). QC: data_cache/figs/cv_minimap_probe_qc.png.
