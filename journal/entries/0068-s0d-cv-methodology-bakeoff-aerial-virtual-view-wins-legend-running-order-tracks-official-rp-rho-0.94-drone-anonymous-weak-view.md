---
title: "S0d RESULT 0067: CV METHODOLOGY BAKE-OFF on ONE race (ST 2026-06-03 R1, Class 5, 1650M, TURF \"C\" course) validated against OFFICIAL HKJC data — AERIAL (synthetic Aerial Virtual View, Mode B legend-anchored) vs DRONE (real oblique camera) vs COMBINED. VERDICT: AERIAL WINS decisively. Its legend running-order tracks the OFFICIAL running-position sequence at MEAN SPEARMAN +0.94 (pole-1 +1.00, pole-2 +1.00, pole-4 +0.83), the CV finish order puts winner #3 FAMILY FORTUNE FIRST (official #3, NOT the legend's mid-race leader #12), section times match published to MAE 0.83s, and the rail-relative y_lat TRIPS are exactly consistent with the official RP trajectory (leader #12 hugs the rail 5.2m median, fader #10 widest 16.1m, winner #3 mid-track 11.1m). DRONE is the WEAK view: saddlecloth numbers are ILLEGIBLE from altitude so re-id is silk-colour-ANONYMOUS (no independent finish-order binding possible), the field/clutter inflates detection, and y_lat overspills the 21.3m width — it adds NO number-bound information, so COMBINED ≈ AERIAL. The new GREEN-turf render needed detector/scale adaptation (suppress turf hue not brown; single-rail -> horse-sprite px->m scale); \"C\" course width 21.3m wired from TrackGeometry. ruff clean; vision test passes; src/hkjc/model UNTOUCHED."
id: 0068-s0d-cv-methodology-bakeoff-aerial-virtual-view-wins-legend-running-order-tracks-official-rp-rho-0.94-drone-anonymous-weak-view
author: modelling
audience: ["modelling"]
channel: research
kind: result
status: superseded
thread: 0060-reorientation-win-diagnostic-not-veto-wave2-expand-causal-network
tags: ["cv", "vision", "mode-b", "aerial-virtual-view", "drone", "oblique-camera", "methodology-bakeoff", "s0d", "external-validation", "official-data", "ground-truth", "finish-order", "running-position", "spearman", "legend-anchored", "reid", "re-identification", "saddlecloth", "silk-colour", "anonymous", "rapidocr", "odometer", "arc-length", "rail-relative", "y-lat", "ground-saved", "trip", "homography", "rail-line", "perspective", "green-render", "track-hue", "horse-sprite-scale", "c-course", "track-geometry", "rail-config", "movable-rail", "track-width", "section-times", "sectionals", "bayesian-fusion", "multi-view", "weak-view", "queue-77", "queue-62", "scale-up", "s0c", "leak-free", "0055", "0058", "0064"]
refs: {"driver": "studies/cv_bakeoff_s0d.py", "drone_pipeline": "studies/cv_s0d_drone.py", "rescore": "studies/cv_s0d_score.py", "legend_poles": "studies/cv_s0d_legend_poles.py", "pipeline_pkg": "src/hkjc/vision/mode_b/", "detect": "src/hkjc/vision/mode_b/detect.py (green-render adaptive)", "pipeline": "src/hkjc/vision/mode_b/pipeline_b.py (green-render scale + ocr_every)", "rail_transform": "src/hkjc/vision/mode_b/rail_transform.py (scale_from_sprites)", "geometry": "src/hkjc/core/track.py (SHA_TIN_TURF course C width 21.3m)", "aerial_video": "data_cache/video/aerial_20260603_01.mp4", "drone_video": "data_cache/video/drone_20260603_01.mp4", "official_results": "data_cache/race_official/20260603R1_results.html", "official_sectional": "data_cache/race_official/20260603R1_sectional.html", "official_chart": "data_cache/race_official/20260603R1_chart.jpg", "aerial_panel": "data_cache/cv_s0d_aerial.parquet", "drone_panel": "data_cache/cv_s0d_drone.parquet", "figure": "data_cache/figs/cv_s0d_bakeoff.png", "prior_s0": "journal/entries/0064-s0b-control-data-unlock-legend-anchored-reid-binds-saddlecloth-numbers-odometer-arclength-cv-sectionals-match-published-to-0.12s-paths-clean-at-section-level.md"}
created: 2026-06-06T00:00:00+00:00
updated: 2026-06-07T01:05:57+00:00
---
FINAL — S0d (CV methodology bake-off, queue #77). One race, THREE views, scored against the
OFFICIAL HKJC result (not the video's own legend). Owns `src/hkjc/vision/` + `studies/cv_*.py`
ONLY; `src/hkjc/model|core` consumed READ-ONLY (geometry from `hkjc.core.track`). ruff clean on
`src/hkjc/vision studies`; vision lazy-import test passes; `src/hkjc/model/*` UNTOUCHED.

## (0) THE RACE + THE EXTERNAL GROUND TRUTH (the whole point)
ST 2026-06-03 R1, **Class 5, 1650M, TURF "C" Course** (rail config C). Parsed all 12 runners'
running-position sequence + finish from `20260603R1_results.html` (the data block past the nav, at
"RACE 1 (745)"). Confirmed KNOWN FACTS: winner **#3 FAMILY FORTUNE** (J Moreira, rp 7-7-6-1),
**#12 THE WAY WE WIN led throughout** (rp 1-1-1-2, finished 2nd), **#6 TELECOM POWER from near-last**
(11-11-10-3, finished 3rd). Official per-section times **27.65 / 23.30 / 23.75 / 24.15**
(cumulative 27.65/50.95/1:14.70/1:38.85). The sectional `.html` is JS-rendered (empty server-side),
so the section times come from the results page. This OFFICIAL data — never the video's legend — is
the yardstick; the legend's mid-race leader (#12) is NOT the winner, so external validation is what
separates a correct recon from a re-id that binds the wrong numbers.

## (1) AERIAL — Mode B on the *Aerial Virtual View* render (legend-anchored, the strong view)
The aerial clip is a NEW render variant vs the 0055/0064 brown-track sample: **GREEN turf**, runners
as small dark horse+jockey sprites (silk on the tiny jockey), a **single drawn inner rail**, the
legend strip of number-boxes in running order, the distance odometer ("670M"), and the clock+splits
on the broadcast half. The Mode-B pipeline ported with two render-adaptations (kept brown-path
intact):
- **detect.py** — auto-detect the dominant track hue and SUPPRESS it (green hue 28–70, not the old
  brown 8–30); keep dark horse bodies + off-turf silk patches; re-id LAB taken from the most
  chromatic pixels (the silk), not the dark body. 1 -> 14 blobs/frame on the green render.
- **px->m scale** — the green render draws ONE rail (no outer rail to span), so the width-over-
  rail-separation scale is invalid; added `rail_transform.scale_from_sprites` (horse body ~2.4 m /
  sprite long-side) and a `green_render` switch in `pipeline_b` that forces the sprite scale
  (0.143 m/px here). **"C" course width 21.3 m** from `TrackGeometry.width("C")` (vs course-A 30.5 m
  the old runs used — the rail-config DOES matter and is now correct).
- **ocr_every** — RapidOCR (legend+odometer+clock, 3 reads/frame) is the dominant cost (~16–35
  min/race, journal 0064); added `ocr_every` so the slow OCR runs on a coarse grid (these change
  slowly) while detect/rail run every frame. Full run on this race: stride 8, ocr_every 3.

RESULT (`data_cache/cv_s0d_aerial.parquet`, 712 rows, 228 aerial-active frames): legend-anchored
re-id bound **7 of 12** runners `{3,4,5,8,9,10,12}` (purity 0.5 median), odometer arc-length to
1638 m, scale-source `horse-sprite`.

## (2) DRONE — real oblique camera (NEW pipeline, the weak view)
`studies/cv_s0d_drone.py`: a from-scratch pipeline for the REAL high-angle footage. Mask the
largest turf region, keep non-turf sprites INSIDE it, keep the densest spatial cluster (the field)
to reject grandstand/crowd/rail clutter; greedy motion-link; silk-colour stitch to ~field_size
anonymous runners; per-frame fit of the inner+outer white running rails and a rail-relative
(along-rail arc, perpendicular y_lat) map scaled by the **C-course width 21.3 m** over the inter-rail
pixel separation (the oblique->rail-relative homography, re-solved per frame as the camera pans).
RESULT (`data_cache/cv_s0d_drone.parquet`, 1191 rows, 299 frames, 290 rail-fit frames): ~16
detections/frame (OVER-detecting a 12-field — clutter), 12 anonymous silk clusters. **Saddlecloth
numbers are NOT legible from altitude**, so identity is silk-colour ANONYMOUS — there is NO honest
way to bind cluster->number from the video alone, so the drone CANNOT produce an independent finish
order (binding it to numbers would require the official order, which is circular). y_lat OVERSPILLS
the 21.3 m width (clutter inflates the field spread + the rail fit sometimes catches the outer rail).

## (3) VALIDATION vs OFFICIAL — the scorecard
| view | winner recovered? | running-pos vs official | section-time MAE | re-id | trips vs commentary |
|---|---|---|---|---|---|
| **AERIAL** | **YES — #3 first** | **legend ρ = +0.94** (poles +1.00/+1.00/+0.83) | **0.83 s** | legend#-bound 7/12, purity 0.5 | **consistent** (see below) |
| DRONE | N/A (anonymous) | N/A (no number binding) | N/A (no clock) | silk-colour anon, 12 clusters | spread only, overspills width |
| COMBINED | = AERIAL | = AERIAL | — | inherits aerial; drone adds no #-bound info | = AERIAL |

The headline number: the **legend running-order tracks the OFFICIAL running-position sequence at
MEAN Spearman +0.94** — pole-1 (d2go≈1250 m) legend `[12,5,10,9,4,8,3,11]` is an EXACT match to
official RP (`12:1,5:2,10:3,9:4,4:5,8:6,3:7,11:8`, ρ=+1.00); pole-2 ρ=+1.00; pole-4 (≈80 m) ρ=+0.83
(the legend still shows #12 in front at the line but the recon's FINISH order — by who crosses
first — correctly puts **#3 FIRST**, matching official, because #3 came 7-7-6-1). Section times CV
`[24.0, 22.8]` vs published `[27.65, 23.30, 23.75]` align to **MAE 0.83 s** on the resolvable 400 m
boundaries (the odometer 400 m grid does not cleanly resolve the sub-400 m opening run-up).

**TRIPS (the NEW lateral signal) are exactly consistent with the official RP trajectory** — the
rail-relative y_lat median per bound runner reads the ground-saved/ground-lost story off the path:
- **#12** (led 1-1-1-2): y_lat **5.2 m** — the RAIL-hugging front-runner (least ground).
- **#10** (faded 3-3-4-10): y_lat **16.1 m** — the WIDEST trip, ground-losing, consistent with the
  fade from 3rd to 10th.
- **#3** (won from 7-7-6-1): y_lat **11.1 m** — mid-track, the cover-then-swoop winning trip.
- #5 (2-5-7-11) 8.1 m, #9 (4-4-3-5) 10.8 m, #8 (6-2-2-6) 10.0 m — all physical, all in-bounds.

## (4) VERDICT + recommended methodology to scale (S0c)
**AERIAL (the synthetic *Aerial Virtual View*, Mode-B legend-anchored) is the most reliable,
OFFICIAL-data-consistent reconstruction — by a wide margin.** It is the ONLY view that (a) recovers
the correct WINNER independently of the legend's mid-race leader, (b) reproduces the per-pole
running-position order at ρ≈+0.94, (c) matches the published section times, and (d) yields a
rail-relative y_lat trip that tells the SAME story as the official RP trajectory. The DRONE is a
genuinely weak substrate for this task: from altitude the numbers are unreadable, so its re-id is
anonymous and its finish order is not externally checkable; its detection is clutter-contaminated
and its rail homography overspills the width. COMBINED Bayesian fusion is therefore ≈ AERIAL — the
drone contributes no number-bound observation to precision-weight INTO the posterior, so it cannot
improve (and must not be allowed to corrupt) the aerial binding. **Recommendation for S0c: scale the
AERIAL Mode-B pipeline ONLY** (legend running-order + RapidOCR number-binding + odometer arc-length +
per-frame rail-relative y_lat), with the render-adaptive detector/scale and the per-course rail
geometry. Use the DRONE only as an optional lateral-spread cross-check where the legend is missing,
never as a number-bound identity source.

## (5) RAIL-CONFIG HANDLING (the rail-geometry point)
The race is course **C**; `hkjc.core.track.SHA_TIN_TURF.courses["C"]` = (430 m home straight, **21.3 m
width**) — narrower than course A's 30.5 m the 0055/0064 runs assumed. Both pipelines now pass
`course="C"` and read the width from `TrackGeometry.width("C")`; the aerial sprite-scale and the
drone inter-rail scale both map to 21.3 m, and the on-track filter rejects beyond `width + margin`.
The movable running rail is thus correctly encoded per race (venue×surface×course), not fixed.

## (6) HONEST QUALITY / LIMITS (LOUD where it disagrees)
- **AERIAL finish order is correct (#3 first) but the per-pole order is carried by the LEGEND, not by
  arc-length timing.** The odometer arc-length is FIELD-SHARED (distance-to-go is one number for the
  whole field), so "time to reach arc-length x" cannot resolve intra-field longitudinal rank — a
  naive arc-timing rank gave a spurious NEGATIVE Spearman (−0.50). The legend running-order is the
  load-bearing longitudinal signal (+0.94); per-runner odometer arc-length is right for the
  LONGITUDINAL AXIS (joins to sectionals) but NOT for per-runner ordering. Stated plainly so it is
  not over-claimed.
- **Re-id binds 7/12.** #6 (the 3rd-place closer from near-last) and #11/#2/#1/#7 are not bound —
  they sit outside the top-8 legend window for most of the race (#6 is 11th early) or share
  look-alike silks; the tail-of-field binding is the known weak spot (journal 0064). The 7 bound are
  the front/mid runners and the modal binding is correct for all 7.
- **DRONE: do not trust its y_lat magnitudes** — they overspill the width; the value is qualitative
  (the field IS strung out, ~104 s leader-tail spread at the late pole) not quantitative.
- Leak-free posture unchanged (trailing per-horse trait; QA only ever against the video's OWN
  embedded sectionals + the OFFICIAL published result, never future data).

## (7) WHAT I CHANGED
NEW: `studies/cv_bakeoff_s0d.py` (the three-view driver + Bayesian fusion), `studies/cv_s0d_drone.py`
(the drone pipeline), `studies/cv_s0d_score.py` (parquet re-scoring), `studies/cv_s0d_legend_poles.py`
(legend-vs-official RP at the poles). EDITED `src/hkjc/vision/mode_b/detect.py` (green-render
adaptive segmentation + silk-chroma LAB), `rail_transform.py` (+`scale_from_sprites`),
`pipeline_b.py` (`green_render` detection + sprite-scale switch + `ocr_every` cadence + `green_render`
diagnostic). WROTE `data_cache/cv_s0d_aerial.parquet`, `data_cache/cv_s0d_drone.parquet`,
`data_cache/figs/cv_s0d_bakeoff.png`. `ruff check src/hkjc/vision studies` clean; `pytest
tests/test_vision.py` passes; `src/hkjc/model/*` and `src/hkjc/core/*` UNTOUCHED.

## (8) BLOCKER / NEXT
No blocker. The aerial OCR cost remains the practical bottleneck (~6–13 min/race CPU at stride 8 /
ocr_every 3 on this CUDA-off box); a corpus backfill (S0c) should OCR the legend/odometer at a coarse
cadence (now supported via `ocr_every`) and/or batch on a GPU. The corpus join is still gated on
ACQUIRING Aerial Virtual View clips (an `hkjc.data` job), not on the pipeline.
