---
title: RF/race-film 4th feed #89 PARTIAL — finish MARGINS validate at MAE 0.41 len / order rho +0.90 vs official chart (lead group, 8/12 OCR'd side-on); the per-horse longitudinal FINISHING-GAP the 3-feed fusion lacked -> the S3 (#60) likelihood surface; slope 0.70 honest back-marker under-read (leave frame before the gap opens). STEP 3 backbone pending.
id: 0085-rf-race-film-4th-feed-89-partial-finish-margins
author: modelling
audience: ["all"]
channel: research
kind: result
status: superseded
thread: 0060-reorientation-win-diagnostic-not-veto-wave2-expand-causal-network
tags: ["cv", "vision", "fusion", "four-feed", "rf", "race-film", "task-89", "task-60", "margins", "finish-margins", "lbw", "s3", "longitudinal", "identity", "saddlecloth", "ocr", "rapidocr", "legend-strip", "photo-finish", "shared-clock", "validation-only", "study-only", "partial", "0077", "0084", "wave-2"]
refs: {"study": "studies/cv_rf_longitudinal.py", "panel": "data_cache/cv_rf_longitudinal.parquet", "figure": "data_cache/figs/cv_rf_qc.png", "video": "data_cache/video/full_20260603_01.mp4", "official": "data_cache/race_official/20260603R1_{results.html,comments.txt} (VALIDATION ONLY)", "margins_vs_official": "MAE 0.41 len, rho +0.90, slope 0.70; #12 neck 0.28 vs 0.3", "identity": "8/12 OCR side-on finish window; whole-race via legend name->number LUT"}
created: 2026-06-07T05:38:36+00:00
updated: 2026-06-07T05:41:40+00:00
---
PARTIAL (identity + finish MARGINS) — task #89, RF/race-film as the 4th fusion feed.
studies/cv_rf_longitudinal.py (study-only; src/hkjc untouched; no existing parquet mutated;
official chart = VALIDATION ONLY). ruff clean. ST 2026-06-03 R1, Class5 1650m turf C, 12 runners.
Shared clock (0077): race_t = RF_vid_t - 1.34; winner #3 crosses race_t 98.85 (1:38.85) == RF_vid ~100.19.

RF is REAL side-on broadcast footage (1920x1080, 25fps, 143s). Unlike the from-behind mid-race body
and unlike the stylised AVV, near the finish the camera is side-on at ~real scale and saddlecloth
NUMBERS OCR on the horses themselves (RapidOCR dual-scale on the track-band crop).

STEP 1 — IDENTITY. Finish-window (vid 96.5-100.5) OCR binds 8/12 saddlecloths on the horses
themselves: {3,4,5,6,7,8,9,12} (the lead+mid group). The 5 deep back-markers {1,2,10,11 + #4/7 fall
out of the margin set} are too distant/blurred side-on to OCR reliably in this window. BACKUP/whole-race
identity comes from the bottom LEGEND STRIP (STEP 3): its english NAMES OCR cleanly and bridge 1:1 to
runner numbers via the fixed pre-race name->number LUT.

STEP 2 — FINISH MARGINS (the priority deliverable; the S3 likelihood surface, #60). Photo-finish
geometry: a horse's horizontal pixel position is its along-track position; the WITHIN-FRAME pixel gap
to the winner / a per-frame body-length px scale (2.4m) is the margin in lengths -- PAN-ROBUST because
it is relative within a single frame. Aggregated open-gap (75th-pct) per horse, n>=2.
  HEADLINE vs the official chart LBW (lengths-behind-winner): margin MAE = 0.41 lengths;
  margin Spearman rho = +0.90; ORDER Spearman rho = +0.90; pairwise order 9/10; fit RF=0.70*official+0.23.
  Per-horse (RF len vs official LBW): #3 0.00/0 (winner ref); #12 0.28 / 0.3 (NECK -- nailed);
  #9 1.82 / 2.25; #6 1.87 / 1.5; #8 3.02 / 4.25. 5 horses co-framed with the winner near the line.
  HONEST RESIDUAL: slope 0.70 (not 1.0) -- back-markers leave the (leader-tracking) frame BEFORE the
  gap fully opens, so the deeper the horse the more its line-margin is under-read (#8 the worst). This
  is a real sampling/geometry limit, not a fit knob; the lead-group close finish is excellent.

S3 IMPLICATION (#60): the RF margins are a per-horse FINISHING-GAP observation the 3-feed fusion (0084)
entirely lacked (its only longitudinal obs was the FIELD-SHARED aerial odometer, per-horse stagger
~1.2m unobserved, order-vs-RP rho~0 at the poles). The margins distribution is the S3 likelihood
surface that sharpens W'_0 + the spread: a TIGHT close finish (here #3/#12 within a neck = ~0.3 len ~0.7m,
then a 1.5-len step to #6) constrains the front-group W'-burn / kick dispersion to be SMALL near the line
(low diffusion D in the run-in), i.e. it pins the upper tail of the finishing-time spread; the longer
4-5+ len tail to the back-markers sets the lower tail. Validated at MAE 0.41 len, this is a usable
per-race S3 constraint on the spread (W'_0 sharpening) for the lead group -- the first per-horse
longitudinal observable in the fusion.

STEP 3 (per-horse longitudinal backbone over the whole race) is computing now (legend-name order +
finish gaps); full entry to follow. Artifacts so far: studies/cv_rf_longitudinal.py;
data_cache/cv_rf_longitudinal.parquet; data_cache/figs/cv_rf_qc.png.
