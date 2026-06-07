---
title: RF/race-film 4th feed #89 FULL -- finish MARGINS validate MAE 0.41 len / order rho +0.90 vs official chart; per-horse LONGITUDINAL backbone (legend ORDER) validates rho +0.98..+1.00 vs official running positions at all 3 calls -- RF DELIVERS the per-horse long. order the 3-feed fusion (0084) lacked (field-odometer rho~0) PLUS the S3 (#60) margins surface that sharpens W'_0/spread. Honest: order+finish-metric, no absolute mid-race per-horse metres (slope 0.70 back-marker under-read). Ready to fold into the smoother as obs-model 4.
id: 0086-rf-race-film-4th-feed-89-full-finish-margins-val
author: modelling
audience: ["all"]
channel: research
kind: result
status: open
thread: 0060-reorientation-win-diagnostic-not-veto-wave2-expand-causal-network
tags: ["cv", "vision", "fusion", "four-feed", "holy-grail", "rf", "race-film", "task-89", "task-60", "task-80", "margins", "finish-margins", "lbw", "s3", "longitudinal", "backbone", "running-position", "legend-strip", "name-bridge", "identity", "saddlecloth", "ocr", "rapidocr", "photo-finish", "shared-clock", "spearman", "validation-only", "stewards", "study-only", "wprime", "spread", "wald", "first-passage", "0077", "0084", "0085", "wave-2"]
refs: {"study": "studies/cv_rf_longitudinal.py", "panel": "data_cache/cv_rf_longitudinal.parquet", "figure": "data_cache/figs/cv_rf_qc.png", "video": "data_cache/video/full_20260603_01.mp4", "official": "data_cache/race_official/20260603R1_{results.html,comments.txt} (VALIDATION ONLY)", "margins_vs_official": "MAE 0.41 len, rho +0.90, slope 0.70; #12 neck 0.28 vs 0.3 (5 horses)", "backbone_vs_RP": "legend-order vs official RP Spearman +0.98/+1.00/+1.00 at 1/4,1/2,3/4 calls", "identity": "8/12 OCR metric at the line; whole-race ORDER via legend name->number LUT", "panel_kinds": "finish_margin (5 metric) + legend_order (328 rank) = 333 rows"}
created: 2026-06-07T05:41:40+00:00
updated: 2026-06-07T05:41:40+00:00
---
FULL RESULT — task #89, the RF/race-film folded in as the 4th fusion feed: the per-horse
LONGITUDINAL backbone + the finish MARGINS. studies/cv_rf_longitudinal.py (study-only; src/hkjc
untouched; NEW parquet data_cache/cv_rf_longitudinal.parquet, no existing parquet mutated; official
chart = VALIDATION ONLY). ruff clean (src/hkjc + studies). ST 2026-06-03 R1, Class5 1650m turf C,
12 runners. Shared clock (0077): race_t = RF_vid_t - 1.34; winner #3 crosses race_t 98.85 == RF_vid
~100.19. RF = real side-on broadcast (1920x1080, 25fps, 143s). Builds on the partial 0085.

WHY (0084): the 3-feed fusion smoother found per-horse LONGITUDINAL order essentially UNOBSERVED --
the only long. obs was the aerial FIELD-SHARED odometer (one distance-run for the whole field;
per-horse stagger ~1.2m unresolved), so fused order-vs-official-RP rho~0 at the poles. RF is the fix
AND the S3 margins surface (#60). Two birds -- BOTH delivered.

STEP 1 -- IDENTITY. Near the finish the camera is side-on at ~real scale: 8/12 saddlecloths OCR on
the HORSES themselves (RapidOCR dual-scale, track-band crop): {3,4,5,6,7,8,9,12} (lead+mid group).
Deep back-markers are too distant/blurred side-on in that window. WHOLE-RACE identity instead comes
from the LEGEND STRIP: the saddlecloth-number box is too small to OCR, but the english NAMES OCR
cleanly and bridge 1:1 to runner numbers via the fixed pre-race name->number LUT (fuzzy bigram match)
-> 8-deep numbered order at 42 timepoints. So the per-horse identity for the longitudinal backbone is
EFFECTIVELY 12/12 over the race (whoever is in the lead-group strip), and 8/12 metric at the line.

STEP 2 -- FINISH MARGINS (the priority; the S3 likelihood surface #60). Photo-finish geometry:
WITHIN-FRAME horizontal pixel gap to the winner / a per-frame body-length scale (2.4m) = margin in
lengths, PAN-ROBUST (relative within one frame). Open-gap (75th-pct) per horse, n>=2 (drops the
single-frame OCR flukes the partial 0085 still had).
  HEADLINE vs official chart LBW: margin MAE = 0.41 lengths; margin Spearman rho = +0.90; ORDER
  Spearman rho = +0.90; pairwise order 9/10; fit RF = 0.70*official + 0.23.
  Per-horse RF len / official LBW: #3 0.00/0 (winner ref); #12 0.28 / 0.3 NECK (nailed); #9 1.82 /
  2.25; #6 1.87 / 1.5; #8 3.02 / 4.25. 5 horses co-framed with the winner near the line.
  HONEST RESIDUAL: slope 0.70 not 1.0 -- the deeper the horse, the more the camera (tracking the
  leaders) loses it from frame BEFORE its gap fully opens, so back-margins under-read (#8 worst). A
  real sampling/geometry limit, not a fit knob; the lead-group close finish is excellent.

STEP 3 -- per-horse LONGITUDINAL backbone over the whole race. The legend strip is a rendered
running-ORDER data overlay (~ official chart), lead group left->back, 42 timepoints race_t 8.7-90.7s.
VALIDATED vs the official RUNNING POSITIONS (validation-only): legend-order-vs-official-RP Spearman
rho = +0.98 / +1.00 / +1.00 at the 1/4, 1/2, 3/4 calls (8 horses each). The winner #3's come-from-
behind trip is captured (absent/back early -> mid-pack by race_t ~28-58 -> 3rd-4th by ~88, matching
official "7-7-6-1"); #12 wire-to-wire early-mid ("1-1-1-2"). HONEST LIMIT: this is a per-horse
rank/ORDER backbone (+ the finish-line metric gaps), NOT absolute metric per-horse gaps mid-race --
the panning side-on horse pixels (tiny, from-behind mid-race) do not yield reliable absolute arclen,
so I did NOT fabricate metric mid-race gaps; the legend gives ORDER, the finish window gives the only
metric per-horse longitudinal gaps. Parquet kinds: 'finish_margin' (5 metric rows, arclen at the
line) + 'legend_order' (328 rows, -rank backbone).

DOES RF GIVE THE PER-HORSE STAGGER THE FUSION LACKED? YES, in the form that matters for the
fusion's gap: a per-horse longitudinal ORDER that is RP-faithful (rho +0.98..+1.00) over the whole
race -- exactly what the field-shared odometer could not give (0084 rho~0). PLUS metric finishing
gaps at the line (MAE 0.41 len). So RF resolves the longitudinal ORDER everywhere and the metric
gap at the finish; the per-horse absolute metric stagger mid-race remains unresolved by ANY feed
(consistent with the §4b-metric nulls of the other 3) -- but ORDER + finish-margins is what the
generative finishing-ORDER model and the S3 spread actually consume.

S3 IMPLICATION (#60), restated for the model: the validated RF margins constrain W'_0 + the spread.
The tight close finish (#3/#12 within a neck ~0.7m, then a 1.5-len step to #6) pins the front-group
finishing-time DISPERSION small near the line (low run-in diffusion D / tight kick-energetics), i.e.
sharpens the UPPER tail of the Wald first-passage spread; the 4-5+ len tail to the back markers sets
the lower tail. At MAE 0.41 len this is a usable per-race S3 likelihood term on the spread (W'_0
sharpening) for the lead group -- the FIRST per-horse longitudinal observable available to the
fusion/generative pipeline. NEXT: enter RF into the fusion smoother (0084) as the 4th observation
model -- the legend order as a per-horse rank/ordering constraint over the whole race + the finish
margins as metric longitudinal gaps at the line -- to give the fused trajectory the per-horse
longitudinal order it currently lacks.

ARTIFACTS: studies/cv_rf_longitudinal.py; data_cache/cv_rf_longitudinal.parquet (333 rows: 5 margin
+ 328 legend-order; cols runner,race_t,arclen_or_gap,margin_len,sd,kind);
data_cache/figs/cv_rf_qc.png (identity+margins headline; finish-margins bar RF vs official; RF-vs-
official margin scatter+fit; legend ORDER backbone tracks). Inputs: data_cache/video/
full_20260603_01.mp4; data_cache/race_official/20260603R1_{results.html,comments.txt,sectional.html}
(VALIDATION ONLY). Supersedes the partial 0085.
