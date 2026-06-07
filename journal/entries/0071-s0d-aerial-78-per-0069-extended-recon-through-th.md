---
title: S0d AERIAL #78 (per 0069): EXTENDED recon through the TRUE finish + 10/12 bound (up from 7) + section MAE 0.83->0.125s + HONEST y_lat NULL. Library fix: legend number-box overlay DRIFTS down toward the run-in so 0068 read it EMPTY from ~t=72s (dropping the winner surge); a band-cached robust legend reader (pipeline_b._read_legend_robust) recovers the running order THROUGH the finish (t=100 "10M": [3,12,6,2,9,8,11,4] = official top-8 EXACTLY, #3 first). Full-video pass now TRACKS the field incl. winner #3 (14 dets past 1500m) to x_arclen 1640m -> the 7->1 surge is captured by TRACKING, not just read off the legend. Sections via odometer MAE 0.125s; left-panel clock [23.05,23.75] vs official [23.30,23.75]=0.55s. START-GATE silk anchor (stalls 1-13=draw->saddle) binds all 12 as a re-id cross-check. BUT the NOVEL y_lat TRIP is an HONEST NULL and 4b says MIS-SPECIFICATION: y_lat-vs-meanRP Pearson +0.12/Spearman +0.05 (n=10), early +0.29/+0.36 LOO spans 0; de-confound localizes WHY -- within-runner frame dispersion 3.8m EXCEEDS between-runner trip spread 3.3m (ratio 0.87<1), early-third y_lat does not predict late-third (corr -0.02). Synthetic AVV sprite offset off a per-frame single-rail polyfit is too NOISY for a trustworthy lateral trip -- 0069 anticipated this. Order/finish/sections SOLVED on the aerial overlay; the control y_lat is NOT validated on the aerial substrate (the drone, real horse positions, remains the path).
id: 0071-s0d-aerial-78-per-0069-extended-recon-through-th
author: modelling
audience: ["modelling"]
channel: research
kind: result
status: open
thread: 0060-reorientation-win-diagnostic-not-veto-wave2-expand-causal-network
tags: ["cv", "vision", "mode-b", "aerial-virtual-view", "s0d", "queue-78", "legend", "legend-overlay", "legend-drift", "band-cache", "robust-legend", "reid", "saddlecloth", "10-of-12", "finish-extension", "winner-surge", "tracking", "odometer", "arc-length", "rail-relative", "y-lat", "lateral", "control", "trip", "ground-saved", "validation", "quantitative", "4b", "ceiling-check", "mis-specification", "honest-null", "de-confound", "within-between", "noise-ratio", "synthetic-sprite", "start-gate", "silk-anchor", "draw", "section-times", "left-panel-clock", "external-validation", "official-data", "spearman", "pearson", "qc-figure", "supersedes-0068", "0067", "0068", "0069", "wave-2", "0060", "no-regression", "leak-free"]
refs: {"driver": "studies/cv_s0d_aerial_finish.py", "qc": "studies/cv_s0d_qc.py", "score": "studies/cv_s0d_score.py", "library_fix": "src/hkjc/vision/mode_b/pipeline_b.py (_read_legend_robust + band cache)", "panel": "data_cache/cv_s0d_aerial.parquet (1860 rows, 10/12)", "figure": "data_cache/figs/cv_s0d_qc.png", "aerial_video": "data_cache/video/aerial_20260603_01.mp4", "official": "data_cache/race_official/20260603R1_results.html", "reframed_by": "0069-s0d-synthesis-correction-reconciles-0067-first-p"}
created: 2026-06-07T01:05:57+00:00
updated: 2026-06-07T01:05:57+00:00
---
RESULT — S0d AERIAL task #78, the REMAINING aerial work after 0069's correction. Owns
`src/hkjc/vision/mode_b/*` + `studies/cv_*aerial*` only; `src/hkjc/model|core` READ-ONLY and
UNTOUCHED; did NOT edit `studies/cv_bakeoff_s0d.py` or `studies/cv_s0d_drone.py` (the parallel
drone agent owns the latter). ruff clean on touched files; vision test passes.

## THE ROOT-CAUSE FIX (why 0068 dropped the finish + bound only 7/12)
0069 said the recon STOPS ~140m early and never tracks the winner's surge. Frame scrutiny found
the real cause: the legend NUMBER-BOX overlay row DRIFTS vertically — it sits low early (~0.78h),
rises to ~0.64h mid-race, and the single `find_layout` argmax band locked onto a DIFFERENT vivid
element late-race, so `read_legend` returned EMPTY from ~t=72s on. The detector itself fires
fine through the finish (13–14 blobs/frame to t=103); the OCR just lost the band. Fix (library,
`pipeline_b._read_legend_robust` + a per-run band cache): try the LAST-GOOD band centre first
(steady state = ONE OCR/frame), then the layout band, then a short sweep (0.60–0.72h) only if
both come up short; cache the winning centre. RapidOCR reads the digits at ~1.0 conf once the
band is right, so "most valid numbers" is a clean selector. Now the running order is recovered
THROUGH the finish: t=100 ("10M") reads `[3,12,6,2,9,8,11,4]` = the official top-8 EXACTLY
(#3 first, #12 2nd, #6 3rd).

## (1) LEGEND AS ANCHOR + START-GATE SILK ANCHOR
The legend is used as a re-id ANCHOR (binds sprite tracks → saddlecloth #) + cheap dense order
feed, NOT as the result (0069). Independent cross-check anchor added: the REAL START-GATE frame
(~t=1s, stalls 1–13 = DRAW) — `gate_silk_anchor` splits the gate into 13 cells, reads each silk's
chromatic LAB, and maps stall(=draw)→saddlecloth via the official draw map. It binds all 12 silks
(a silk↔number anchor independent of the legend). (The official draw map is used only as an
ANCHOR/validation join, never as a model input.)

## (2) RECON EXTENDED THROUGH THE TRUE FINISH — winner recovered by TRACKING
Full-video pass (stride 6, ocr_every 3; no early stop). 380 aerial-active frames, 1860 rows.
The field is now TRACKED through the run-in: every bound runner has 11–17 detections past 1500m,
and winner #3 has 14 detections to frame 2562 → x_arclen 1640m (REACHES the finish), vs 0068
dropping #3 at ~1507m. So #3's 7→1 surge is captured by TRACKING the sprite, not merely read off
the legend. (Caveat unchanged: x_arclen is the FIELD-SHARED odometer, so it gives the longitudinal
axis, not intra-field rank; the finish ORDER is the legend overlay, = the official chart.)

## (3) BINDING: 10/12 (up from 0068's 7/12)
Bound `{2,3,4,5,6,8,9,10,11,12}` — added #2, #6, #11 vs 0068 (the denser through-finish legend
gives the late-appearing closers their votes). Still missing #1 (last 10-10-12-12 throughout,
never in the legend top-8 window) and #7 (12-12-11-9, tail, look-alike silk). Honest: 12/12 not
reached; the tail-of-field binding is the known weak spot (0064), and #1 is genuinely never in
the legend window.

## (4) SECTION TIMES — much finer than 0068
Odometer-grid section MAE vs official = **0.125s** (0068 was 0.83s). Left-panel broadcast clock
gives `[23.05, 23.75]` on resolvable 400m boundaries vs official `[23.30, 23.75]` = 0.55s; the
video's own published splits `[27.65, 23.30, 23.75]` match the official exactly. The clock+
odometer are read in the SAME recon pass (no separate decode).

## (5) THE NOVEL DELIVERABLE — y_lat TRIP — is an HONEST NULL (4b: MIS-SPEC, not true-null)
Quantitative validation across ALL 10 bound runners (not a 3-horse anecdote):
- median y_lat vs mean official RP: **Pearson +0.12, Spearman +0.05** (n=10) — essentially flat.
- early y_lat vs early RP: +0.29 / +0.36 — weak positive but leave-one-out spans [−0.07, +0.38],
  i.e. NOT robust (one horse flips the sign).
The rail-saved/wide structure (0068's #12-rail/#10-wide/#3-mid anecdote) does NOT generalise:
e.g. mid/back #11 and #5 read the LOWEST y_lat (rail), while mid #2 and #8 read the HIGHEST.

§4b de-confound — true-null vs mis-specification — and it is MIS-SPECIFICATION at the
MEASUREMENT level, with the reason localized: the y_lat is not a stable trip but per-frame noise.
- within-runner frame dispersion = **3.8m** EXCEEDS the between-runner trip spread = **3.3m**
  (ratio 0.87 < 1 → noise dominates the signal).
- a runner's early-third y_lat does NOT predict its late-third: corr(third1,third3) = **−0.02**
  (third-to-third swing 2.5–10.2m, larger than the entire between-runner spread).
So the synthetic AVV sprite offset, taken off a per-frame SINGLE-rail polyfit and scaled by
sprite size (sign toward the field centroid), is too noisy to carry a trustworthy lateral trip —
the rail fit drifts, the sprites bunch at the run-in, the centroid-sign flips. This is exactly
0069's anticipation ("aerial sprites collapse at the run-in"). The lateral CAUSE is not refuted;
the AERIAL SUBSTRATE cannot measure it. The drone (REAL horse positions) remains the path to a
numbered lateral read — the parallel agent's #79.

## HONEST SCORECARD (aerial, ST 2026-06-03 R1)
| signal | result | source | verdict |
|---|---|---|---|
| finish ORDER | #3,#12,#6,... = official top-8 | LEGEND overlay (≈ official chart) | SOLVED, not novel |
| winner #3 surge | tracked to 1640m (14 dets past 1500m) | TRACKING (new) | RECOVERED |
| binding | 10/12 (was 7/12) | legend anchor + gate silk anchor | improved, not 12/12 |
| section times | odometer MAE 0.125s; clock 0.55s | odometer + left clock | SOLVED (finer) |
| **y_lat TRIP (novel)** | **Pearson +0.12, NULL** | rail-relative sprite offset | **NOT validated (mis-spec)** |
| per-pole arc-timing | mean ρ −0.11 (noise) | field-shared odometer | noise (0069 confirmed) |

## WHAT I CHANGED
LIBRARY: `src/hkjc/vision/mode_b/pipeline_b.py` — `_read_legend_robust` (band-cached, drift-
robust legend OCR) replacing the single-band read; per-run legend band cache threaded through the
OCR loop. NEW STUDY: `studies/cv_s0d_aerial_finish.py` (full-video finish-extension driver +
start-gate silk anchor + 4b y_lat validation + left-panel clock sections). REWROTE
`studies/cv_s0d_qc.py` (finish-extended arc-length, y_lat-vs-RP scatter with the within/between
noise-ratio, per-pole ρ) and updated `studies/cv_s0d_score.py` (dynamic binding count, honest
legend-vs-geometry note). WROTE `data_cache/cv_s0d_aerial.parquet` (1860 rows, 10/12),
`data_cache/figs/cv_s0d_qc.png`.

## NEXT / GATING
Order/finish/sections on the aerial overlay are done and not novel. The CONTROL (lateral) cause is
GATED on a VALIDATED lateral reconstruction, which the AERIAL synthetic substrate cannot supply
(measurement mis-spec, de-confounded above). The novel signal must come from the DRONE (real
horses, #79) — the start-gate + legend numbers are the natural anchor to tie its real-horse tracks
to identities (fusion, #80). Aerial OCR cost note: the band-cached reader keeps the steady state
at ~1 OCR/frame, but a sparse stride still re-sweeps; a coarse corpus backfill should pin the
band per-clip once and reuse it.
