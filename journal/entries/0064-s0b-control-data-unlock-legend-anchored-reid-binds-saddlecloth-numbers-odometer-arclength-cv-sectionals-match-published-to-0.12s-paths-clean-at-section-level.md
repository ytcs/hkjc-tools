---
title: "S0b RESULT 0063: the CONTROL dimension's DATA UNLOCK — solved the 0055 re-id bottleneck by LEGEND-ANCHORED number-binding. The rendered legend lists saddlecloth #s in current RUNNING ORDER every frame (RapidOCR, 0058); fusing that order + along-track rank (global orientation anchored on the leader silk) + silk-LAB + motion BINDS each tracked blob to a stable SADDLECLOTH NUMBER — the panel is no longer anonymous (rid{N}→saddlecloth#, the join key). The ODOMETER ('420M') is OCR'd per-frame for a TRUE per-runner arc-length axis (replaces the 0055 centroid-integral proxy). QA / KILL-CRITERION PASSES: CV-derived section times [22.6,23.59,24.0] match the video's own PUBLISHED splits [22.76,23.78,24.01] to MAE 0.12s / max 0.19s; CV finish order [9,10,6,8,2,4,11,3] exact. Re-id: 9/11 runners bound (vs 0055 anonymous), vote-purity 0.49 median (leader #9 0.72), per-frame |dy|=0.63m median BUT p90≈5m — residual swaps remain, so per-frame paths are NOT pristine; SECTION-LEVEL median-y_lat profiles ARE clean+physical (winner #9 hugs the rail 8.5→6.2→8.8→11.8m; #7 wide-trip 27.5m) = the usable ground-saved/control signal for S8/S9. Panel: 1,140 rows × 14 cols, 100% on-track. src/hkjc/model UNTOUCHED."
id: 0064-s0b-control-data-unlock-legend-anchored-reid-binds-saddlecloth-numbers-odometer-arclength-cv-sectionals-match-published-to-0.12s-paths-clean-at-section-level
author: modelling
audience: ["modelling"]
channel: research
kind: result
status: final
thread: 0046-causal-rebuild-design-cause-not-effect-strip-metasignals-reemergence-validation
tags: ["cv", "vision", "mode-b", "aerial-virtual-view", "control-cause", "lateral-path", "y-lat", "ground-saved", "rail-relative", "reid", "re-identification", "legend-anchored", "saddlecloth", "number-binding", "running-order", "silk-colour", "lab", "motion-linkage", "global-orientation", "leader-anchor", "rapidocr", "odometer", "arc-length", "true-arclength", "clock", "published-splits", "sectional-qa", "kill-criterion", "finish-order", "section-level", "purity", "swap", "wave-2", "s0", "s0b", "s8", "s9", "queue-62", "join-key", "trajectory-dataset", "leak-free", "0055", "0058", "0060"]
refs: {"pipeline_pkg": "src/hkjc/vision/mode_b/", "reid": "src/hkjc/vision/mode_b/reid.py (assign_by_legend)", "ocr": "src/hkjc/vision/mode_b/ocr.py (read_legend/read_odometer/read_clock_splits)", "layout": "src/hkjc/vision/mode_b/layout.py (odo_roi/clock_roi)", "pipeline": "src/hkjc/vision/mode_b/pipeline_b.py", "driver": "studies/cv_path_extraction.py", "panel_out": "data_cache/cv_lateral_paths_sample.parquet", "figure": "data_cache/figs/cv_lateral_paths_sample.png", "video": "data_cache/aerial_sample.mp4", "join_target": "data_cache/trajectory_dataset_full.parquet", "qa_published": "data_cache/results_rich.parquet", "prior_s0": "journal/entries/0055-s0-cv-lateral-path-pipeline-mode-b-aerial-virtual-view-rail-relative-y-lat-extracted-prototype-reid-is-the-bottleneck.md", "ocr_bakeoff": "journal/entries/0058-s0b-ocr-bakeoff-on-our-aerial-virtual-view-legend-rapidocr-pp-ocrv5-wins-100-percent-f85-incl-two-digit-10-easyocr-tesseract-pruned.md", "reorientation": "journal/entries/0060-reorientation-win-is-a-diagnostic-of-causal-coverage-not-a-veto-longitudinal-is-one-causal-dimension-reject-form-as-measurement-wave2-expand-the-causal-network.md"}
created: 2026-06-06T00:00:00+00:00
updated: 2026-06-06T00:00:00+00:00
---
FINAL — S0b (the CONTROL dimension's data unlock; Wave-2 frontier per 0060). Turns the 0055
anonymous prototype into a per-runner LATERAL-PATH panel keyed by SADDLECLOTH NUMBER, by
solving the re-id bottleneck. Owns `src/hkjc/vision/` + `studies/cv_*.py` ONLY;
`src/hkjc/model/` (the concurrent field.py agent) and the corpus panels are UNTOUCHED
(geometry + join targets consumed READ-ONLY). ruff clean on `src/hkjc/vision studies`; full
suite **181 passed**.

## (0) THE DELIVERABLE & WHY (per 0060)
0060 set the CONTROL cause (jockey) — the CV LATERAL path — as the Wave-2 frontier: an
orthogonal signal NOT in the longitudinal sectionals and invisible to form's aggregate. 0055
extracted a clean rail-relative `y_lat` but the panel was ANONYMOUS (`rid{N}` silk clusters,
~52% colour-only stability) and `x_arclen` was a centroid-integral PROXY — neither joinable
nor per-runner. This task bound the identity to the saddlecloth NUMBER and put `x` on a true
arc-length axis, so S8 (control latent) / S9 (draft/geometry) can be built on real paths.

## (1) THE RE-ID SOLVE — legend-anchored number-binding (`reid.assign_by_legend`)
The render's bottom legend lists the saddlecloth numbers in CURRENT RUNNING ORDER, left→right,
EVERY frame (read by `ocr.read_legend`, the RapidOCR winner of 0058). That running order is the
STRONG anchor — it is *rendered*, not inferred. The method (the user's confirmed recipe: crop
per-region, OCR per-crop, never full-frame):
1. **Motion link** blobs → tracklets (existing `link_tracklets`).
2. **Per frame**, order the detections ALONG-TRACK (PCA major axis = local track tangent) and
   align that ordering to the legend; cast a (tracklet → saddlecloth#) vote per detection.
3. **GLOBAL orientation** (the load-bearing fix): the PCA sign flips frame-to-frame; left alone
   it was the dominant noise source (purity 0.43, and the QA mis-aligned at MAE 3.3s). I
   bootstrap the LEADER's silk-LAB (the densest colour recurring at an along-track extreme —
   the leader, #9, is stable all race) and orient every frame so rank-0 is the extreme matching
   the leader colour. This single change lifted purity 0.43→0.49 and (with the QA align fix)
   dropped section-time MAE 3.30s→**0.12s**.
4. **Bind** each tracklet to its modal voted number; **greedy one-number-per-runner bijection**
   (highest-vote/longest claims first); fold leftover fragments by silk colour.
5. **Silk-LAB consistency cleanup**: a runner's silk colour is stable, so a folded detection
   far from the runner's own robust LAB centroid is a swap — dropped. This is what cleans the
   per-runner PATH (rows 1,206→1,140; lifts section-level robustness).

The colour-only clusterer (`reid_by_colour`, the 0055 prototype) is RETAINED as the fallback
for clips where the legend is unreadable.

## (2) TRUE ARC-LENGTH — odometer OCR (`ocr.read_odometer`, `layout.odo_roi`)
The distance-to-go odometer ("420M", top-right map box) reads at conf≈1.0 nearly every frame
(1630M→20M monotone over the clip). `x_arclen = distance_m − dist_to_go`, monotonicity-cleaned
(reject OCR back-jumps). This REPLACES the 0055 centroid-integral proxy with a true per-frame
per-runner longitudinal axis. `diagnostics["arclen_source"] = "odometer"`; 97% of steps
non-decreasing.

## (3) QA / KILL-CRITERION — CV sectionals vs the video's OWN published splits: PASS
The LEFT (broadcast) panel prints HKJC's official clock ("1:23.23") and cumulative sectional
splits as each boundary is passed (`ocr.read_clock_splits`, `layout.clock_roi`). Reading the
clock at each 400 m odometer boundary gives CV-derived section times; `_best_align` matches them
to the published list (which leads with a sub-400 m "START-1600M" run-up the 400 m grid can't
resolve):

| | section A (1600→1200) | B (1200→800) | C (800→400) |
|---|---|---|---|
| CV-derived | 22.60 s | 23.59 s | 24.00 s |
| published (video) | 22.76 s | 23.78 s | 24.01 s |
| |Δ| | 0.16 | 0.19 | 0.01 |

**Section-time MAE 0.12 s, max 0.19 s.** **CV finish/running order [9,10,6,8,2,4,11,3] is
EXACT** (matches the 0058-recorded finish, leader #9 throughout). The arc-length axis and timing
are physically correct — the kill-criterion passes. (Cross-check is against the video's *own*
embedded sectionals; the sample race is post-corpus, so no `trajectory_dataset_full` row exists
to join — the join SCHEMA is in place for in-corpus races, see §5.)

## (4) HONEST QUALITY — clean at SECTION level, swap-contaminated per FRAME
- **Number-binding: SOLVED, the big win vs 0055.** 9 of 11 distinct runners bound to a
  saddlecloth number (`{1,3,4,5,6,7,8,9,10}`). #2/#11 appear only briefly at the tail of the
  top-8 legend window and don't clear the vote threshold; #12 never appears in the legend (a
  12-field where the backmarker is never top-8). The panel is no longer anonymous.
- **Vote-purity: 0.49 median** (leader #9 0.72, #3 1.00, tail runners 0.28–0.43) — the fraction
  of a runner's legend-rank votes agreeing with its bound number. Materially above the 0055
  colour-only 52% *stability* on a HARDER metric (per-frame rank agreement, not cluster
  persistence), and the MODAL binding is correct for all 9.
- **Per-frame path: median |Δy_lat| = 0.63 m/step (~0.2 s)** — physical — **but p90 ≈ 5 m**:
  ~10% of steps still swap. So the frame-by-frame `y_lat` is NOT pristine; fine lateral control
  read straight off raw frames would be noisy.
- **Section-level path: CLEAN and PHYSICAL** — this is the usable signal. Per-runner MEDIAN
  `y_lat` by 400 m section (m outward from inner rail) differentiates trips: winner **#9 hugs
  the rail** (8.5→6.2→8.8→11.8 — a ground-saving inside trip), **#7 runs wide** (→27.5 wide,
  ground-losing), **#10 starts widest then tucks in** (26.3→11.2). Symmetric swap noise washes
  out under the per-section median; this is exactly the ground-saved / rail-position input S8/S9
  consume.
- **Lateral geometry: GOOD** (unchanged from 0055): 100% of rows within [0, track_width 30.5 m],
  median 14.2 m. **Leak-free by construction** (trailing per-horse trait, same posture as form).

## (5) THE PANEL — `data_cache/cv_lateral_paths_sample.parquet` (extended schema)
**1,140 rows × 14 cols**: the 0055 schema + `runner` now an **int saddlecloth NUMBER** (was
`rid{N}` string) + new **`reid_purity`** (0..1 vote-purity backing the binding). Tidy
`(venue,date,race_no,runner,t,frame, x_arclen, y_lat, rail_offset_px, silk_lab_{l,a,b},
confidence, reid_purity)`. **JOINS to `trajectory_dataset_full.parquet` on
`(venue,date,race_no,runner)`** with `x_arclen ↔ section_dist` (corpus = 4×400 m sections); the
saddlecloth `runner` is the matching key. For the control latent, aggregate `y_lat` to per-400 m-
section medians (the clean signal) and weight by `confidence`×`reid_purity`.

## (6) SCALE-UP STATUS
- **Re-id reliability NOW**: legend-anchored binding is reliable for the top-of-field /
  distinct-silk runners (leader near-pristine); tail-enders with few top-8 legend appearances
  and look-alike silks remain the weak spot (low purity, occasional non-binding). For
  section-level features this is adequate; for frame-level control it is not yet.
- **REMAINING GAPS**: (a) Kalman/RTS smoothing across occlusions + a continuity-gated swap
  rejector would push p90 |dy| down and rescue the brief-appearance runners; (b) the legend caps
  at top-8 — runners outside it need pure colour/motion bridging. Both are bounded follow-ups,
  not blockers.
- **CORPUS BACKFILL — BLOCKED on DATA, not on the pipeline.** The Aerial Virtual View render is
  the only Mode-B substrate, and locally we have exactly ONE such clip (`aerial_sample.mp4`).
  The other `data_cache/video/*.mp4` are a real drone view + broadcast/patrol pans
  (`aerial_active=False`; the YOLO-fails Mode-A substrate of the feasibility probe). So there is
  NO cheap local backfill — more races require ACQUIRING more Aerial Virtual View clips, an
  `hkjc.data` acquisition job outside this package's ownership. COST per race is also non-trivial:
  ~16–35 min wall (3 RapidOCR calls/aerial frame on a CUDA-off box) — a corpus backfill should
  OCR the legend/odometer at a coarser cadence (both change slowly) and/or batch on a GPU.

## (7) WHAT I CHANGED
NEW behaviour in `src/hkjc/vision/mode_b/`: `ocr.py` (+`read_odometer`, +`read_clock_splits`),
`layout.py` (+odometer/clock ROIs), `reid.py` (+`assign_by_legend`, `NumberedRunner`,
`_leader_lab`, `_along_track_order`), `pipeline_b.py` (legend/odometer/clock per-frame, numbered
output, odometer arc-length, sectional QA). EDITED `studies/cv_path_extraction.py` (re-id +
QA reporting). REWROTE `data_cache/cv_lateral_paths_sample.parquet` (numbered) +
`data_cache/figs/cv_lateral_paths_sample.png`. `ruff check src/hkjc/vision studies` clean;
`PYTHONHASHSEED=0 pytest -q` → **181 passed**. `src/hkjc/model/*` UNTOUCHED.

## (8) BLOCKER
Corpus backfill is blocked on Aerial-Virtual-View clip ACQUISITION (only one clip on disk;
`hkjc.data`'s job, not this package). The pipeline itself is ready and validated on the sample.
