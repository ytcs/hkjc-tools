---
title: S0d · CV METHODOLOGY BAKE-OFF (aerial vs drone vs combined) on ONE race — ST 2026-06-03 R1, Class 5, 1650M, TURF "C" course (off. winner #3 FAMILY FORTUNE, rp 7-7-6-1 from off the pace; #12 wire-to-wire 1-1-1-2) — validated against EXTERNAL official data (running positions / sectionals), NOT the video's own legend. VERDICT (§4b-de-confounded): NO view reconstructs the finish/running-order against official on this race yet — but the failures are LOCALIZED MIS-SPECS, not view-level nulls. AERIAL is the only scoreable view (legend-OCR numbers + a SOUND geometry layer: odometer arc-length to 1644m, rail-width scale 0.18 m/px, section-time MAE ~0.49s) — it binds 8/12 at 0.775 purity and captures the wire-to-wire leader #12, but its positional ρ vs official is NOISE-DOMINATED (sampling-fraction sweep swings [-0.82,+0.77]; pole-2 bootstrap 95%CI [-1.00,+0.04]; mean ~0) and the WINNER #3 DROPS OUT at 1507m — the final-140m run-in re-id collapse, exactly where it surges 7th->1st through traffic; 4/12 (incl. closers #6 3rd, #2 4th) never bind. So aerial's bottleneck is RE-ID ROBUSTNESS AT THE CONGESTED FINISH (fixable), not the view. DRONE has the richer real-silk substrate and a CLEAN tracking substrate (16 det/frame, 12 coherent anonymous tracks ~193 frames, monotonic, homography on 701 frames) BUT three blockers: arc-length mis-scaled ~3-4x (x_max 4700-6400m for a 1650m race — rail-homography scale bug), y_lat exceeds the 21.3m track width, and it is ANONYMOUS (silk-colour clusters, saddlecloth numbers not legible from altitude) so it CANNOT be scored against numbered official results at all without a silk->number bridge. COMBINED is meaningless until both views are individually scoreable. CRITICAL META-FINDING: this VINDICATES the external-validation requirement — the prior CV 'success' (0064: sectionals to 0.12s, finish EXACT) was SELF-REFERENTIAL (scored vs the video's OWN legend, on a DIFFERENT race: aerial_sample.mp4 winner #9 != R1 winner #3); against EXTERNAL official data the single-race reconstruction does NOT hold at the finish. The §4b ceiling discipline + 'check official data' caught what self-validation hid. CONTROL (S8) and the CV-corpus scale-out (S0c) STAY GATED — do not scale a re-id that loses the winner. Study-only; nothing wired into the model; the only src/ change is a behaviour-preserving `ocr_every` perf knob on mode_b.pipeline_b.
id: 0067-s0d-cv-methodology-bake-off-aerial-vs-drone-vs-c
author: modelling
audience: ["modelling"]
channel: research
kind: result
status: open
thread: 0060-reorientation-win-diagnostic-not-veto-wave2-expand-causal-network
tags: ["s0d", "cv", "computer-vision", "vision", "mode-b", "methodology", "bake-off", "aerial", "drone", "combined", "fusion", "bayesian-fusion", "multi-view", "re-id", "reid", "legend-anchored", "silk-colour", "anonymous", "rapidocr", "ocr", "ocr-every", "homography", "rail", "rail-homography", "rail-config", "c-course", "track-geometry", "arc-length", "odometer", "y-lat", "lateral", "control", "control-dimension", "jockey", "section-times", "sectionals", "running-position", "finish-order", "spearman", "bootstrap", "noise-dominated", "external-validation", "official-data", "ground-truth", "self-referential", "0064", "0058", "winner-dropout", "run-in", "occlusion", "mis-spec", "mis-specification", "true-null", "ceiling-check", "s4b", "de-confound", "gated", "s8", "s0c", "0076", "scale-bug", "calibration", "study-only", "dont-bake", "no-regression", "wave-2", "causal-network", "0060"]
refs: {"orchestrator": "studies/cv_bakeoff_s0d.py", "drone_pipeline": "studies/cv_s0d_drone.py", "qc_figure_script": "studies/cv_s0d_qc.py", "qc_figure": "data_cache/figs/cv_s0d_qc.png", "aerial_panel": "data_cache/cv_s0d_aerial.parquet", "drone_panel": "data_cache/cv_s0d_drone.parquet", "aerial_pipeline": "src/hkjc/vision/mode_b/pipeline_b.py (ocr_every perf knob added)", "official": "data_cache/race_official/20260603R1_results.html", "prior_cv_journal": "journal/entries/0064-s0b-control-data-unlock-legend-anchored-reid-binds-saddlecloth-numbers-odometer-arclength-cv-sectionals-match-published-to-0.12s-paths-clean-at-section-level.md", "thread_journal": "journal/entries/0060-reorientation-win-is-a-diagnostic-of-causal-coverage-not-a-veto-longitudinal-is-one-causal-dimension-reject-form-as-measurement-wave2-expand-the-causal-network.md", "videos": "data_cache/video/aerial_20260603_01.mp4, data_cache/video/drone_20260603_01.mp4"}
created: 2026-06-06T23:40:37+00:00
updated: 2026-06-06T23:40:37+00:00
---
S0d — the CV METHODOLOGY BAKE-OFF the user asked for: "try a combined drone and aerial recon for
a SINGLE race first to nail down the correct methodology; study which video gives a more reliable
reconstruction (aerial / drone / combined)" + "check consistency with known section times and posted
race commentary so our recon is consistent with official data." Single race: **ST 2026-06-03 R1**
(Class 5, 1650M, TURF **"C" course**). Two views of the SAME race were downloaded (deterministic
auth-free Akamai MP4s): the synthetic top-down **aerial** (AVV render) and a real high-angle **drone**
camera. Official ground-truth parsed from `data_cache/race_official/20260603R1_results.html`.

## What was built (and why the first run produced nothing)
The dispatched S0d agent built the FULL harness correctly: a new 400-line real-horse **drone**
pipeline (`studies/cv_s0d_drone.py` — HSV field-detect -> connected-component sprites -> nearest-
neighbour link -> per-frame white-rail homography -> rail-relative coords, NO OCR), the **aerial**
path via `hkjc.vision.mode_b.extract_lateral_paths`, a **Bayesian two-view fusion** (precision-
weighted by re-id purity x coverage), and a **scorecard** that validates EVERY view against official
(winner must come back #3, per-pole running-position Spearman, section-time MAE). It then STALLED on
runtime: the aerial pass does **3 RapidOCR reads/frame** (~16-35 min/race, the cost 0064 warned of),
the agent wrapped it in a 590s timeout, and it was KILLED mid-aerial before writing anything — no
journal, no scorecard. Recovered by adding an **`ocr_every` decimation knob** to `mode_b.pipeline_b`
(legend/odometer/clock change slowly — read OCR every Nth processed frame; behaviour-preserving at
`ocr_every=1`) and rerunning uncapped at stride=8/ocr_every=3 (~1Hz OCR). It then completed.

## The raw scorecard was DEGENERATE — so §4b says STOP, de-confound before concluding
Raw bake-off output: aerial `cv_finish_order=[7]` (a single-element list — and #7 isn't even in the
bound set!), drone `cv_finish_order=[]`, and NEGATIVE running-position correlations across BOTH views
(-0.24 aerial, -0.48 drone). Two INDEPENDENT pipelines both landing anti-correlated is the tell of a
SHARED harness/orientation bug, not two reconstruction failures. Per PHILOSOPHY §4b (a null is not a
conclusion until a de-confounded ceiling separates true-null from mis-specification), I re-scored the
written parquets directly with a correct distance-at-time ordering.

## DE-CONFOUNDED READ (re-scoring `data_cache/cv_s0d_{aerial,drone}.parquet`)
**AERIAL** — the only view that can be scored (it has numbers via legend OCR):
- GEOMETRY LAYER IS SOUND: odometer arc-length reaches 1644m of 1650; rail-width scale 0.181 m/px;
  section-time MAE ~0.485s (but only **2 of 4** sections detected — missed the 27.65s opening + a late
  split). reid_purity_median 0.775.
- RE-ID BINDS 8/12: [3,4,5,8,9,10,11,12]. It DOES capture the wire-to-wire leader #12 (official
  rp 1-1-1-2) — that horse is first/near-first in the CV order throughout.
- BUT POSITIONAL SIGNAL IS NOISE: CV-vs-official running-position Spearman, swept over the sampling
  fraction, swings between **-0.82 and +0.77**; a pole-2 bootstrap (n=7 runners) gives mean -0.78 with
  **95% CI [-1.00, +0.04]** — i.e. fully consistent with ZERO. Any single "ρ=+0.38 mid-race" read is a
  noise draw from a 7-8 runner Spearman, not signal.
- THE WINNER IS LOST: #3 (FAMILY FORTUNE, rp 7-7-6-1, came from 7th) DROPS OUT of the CV track at
  **1507m** (t=83.5s) — the final ~140m run-in, exactly where it surges through traffic to win. 4/12
  never bind, including closers #6 (3rd) and #2 (4th). So the CV "finish" is dominated by who STAYED
  bound, not who won; finish-order ρ ~ 0.
- DIAGNOSIS: **mis-specified re-id at the congested business end**, fixable. NOT a view-level null —
  the geometry works and the leader is captured.

**DRONE** — richer real-silk appearance, clean tracking substrate, but three blockers:
- SUBSTRATE WORKS: 16 detections/frame (median), 12 coherent ANONYMOUS silk-colour tracks (~193
  frames each, fully monotonic arc-length), homography fitted on 701/716 frames.
- ARC-LENGTH MIS-SCALED ~3-4x: cluster x_max values 4735 / 5415 / 6346 / 6410m for a **1650m** race;
  y_lat ranges to ~29m, exceeding the 21.3m "C"-course width -> the oblique rail-homography SCALE is
  wrong (monotonic but mis-calibrated). A `scale_source=rail-width(C)/inter-rail-px` bug.
- ANONYMOUS: saddlecloth numbers are NOT legible from the drone altitude -> silk-colour clusters only,
  NO binding to saddlecloth#, so it CANNOT be scored against numbered official results at all
  (`cv_finish_order=[]`, sections []). It needs a **silk->number bridge** (borrow the aerial legend
  map for the same race, or read numbers in the parade / early when horses are large and separated).

**COMBINED** — the Bayesian fusion is meaningless while one view is degenerate-finish and the other
is unlabelled; `nan`. Revisit once both views are individually scoreable.

## VERDICT (the question asked: which view is more reliable?)
On this race, **no view reconstructs the finish/running-order against official yet.** AERIAL is the
only view that can be VALIDATED at all today (numbers + a working geometry layer) and its limit is a
fixable re-id collapse at the finish. DRONE has the better raw substrate (real silks, clean tracking)
and is the stronger LONG-TERM bet for the CONTROL/lateral dimension, but is currently UNVALIDATABLE
(scale bug + anonymity). The right move is NOT to crown a view now — it is to fix the two localized
mis-specs and re-run S0d:
  1. AERIAL: harden re-id through the run-in (occlusion-robust silk matching + motion-bridge gaps;
     bind all 12; recover the winner's final 140m); add the missing 2 section splits.
  2. DRONE: fix the rail-homography scale (calibrate inter-rail-px -> 21.3m correctly; clamp y_lat to
     track width); build the silk->number bridge so the 12 clean tracks become numbered.
  3. Re-run the bake-off; only THEN compare aerial / drone / combined reliability.

## CRITICAL META-FINDING — external validation vindicated; the prior CV "win" was self-referential
0064 reported aerial CV sectionals matching published to 0.12s and finish order EXACT. That was scored
against the **video's OWN legend**, on a DIFFERENT race (`aerial_sample.mp4`, winner #9, != this R1's
winner #3). Against EXTERNAL official data, the single-race reconstruction does NOT hold at the finish
(winner lost, positional ρ ~ 0). This is exactly the gap the user's "check official data" requirement
and the §4b ceiling discipline are designed to catch — and they did. Self-consistency (CV vs the
render's own overlay) is NOT validation; only external official ground-truth is.

## CONSEQUENCES for the Wave-2 board
- **CONTROL latent (S8, #66)** and **CV-corpus scale-out (S0c, #75)** STAY GATED. Do not scale a re-id
  that loses the winner and binds 8/12 on a clean synthetic render; a noisier multi-race corpus will be
  worse. The CV CONTROL dimension is real but the EXTRACTION is not yet trustworthy on external truth.
- **Rail-config (#76)** is confirmed necessary and now partly DONE for aerial ("C" width 21.3m wired);
  the drone scale bug is the same geometry concern in the oblique view.
- Nothing was wired into the generative model. The only `src/` change is the behaviour-preserving
  `ocr_every` perf knob; the model's win/path numbers are unchanged.

## Artifacts
- `studies/cv_bakeoff_s0d.py` (orchestrator + scorecard), `studies/cv_s0d_drone.py` (drone pipeline),
  `studies/cv_s0d_qc.py` (the de-confounded QC figure).
- `data_cache/cv_s0d_aerial.parquet` (972 rows, 8 runners), `data_cache/cv_s0d_drone.parquet`
  (2770 rows, 12 anonymous clusters), `data_cache/figs/cv_s0d_qc.png` (3-panel: noise-dominated
  positional ρ; aerial arc-length with the #3 dropout; drone scale-inflation).
- `src/hkjc/vision/mode_b/pipeline_b.py` gained `ocr_every` (perf only).
