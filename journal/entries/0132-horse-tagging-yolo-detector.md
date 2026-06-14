---
title: Horse tagging — YOLO avatar detector (self-supervised on the pipeline's own tags) breaks the classical ceiling; R08 coverage 77%->95%, all 11 tags on 1144/1166 frames
id: 0132-horse-tagging-yolo-detector
author: modelling
audience: ["modelling", "all"]
channel: handoff
kind: finding
status: open
thread:
tags: ["cv", "aerial", "horse-tagging", "detection", "yolo", "neural", "r08"]
refs: {"refs": ["0130", "0131"]}
created: 2026-06-13T00:00:00+00:00
updated: 2026-06-13T00:00:00+00:00
---
## The ceiling, and the call

The classical segmentation detector + global tracker (0130/0131) reached coverage ~77% with
identity correct in the clean zones, but plateaued on TWO detection failures no amount of tuning
fixed: (a) the zoomed-out BEND (f816: tiny tightly-packed avatars -> 1 detection for 11 horses),
and (b) the disclaimer-text-occluded RUN-IN (semi-transparent legal text OVER the strung-out pack
-> 15-20 detections, false positives on text the scale logic can't merge and a mask would erase
the horses under). Owner offered heavier machinery; this is the right use of it — a DETECTION
problem, not an identity one.

## Self-supervised YOLO (reuse, not rebuild)

Ultralytics YOLO is ALREADY a `vision` dependency. The pipeline's own output is the training set:
- **Pseudo-labels** (`cv_htag_nn_labels.py`): the global tracker's FULLY-LABELLED frames (exactly
  11 tags) are high-quality avatar centres — 905 frames, 754 train / 151 val. Box side = 2.4·√area.
  Only fully-labelled frames, so YOLO never sees a real avatar as background. Avatars are CONSISTENT
  synthetic sprites, so a nano model + the clean-zone labels generalise to the hard zones.
- **Train** (`cv_htag_nn_train.py`): `yolo11n`, single class, imgsz 960, CPU (no CUDA at runtime),
  aug = degrees 15 + scale 0.4 to simulate the bend rotation/zoom the clean labels lack, NO flips
  (the render's shadow/orientation geometry is a consistent cue). Converged FAST: epoch 5 val
  recall 0.993 / mAP50 0.993 (stopped there — ~12 min/epoch on CPU, further epochs not worth it).
- **Infer** (`cv_htag_nn_detect.py`): box centres + a silk signature read from the box's vivid/white
  px -> the SAME dets-cache format the classical detector emits, so `cv_htag_track.py` consumes it
  unchanged (`HTAG_DETS` env; the NN cache is now the default when present).

## Result (R08) — a step change

Detection: **mean 11.01 dets/frame (= field size), ZERO over-detecting frames** (classical had
217), 904 frames at exactly 11; the bend f816 1->9, the text-occluded run-in 20->10 with NO text
false positives (the net LEARNED to reject the disclaimer, no heuristic). Tracking on the NN dets:
**coverage 1153/1165/1166 of 1227 (~95%, was 77%); all 11 tags on 1144/1166 frames; every race
phase mean ~11.0.** Identity (silk-consistency audit): **9 of 11 tags spread < 5.1 LAB** (one horse
all race), the 2 high-spread are the known multi-tone silks (magenta+white, yellow+red+white),
visually confirmed correct. The previously-broken bend / run-in / finish all resolve.

## Generalization — this is NOT a per-race-finetune approach (owner's scale-out concern)

Three findings, escalating:
1. **Zero-shot PRECISION transfers.** The R08-only detector on R02A (different race/config/silks,
   no retraining) puts every box on a horse and ZERO on the grandstand CROWD or the disclaimer
   text — the exact furniture that defeated the classical pipeline. It learned "avatar" (a bounded
   synthetic sprite), not "salient blob", so unseen furniture is rejected. Recall was the only gap
   (~10/12, partly real occlusion — the classical detector misses the same horses).
2. **The tracker generalized (not the detector).** Most of R02A's apparent coverage gap was a
   tracker bug: it assumed the full field is always visible, but R02A shows ~10-11 of 12 (odometer
   box / shadows / frame edge). Thresholds now anchor on the per-race MEDIAN visible count -> R02A
   median coverage 603->883, R08 unchanged (no-op where median=field). No per-race tuning.
3. **One JOINT detector for all races.** Pool labels from a few races spanning the bounded AVV
   variation (lighting/shadow/silk/zoom), train ONE model (`RACE=joint`, R08+R02A, 1036 imgs,
   yolo11n CPU ~6 epochs, val mAP50 0.994 / mAP50-95 0.77 — ABOVE the R08-only 0.65). Labels are
   SELF-BOOTSTRAPPING (the detector+tracker auto-label a new race; the "every detection tagged"
   criterion handles partial visibility — R02A 518 frames, zero manual annotation).

**Held-out test (the proof): R03 was NOT in training.** The joint detector zero-shot on R03 ->
tracker: **coverage 696/757 (~92%), 10 of 12 tags silk-consistent** — BETTER than R02A (R03 has
less occlusion). R08 (trained): 1151/1165/1168, no regression. R02A (trained): median 777, 8/12.
So a detector trained on 2 races tags an unseen 3rd at 92% / 10-of-12 — the pool-and-generalize
scale-out works; new races need no manual labels and (once the pool spans the variation) no
training.

## Pipeline + state

`cv_htag_nn_labels.py` -> `cv_htag_nn_train.py` (weights under `runs/`, gitignored) ->
`cv_htag_nn_detect.py` (-> `data_cache/_htag_dets_nn_R08.npz`) -> `cv_htag_track.py` (prefers the
NN cache). The classical detector (`cv_htag_detect.py` / `cv_htag_dets_cache.py`) is retained — it
BOOTSTRAPS the labels and is the fallback. Residual: 22 frames short of full (deepest bend
occlusion f816 = 9/11; 1-frame reliability drops); the multi-tone-silk audit false alarm is a
metric artifact, not an error. Next: generalize the detector to R02A/R03 (retrain or fine-tune;
the crowd-furniture zone is the new test), then the number/identity-binding layer.
