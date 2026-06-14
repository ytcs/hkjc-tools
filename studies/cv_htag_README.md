# Horse tagging — AVV per-horse identity pipeline

Stable per-horse tags on every frame of the *Aerial Virtual View* render. A learned avatar
detector (Ultralytics YOLO, self-supervised on the pipeline's own tags) feeds a global no-teleport
tracker that assigns exactly-N persistent identities. Journals **0130–0132**; methods in
`docs/CV_RECON.md`.

## Current state (committed)

| Race | In detector training? | Coverage (frames tagged) | Identity (silk-consistency) |
|---|---|---|---|
| R08  | yes | ~95% (1151/1166) | 9/11 tags clean; 2 multi-tone, correct |
| R02A | yes | ~78% (occlusion-limited, ~10/12 visible) | 8/12 clean |
| **R03** | **NO (held out)** | **~92% (696/757)** | **10/12 clean** |

The joint **R08+R02A** detector tags held-out **R03 zero-shot** → the approach scales without
per-race fine-tuning (see "Scale-out" below). val mAP50 0.994 / mAP50-95 0.77.

## Files (8 scripts, ~1.8k lines, all `studies/`)

- `cv_htag_detect.py` — CLASSICAL per-frame detector (`detect_core`): class-segmentation +
  venue-model gating. Bootstraps the labels; retained as the fallback. (33k — the big one.)
- `cv_htag_dets_cache.py` — runs the classical detector over a race → `_htag_dets_{RACE}.npz`
  (track-width-anchored avatar scale).
- `cv_htag_rails_cache.py` — per-frame rail curves (racing/fixed) from the rail pipeline →
  `_htag_rails_{RACE}.npz` (the tracker's venue-model gating + scale anchor).
- `cv_htag_track.py` — the GLOBAL TRACKER (`propagate_ntracks`): N no-teleport tracks per stable
  segment, stitched by pooled silk; per-race adaptive visible-count thresholds. Reads the NN
  detection cache by default (classical fallback; `HTAG_DETS` overrides). → tags + overlay render.
- `cv_htag_nn_labels.py` — YOLO pseudo-labels from the tracker's complete-per-frame output →
  `data_cache/htag_yolo_{RACE}/`.
- `cv_htag_nn_train.py` — train `yolo11n` single-class (`RACE=joint` for the multi-race model).
- `cv_htag_nn_detect.py` — YOLO inference → `_htag_dets_nn_{RACE}.npz` (classical-cache format).
- `cv_htag_camera.py` — a turf-flow camera estimate, **built then REVERTED** (kept as a documented
  dead-end: the camera follows the field, so the turf transform predicts horses backwards).

## Pipeline order

Prereq (per race): the rail LOOP-1 cache `data_cache/_railclean_{RACE}.npz` from the rail pipeline
(`studies/cv_rail_exp_pixtag.py`) + the race video `data_cache/video/<file>.mp4`.

**Bootstrap a new race (no trained detector yet):**
```
RACE=R08 uv run --group bayes python studies/cv_htag_rails_cache.py      # rail curves
RACE=R08 uv run --group bayes python studies/cv_htag_dets_cache.py       # classical dets
HTAG_DETS=data_cache/_htag_dets_R08.npz RACE=R08 \
    uv run --group bayes python studies/cv_htag_track.py                 # classical tags
RACE=R08 uv run --group bayes python studies/cv_htag_nn_labels.py        # YOLO labels
```

**Train the (joint) detector** — pool races for generalization:
```
# edit /tmp/make_joint.py-style data.yaml to list the races' images dirs, or RACE=R08 for one race
RACE=joint uv run --group bayes python studies/cv_htag_nn_train.py       # GPU auto-used if present
```

**Steady state (any race, incl. unseen):**
```
RACE=R03 uv run --group bayes python studies/cv_htag_nn_detect.py        # YOLO dets (joint model)
RACE=R03 uv run --group bayes python studies/cv_htag_track.py            # final tags + render
```

## GPU / continuation notes

- Training auto-selects CUDA (`cv_htag_nn_train.py`: `device = "cuda" if available else "cpu"`).
  It was developed CPU-bound (~16 min/epoch at imgsz 960); on a GPU the joint model is a few
  minutes, so iterating the race pool is cheap.
- `HTAG_WEIGHTS` overrides the detector weights; `HTAG_EPOCHS` the epoch count. The joint model is
  the default detector for ANY race (`cv_htag_nn_detect._weights` prefers `htag_yolo_joint`).
- **Data does NOT push** (gitignored): videos, `_htag_*` caches, `htag_yolo_*` datasets, `runs/`
  weights. On a fresh machine: copy the `data_cache/video/*.mp4` + `_railclean_*.npz`, then run the
  pipeline. (Everything downstream regenerates.)

## Scale-out (the generalizable approach — no per-race fine-tuning)

1. Pool labels from a few races spanning the bounded AVV variation (lighting/shadow/silk/zoom).
2. Train ONE joint detector → handles new races zero-shot (R03 held-out: 92% / 10-of-12).
3. New races are SELF-LABELLING: the detector + tracker auto-generate labels (the
   "every detection tagged" criterion handles partial visibility), add to the pool, retrain.
4. The tracker already adapts per-race (median visible count); no per-race code.

Residual ceiling: genuine occlusion (odometer box / shadows / frame edge hide horses both the NN
and classical detectors miss) — not recoverable by training. Next layer: bind saddlecloth numbers
(legend silks + draw order) to the stable tags.
