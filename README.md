# hkjc-tools

Reconstructs every horse's 2D path in a Hong Kong Jockey Club race from public data: the race
facts (`hkjc.data`) plus the four replay videos HKJC publishes per race (aerial virtual view,
drone, side-on race film, patrol). Output: per horse, metres-to-finish along the racing rail and
metres off the racing rail, on the race clock, with posterior uncertainty.

Two products per race:
- `paths_video.parquet`: from the videos and pre-race facts only (distance, course, draw, silks).
  The acceptance gates are scored on it against the official results.
- `paths.parquet` (`run.paths()`): the final product. It also uses the official results
  (placings, finishing times, sectional times).

## Install

```bash
uv sync                    # data interfaces
uv sync --group vision     # + OpenCV / YOLO / video decode / OCR for the reconstruction
```

## Use

```python
from hkjc.data import race_info
from hkjc.recon import RaceRun

run = RaceRun(race_info("20260603", "HV", 8))   # downloads the replays on first use
run.run()                                       # cached stages -> data_cache/recon/20260603_HV_8/
paths = run.paths()                             # final product: horse_no, t, s, d, lat, sd_s, sd_lat, n_obs
```

```bash
uv run --group vision python -m hkjc.recon 20260603 HV 8 [--force fuse_final,validate] [--stages ...]
```

Detector weights are expected in `models/` (`avv_avatar_yolo11n.onnx`/`.pt`, the self-trained
render-avatar detector; `coco_yolov8x.pt` from Ultralytics). They are not distributed in git.
Environment knobs: `HKJC_DEVICE=cuda` (GPU detection), `HKJC_NPROC` (worker processes),
`HKJC_OCR_THREADS` (OCR threads per process). `HKJC_WORLD_MASK=yolo HKJC_FRAME_GRID=1` selects the
masked registration variant (avatar boxes masked out of the world features, frame grid kept); it
is under validation and not yet the default.

## Layout

```
src/hkjc/data/    HKJC interfaces: race_info (results+sectionals), replay videos, silks, track
                  geometry. Other sources (racecards, live odds/pools, trackwork, barrier trials,
                  dividends, form): fetch recipes in docs/HKJC_DATA.md, docs/HKJC_SPECS.md
src/hkjc/recon/   race info + videos -> paths
    pipeline.py   RaceRun: cached stages frames -> features -> pose -> worldmap -> detect -> silk
                  -> hud -> rails -> film -> observe -> finish -> identify -> fuse_video -> fuse_final
                  -> validate
    video.py      frame access, AVV render gating / 10 Hz de-dup / HUD masks
    register.py   turf ORB features, RANSAC similarity + least-squares refit
    pose.py       camera pose graph (sequential, long-baseline, loop closures)
    worldmap.py   mosaic of registered frames (the static world map)
    rails.py      racing rail on the map, track frame (s, lat), marker grid, odometer calibration
    detect.py     self-supervised YOLO avatar detector + silk histograms
    film.py       race-film marker crossings (the distance metric's constant offset)
    ocr.py        recognition-only OCR of the burnt-in clock / odometer / legend
    tracking.py   world-frame multi-horse tracking + stitching (initialisation only)
    identity.py   tracklet -> saddlecloth number (legend order, elimination, silk)
    silkmap.py    learned official-silk -> render-silk appearance map (identity's silk evidence)
    assoc.py      detection-level EM association against the fused continuous paths
    finishcam.py  finish-line camera: cloth OCR, order + spatial gaps, finish identity
    fusion.py     Bayesian path reconstruction from all feeds' proposals (physics + geometry priors)
    validate.py   video-only product vs the official results, against fixed gates
    audit.py      label-free consistency invariants of a reconstruction (audit.json)
    viz.py        synced four-feed video with the reconstructed track map
    drone.py      drone feed (experimental: registration open)
docs/RECON.md     method, gates, current state, open items, rejected approaches
docs/HKJC_*.md    HKJC data sources and track specifications
models/           detector weights (git-ignored)
```

