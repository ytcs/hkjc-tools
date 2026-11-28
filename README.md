# hkjc-tools

Reconstructs every horse's 2D path in a Hong Kong Jockey Club race from public data: the race
facts (`hkjc.data`) plus the four replay videos HKJC publishes per race (aerial virtual view,
drone, side-on race film, patrol). Output: per horse, metres-to-finish along the racing rail and
the offset from the racing rail (render units; metres where validated), on the race clock, with posterior
uncertainty.

The product, `paths.parquet` (`run.paths()`), is one posterior built from all the information
available: the videos, the pre-race facts (distance, course, draw, silks), the official records
(placings, finishing times, sectional times) and the track geometry. `paths_video.parquet` keeps the
reconstruction from the videos and pre-race facts alone; the fixed acceptance gates are scored on it.
A race-fact mask (`hkjc.data.Mask`; `RaceRun(mask=...)`, CLI `--mask`, env `HKJC_MASK`) withholds
official records from every stage, so accuracy is measured where information was held out
(docs/RECON.md).

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
paths = run.paths()                             # the product: horse_no, t, s, d, lat, sd_s, sd_lat, n_obs
```

```bash
uv run --group vision python -m hkjc.recon 20260603 HV 8 [--force fuse_final,validate] [--stages ...]
```

Detector weights are expected in `models/` (`avv_avatar_yolo11n.onnx`/`.pt`, the self-trained
render-avatar detector; `coco_yolov8x.pt` from Ultralytics). They are not distributed in git.
Environment variables (all optional; unset = CPU):

| Variable | Effect |
|---|---|
| `HKJC_DEVICE` | `cuda` or `intel:<device>` (OpenVINO, e.g. `intel:gpu`) for the avatar detector; `intel:` also runs the OCR recogniser on that device. The film detector follows it unless `HKJC_FILM_DEVICE` is set. |
| `HKJC_FILM_DEVICE` | Film detector device, e.g. `intel:gpu` (OpenVINO at FP32: film stage 3–4× faster, crossings within 0.02 s of CPU; needs the export `models/coco_yolov8x_544x960_openvino_model`). |
| `HKJC_MATCH_DEVICE` | `cuda` or `cpu`: brute-force ORB matching in a helper process, index-identical to OpenCV. Unset: OpenCV. |
| `HKJC_OCR_DEVICE` | `cuda`: OCR recogniser on ONNX Runtime CUDA (not bit-exact vs CPU; fails if the CUDA provider is missing). |
| `HKJC_OCR_THREADS` | OCR threads per process (cap it when running races in parallel). |
| `HKJC_NPROC` | Worker processes per race (default: all cores). |
| `HKJC_AVATAR_WEIGHTS` | Path to avatar-detector weights, overriding the lookup in `models/`. |
| `HKJC_MASK` | `none` (default), `boundary=J` or `outcome`: official outcome data withheld from the reconstruction (validation still scores against the full record; CLI `--mask`). Give a masked run its own root. |

Detector boxes, and so camera poses, depend slightly on the device (CPU vs CUDA on HV 06-03 R8 moves
paths by ≤ 0.18 m); the reference baseline is a CUDA run.

## Layout

```
src/hkjc/data/    HKJC interfaces. Other sources (racecards, live odds/pools, trackwork, barrier
                  trials, dividends, form): fetch recipes in docs/HKJC_DATA.md, docs/HKJC_SPECS.md
    race.py       race_info: one race's facts (runners, draw, course) + the official outcome
    historical.py official results page: header, placings, finish times, running positions
    sectional.py  official sectional times + running positions per horse
    video.py      the four replay MP4 feeds (aerial, drone, race film, patrol)
    silks.py      official racing-silk images
    tracks.py     track geometry per venue x surface x course (TrackGeometry, track_for)
src/hkjc/recon/   race info + videos -> paths
    pipeline.py   RaceRun: cached stages frames -> features -> pose -> worldmap -> detect -> silk
                  -> hud -> rails -> start -> film -> observe -> finish -> appearance -> identify -> fuse_video
                  -> fuse_final -> validate
    video.py      frame access, AVV render gating / 10 Hz de-dup / HUD masks
    register.py   turf ORB features, RANSAC similarity + least-squares refit
    pose.py       camera pose graph (sequential, long-baseline, loop closures)
    worldmap.py   mosaic of registered frames (the static world map)
    rails.py      racing rail on the map, track frame (s, lat), marker grid, odometer calibration
    start.py      gate opening in the start film: re-anchors the race clock when the overlay lags
    detect.py     self-supervised YOLO avatar detector + silk histograms
    circles.py    numbered-circle render front end (2026/27 replays: numbered discs, no avatars)
    film.py       race-film marker crossings (the distance metric's constant offset)
    ocr.py        recognition-only OCR of the burnt-in clock / odometer / legend
    bfmatch.py    opt-in brute-force ORB matching (CUDA/CPU), index-identical to OpenCV
    ortsession.py ONNX Runtime sessions without spin-waiting (identical outputs, less idle CPU)
    tracking.py   world-frame multi-horse tracking + stitching (initialisation only)
    identity.py   tracklet -> saddlecloth number (legend order, elimination, silk)
    silkmap.py    learned official-silk -> render-silk appearance map (identity's silk evidence)
    assoc.py      detection-level EM association against the fused continuous paths
    finishcam.py  finish-line camera: cloth OCR, order + spatial gaps, finish identity
    fusion.py     Bayesian path reconstruction from all feeds' proposals (physics + geometry priors)
    validate.py   fixed gates on the video-only paths vs the full official record
    audit.py      label-free consistency invariants of a reconstruction (audit.json)
docs/RECON.md     architecture: stages, artifacts, mask and held-out evaluation, gates, environment, known limitations
docs/HKJC_*.md    HKJC data sources and track specifications
models/           detector and render-appearance weights (git-ignored)
```

