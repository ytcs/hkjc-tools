# CV Trajectory PoC — Results (ONE race: HV 2026-06-03 R1)

**Status:** proof-of-concept, single race, depth-not-breadth.
**Race:** Happy Valley, 2026-06-03, Race 1 (Mount Butler Handicap), **1650 m, "C" course
(turf), Good To Firm, 12 runners**, winning time 1:38.85 (98.85 s). Winner #3 FAMILY FORTUNE.
**Source video:** `data_cache/video/drone_20260603_01.mp4` — 1920×1080, 25 fps, 143 s.
**Code:** `experiments/trajectory/{race_ground_truth,detect_probe,rail_probe,extract_tracks,project_traj}.py`
**Artifacts:** `data_cache/video/poc_out/` (probe PNGs, annotated mp4, top-down/speed plots, npz).

---

## TL;DR / go-no-go

<!-- FILLED AT END -->

---

## 0. The video is NOT a clean overhead drone clip — it's a multi-shot broadcast edit

The single most important finding, established by eyeballing full-res frames across the clip
(`poc_out/probe_*`, and sampled frames):

| time (s) | shot | usable for trajectory? |
|---|---|---|
| 0–4 | overhead drone, horses loading into the **starting gate** (a fixed `d=1650` landmark) | setup |
| ~4–102 | **continuous overhead drone follow-shot**, start → down the straight → bottom bend → back straight → approaching the home turn | **YES** — this is the whole trajectory window |
| ~102–115 | cuts to **ground-level home-turn broadcast camera** (saddlecloth numbers legible) | no (oblique ground cam) |
| ~115–143 | **ground-level finish / photo-finish cameras** (numbers 1–12 clearly legible) | no |

Consequences:
- The homography-friendly overhead window covers roughly **d = 1650 → ~400 m** (~1250 m of the
  1650 m race). The **final ~400 m (home straight) is shown only by ground cameras**, so the
  4th sectional (last 400 m) is **not** recoverable from the overhead footage of this clip.
- Saddlecloth numbers are **never legible in the overhead window** (~30–50 px horses). OCR
  identity is only possible on the out-of-window finish frames. Identity inside the window
  must come from **silk colour**.

---

## 1. DETECTION — what works

Probe (`detect_probe.py`) compared plain YOLO vs SAHI tiling on 7 representative overhead
frames, counting horses found (field = 12):

| frame (frac) | plain yolov8x @1920 | plain @1280 | SAHI 512px tiles | note |
|---|---|---|---|---|
| 0.05 (start) | **12** | 12 | 14 (2 FP) | field laterally spread |
| 0.15 | 9 | 8 | 1 | partial pan/blur |
| 0.30 | 2 | 0 | 0 | field single-file on far bend, going away → one blob |
| 0.40 | 14 (2 FP) | 6 | 14 | spread |
| 0.50 | 2 | 2 | 0 | single-file |
| 0.60 | **12** | 12 | 12 | spread, clean |
| 0.68 | 13 | 12 | 13 | spread |

**Decision: plain Ultralytics `yolov8x.pt` @ imgsz=1920, conf≈0.15. SAHI gave NO accuracy
gain at ~3× the cost (33 s/frame vs ~10 s/frame on CPU) and sometimes did worse** — the horses
are not small *enough* relative to a 1920-px frame to need tiling; the COCO `horse` class fires
well at native 1920. yolov8n/m @640 (the prior known-failure config) detect **zero** because the
downscale shrinks the horses below the detector's small-object floor.

**Binding detection failure mode:** when the field bunches **single-file along the rail going
away from the camera** (mid-bend), 12 overlapping horses read as ~2 blobs. This is geometry, not
resolution — SAHI doesn't fix it. The tracker's motion model must bridge these stretches.

Full overhead-window pass (`extract_tracks.py`, yolov8x@1920, stride 5 ≈ 5 Hz):

<!-- FILLED: det/frame median/mean, %frames>=10, track-length stats -->

---

## 2. IDENTITY

- **Saddlecloth OCR is out** for the trajectory window (numbers sub-legible at 30–50 px). It
  *is* legible on the ground-level finish frames (verified: numbers 1,2,3,5,6,8,9,12 readable),
  but those frames are outside the homography window, so OCR can anchor finish order but not
  in-running identity.
- **Silk colour is the operative ID handle.** Each detection's rider-torso ROI (upper 45 % of
  the box) is reduced to a mean Lab colour; tracklets are clustered (k=12) by mean silk colour.
- **ID fragmentation** (tracklets per true runner) is the measurable quality number:

<!-- FILLED: tracklets, clusters, tracklets-per-cluster -->

ByteTrack alone produces many short tracklets (ID switches at every bunch/occlusion), so raw
track-count ≫ 12; silk-colour clustering is what collapses them toward the 12 runners. Mapping a
colour-cluster to a specific saddle number needs the declared-silks swatches (HKJC racecard) —
not wired in this PoC; the fragmentation metric is reported instead.

---

## 3. HOMOGRAPHY / PROJECTION to (d, off)

A full per-frame 4-point rail-to-rail homography is **brittle here**: the outer rail is heavily
cluttered by background white (grandstand, pavement, road markings — see `rail_probe_*.png`),
so a clean outer correspondence is unreliable. The **inner running rail** (the bright continuous
fence the field hugs) is the one robust landmark. So projection is **rail-anchored**:

- **`off` (lateral m from rail):** signed perpendicular pixel distance from each horse foot-point
  (box bottom-centre) to the fitted inner-rail polyline, × a local metres-per-pixel scale.
- **local scale:** a thoroughbred is ~2.4 m nose-to-tail; metres-per-pixel each frame = 2.4 /
  median box long-side. (The outer rail is too cluttered to scale from rail-width reliably.)
- **`d` (m from finish):** the drone *tracks the field*, so the **static-background pixel pan**
  between frames, in metres, equals the field's ground displacement. We measure the pan with
  sparse Lucas–Kanade optical flow on corners **outside** the horse boxes (true static scene),
  accumulate it, and anchor start = 1650 m. **This is non-circular** — `d`/speed come from the
  background, not from an assumed pace.

<!-- FILLED: topdown + speed plots, what they show -->

**Honest accuracy ceiling:**
- `d` / along-track speed: as good as the background-flow scale (the horse-length scale prior is
  the dominant ~±10–15 % error; foreshortening varies the scale across the oblique frame).
- `off` / lateral: **the weakest axis.** From a near-top-down-but-oblique view, lateral metres
  are the most foreshortening-sensitive, and the rail-fit + horse-length scale compound. Treat
  `off` as a **coarse lane indicator (inner/middle/wide), not a sub-metre measurement.**

---

## 4. VALIDATION vs HKJC sectionals

Ground truth (`race_ground_truth.py`, scraped from the HKJC sectional page): 4 splits/horse
summing to 98.85 s; HKJC standard scheme ⇒ sections ≈ [450, 400, 400, 400] m
(d markers 1650/1200/800/400/0). Only the first **three** sections fall inside the overhead
window; the last 400 m is on the ground-camera (out of window).

<!-- FILLED: recovered vs published section times, error -->

---

## 5. Binding constraints (ranked)

<!-- FILLED -->

---

## 6. Reproduce

```bash
cd /home/ytc/Projects/hkjc-tools
# detection comparison
uv run --with ultralytics --with sahi --with opencv-python-headless --with av \
    python experiments/trajectory/detect_probe.py
# full overhead detect+track pass (~CPU-bound, ~1h)
EX_STRIDE=5 uv run --with ultralytics --with opencv-python-headless --with av \
    --with imageio --with imageio-ffmpeg --with matplotlib \
    python experiments/trajectory/extract_tracks.py
# identity + rail-anchored projection + validation + plots
uv run --with opencv-python-headless --with av --with scipy --with scikit-learn \
    --with matplotlib python experiments/trajectory/project_traj.py
```
