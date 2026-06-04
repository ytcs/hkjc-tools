# CV Trajectory PoC — Results (ONE race: HV 2026-06-03 R1)

**Status:** proof-of-concept, single race, depth-not-breadth.
**Race:** Happy Valley, 2026-06-03, Race 1 (Mount Butler Handicap), **1650 m, "C" course
(turf), Good To Firm, 12 runners**, winning time 1:38.85 (98.85 s). Winner #3 FAMILY FORTUNE.
**Source video:** `data_cache/video/drone_20260603_01.mp4` — 1920×1080, 25 fps, 143 s.
**Code:** `experiments/trajectory/{race_ground_truth,detect_probe,rail_probe,extract_tracks,project_traj}.py`
**Artifacts** (all in `data_cache/video/poc_out/`):
- `probe_{plain1920,plain1280,sahi512}_*.png` — detection-config comparison frames.
- `rail_probe_*.png` — white-rail segmentation (homography landmark) sanity.
- `ovr_montage_*.png` — YOLO boxes drawn on overhead frames through the run (incl. the
  det=0 dead-zone frames).
- `annotated_overhead_<tag>.mp4` — the early detect+track overlay video.
- `ovr_tracks_<tag>.png` — pixel-space foot-point tracks (shows fragmentation).
- `topdown_<tag>.png` — **the money plot**: recovered `(d, off)` trajectories.
- `speed_<tag>.png` — recovered speed-vs-distance per tracklet.
- `tracks_raw_<tag>.npz` (per-detection boxes+silk Lab), `traj_<tag>.npz` (projected `(d,off)`).

---

## TL;DR / go-no-go

**Conditional NO-GO for using THIS public drone feed to sharpen v_c/W' — for two
structural reasons, only one of which is fixable with more engineering:**

1. **(Fixable) Detection of small overhead horses is solved** by plain YOLOv8x @ imgsz=1920
   (no SAHI needed) — it hits the full 12-horse field cleanly *when the field is laterally
   spread*. So "can we see the horses at all" is a YES.
2. **(NOT fixable from this clip) The drone clip is a broadcast edit, not a continuous
   rectified overhead.** Two hard limits fall straight out of the footage:
   - It cuts to **ground-level cameras for the final ~400 m** (the home straight), so the
     last sectional — the one where W′ depletion is most visible — is **never overhead**.
   - During the **bottom bend (~t 18–47 s, d≈1320→1000 m)** the field runs **single-file
     directly away from the camera**: 12 overlapping horses read as 0–2 detections. This is
     a ~27 s / ~300 m **detection dead-zone** that no detector setting fixes (it is occlusion
     + viewing angle, not resolution). Even on the "good" back straight, live detection
     averages well below 12 because the field stays bunched and partly tail-on.
3. **`off` (lateral / ground-loss) — the single highest-value hidden signal — has the lowest
   accuracy ceiling here.** The oblique view + a horse-length scale prior + the fact that the
   running rail (not the outer) is the only reliable landmark make lateral metres a
   *coarse lane indicator at best*, exactly the axis the energetics model would most want.

**Net:** the overhead window yields a usable **along-track speed-vs-distance curve for the
middle of the race** (validated against sectionals below), but **not** a clean full-race
per-horse `(d, off)` trajectory, and **not** the final-400 m kick where W′ lives. To get
those you would need either (a) the raw Trakus feed (ruled out — too expensive), (b) a
purpose-shot continuous overhead (HKJC doesn't publish one), or (c) Mode-A broadcast CV with
per-frame homography (the months-long research project). **Recommendation: do NOT build a
production drone-CV trajectory pipeline for the energetics model on the strength of this clip.**
Keep the Tier-1 sectional features as the energetics input; revisit only if a continuous
rectified overhead source appears.

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

Full overhead-window pass (`extract_tracks.py`, yolov8x@1920, t∈[3,102]s, 207 frames @ ~2 Hz):

- **580 detections over 207 frames; det/frame median = 5.0, mean = 5.2, max = 12, min = 1.**
- **Only 13 % of frames (14/111 with any detection) reach ≥10 horses.** The full field is
  seen only in the brief windows where it spreads laterally (start straight, parts of the
  back straight / home turn). The modal state is ~5 visible because the field bunches
  single-file along the rail and is partly tail-on.
- Throughput: ~12–17 s/frame on CPU (no GPU); the whole overhead window ≈ 50–90 min.

See `poc_out/ovr_montage_*.png` for boxes drawn on frames and
`poc_out/ovr_tracks_<tag>.png` for the pixel-space foot-point tracks (short, central,
camera-following — visually confirms the fragmentation).

---

## 2. IDENTITY

- **Saddlecloth OCR is out** for the trajectory window (numbers sub-legible at 30–50 px). It
  *is* legible on the ground-level finish frames (verified: numbers 1,2,3,5,6,8,9,12 readable),
  but those frames are outside the homography window, so OCR can anchor finish order but not
  in-running identity.
- **Silk colour is the operative ID handle.** Each detection's rider-torso ROI (upper 45 % of
  the box) is reduced to a mean Lab colour; tracklets are clustered (k=12) by mean silk colour.
- **ID fragmentation** (tracklets per true runner) is the measurable quality number:
  **ByteTrack produced 134 track-ids for the 12 runners** (raw fragmentation ≈ 11× — IDs
  switch at every bunch/occlusion; no tracklet survives ≥20 frames). Restricting to the 44
  tracklets ≥5 frames and **clustering their mean silk-colour into k=12 groups gives a median
  of 3.5 tracklets per colour-cluster** (mean 3.7) — i.e. colour clustering recovers the
  12-runner structure but each runner is still split across ~3–4 fragments that must be
  stitched. With true per-runner silk swatches (HKJC racecard) the colour map could assign
  clusters to saddle numbers; that join is not wired here.

So identity is **partially recovered** (12 colour groups, ~3.5 fragments each), not the clean
1-track-per-horse the energetics model would want.

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

**What came out** (`poc_out/topdown_<tag>.png`, `speed_<tag>.png`): the recovered `(d, off)`
trajectories appear as **three disconnected clusters** at d≈1620 m (start straight),
d≈1340 m, and d≈900 m — exactly the three frame-segments where detection worked, separated
by the bottom-bend detection dead-zone (over which `d` only coasts at a nominal speed).
`off` spans ~0–16 m (a believable lane band), with occasional 30 m+ excursions that are
inner-rail-fit artifacts. Recovered speeds sit at **~5–9 m/s**, i.e. roughly **half** the
true ~16 m/s gallop — see the accuracy note.

**Honest accuracy ceiling (measured):**
- `d` / along-track speed: the background-flow camera-pan IS recovered (≈60–150 px/step,
  consistent), so the *method* is sound — but the **metre scale is biased ~1.8× too small**.
  The bias is the horse-length prior (2.4 m / median box long-side): on the oblique overhead
  the box "length" is mostly the foreshortened body width, so metres-per-pixel is
  underestimated and the integrated distance/speed comes out low. Result: recovered speeds
  ~5–9 m/s vs true ~16 m/s. A per-segment scale calibration (e.g. from the evenly-spaced
  white rail posts as a metric ruler, or anchoring to the published split at one marker)
  would remove most of this — but it is a real, open calibration gap, not a solved one.
- `off` / lateral: **the weakest axis** and not reliably better than the `d` axis here.
  Lane assignment (inner / mid / wide) is plausible; sub-metre ground-loss is **not**
  achievable from this angle with this scale source.

---

## 4. VALIDATION vs HKJC sectionals

Ground truth (`race_ground_truth.py`, scraped from the HKJC sectional page): 4 splits/horse
summing to 98.85 s; HKJC standard scheme ⇒ sections ≈ [450, 400, 400, 400] m
(d markers 1650/1200/800/400/0). Only the first **three** sections fall inside the overhead
window; the last 400 m is on the ground-camera (out of window).

**Measured:** the overhead window's recovered `d` reaches only **d ≈ 880 m** (the field
"travels" 1650→880 m before detection runs out / the clip cuts), so only the **1650→1200 m
marker crossing** lands inside it. The field crosses **d = 1200 m at a recovered t ≈ 52.5 s**
from the start, vs the winner's published **cumulative 28.2 s** at the same marker — an error
of **+24 s (~1.85×)**. That factor is the same ~1.8× scale bias as the speed curve: the
section-time check therefore **fails the sectional-tolerance gate by a wide margin**, and it
*localises the failure to the metric-scale calibration* (the camera-pan trajectory shape is
right; its metre scale is wrong). Markers at 800 / 400 / 0 m are outside the recoverable
window entirely (dead-zone + ground-camera final 400 m).

> Interpretation: this is **not** "CV can't see the race" — detection and the pan-based
> motion model both work. It is "the public drone clip doesn't give a continuous,
> scale-calibratable overhead of the *whole* race," so the one number we can check against
> ground truth is off by ~1.8× and the other three markers can't be checked at all.

---

## 5. Binding constraints (ranked)

1. **Source coverage (fatal for the kick).** The clip is a multi-camera broadcast edit; the
   final ~400 m is ground-level only. The most W′-informative part of the race is
   un-extractable from the overhead. *Not fixable without a different video source.*
2. **Bottom-bend detection dead-zone (~300 m).** Field single-file, tail-on → ~0 detections
   for ~27 s. Tracking can only motion-coast `d` through it (via the background-flow pan);
   per-horse `off` is simply absent there. *Not fixable with detector tuning.*
3. **Lateral (`off`) accuracy ceiling.** Oblique view + horse-length scale + single-rail
   anchor → `off` is a coarse lane band, not metres. The ground-loss term the model wants
   most is the least trustworthy output. *Partially improvable with a learned track-template
   homography; never sub-metre from this angle.*
4. **Identity is silk-colour-only in-window** (numbers illegible overhead) and ByteTrack
   fragments heavily at every bunch. Colour clustering recovers ~12 groups but the
   cluster→saddle-number map needs the declared-silks racecard (not wired here).
5. **Compute.** CPU-only, ~15 s/frame at imgsz 1920; a single race's overhead window is ~1 h.
   Fine for a PoC, a real bottleneck for scale-out (needs a GPU).

---

## 6. Reproduce

```bash
cd /home/ytc/Projects/hkjc-tools
# detection comparison
uv run --with ultralytics --with sahi --with opencv-python-headless --with av \
    python experiments/trajectory/detect_probe.py
# full overhead detect+track pass (CPU-bound ~1h; stride 12 ≈ 2 Hz; checkpoints npz every 40 frames)
EX_STRIDE=12 EX_VIDEO=0 uv run --with ultralytics --with opencv-python-headless --with av \
    --with imageio --with imageio-ffmpeg --with matplotlib \
    python experiments/trajectory/extract_tracks.py
# identity (silk-colour) + rail-anchored projection (off) + background-flow d + validation + plots
uv run --with opencv-python-headless --with av --with scipy --with scikit-learn \
    --with matplotlib python experiments/trajectory/project_traj.py
```

Notes / gotchas encountered (so a re-runner doesn't repeat them):
- **Decode with PyAV**, not `cv2.VideoCapture` (fails to open these H.264 MP4s in this env).
- Ultralytics auto-installs `lap` on first `.track()` and prints a benign "restart runtime"
  warning — it does **not** require a restart; the run continues.
- On 15 GB RAM, writing the annotated mp4 (`EX_VIDEO=1`) while holding all detections pushed
  the box into **swap thrash** and a ~4× slowdown — run with `EX_VIDEO=0` for the data pass.
- `extract_tracks.py` checkpoints the npz every 40 processed frames, so a killed run is still
  usable up to the last checkpoint. Launch detached (`setsid … &`) so a harness/process-group
  kill of the launching shell doesn't take the worker with it.
- NumPy 2.0 removed `ndarray.ptp()` → use `np.ptp(arr)`.
