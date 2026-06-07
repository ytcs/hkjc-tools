---
title: "S0 RESULT 0055: CV lateral-path pipeline (#62) — the aerial_sample IS HKJC's synthetic *Aerial Virtual View* (top-down render, drawn rails, silks+number legend), so Mode B is a CLASSICAL colour-seg problem, not a deep-detector one (generic-YOLO 0-det failure stands). Built a runnable modular pipeline (layout→detect→rail→reid→rail_transform) + driver on the 1600m ST sample: 1,187 rows, 12 runners, x_arclen 0→1597m, y_lat[1.8,31.9]m (100% within the 30.5m track width, median 14.4m). The RAIL-RELATIVE lateral coordinate extracts cleanly; the BOTTLENECK is colour-only re-id at ~15px (per-runner coverage 7–50%, ID-swaps between similar silks) — number-binding via legend OCR is the gating scale-up step. Leak-free by construction (trailing per-horse trait). src/hkjc/model UNTOUCHED."
id: 0055-s0-cv-lateral-path-pipeline-mode-b-aerial-virtual-view-rail-relative-y-lat-extracted-prototype-reid-is-the-bottleneck
author: modelling
audience: ["modelling"]
channel: research
kind: result
status: final
thread: 0046-causal-rebuild-design-cause-not-effect-strip-metasignals-reemergence-validation
tags: ["cv", "vision", "mode-b", "aerial-virtual-view", "lateral-path", "rail-relative", "y-lat", "arc-length", "homography", "stabilisation", "reid", "re-identification", "silk-colour", "saddlecloth-ocr", "colour-segmentation", "classical-cv", "yolo-fails-on-render", "trajectory", "ground-lost", "c_da", "drafting", "jockey-control-latent", "geometry-x-control", "surface-3", "leak-free", "trailing-trait", "prototype", "scale-up", "s0", "queue-62", "trajectory-dataset", "join-key"]
refs: {"pipeline_pkg": "src/hkjc/vision/mode_b/", "driver": "studies/cv_path_extraction.py", "panel_out": "data_cache/cv_lateral_paths_sample.parquet", "figure": "data_cache/figs/cv_lateral_paths_sample.png", "video": "data_cache/aerial_sample.mp4", "geometry": "src/hkjc/core/track.py", "join_target": "data_cache/trajectory_dataset_full.parquet", "feasibility_probe": "studies/cv_feasibility_probe.py", "prior_overhead_log": "data_cache/video/poc_out/extract_run.log"}
created: 2026-06-06T00:00:00+00:00
updated: 2026-06-06T00:00:00+00:00
---
FINAL — S0 (CV lateral-path extraction, queue #62). A first prototype that turns the
aerial race video into rail-relative runner paths, plus an honest quality read and a
scale-up plan. Owns `src/hkjc/vision/` + `studies/cv_*.py` ONLY; `src/hkjc/model|core`
and `data_cache` panels are UNTOUCHED (geometry consumed read-only via `hkjc.core.track`).

## (0) THE SUBSTRATE — `aerial_sample.mp4` is the *Aerial Virtual View*, not a drone
The clip (1920×1080, 25 fps, 118 s) is HKJC's **synthetic "Aerial Virtual View"** product,
not raw aerial footage. Each frame is split: the LEFT half is a real broadcast pan
(occluded / motion-blurred — ignored); the RIGHT half is a **top-down rendered animation**
of the runners on a uniform brown track, with **both rails drawn as white curves**, a
**distance odometer** ("1600M"→"100M"), and a **bottom legend strip** of silk icons +
saddlecloth numbers in current running order. HKJC itself labels it an approximation
("refer to Real Replay for actual results").

This reframes the task. The prior recon (journal trail + `studies/cv_feasibility_probe.py`,
`poc_out/extract_run.log`) showed generic YOLO returns **0 detections on 88/111 overhead
frames** — that verdict STANDS (CUDA is off here; a COCO `horse` class does not fire on a
~15 px synthetic sprite). But on the *render* the right tool is **classical colour
segmentation**: runners are vivid silk blobs against a low-saturation track, the rails are
literally drawn, and the camera-motion problem dissolves because we work in **rail-relative
coordinates measured per frame** — the per-frame rail fit IS the lateral-axis stabiliser;
no global homography is needed for `y_lat`.

## (1) THE PIPELINE — modular, in `src/hkjc/vision/mode_b/`
Import-safe with no heavy deps (cv2/av/numpy lazy). Five stages + driver:

| module | does |
|---|---|
| `layout.py` | per-frame ROIs of the right panel (legend band located from the data; the track band drifts with the camera); flags `aerial_active` (start-gate frames are not) |
| `detect.py` | vivid silk blobs in the track ROI; **down-weights the brown track hue** so texture/infield is suppressed; caps to `field_size+2` largest |
| `rail.py` | fits the **inner (runner-side)** rail: per scan-line takes the white band NEAREST the field centroid, robust quadratic — NOT a global fit through all white (which lands mid-track) |
| `reid.py` | greedy motion linkage → tracklets; then **motion-anchored** stitch (keep long tracklets as identity anchors, fold only short fragments by silk colour) |
| `rail_transform.py` | px→m scale from drawn track WIDTH / rail separation; `y_lat` = perp offset×scale; `x_arclen` = field-centroid path integral anchored to race distance |
| `studies/cv_path_extraction.py` | driver: runs on the sample, writes the panel + a sanity diagnostic + figure |

Driver runs SYNCHRONOUSLY: `uv run --group vision --group dev python studies/cv_path_extraction.py`.

## (2) RESULT — the panel (one race, ST 1600m)
`data_cache/cv_lateral_paths_sample.parquet`, **1,187 rows × 13 cols**, tidy
`(venue,date,race_no,runner,t,frame, x_arclen, y_lat, rail_offset_px, silk_lab_{l,a,b}, confidence)`.

| metric | value | read |
|---|---|---|
| runners recovered | 12 | matches a full field |
| `x_arclen` range | 0 → **1597 m** | recovers the **1600 m** race length (centroid integral, anchored) |
| `y_lat` | min 1.8 / **median 14.4** / max 31.9 m | spread across the track, sensible |
| `y_lat` within [0, width] | **100.0%** | track width (course A) = 30.5 m |
| arc-length monotone | **100%** of steps non-decreasing | the longitudinal axis is well-behaved |
| mean confidence | 0.93 | detection+rail+scale composite |

`y_lat` is the **NEW** signal this unlocks; `x_arclen` is the longitudinal axis that
**JOINS to `data_cache/trajectory_dataset_full.parquet`** on `(venue,date,race_no,runner)`
with `x_arclen ↔ sectional d`. (Sample `runner` is currently a silk-cluster id `rid{N}`;
binding to the saddlecloth number is the gating step — see §3.)

## (3) HONEST QUALITY — the re-id is the bottleneck, not the geometry
- **Lateral coordinate: GOOD.** Bounded, physical, 100% on-track after fitting the *inner*
  rail per frame and rejecting infield-side detections. The early naive global rail fit
  gave `y_lat` up to 271 m (mid-track reference); the near-band fit + on-track filter
  closed it to [0,30.5]. This is the load-bearing correctness fix.
- **Re-identification: WEAK — the real limit.** Colour-only re-id at ~15 px cannot cleanly
  hold 12 identities for a full race: per-runner coverage is **7–50%** of frames and paths
  still ID-swap between similar silks (visible as lane-jumps in the figure). The render's
  legend HAS the silk→number key every frame, but binding it needs digit OCR, which is the
  missing piece (`easyocr` is NOT installed — see §5).
- **Detection recall: MODERATE.** Dark/low-sat silks and the horse body (vs the silk cap)
  are missed; we kept recall clean-over-complete on purpose (a missed runner beats an
  infield false positive). Mean ~7 blobs/frame for a 12-field.
- **Arc-length: PROXY.** Centroid-integral anchored to the known distance, NOT the odometer
  (OCR again). Monotone and right-scaled, but it is a field-level series, not per-runner —
  fine for joining, not yet for per-runner ground-lost.
- **Leak-free by construction.** Like form, the (d, y_lat) path is a TRAILING per-horse
  trait computed from the race itself; averaged over a horse's PAST runs and joined on
  `horse_id` it is leak-free for future races (same argument as the feasibility probe).

## (4) WHY IT MATTERS (RESEARCH_PROGRAM §1/§2/§3)
`y_lat` is the lateral path the longitudinal sectional dataset cannot see. It is the raw
material for: **ground-lost / path efficiency** (extra metres run wide on bends),
**C_dA / drafting** (lateral spacing vs the runner ahead), the **jockey-control latent a**
(how a rider places the horse laterally), and **geometry×control** de-confounding. RQ-ctl1/
ctl2 (control vs ground-saved) need exactly this lateral axis.

## (5) SCALE-UP PLAN for the full-arc backfill
1. **Number-binding (the gate).** Add `easyocr` (or a tiny digit CNN) to OCR each legend
   cell's number box, build a stable silk→number map per race, and key `runner` to the
   saddlecloth number. This is the single highest-value next step; without it the panel is
   anonymous. `easyocr` is **not in the `vision` group** — propose adding it (NOTED, not
   installed globally).
2. **Re-id robustness.** Fuse the legend running-ORDER (left→right = current order) with
   silk colour and motion → Hungarian assignment under one-number-per-runner, the
   `identify.assign_runners` contract already sketched in `src/hkjc/vision/identify.py`.
3. **Odometer OCR** for true per-frame `x_arclen` (replaces the centroid-integral proxy),
   enabling per-runner ground-lost.
4. **Coverage.** The Aerial Virtual View covers the FULL arc (bends + straight), unlike the
   broadcast pan — so once re-id holds, coverage is a render-availability question, not a
   camera-coverage one. Per-race the render exists for most cards; the corpus join is on
   `(venue,date,race_no)`.
5. **Occlusion.** The pack bunches on bends; bridge gaps with the existing motion model +
   Kalman/RTS smoothing (the `project.build_trajectory` contract) once IDs are stable.
6. **Validation.** Wire `export.qa`: integrate recovered speed to each sectional boundary,
   diff vs published sectionals (the kill criterion); check finish-order vs official result.

## (6) WHAT I CHANGED
NEW: `src/hkjc/vision/mode_b/{__init__,layout,detect,rail,reid,rail_transform,pipeline_b}.py`,
`studies/cv_path_extraction.py`. WROTE `data_cache/cv_lateral_paths_sample.parquet` +
`data_cache/figs/cv_lateral_paths_sample.png`. The existing `src/hkjc/vision/*.py` stubs
(broadcast Mode-A scaffold) are LEFT in place — they target the side view and a different
detector path; Mode B is the tractable substrate for THIS video. `ruff check
src/hkjc/vision studies` clean; full suite **172 passed**. `src/hkjc/model|core` untouched.

## (7) BLOCKER
`easyocr` / a digit model is missing from the `vision` dep group; the legend number-binding
(step 1, the gate to a usable backfill) needs it. Everything else (cv2/av/ultralytics/numpy)
is present. CUDA is off — fine for classical Mode B, but a fine-tuned detector path (Mode A)
would want a GPU.
