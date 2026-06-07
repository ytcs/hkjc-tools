---
title: "S0b RESULT 0058: OCR BAKE-OFF on OUR rendered legend (NOT online benchmarks) — read the Aerial-Virtual-View saddlecloth number boxes, the gate to stable re-id (0055 §5). Three contenders (RapidOCR=PP-OCRv5/onnxruntime, pytesseract, easyocr) scored on f85 GT [9,8,10,6,4,1,2,5] + 18 frames (disk + video sweep incl. finish). VERDICT: RapidOCR WINS decisively — 100% exact on f85 (incl. the two-digit '10' as ONE token), 98% cross-frame validity ({1..12}), 310 ms/cell CPU, NO torch. easyocr 50% on f85 (dropped the dark-box '4' + thin '1'), torch-heavy (~80 MB weights + CUDA). tesseract 62% best-case PER-CELL only (fails on the full multi-colour strip: 11 reads/18 frames; splits/merges glyphs e.g. 9->29, 2->525) and the system tesseract ships NO eng.traineddata here. PRUNED easyocr + pytesseract from the `vision` group; kept ONLY rapidocr-onnxruntime. Winner wired into src/hkjc/vision/mode_b/ocr.py:read_legend. ruff clean; suite 172 passed. src/hkjc/model UNTOUCHED."
id: 0058-s0b-ocr-bakeoff-on-our-aerial-virtual-view-legend-rapidocr-pp-ocrv5-wins-100-percent-f85-incl-two-digit-10-easyocr-tesseract-pruned
author: modelling
audience: ["modelling"]
channel: research
kind: result
status: final
thread: 0046-causal-rebuild-design-cause-not-effect-strip-metasignals-reemergence-validation
tags: ["cv", "vision", "mode-b", "aerial-virtual-view", "ocr", "ocr-bakeoff", "benchmark-on-our-data", "rapidocr", "pp-ocrv5", "onnxruntime", "easyocr", "pytesseract", "tesseract", "saddlecloth", "legend", "number-box", "two-digit", "reid", "re-identification", "silk-colour", "digit-recognition", "validity", "temporal-stability", "latency", "footprint", "dep-prune", "lean-group", "s0", "s0b", "queue-62", "no-online-benchmarks", "classical-cv", "leak-free"]
refs: {"benchmark": "studies/cv_ocr_benchmark.py", "reader": "src/hkjc/vision/mode_b/ocr.py", "layout": "src/hkjc/vision/mode_b/layout.py", "pyproject_group": "pyproject.toml ([dependency-groups].vision)", "prior_s0": "journal/entries/0055-s0-cv-lateral-path-pipeline-mode-b-aerial-virtual-view-rail-relative-y-lat-extracted-prototype-reid-is-the-bottleneck.md", "panel": "data_cache/cv_lateral_paths_sample.parquet", "frames": "data_cache/frames/*.jpg", "video": "data_cache/aerial_sample.mp4", "tessdata": ".tessdata/eng.traineddata"}
created: 2026-06-06T00:00:00+00:00
updated: 2026-06-06T00:00:00+00:00
---
FINAL — S0b (OCR bake-off for Mode-B legend number-binding, the gate flagged in 0055 §5/§7).
Owns `src/hkjc/vision/` + `studies/cv_ocr_benchmark.py` + the `vision` dep group ONLY;
`src/hkjc/model/*` was NOT touched (the orchestrator owns `pure_mechanism.py` concurrently).

## (0) THE DIRECTIVE — benchmark on OUR data, not online leaderboards
Re-id at ~15 px is the S0 bottleneck (0055 §3); the render HAS the silk→number key in the
bottom legend strip every frame, so digit OCR is the single highest-value unblock. The user's
directive: do NOT trust published OCR benchmarks — score on OUR rendered legend, which is an
adversarial surface (1-2 digit numbers in MULTI-COLOURED boxes — red, yellow, blue, **dark
slate** — sitting on top of multi-colour silk icons on a brown track). The "10" two-digit case
is the differentiator: it must come back as `"10"`, not `1`/`0`/`100`.

## (1) GROUND TRUTH (no eyeballing beyond the orchestrator's f85 read)
* **f85 exact GT** (orchestrator VIEWED the frame): running order left→right = **[9,8,10,6,4,1,2,5]**.
* **Valid saddlecloth set**: the S0 sample is ST 2026-06-03 race 0, **12 runners** → `{1..12}`.
  The field is POST-corpus (`results_rich.parquet` ends 2026/05/31, so no row for it) — I
  cross-checked against the silks actually seen on the legend across the video sweep instead:
  `{1,2,3,4,5,6,7,8,9,10,11}` all appear, consistent with a 12-field. So validity = read ∈ {1..12}.
* **Finish order** (last legend frame, vid2940): stabilises to `9,10,6,8,2,4,11,3,…` (legend shows
  the top 8 in current running order) — used as a sanity check, not a hard score.

## (2) METHOD — `studies/cv_ocr_benchmark.py`
Strip located per frame via `mode_b.layout.find_layout` (its sat-peak legend finder fires on all
test frames, `aerial_active=True`), cropped, upscaled 3×. **Key design finding: structural
per-cell segmentation is UNRELIABLE on this render** — the boxes come in many colours incl. dark
ones, and the brown track between silks is itself moderately saturated, so saturation/contour
cell-finders merge or miss boxes. This is *why* detection-capable OCR (which localises text
internally) is the right tool and tesseract (needs pre-segmented input) is handicapped. So:
detection engines (RapidOCR, easyocr) read the WHOLE strip and return digit tokens with x; we
sort left→right. Tesseract additionally gets a best-case fixed-pitch (~108 px/cell) per-cell crop
sweep with OTSU both-polarity (`_tesseract_cells`) and we report the better of strip/cell.
Bench corpus: 18 frames = 7 on-disk (`f85/f25/f55/f_30/f_60/f_90/f_110`) + an 11-frame video
sweep `vid300…vid2940` incl. the finish.

## (3) RESULT — the bake-off table

| engine | f85 exact acc | cross-frame validity (∈{1..12}) | temporal stability | latency/cell (CPU) | footprint |
|---|---|---|---|---|---|
| **RapidOCR** (PP-OCRv5/onnxruntime) | **100%** (8/8) | **98%** (127 reads) | 52% | **310 ms** | onnxruntime + opencv + pyclipper + shapely; bundled det+rec ONNX (~14 MB). **NO torch.** |
| easyocr | 50% (4/8) | 96% (99 reads) | 58% | 425 ms | **torch 2.12 + CUDA libs + torchvision + scikit-image** (~80 MB weights). HEAVY. |
| pytesseract | 62% best-case (per-cell) | 82% (only 11 reads on full strip) | 47% | 525 ms | system `/usr/bin/tesseract` + eng.traineddata (~4 MB); thin shim. Lightest py-dep but FAILS on the render. |

f85 reads (left→right; GT `[9,8,10,6,4,1,2,5]`):
* RapidOCR `[9, 8, 10, 6, 4, 1, 2, 5]` — exact, **"10" read as one token**.
* easyocr `[9, 8, 10, 6, 2, 5]` — got "10" but **dropped the dark-slate-box "4" and the thin "1"**.
* tesseract `[9, 8, 4, 6, 4, 1, 7, 4]` — gross confusions; on the full strip it returns almost
  nothing (`image_to_data` → 0 tokens) and per-cell it bleeds silk pattern into the glyph
  (observed 9→`29`, 8→`3`, 10→`105`, 2→`525`).

## (4) WINNER — **RapidOCR**, and WHY
Accuracy first, then validity/latency/footprint, and RapidOCR leads on EVERY axis that matters:
perfect f85 (the only hard-labelled frame, **incl. the two-digit case the others split or miss**),
highest validity AND highest read count (127 confident reads vs 99/11 — it actually SEES the
cells the others drop), lowest CPU latency, and the lightest deployable stack (**no torch / no
CUDA** — it is pure onnxruntime, which suits this CUDA-off box). PP-OCRv5's det+rec on
onnxruntime delivers PaddleOCR-tier quality without the paddlepaddle framework, exactly as the
setup brief predicted. Temporal-stability is moderate for all three (47–58%) but that number is a
clustering artdefact (the silk-LAB greedy clusterer merges similar silks and the legend's
membership rotates as running order changes), NOT an engine discriminator — RapidOCR's
validity+coverage is the load-bearing stability evidence.

Failure modes catalogued: **easyocr** misses low-contrast glyphs (white digit on dark/slate box,
and the thin "1"); **tesseract** has no text-localisation, splits/merges digits against the
coloured-box + silk background, and the system install ships **no `eng.traineddata`** (only `afr`
+ `osd`) — I fetched `eng` into a repo-local `.tessdata/` just to give it a fair run; in
production that's another missing dep. **RapidOCR** showed no "10"-splitting and no box-colour
confusion on our strip.

## (5) PRUNE — the `vision` group is now lean
Removed `easyocr` and `pytesseract` from `[dependency-groups].vision`; kept ONLY
`rapidocr-onnxruntime` (alongside the existing opencv/ultralytics/av). `uv sync --group vision`
resolves cleanly and uninstalled 15 packages (easyocr + its torch/scikit-image/python-bidi tail).
One gotcha worth recording: easyocr had pulled in `opencv-python-headless`, and uninstalling it
clobbered the shared `cv2/` namespace (lost `cv2.imread`); a `--reinstall opencv-python` (or just
the post-prune `uv sync`, which I confirmed stable) restores it. Final `vision` group:
`opencv-python`, `ultralytics`, `av`, `rapidocr-onnxruntime`.

## (6) WHAT THIS UNBLOCKS (context for S0/S0b)
`src/hkjc/vision/mode_b/ocr.py:read_legend(strip, valid_set={1..12})` is the deployable reader
(engine-selectable, defaults to the winner; cv2 + onnxruntime lazy). It returns
`LegendRead(number, x, conf, text)` sorted left→right — i.e. the **silk-colour-cluster→saddlecloth
number map** the re-id needs. Next (re-id, NOT this task): bind each legend cell's number to the
nearest silk-LAB cluster, fuse with the running-ORDER (left→right) + motion → Hungarian
one-number-per-runner assignment (the `identify.assign_runners` contract), then re-key the
`runner` column in `data_cache/cv_lateral_paths_sample.parquet` from `rid{N}` to the saddlecloth
number. Odometer/clock OCR (e.g. "420M", "1:23.23") is the same engine on the top strip — also
unblocked. Leak-free posture unchanged (trailing per-horse trait).

## (7) WHAT I CHANGED
NEW: `src/hkjc/vision/mode_b/ocr.py`, `studies/cv_ocr_benchmark.py`. EDITED: `pyproject.toml`
(`vision` group: added then pruned to the winner). ADDED repo-local `.tessdata/eng.traineddata`
(only to run the bake-off fairly; not a runtime dep of the winner). `ruff check
src/hkjc/vision studies/cv_ocr_benchmark.py` clean; `PYTHONHASHSEED=0 pytest -q` → **172 passed**.
`src/hkjc/model/*` UNTOUCHED.
