---
title: Aerial frame classification — render_active now rejects the live->render cross-dissolve frames
id: 0111-aerial-render-active-dissolve-frames
author: modelling
audience: ["modelling", "all"]
channel: handoff
kind: finding
status: open
thread:
tags: ["cv", "aerial", "classification", "render_active", "handoff"]
refs: {"updates": "0110", "task": 104}
created: 2026-06-08T11:30:00+00:00
updated: 2026-06-08T11:30:00+00:00
---
## Frame classification — the recon was running on blended dissolve frames

Debugging the pipeline from the front: `render_active` (the gate that decides a frame is the AVV
render, not the live broadcast intro) **let cross-dissolve frames through**. The replay opens with a
full-width live broadcast and CROSS-DISSOLVES (~5 frames) into the split-screen render. The old gate
keyed only on the bottom SILK-LEGEND strip, which **fades in DURING the dissolve** — so it flickered
on for blended frames whose track region was still live footage (R08 frame 331 then 333/334/335; the
big low-angle live horses ghosted over the small render blobs). The rail + horse detection then ran on
garbage. Frame 331 was the user-cited example.

### Root cause + the two complementary gates (validated on R08, R03, R02A; src `aerial_layout`)
A full-race feature scan (`/tmp/feat_scan.py` → `data_cache/_feat_<RACE>.npy`) over every frame found
two structural invariants that separate a CLEAN render from a blend, catching DIFFERENT signatures:
- **TOP_GREEN_MAX = 0.30** — the render's top strip is the dark odometer/clock box (green frac
  ~0.1–0.2); a blend leaks turf+horses there (up to ~0.95). Catches **R02A 336–338** (topgreen ~0.83)
  which texture alone PASSES (lap 279–465).
- **TRACK_LAP_MIN = 240** — a clean top-down render is sharp/detailed (Laplacian var of the track band
  ≥266; mid-race 550–820); a blend is smooth/ghosted (≤216). Catches **R08 334/335** (lap 161/216)
  which the top-strip gate PASSES (topgreen already 0.14/0.23).

Both are required (AND-ed with the existing legend ≥1). Either alone leaks one race's dissolve.

### Validation (whole race, all three)
- R08: rejects exactly {331, 333, 334, 335}; 0 other frames; first clean = 336.
- R02A: rejects exactly {336, 337, 338}; first clean = 339.
- R03: rejects 0 extra (its dissolve never tripped the legend); first clean = 300.
- **No CLEAN frame wrongly rejected** in any race: min clean lap 266 / 435 / 559 (all ≥240), max
  clean topgreen ≤0.20 (< 0.30). End-of-race frames stay clean (R08 last passed 2788, lap ~820).
- The texture threshold is the tighter one (R08 gap 216→266); the top-green gate gives the margin.
  Thresholds are calibrated to the AVV product resolution — re-check for a new venue/source.

### Knock-on
- Stale caches deleted (`cv_trackid_pass1_*`, `cv_tracks_*`): they were built including the blended
  frames, so the recon must rebuild. The odometer obs are robust (robust line-fit absorbs the few
  blended reads) but should rebuild too on next `cv_build_obs` run.
- New scratch `studies/cv_frame_dump.py`: dumps EVERY clean-render frame with the fitted running rail
  (green) + outer rail (blue) + s-ranked horses (red) + off-track rejects (magenta) + a per-frame
  header (in-lane count, bend/straight, outer found, lap, rail OK/FAIL) to `data_cache/diag/frames/`
  for manual frame-by-frame scrutiny.
