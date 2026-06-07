---
title: "DRAFT 0041 — Fork-3 CV frontier scoping: #34 full-arc lateral (x,y,t) + #22 gait; binding constraints + a working lateral-path POC"
id: 0041-fork3-cv-frontier-scoping-34-lateral-22-gait-poc
author: modelling
audience: ["all"]
channel: research
kind: scoping
status: draft
thread:
tags: ["vision", "cv", "lateral-path", "trajectory", "gait", "biomechanics", "geometry-control", "theory-3c", "draw", "re-identification", "homography", "poc", "queue-34", "queue-22", "fork-3", "factor-audit-39"]
refs: {"path_sample": "data_cache/path_sample.parquet", "extractor": "studies/cv/extract_path.py", "poc": "studies/cv/lateral_poc.py", "video_drone": "data_cache/video/drone_20260603_01.mp4", "vision_pkg": "src/hkjc/vision"}
created: 2026-06-06T00:00:00+00:00
updated: 2026-06-06T00:00:00+00:00
---

Scoping Fork-3 (the CV frontier — the only path to genuinely NEW observables after the #39
factor audit showed the race-panel channels collapse to one ability latent). Two queue items:
**#34** full-arc lateral (x,y,t) path extraction; **#22** gait/biomechanical condition.

## 1. What CV infra + data already exists

- **`src/hkjc/vision/` is a clean SKELETON** — schema (pure-stdlib dataclasses: `RaceContext`,
  `Detection`, `Tracklet`, `IdentityVote`, `Homography`, `TrajPoint`, `RunnerTrajectory`,
  `RaceTrajectory`; canonical output `(t, d=m-from-finish, off=rail-offset, speed)`) is real and
  import-safe with no heavy deps; **every stage fn (`detect/track/identify/homography/project`) is
  a `NotImplementedError` stub.** The working code lives in `studies/cv/`. Vision deps (`opencv`,
  `ultralytics`, `av`) are the `vision` group, lazy-loaded.
- **#21 IS BUILT and REPRODUCES** (`studies/cv/extract_path.py`, output committed to
  `data_cache/path_sample.parquet`). Classical-CV pipeline on the HKJC **drone/aerial** replay
  (`drone_20260603_01.mp4`, 1920×1080, 25fps; HV 2026-06-03 R1, 1650m, "C" course): ingest →
  **keyframe ORB+RANSAC background homography** (registers the panning camera to a common ground
  frame, bounded drift, 63/63 frames, bg-align MAE 0→~50) → per-frame **rail fit** (Hough white
  lines validated as turf boundaries, locked to the 22.5 m course width → 32 px/m) → classical
  foreground detection in the rail-bounded lane (off-the-shelf YOLO finds ~0 horses overhead, so
  detection is colour/temporal-foreground + watershed) → gap-tolerant NN tracking in ground coords
  → curvilinear projection to `(x_long, y_lat)`. **It produced 3 horse tracks over a 5 s / ~28 m
  start-straight window**, x monotone (mono=1.00), y bounded by track width. This SUPERSEDES the
  earlier 0024 "drone=defer / homography broken / 0 tracks >19 frames" verdict — homography and
  short-arc tracking are now solved on the straight.
- **#19 (control-quality / O3) is NOT a CV trait** — it is the `trajectory_dataset_full`
  rank-trajectory probe (section velocity + longitudinal rank, **no lateral coord**); the closest
  built feature is `features.trajectory_trait_table` (`traj_sustain`, `traj_fade`). The §3c O3
  ground-saved index from the rank panel is still open. CV is the only source of true `y_lat`.
- **Data on disk:** `drone_20260603_01.mp4` (47M, the working substrate), `aerial_sample.mp4`
  (20M), `rf_20250406_09.mp4` / `patrol_…` (20M each, broadcast SIDE view), plus `path_sample.parquet`
  (the #21 output) and `data_cache/video/poc_out/` (annotated videos, montages, npz). **Coverage =
  ONE race, partial arc. There is no backfilled path corpus.**

## 2. The gap + BINDING CONSTRAINT

### #34 — full-arc lateral (x,y,t). Delta from #21:
#21 gives a clean `(x,y,t)` on a **straight** segment for **a few** horses with **no stable named
identity** (tracks are `trkN_hueM` — anonymous colour blobs). #34 needs three things #21 does not have:
1. **Silk/saddle re-identification** — bind each track to a *named horse* (saddlecloth #) and hold it
   across the whole race. #21's overhead view has horses ~11–30 px → numbers/silks are **unreadable**
   overhead. Only 3 of ~12 runners even survive ≥22 frames.
2. **Full-arc homography on the BENDS** — #21's rail-fit assumes near-straight white rails; on the
   ~470–530 m of HV turn the rail is curved and the perspective rail-separation ruler degrades. The
   curvilinear `(d, off)` projection on a bend (where §3c actually lives — the arc-length tax) is
   unbuilt.
3. **Full-arc backfill** — many races, not a 5 s window.

**BINDING CONSTRAINT for #34 = re-identification, not homography or video.** Homography is largely
solved (#21). Video is interceptable as pre-rendered MP4 (confirmed; no structured (x,y) feed exists).
Compute is modest (classical CV, ~1 min/race-window). The wall is **maintaining stable named identity
across a full race on a ~20 px overhead target** where silks/numbers are sub-resolution. Without it,
paths cannot be joined to a horse_id and so cannot become a leak-free trailing trait. The likely fix is
NOT overhead OCR but **cross-view fusion**: read identity from the **broadcast SIDE view** (YOLO finds
the full field + on-screen saddlecloth numbers + a silks legend there — verified in `probe_side.png`)
and transfer it to the overhead track via finish-order / time alignment.

### #22 — gait/biomechanical condition. Feasibility:
Gait (stride frequency/length, symmetry, soundness) needs a **slow, isolated, side-on** subject and
clean leg visibility for animal pose estimation. **In-race footage fails this**: `probe_side.png` /
`biggest_horse_crop.png` show racing-pace motion blur on the legs, **rail-occluded lower legs**, and
heavy horse-on-horse occlusion. Resolution (1080p side view, 200–460 px boxes) is adequate for *bodies*
but not for clean *limb* tracking at gallop.

**BINDING CONSTRAINT for #22 = FOOTAGE ACQUISITION (a parade/walk feed), not the model.** The natural
place to read soundness is a pre-race **parade-ring / walk-up**, where the horse is slow and isolated —
and that is **leak-free pre-race by construction** and the only CV signal that helps **debutants**
(blank form). But HKJC's documented replay types are `replay-full | replay-aerial |
replay-aerial-virtual | trackwork` — **no parade/paddock feed in the URL scheme** (`docs/HKJC_DATA.md`).
So #22 is blocked on *sourcing the right footage* before any model question is even reachable.

## 3. POC result (what was demonstrated end-to-end)

Reproduced #21 and added `studies/cv/lateral_poc.py`, which turns the real extracted path into the
**§3c lateral observables** the section-rank panel structurally cannot produce:

| track | n | mean_off (m) | off range | v_med |
|---|---|---|---|---|
| trk1  | 22 | 0.82 | 1.9 | 7.6 |
| trk3  | 31 | 0.83 | 5.7 | 9.7 |
| trk46 | 24 | **6.62** | 13.2 | 7.5 |

**The §3c rail-saver vs wide-and-clear split is RESOLVED end-to-end from pixels:** two rail-huggers
(mean_off ≈ 0.8 m) vs one wide horse (mean_off ≈ 6.6 m). That is exactly the `y_i(t)` lateral channel
THEORY §3c needs and that `trajectory_dataset_full` (velocity + longitudinal rank, no `y`) cannot give.
**Honest accuracy notes:** (a) speeds read 7.5–9.7 m/s vs 15–19 physical — this is partly the standing
start but also the displacement is under-counted; (b) the watershed over-splits (blobs ~0.3 m vs a
~2.5 m horse+rider), so detections are fragments; (c) the wide track's `y_lat` is jittery → the derived
`ground_lost` is dominated by re-id/detection noise, not real weaving, so its ABSOLUTE value is not yet
trustworthy. What is DE-RISKED: video → camera-motion homography → rail-relative `(x_long, y_lat)` → a
per-horse lateral statistic, on real HKJC drone pixels.

## 4. Staged build plan

### #34 (lateral path) — the higher-value, lower-risk first investment
- **S1. Cross-view identity (the binding constraint). ~1–2 wk.** Run YOLO+OCR on the broadcast SIDE
  view to read saddlecloth numbers + silks per tracklet; align to the overhead track by finish-order /
  timing. De-risks re-id before any backfill. *Acquire:* the matching `replay-full` side video for the
  same race already on disk (drone HV R1) — 1 file, ~50 MB.
- **S2. Bend homography + curvilinear (d, off). ~1–2 wk.** Extend the rail fit to curved rails (fit the
  rail polyline, arc-length parametrise) so `(d, off)` is valid through the turns — where §3c's
  arc-length tax lives. Reuse the keyframe-chain homography.
- **S3. Full-arc single race. ~1 wk.** Chain straight+bend, smooth `(x,y)` with a Kalman filter, output
  all ~12 named runners for the whole 1650 m. Validate `off` against draw at the start and finish order.
- **S4. Backfill + leak-free trailing trait. ~2–4 wk + compute.** Scrape N drone replays (start with
  ~100 HV races — HV is bend-dominated, the strongest draw bias, smallest fields → easiest), wire a
  `vision.export → parquet` corpus, build a `mean_off / ground_saved` per-horse **trailing** trait
  (strictly prior, `features.py` discipline), couple it as the §3c O3/O4 control-quality input.
- **Effort: ~6–9 wk to a first leak-free lateral feature. Binding constraint: S1 re-id.**
- **Unblocks:** the geometry × control game (§3c) with REAL lateral paths — draw × distance done
  structurally (arc-length tax measured, not the scalar `DRAW_GAIN` shadow); the O3/O4 ground-saved /
  jockey-policy-type traits; eventually the 2-D MFG for place/exotics (O5).

### #22 (gait) — gated on footage, defer behind #34
- **S0. Source a parade/walk feed (the binding constraint).** Investigate whether `type=trackwork` /
  the barrier-trial (`btresult`) videos carry a slow walk-up, or whether paddock/parade footage is
  scrapeable at all. **If none exists, #22 is not actionable — that is the finding.**
- **S1. (only if S0 succeeds)** Animal-pose model (e.g. ViTPose-animal / DLC) on isolated walk crops →
  stride frequency/length + L/R symmetry → a single-frame/short-clip **soundness** read; pre-race,
  leak-free, helps debutants. ~3–4 wk after footage.
- **Effort: unknown until S0; the model is the easy part, the footage is the wall.**
- **Unblocks:** the physiology condition/health cell (scorecard §5, currently ✗) — soundness/condition
  latent for debutants the form channel can't reach.

## 5. Recommendation — sequence #34 first

**#34 is the higher-value AND lower-risk first investment.** Its binding constraint (re-id) is a
known, attackable CV problem with a concrete cross-view fix, its homography is already solved, its
substrate (drone replays) is on disk and scrapeable, and it unblocks the single most-promising open
theory (geometry × control / draw, deferred-not-dead per 0027). **#22 is blocked upstream on data
acquisition** (no parade feed in the HKJC URL scheme) — do S0 (cheap footage recon) in parallel, but
do not invest in the gait model until a walk-up feed is confirmed. Concretely: **build #34 S1 (cross-view
re-id) next** — it is the one piece that converts the working straight-segment PoC into a real, named,
full-race lateral path.
