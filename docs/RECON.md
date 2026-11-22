# Race reconstruction

`hkjc.recon` turns one race's facts (`hkjc.data.race_info`) and its four HKJC replay videos into
per-horse paths: `d(t)` metres to the finish along the racing rail, `s = distance − d` metres run,
and `lat(t)` metres outward of the racing (movable) rail, on the race clock (`t = 0` when the gates
open), each with a posterior sd.

```python
from hkjc.data import race_info
from hkjc.recon import RaceRun
run = RaceRun(race_info("20260603", "HV", 8)); run.run(); run.paths()
```

CLI: `uv run --group vision python -m hkjc.recon 20260603 HV 8 [--force stage,...] [--mask ...]`. Stages
are cached under `data_cache/recon/<date>_<venue>_<no>/`; a stage is skipped when its outputs exist.
Runtime environment: `HKJC_DEVICE` (`cpu`, `cuda`, `intel:…` for OpenVINO), `HKJC_NPROC` (workers per
race), `HKJC_OCR_THREADS`, `HKJC_OCR_DEVICE`, `HKJC_MATCH_DEVICE`, `HKJC_AVATAR_WEIGHTS` (detector weights),
`HKJC_FILM_DEVICE` (the film detector alone, e.g. `intel:gpu`), `HKJC_MASK` (§3), `HKJC_PERTURB_SEED` (§4 observe, §6).

Race identifiers are `venue MM-DD Rn` for 2026 meetings (ST 05-31 R1 is Sha Tin, 2026-05-31, race 1)
and carry the full date otherwise.

## 1. What the library produces

The reconstruction is one posterior of what happened in the race, built from all the information
available: the four replay videos, the pre-race facts, the official records and the track geometry.
Each video is an incomplete, noisy proposal of each horse's path; the path is the posterior of a
Bayesian fusion under physics (bounded acceleration) and geometry (rails, course width) priors, with
no feed acting as a master that the others correct.

| Artifact | Content |
|---|---|
| `paths.parquet` (`run.paths()`) | **The product**: per horse `horse_no, t, s, d, lat, sd_s, sd_lat, n_obs` on a 0.1 s grid, from every source, including the official records the run's mask leaves visible |
| `paths_video.parquet` | The reconstruction from the videos and pre-race facts only (no official outcome record); the fixed gates score it, and it is the reference for what the videos alone determine |
| `validation.json` | fixed gates on `paths_video.parquet` against the full official record, plus the product's residual against the official times (`final_official_resid_mae`) |
| `audit.json` | label-free invariants (§4, validate) |
| `status.json` | `ok`, `unhealthy` or `failed` (§4, status) |
| `timings.jsonl` | per stage: wall s, CPU s, peak RSS |
| `mask.txt` | the run's mask, when not empty (§3) |

Computer vision is validated by reprojecting the reconstruction onto raw frames (`viz.reproject`, the
synced video), because aggregate numbers hide failures. Gates are fixed and never loosened after seeing
results; failures are reported as failures.

## 2. Information and where it enters

| Source | Content | Enters |
|---|---|---|
| aerial feed | split screen: live race film (left, with the race clock) and the AVV 3D render (right) with a HUD odometer and a top-8 running-order legend | every in-race position (render), world frame, HUD clock/odometer/legend, painted-marker crossings in the film, gate opening in the start film |
| full feed | TV broadcast; after the race it replays a fixed finish-line camera | finish order and spatial gaps from saddlecloth OCR (`finish`, `identify`, every fuse) |
| patrol feed | head-on camera | synced viewer only |
| drone feed | one shot gate to finish | synced viewer only |
| pre-race facts | distance, course and rail width, runners, official silk images | every stage (course shape, lateral bounds, silk evidence, runner numbers) |
| draw | stall per runner | `fuse_final` (lateral at t = 0) |
| official records | cumulative sectional times, finish times, placings | `fuse_final` only (`pipeline.READS_OUTCOME`): official observations, shared metric bias, final identity pass, re-timing, placing ties |
| track geometry | `hkjc.data.track_for`: handedness, course widths, home-straight length | `rails` (inner side, movable-rail check, expected fixed-rail gap), `film` (markers inside the home straight), fusion's lateral box |

## 3. The mask and held-out evaluation

Every stage reads the race through `RaceRun.info`, a `hkjc.data.MaskedRace`: the pre-race facts plus the
official outcome records that the run's `hkjc.data.Mask` leaves visible, read only through
`official_boundaries()` (cumulative sectional times), `official_finish()` and `official_order()` (placings).
`validate` scores against the full record (`RaceRun.truth`).

- `Mask()` withholds nothing (the default).
- `Mask(boundary=j)` withholds the cumulative official time at sectional boundary j for every runner (0 is the
  first section end; the last index is the finish, which also withholds finish times and placings).
- `Mask(outcome=True)` withholds all official outcome data (results, finish times, placings, sectionals,
  running positions); pre-race facts, the draw included, stay visible.

The mask also withholds video evidence, from `observe` on (`pipeline.READS_EVIDENCE`), for held-out scoring
of the reconstruction where the videos are silent:
- `Mask(dets={i, ...})` removes those render detections (`dets.npz` row indices) in `observe`, as if the
  detector had missed them: tracking, identity and every fusion pass run without them. `rails` (upstream)
  still sees them.
- `Mask(discs=P)` counts every disc-number read (circle render) on the race clock's odd `P`-second blocks
  `[kP, (k+1)P)` as unread (`discs_phase=0`: the even blocks), in all three consumers: `observe`'s number
  split, `identify`'s votes and the fusion's blind set (`RaceRun._number_reads`).

Select it with `RaceRun(info, mask=...)`, CLI `--mask SPEC`, or env `HKJC_MASK`; a spec is `none` or
`;`-joined parts `boundary=J` | `outcome`, `dets=I,J,...`, `discs=P` | `discs=P@PHASE`
(`HKJC_FINAL_HOLDOUT=j` is an alias for `boundary=j`; setting both is an error). A masked run writes
`paths.parquet` and `validation.json` like any other, so give it its own `root`: the run dir records a
non-empty mask in `mask.txt`, and a run refuses a dir whose mask-dependent artifacts were made under another
mask (the outcome part governs `fuse_final`/`validate`, the evidence part `observe` and everything downstream;
an evidence-masked run therefore starts from a dir without `observe`'s outputs). A withheld boundary is absent
from the official observations, sets no metric bias and is not re-timed, so the product's error there is
independent of that record.

**Held-out baseline.** 196 races (145 silk-render races from a 200-race sample of 2023-09 to 2026-11 turf
meetings, the 26 silk-render development races, 25 numbered-disc races); one official boundary withheld per
race, rotating over the boundaries; absolute error of the product's crossing time at the withheld boundary.

| Withheld boundary | Median \|error\| | p90 |
|---|---|---|
| first boundary | 0.131 s | 0.315 s |
| later interior boundaries | 0.097 s | 0.269 s |
| finish | 0.125 s | 0.510 s |

- Horses without a crossing at the withheld boundary: 10 (finish withheld; the product's grid runs past the
  video-only grid's end).
- The error is almost all shared by the whole field: relative to the race's shared shift at the withheld
  boundary the median is 0.032 s, p90 0.131 s. Order inversions at the withheld boundary: 3.5 % of the pairs
  with an official gap > 0.05 s.
- Boundary speed kinks in the product: the speed jump at an official interior boundary has median 0.032 m/s,
  p95 0.216 m/s; at the same instants the video-only paths give 0.004 / 0.023 m/s. Speed and acceleration
  derived from the product near the official 400 m boundaries are fusion artefacts.
- Unmasked product on the same 196 races: every winner and finishing order reproduced, every official
  boundary within 0.02 s.

## 4. Pipeline

Stages run in this order: `frames → features → pose → worldmap → detect → silk → hud → rails → start → film →
observe → finish → identify → fuse_video → fuse_final → validate`. `hud` and `finish` read only the videos and
run in background processes until the first stage that depends on them. `pipeline.DEPS` declares each stage's
inputs: forcing a stage re-runs every requested stage downstream of it. A requested stage that has never run
(none of its outputs exist) counts as forced, since its consumers' artifacts predate it; a stage missing only
some outputs (large files deleted for space) is regenerated without invalidating its consumers (the pipeline
is deterministic). Artifacts are written to a temporary file and renamed, so a killed stage never reads as
done; a failed stage terminates the background stages.

ONNX Runtime sessions (avatar detector, OCR) run with intra-op spin-waiting off (`ortsession.no_spin`): same
threads and work split, identical outputs, CPU about a third (avatar detector) and a half (OCR).
`HKJC_OCR_THREADS=1` and `HKJC_MATCH_DEVICE=cuda` (ORB brute-force tables on the GPU in a helper interpreter:
integer popcounts, OpenCV's tie order) give artifacts identical to 8 OCR threads and OpenCV matching (3 races,
every artifact). `HKJC_OCR_DEVICE=cuda` (ONNX Runtime's CUDA provider, needs onnxruntime-gpu) is not bit-exact:
finish-camera reads, a legend read and identity votes move; on the 25 numbered-disc development races no gate
changes, 8 validations are identical and the rest differ in the third decimal. With all three, per-race CPU
falls several-fold (OCR dominates; on the disc render the number reads are most of `features`).

### frames

Two render styles: the avatar render with the silk legend (to 2025/26) and, from 2026/27, numbered blue discs
(radius ~25 px, the runner's number printed on each) with no legend. `video.layout` samples 12 frames across
the replay: a frame with ≥ max(3, 0.4 × runners) discs votes circle (tested first: discs crossing the legend
strip pass the silk gate), a frame passing the relaxed legend gate votes silk; ≥ 2 votes and a strict
plurality decide, else the race stops with "unknown render style" (213/213 replays classified correctly). The
style is stored in `frames_idx.npz`.

The panel box is a constant of the broadcast layout: the voting frames give a median box, and the first render
frame keeps its own box unless it differs from that by more than 3 px (antialiased edges jitter 1–2 px). The
first accepted frame can be the crossfade from the full-screen live shot, whose letterbox is not yet black:
its box spans the full height (rows 1–1076 instead of 271–806), and with that box the static-overlay mask
lands in the letterbox and the pose locks on the disclaimer and minimap (HV 10-14 R3 with that box: oval
folded, 283 tracklets).

Silk render: a frame is render-active when the legend band (0.70–0.86 of the panel) shows a saturated box,
the top strip is the dark HUD box (turf-green fraction ≤ 0.30) and the track band is sharp (Laplacian variance
≥ 240). Once the strict test has passed, ≥ 5 boxes in a band widened to 0.60 also pass (on ST 04-19 R4 and ST
04-26 R9 the legend sits higher and the strict gate drops up to 13 s of race). The render updates at 10 Hz in
25 fps video; a frame within mean |diff| 3.0 of the last kept frame is a duplicate. Panels are cached as JPEG
with their video index as timestamp. `video.grid_keep` drops off-grid re-encodes of a kept tick (HV 04-22
R1/R9 have a keyframe every 100 frames that would enter as a zero-motion tick).

Circle render: the render's yellow odometer text (fraction ≥ 0.05 of its box; live film and the dissolve
≤ 0.02, the render 0.10–0.33; without it the start gate's blue banner passes as discs) plus ≥ 2 discs; the
top-strip test stays and there is no texture floor (the smoother 2026/27 render has median Laplacian variance
~230). Once the render has started (`relaxed`) the odometer text is not required: it leaves ~20 m out while
the render runs on across the finish line to the end of the replay.

### features

Circle render: disc centres replace the avatar detector (a disc-template filter on the closed blue mask,
peaks with fill ≥ 0.8 at least 18 px apart, so discs 25 px apart, about 1.3 m, still split); the number inside
each disc is read with recognition-only OCR (`numbers.npz`, row-aligned with `yolo.npz`): 96–99 % of discs
give a valid read, 96.5–99.8 % agree with their tracklet's majority. `detect` uses the disc detections only.

Silk render: the avatar detector runs once per panel (`yolo.npz`: boxes + silk histograms, read by `detect`;
a run dir with only a YOLO-only `dets.npz` recovers `yolo.npz` from it). ORB (3000 features) runs on a world
mask: HUD removed, dark (V < 45) and vivid blobs dilated, and the avatar detector's boxes, grown 1.6×, masked.
The colour mask misses brown and pastel avatars, which are static in the image because the camera follows the
field, and step matches would lock onto them (ST 04-19 R4: camera "stopped" for 3 ticks, identities rotated).
Steps between frames ≤ 6 video frames apart are RANSAC similarities (1.5 px) refitted by least squares (Umeyama):
the camera is near-nadir with no tilt and a slow zoom.

Poses depend on the detector's device, so a CPU result is compared only with a CPU control (the reference
runs are CUDA). The OpenVINO detector's boxes differ (ST 04-19 R4 9122 vs 9129) and change the ORB steps on
603/705 frames. CPU vs CUDA: HV 06-03 R8's product moves by ≤ 0.18 m (median 0.01 m); on ST 05-03 R7, HV 05-27
R9 and ST 05-24 R1 the frames are byte-identical and the boxes match one to one (1 of 29,862 unmatched) within
1.4 px, and the products agree to a median 0.04–0.05 m on the first two (max 0.44 m; 1.6 m on HV 05-27 R9's
loose video-only #12), while on ST 05-24 R1 #2 and #3 exchange identities over most of the race (mirror-image
9 m / 4.5 m differences, tight in both runs) and #13 and #5 differ over the last 20 s.

### pose

The pose graph has sequential edges (a failed step, < 60 inliers or residual > 1.3 px, becomes an
interpolated edge of weight 0.05), long-baseline edges at d = 2/5/12/24 frames and loop closures (frames > 120
apart, matched unguided because chain drift can exceed any gate, kept if within 400 px of the chained
prediction; weight 3). A similarity is z → Az + B in complex numbers, so the probe residuals are linear: IRLS
with Huber 3, gauge at the mid-race frame (the first frame after the live-to-render dissolve is unreliable),
edges > 10 px pruned twice.

### worldmap

The mean mosaic of all registered frames: horses average out, rails and painted markers stay sharp.

### detect

The detections of a self-supervised YOLO avatar detector (conf 0.25, IoU 0.5), made in `features` on the
render panel; the legend icon band is dropped. Below a median of 0.6 N detections per frame (N runners;
all-weather dirt) a map-difference blob detector is added. Inside the disclaimer band only conf ≥ 0.45
survives, and a box recurring at one panel cell on ≥ 4 % of frames there is overlay text.

### silk

`detect.silk_feature` per detection drops turf and brown pixels and keeps 12 hue bins (pastel and vivid),
white and dark. Matching collapses pastel and vivid, because the render desaturates (correct group in the top
3: 73/79 hand crops). `silkmap.SILK_MAP` maps each runner's official silk prototype
(`detect.silk_prototype`) into render appearance: a ridge fit on 187 legend-confident tracklets, legend labels
only (leave-one-race-out top-1 47 %). Consumers: the appearance split in `observe` and the silk
likelihood ratio in `identify`.

### hud

Recognition-only OCR on fixed crops: the race clock on the film panel (every 10th frame), the render's
odometer and its 8 legend slots (re-read when the crop changes). Race-clock origin:
`clock_f0 = median(frame − 25·clock)`.

### rails

Stations every 2 world px follow the camera path, then the field's inner edge (90th percentile of inward horse
offsets, smoothed by `rails.robust_offset`, a Huber Whittaker smoother, so an outlier run cannot fold every
ridge into a V as on HV 05-20 R9). Thin bright ridges on the inward normals are linked by a Viterbi DP
(`rails.dp_track`) that scores +1 for any ridge above threshold and −1 for none, and charges 0.002 per px
inside the field edge. It rewards continuity, not brightness, so it stays on the thin movable rail instead of
the brighter fixed rail ~6.6 m inside. Then a world-space despike (> 0.55 m off a 26 m quadratic trend),
sub-pixel refinement and `rails.smooth_rail`, a support-weighted Huber Whittaker smoother with a
third-difference penalty (arcs are free; ~20 m cutoff; unsupported stations weight 0.1; end runs continue
straight): no 4 m-chord kink > 8° on ST 05-24 R1, marker grid found on all 26 development races.
`rails.RailFrame` takes positions and normals from a 121-station (~12 m) cubic fit, because snap wobble
(~0.1 m) makes normals noisy at wide lanes; projection uses a time-ordered station hint for revisited ground.

The inner side is the measured turning sign when the camera path turns at least a quarter turn (range of the
cumulative turning, so a fold-back at one end cannot cancel a bend; ovals 2.8–9.1 rad, all clockwise;
straights ≤ 0.28), else the race's handedness (`rails.inner_side`). On straights (the Sha Tin 1000 m) the
mosaic shows the fence only at d ≈ 400–550, so `rails.straight_datum` makes the field's inner-edge scaffold the
rail and adds a measured per-race fence offset `lat0_px` (median over the longest supported DP run: 111, 111,
61 px on the three development straights). Straight-course lateral is pack-edge-relative plus that constant;
pack-to-fence drift (up to ~1.8 m on ST 06-07 R8) is not modelled.

`rails.movable_rail_check` moves the rail out when a supported ridge sits at the course-table gap toward the
field and the innermost horses are still outside it (the extracted ridge was the fixed rail). Painted markers
(FINISH/50…300 m) appear only near the end, so they are profiled on a mosaic of the final 9000 world px.
`rails.fixed_rail_gap` measures the lateral-gate gap on a padded window (the ridge filter wraps at window
edges; unpadded, ST 06-07 R6 reads 11.27 m instead of 9.73 m).

### start

The race clock is the aerial's overlay clock, except where that clock starts late. Before the 2025/26 season
the overlay clock started about 0.4 s after the gates opened (up to ~3 s on some races); from 2025/26 it starts
with the gates. `start.lag` finds the gate opening in the live start film that opens the aerial video: the
closed stalls' doors are a band of dark pixels that drops when they swing open. The door colour depends on the
light: dark green in sun (OpenCV hue 35–95, saturation > 60, value < 110), dark teal in shade (hue 96–107 on
evening Happy Valley films). The teal band (hue 60–105) is tried only when the green band finds no opening; it
excludes the turf (hue ≤ 55), which the green band admits where the turf is shaded, and the blue banner and
night sky (hue ≥ 105). On an 8 × 6 grid of the quarter-resolution frames it takes the row strip with the
largest relative drop (≥ 25 %, ≥ 1 point) within 100 frames before the clock zero, provided the drop persists
over 40 frames (doors stay open) and is shared by ≥ 60 % of the strip's door-bearing columns (not a person
crossing). The onset is the 25 % crossing of that step. The detector's convention lag (clock zero − onset where
the clock is synced) is 1.4 frames at Sha Tin and 2.55 at Happy Valley; a net lag of more than 4 frames
re-anchors the clock (`RaceRun.clock_f0` = overlay zero − lag, used by `observe` and every stage downstream).
`film.npz` keeps overlay-clock times and `observe` shifts them by the lag, so `film` does not depend on
`start`. No detection, or a lag within 4 frames, keeps the overlay clock. `start.npz` records the clock zero,
onset, lag and status (`late`, `synced`, `none`).

Measured (film 400 m read minus official, 200-race sample): on 66 corrected pre-2025/26 races the residual is
−0.12 s median (the film read's own bias, as on synced races), MAD 0.06 s, 1 race in 35 beyond 0.2 s; 6 of 74
pre-2025/26 races give no detection and keep the late clock; no correction on 54 synced-era races. On the 26
development races there is one false correction (HV 06-03 R3, 0.34 s: its doors visibly open about 11 frames
before a clock zero that matches the official start).

On the 196 held-out races the green band finds no opening on 14; the teal band recovers 8 (3 late with lags
7.4–10.4 frames, 5 synced), each onset within about a frame of the first door motion seen in the frames. The 3
late lags (0.30, 0.29, 0.42 s) match those races' uncorrected video-only whole-race errors (field-median
sectional error −0.51, −0.30, −0.39 s); the 5 synced races (2025/26 or later) have errors of −0.01 to +0.09 s.
Of the 6 left `none`, 5 have a start film off the overlay clock's timeline (the clock zero extrapolates to
frames 0–2.5, the doors open 25–60 frames later; two visibly slow motion after the opening), and HV 2026-01-14
R8 is filmed from behind the stalls. On 50 held-out races where the green band works, the teal band alone also
finds every opening, a median 0.56 frames later (IQR 0.36–1.03; a teal lag reads about 0.02 s short); 2 differ
by 12–13 frames (ST 2025-01-05 R6, HV 2023-10-11 R4).

### film

The aerial's left half is a live race film on the broadcast clock; in the straight a front-on camera sees the
painted 400–50 m stripes. `film.crossings` tracks paint lines that sweep ≥ 80 px up the image, takes the
lowest COCO horse box as leader (the detector runs on a track's frames only up to the crossing the track is
read at: the first with a readable odometer), and names the marker from the HUD odometer (0 to 40 m behind).
Crossings must imply 12–22 m/s to a neighbour and lie ≥ 20 m inside the home straight (Happy Valley's 400 m
marker is on the bend). Sha Tin 400 m crossings average −0.11 s against official (early on 9/10 races); the
bias is not corrected.

`HKJC_FILM_DEVICE=intel:gpu` (or `HKJC_DEVICE=intel:*`) runs this detector as the OpenVINO export on the Intel
iGPU, compiled at FP32 (the GPU plugin's FP16 default loses the 200 m crossing on ST 05-09 R1; a missing device
raises). It is not bit-identical to the CPU default: on 26 races the marker sets match and every crossing is
within 5.4e-4 s of the references; on the 4 CPU controls, within 4e-7 s. The film stage takes 35–127 s instead
of 98–484 s, and 47–93 CPU-s instead of 570–3,440. Each process holds about 2 GB of system RAM for it (1.1 GB
of that shared iGPU buffers). The other `intel:` paths (avatar detector, OCR recogniser, `horse_scale`)
compile at the GPU default (FP16) and are untested at FP32.

### observe

Detections go to world px and to (s, lat) on the rail. The per-frame leader ignores a front detection > 8 m
clear only if the gap is not persistent over 11 frames (HV 05-20 R2's winner was 13 m clear).

**Distance metric** (`rails.calibrate_distance_ticks`). The odometer ticks V+10 → V as the leader reaches V.
Tick events give scale and finish through a robust line plus a cubic B-spline (150 m knots) for map-scale
drift; where ≥ 4 consecutive smoothed residuals exceed 10 m, knots densify to 25 m (the 1000 m straights, where
the camera swings 180° at t ≈ 28–32 s and the pose steps ~2.5 % in scale). Lateral keeps the base fit's px/m.
The constant comes from the film (`film.anchor_offset`): with ≥ 3 crossings the ticks are unpinned and shifted
by the median offset (+2.4…+11 m) that puts the render leader on the film's crossings (against the film the
render's painted FINISH pin is a step in the last 50 m). With fewer crossings, a tick-seeded marker grid
(`rails.fit_marker_grid_seeded`: paint scale ≈ tick scale / 1.035 within ±8 %, FINISH within ±60 m, ≥ 4
markers including FINISH) pins d = 0 when its offset after resolving whole-marker shifts is < 6 m and the
scales agree within 8 %; a pinned grid also sets the per-race tick convention (Sha Tin ≈ −3.2 m, Happy Valley
≈ 0). `rails.time_local_shift` subtracts a per-frame along-track shift where ≥ 3 consecutive ticks keep
|residual| > 10 m (a pose slip in the straights' camera swing). Lateral = (lat_px + lat0_px) / local px/m;
detections > 1 m inside the rail are dropped.

**Timestamps.** The render updates at 10 Hz in 25 fps video, so new ticks start 2 and 3 frames apart while
their content advances a uniform 0.1 s. Each render frame is stamped with its tick time (`video.tick_frame`):
the residue followed by a 2-frame step moves −0.25 frame, the other +0.25 (mean zero; off-grid frames
unchanged). All observation times (tracks, tracker, film anchor, time-local shift) use it: the along-track
residual on the fused path is +0.006 / +0.004 m on the two residues (−0.17 / +0.18 m with video-index time).

**Tracking.** Constant-velocity prediction and per-tick Hungarian assignment in metres (gate 3σ,
σ = 0.9/0.45 m, coast 20 ticks), then kinematic stitching across gaps ≤ 6 s. `tracking.split_by_appearance`
cuts a tracklet where its mean silk signature changes (total-variation distance > 0.30, ≥ 40 silk detections
per side; legend-pure tracklets reach at most 0.28). Circle render:
`tracking.split_by_number` splits tracklets where the disc number changes and stays changed (≥ 8 of the last
10 confident reads name one other runner number that is not a dropped digit of the current one, and it holds
for the next 20 reads, ~2 s), because the motion tracker swaps identities where discs touch (ST 2026-10-19 R3:
#11 and #12 swap at ~60 s, #2's reads become '9' at ~90 s). Swaps between 1 and 11 (or 2 and 12, ...) look like
dropped digits and are not caught.

`HKJC_PERTURB_SEED=k` drops 1 % of detections here (seeded) for robustness ensembles.

### finish

The full feed is scanned from `distance/18` s for 70 s for the finish camera: a white vertical line at a stable
x, slow motion at a variable rate, so unique-frame index is not time. Saddlecloths are OCR'd per unique frame
of a run that reaches 40 unique frames (shorter runs are dropped unread): yellow on dark (Happy Valley,
overhead) and, at Sha Tin only, dark digits on white cloths inside a white ring. A COCO horse-box ruler (2.6 m
nose to tail) gives camera px/m; the box includes extended legs, so the ruler-based C reads ~×1.17 high. C is
not calibrated on official finish gaps.

`finishcam.tracks` keeps, per number, the lateral cluster read at the most distinct places (a rail post read
60 times at one spot outvoted the real cloth on ST 06-07 R8) and needs ≥ 4 reads and ≥ 100 px of motion. Digit
drops ("1" on the "11" cloth) are removed when most of the shorter number's reads lie on the longer number's
track; coinciding numbers keep the longer one. On the overhead camera (Happy Valley; `RaceRun.finish_sc`
carries the geometry) reads of any number lying on a much better-read track (≥ 10 reads, ≥ 3× the reads it
claims, on its line within 60 px, its lane within 30 px) are that cloth's misreads and are removed before
tracks are built (HV 2026-11-18 R6: '3' reads on the #7 cloth would merge with the real #3's reads into a
track that puts #3 second; official: last). On 197 races camera-order inversions against official are 217, no
race worse for the rule. On the Sha Tin side-on camera, where horses abreast share one image band,
the rule makes 4–5 races worse at every threshold, so it is not applied there.

`finishcam.finish_gaps` gives spatial gaps at an instant: crossers are chained from the crosser that links the
most, and other horses get `gap_h = gap_r + x_h − x_r` from ≥ 2 shared frames, which recovers horses occluded
at the line (sd = robust spread / √n, floor 15 px).

### identify

`identity.legend_votes` uses frames where every runner is tracked by distinct tracklets: the tracklet at rank r
votes for legend slot r, and ranks 9+ get set-valued elimination evidence ("one of the N − 8 unlisted
numbers"). It takes the runners' saddlecloth numbers, so withdrawn numbers (gaps) neither block the whole-field
test nor enter the unlisted set (HV 04-15 R6: #2 and #4 withdrawn, no slot
votes for either). The finish camera names tracklets by 1D registration
(`finishcam.identity_match`) of their line crossings to `t_w + gap/C`, scanning C ± 25 % around the ruler
prior; a name needs residual < 0.08 s with every other prediction ≥ max(2× residual, 0.15 s) away, and adds
40 slot votes. Circle render: each confident disc-number read (OCR confidence ≥ 0.6, a runner's number) adds
one vote for that number to its tracklet, about one legend frame's worth; late-frame numbers count as shown for
the blind set in the fusion.

`identity.assign_global` solves the global maximum-likelihood assignment as a MILP: a vote is right with
probability ρ, else uniform, so evidence grows with the vote count; elimination frames score log(ρₑN/m) for
each unlisted number; ρ and ρₑ are re-estimated from labelled tracklets (camera votes excluded); the silk LLR
`2.24 − 6.37·TV` (clipped ±3) counts where the legend has spoken; one number per tracklet and per frame; gain
≤ 0 stays unlabelled. Unlabelled line-reaching tracklets are then matched by elimination (`finishcam.eliminate`) to
numbers with no line-reaching tracklet. Camera and elimination names are hard constraints downstream.

### fuse_video

Writes `paths_video.parquet` and persists its observations and paths (`fuse_video.pkl`); `fuse_final` reads
them, so a change to the official-record stage re-runs only `fuse_final`.

**Model** (`fusion.smooth`). Per horse, s and lat on a 0.1 s grid with second-difference priors: along-track
acceleration sd 6 m/s² for t < 6 s, 1.2 m/s² after; lateral 0.25 m/s² plus a 2 m/s lateral-velocity term; no
along-track velocity term. Observations enter at their own times by linear interpolation; render sd 0.8 m
along, 0.35 m lateral. Lateral is boxed to [0.3, width − 0.3] m. The MAP is a banded Cholesky solve with
violating nodes pinned by a stiff penalty (≤ 8 rounds); it matches an exact sparse LU to ~1e-8 m in 1/27 of the
time (declared non-exact: labels differ on 3/26 races by 0.1–3.2 %). Posterior sd comes from the unconstrained
normal equations at every 10th node, interpolated. The grid ends at the race's latest observation. When
`paths_video.parquet` is written, each path gets a 0.5 s prior-only tail (`fusion.with_tail`): s continues at
the end-node speed, lat is held, and sd_s grows with the acceleration prior; the fusion, EM and
`fuse_video.pkl` never see the tail. It gives a crossing to a horse a few cm short of the line at the last
observation (HV 04-15 R1 #2 ends between 4 cm past and 3 mm short of the line across runs). Tail crossings
are extrapolations: up to 7.6 m (ST 05-09 R2 #11, ST 05-09 R1 #8), most under 1.5 m.

**Association** (`fusion.fuse_race`). Anonymous tracklets are scored against each posterior by mean χ² plus
log-det ratio; the best must be < 9, beat the runner-up 2×, not overlap that horse's same-feed tracklets in time
and not be in the tracklet's forbidden set. Orphan tracklets are chained and given to a path-less horse by its
finish anchors (and official anchors in `fuse_final`), or by elimination when one horse is missing. The anchor check demotes up to 3 render tracklets
whose removal at least halves a horse-specific anchor conflict above 4σ (field median residual subtracted;
never below half the horse's render data), then re-fuses once.

**Finish observations** (`RaceRun.finish_observations`, a second pass inside every fuse). Each gapped horse
crosses d = 0 at `t_w + gap/C`. `finishcam.anchor` fits t_w and C (800–4000 px/s) to fused crossings with sd ≤
3 m, with a soft prior C = line speed × ruler px/m (relative sd 0.30). If the label-free registration disagrees
by > 0.4 s or > 30 % in C, the registration is used (t_w sd 0.15 s, its measured bias); if both fail, no finish
observations are emitted. Per-horse sd is conditional on the fitted clock.

**EM identity pass** (`RaceRun._em_identity`). Legend votes on frames with undetected horses, ranks completed by
the first posterior where every sd < 2 m, are added and `assign_global` re-run.

**Detection-level EM** (`assoc.assign_frames`). The tracker fragments are only the initialisation. Per frame,
one Hungarian assignment of all detections to all horses, clutter cost 16 (`assoc.CLUTTER`, 4σ). A detection's
rank is 1 + the number of detections more than 0.6 m ahead of it in the frame. Costs:
- Gaussian −2 log-likelihood against each posterior including the log-det ratio, so a vague out-of-view path
  pays for its vagueness.
- Hard legend rank: a detection with ≥ q detections more than 4.5 m ahead (4σ on a difference of two
  detections) cannot be the horse in legend slot q (+50, forbidden). At 0.6 m two detections 0.9 m apart
  hard-exclude correct horses (ST 04-26 R1).
- Soft legend rank: +9 for slot q where the detection's rank, confirmed by the count of tight posteriors
  ahead, differs from q.
- Negative legend evidence: a horse missing from a fully read top 8 is barred from detections surely inside the
  top 8 (detections ahead + undetected horses + 1 < 8) or > 8 m ahead of the rearmost tight listed horse.
- Finish-camera negative evidence: with recall r ≥ 0.5 (capped 0.95), detections in the last 15 m cost
  −2 log(1 − r) for unread horses; a horse unread and absent from the legend in the 8 s before the first
  crossing ("blind") pays CLUTTER, hard, in the last 50 m.
- Rank attraction: in the last 100 m a detection whose rank equals a listed horse's slot costs that horse at most
  10. Slowing unread horses' real detections sit 7–13 m behind their extrapolated paths at χ² ≈ 22 (ST 04-26
  R9).

`assoc.release_post_finish` makes anonymous any detection of a horse later than its first detection past the
line and > 3.2 m before it (ST 05-24 R1: two such detections would drag the camera clock and lose the winner).
`assoc.runs_to_obs` splits runs at gaps > 1 s and drops runs of < 3. Unassigned detections stay anonymous per
tracker fragment and carry the hard exclusions held on ≥ 50 % of their detections, else fusion's association
would reassign them to an excluded horse. Up to 8 iterations, stopping below 0.2 % label change.

`RaceRun._finish_seeds`: horses the camera read but still path-less after the EM get an E-step-only seed path
(camera crossing, field line speed, sd growing 1 m/s away from the line) and the EM runs again. Shadow release:
two tight paths within 0.8 m along and 0.5 m lateral for ≥ 0.5 s in the last 3 s lose the weaker horse's
detections there (the unread one, else fewer detections).

### fuse_final

The product: the video-only detections plus the draw and the official records the mask leaves visible.
1. `RaceRun.official_observations`: the draw fixes lateral at t = 0 (stall 1 at 2.5 m, pitch 1.06 m, measured
   on HV 06-03 R8's drone start frame; sd 0.4 m; `STALL1_LAT`, `STALL_PITCH`, `DRAW_SD`) and official cumulative
   sectional times fix crossings at d = 400k with `OFFICIAL_SD` = 0.1 m (their 0.01 s publication precision);
   runners without sectional times are held to their official finish time. Near-hard official times make a
   mislabelled render tracklet conflict with the record (and be demoted) instead of bending it.
2. `RaceRun._metric_bias`: at each visible boundary, b = median over ≥ 3 horses of (video-only crossing delay
   × speed); render s is shifted by b interpolated along s. The render metric error is shared by all horses,
   and hundreds of render observations would otherwise outvote one official time (residual ~5× its sd).
3. Fuse with the grid running 0.5 s past the later of the last visible official time and the video-only grid's
   end (ending at the last observation leaves HV 04-22 R9 #11 0.8 m short of the line; with the finish withheld
   the last visible official time is 400 m out). Held-out finish: §3.
4. Field-relative start prior: until the first render frame + 1 s there is no video, so acceleration has sd
   0.63 m/s² per node about the field's median curve from step 3 (the clean races' horse-specific
   acceleration). The zero-mean 6 m/s² prior would extrapolate a late-seen horse backwards out of the gate (ST
   05-31 R1 #6 26 m out at 2 s, the field 8–15 m).
5. Final identity pass: `RaceRun._skeleton_merge` keeps the step-4 path except where it conflicts with the
   horse's official-only skeleton by > 4 combined σ (flat at 3–5); the E-step is re-run from there with the
   official observations in every fuse and bias-shifted detections (`det_labels_final.npz`). With every official
   time withheld there is no skeleton.
6. `RaceRun._official_retime`: each fused path is re-timed so it passes every visible official boundary at the
   official time. The re-timing (`RaceRun._monotone_warp`) is a strictly increasing C1 cubic Hermite through the
   (fused crossing, official time) knots with Fritsch–Butland slopes, fixed at the path's first node and
   continued past the last boundary at the last section's rate, so it adds no speed jump at a boundary;
   lateral and sd follow the same re-timing. Horses sharing an official finish time keep the official placing,
   2 ms apart (`TIE_STEP`). Shifts over 0.05 s are logged (they mark identity errors left in the video evidence).

`fuse_final` leaves `paths_video.parquet` byte-identical.

### status

`run()` always writes `status.json` (not a stage artifact): `failed` with the stage and error when a stage
raises, else `unhealthy` with the failed video-only checks, else `ok`. Checks: render ticks per expected race
second (unique render frames / (distance / 16.5 m/s)) ≥ 4.0; median detections per render frame per runner ≥
0.4; ≥ 3 race-clock reads; the rails stage's course shape (straight writes `lat0_px`) matches the race facts
(straight only for Sha Tin 1000 m). The `start` status is recorded but not a failure. All 26 development races
and every completed silk-render race of the 200-race sample are healthy.

### validate

`recon.validate` writes `validation.json` (gates on `paths_video.parquet` plus the product's residual against
the official times). `recon.audit` writes `audit.json`: label-free invariants (rail off the map, marker grid and
spacing, kinks; field-to-rail gap, field-wide lateral slides; legend negatives and ranks, ghost crossings,
finish-order inversions). A contiguous off-map rail run over 100 stations blocks an ensemble (clean rails reach
44).

## 5. Feed synchronisation

`viz.sync_video` draws the four feeds beside the reconstructed field on a map built from the run's own rail.
Offsets are video seconds at race clock 0, a constant or `[(race t, offset), ...]` interpolated.
- aerial: `clock_f0 / 25` (the on-screen race clock; 25.000 fps on every race).
- full: `viz.full_offset`, the lag maximising thumbnail correlation between the aerial's left panel (a scaled
  copy of the TV picture) and the full feed (correlation 0.66–0.92 on 26 races; two windows agree within
  0.08 s); full sits at clock + 1.21–1.51 s.
- patrol and drone: gate opening read by eye from 0.2 s contact sheets, assuming the gates open at clock
  −0.12 s (measured on HV 06-03 R8 only). Patrol opens at 1.15–1.4 s, drone at 2.1–3.1 s, with −0.35 s for
  drone horses becoming visible clear of the gate. The drone feed loses time during the race: its offset at the
  finish board is ~1.5 s smaller than at the gate (HV 06-03 R8 0.96 vs 2.46 s; ST 04-19 R4 1.16 vs 2.67; ST
  06-07 R6 1.16 vs 2.77), so it is anchored at both. Precision is a few tenths of a second. ST 05-03 R7 has no
  drone feed.

Display: a ring marks sd 1.5–3 m, a hollow marker 3–10 m; positions with sd > 10 m are omitted and listed as
"not placed"; the leader and window use sd < 3 m. A small sd does not flag a confident error (ST 04-19 R4 #3 at
sd 0.17 m on wrong detections).

## 6. Acceptance

### Fixed gates (`recon.validate`)

Scored on `paths_video.parquet` against the full official record, unchanged by the mask. A per-horse crossing
is scored only where the posterior sd_s < 1.5 m (`SD_MAX`) on both bracketing nodes. Boundaries are the
official sectional ends, 400 m apart.

| Gate | Threshold | Definition |
|---|---|---|
| `leader_timing_mae` | ≤ 0.25 s | identity-free: running minimum of raw detection d per frame, crossing time at each boundary vs the official leader |
| `sectional_mae` | ≤ 0.25 s | per-horse crossing times at every boundary vs official |
| `finish_rho` | ≥ 0.90 | Spearman of reconstructed vs official finish times (≥ 3 scored) |
| `winner_ok` | == official | earliest scored crossing |
| `behind_winner_mae` | ≤ 0.20 s | time behind the winner at the line |
| `sectional_coverage` | ≥ 0.80 | share of (horse, boundary) crossings scored |
| `horses_with_path_short` | ≤ 1 | runners without a fused path |
| `finish_n_short` | ≤ 2 | runners without a scored finish |
| `lateral_scale_err` | ≤ 0.75 m | \|racing-to-fixed rail gap on the map − course table\|; n/a where unmeasurable (A course, no supported ridge) |

The last four were declared before any held-out race was run. "Timing gates" are the first five; "non-lateral
gates" the first eight.

### Yardsticks

- **Held-out boundary** (§3): the product's absolute error at an official boundary the mask withholds.
- **Hand truth.** Detections hand-labelled at colour-group level against the official silks; a detection counts
  as correct when its label is in the labelled set. Single labeller: the hand truth's own error rate is not yet
  measured.
- **Numbered-disc agreement.** On the disc render, detections with a confident disc read and a label: the share
  whose label equals the read number.
- **Withheld detections** (§3, `Mask(dets=...)`). Per race, two horses × one 3 s stretch in the middle third of
  the race, chosen by a fixed seeded rule from an unmasked reference run: at least 80 % of the stretch's render
  frames carry a detection that both the video-only and the final E-step label as the horse, the horse is never
  the frame's frontmost detection, and stretches where another detection lies within 3 m along and 2 m across
  in at least half the frames are preferred. Every detection either E-step labels as the horse in the stretch is
  withheld (plus unlabelled duplicate boxes within 0.6 m / 0.4 m), and the reconstructed path is scored against
  the withheld detections: along-track and lateral RMS and max error, and whether the path in the gap overlaps
  another horse's firm path (|Δs| < 2.0 m and |Δlat| < 0.7 m).
- **Withheld disc reads** (§3, `Mask(discs=10)` and `Mask(discs=10, discs_phase=0)`). Each confident disc
  read is scored once while hidden: among detections with a confident read inside the withheld blocks, the
  share of labelled ones whose label equals the read, and the share left unlabelled.
- **Lateral yardstick.** AUC of final-path lateral position against the stewards' comments (WIDE > INSIDE, with
  MID between).
- **Perturbation ensembles.** Seeds of `HKJC_PERTURB_SEED` (1 % detection drop) per race; arms are compared
  paired by seed with SE from per-seed differences, under an adoption rule declared before reading results. A
  run never scores a directory another run writes into.
- **Identity score.** Spearman of reconstructed vs official crossing times and the share of horse-boundaries
  within 0.25 s after removing each boundary's common mode.
- **Per-horse census** (what a viewer of the synced video sees). A horse-run is good when its posterior sd < 3 m
  on ≥ 90 % of the span from the first render detection to its crossing, it is scored at the line, and its
  largest boundary error with the boundary's race median removed is ≤ 0.25 s (~4 m). On the product the official
  records are inputs, so the census detects conflict only.
- **Glitch-seconds** (product). Per horse, gap = d − the median d of horses still running; horse-seconds where
  the gap changes by > 3 m/s over a 1 s window, between the first render detection + 1 s and the first finisher
  − 1 s, with sd < 3 m at both ends.
- **Pixels.** `viz.reproject` and the synced video; every change is checked on frames.

The census, glitch, identity-score and ensemble scripts live outside the package.

## 7. Current state

### Data

26 turf development races (inspected). Sha Tin: 04-19 R3 (1000 m straight), 04-19 R4, 04-26 R1, 04-26 R9, 05-03
R7, 05-09 R1 (1000 m straight), 05-09 R2, 05-17 R1, 05-24 R1, 05-31 R1, 06-07 R6, 06-07 R8 (1000 m straight).
Happy Valley: 04-15 R1/R9, 04-22 R1/R9, 04-29 R3/R9, 05-13 R1/R9, 05-20 R2/R9, 05-27 R3/R9, 06-03 R3/R8. Also
inspected: HV 04-15 R6 (two withdrawals). 25 numbered-disc races from 2026/27. A 200-race sample of turf races
from 2023-09 to 2026-11, of which 145 are silk-render races in the evaluation set. The all-weather track is out
of scope. The 26 include the 11 races of the only sealed test (video-only product at an earlier version: all
five timing gates on 9/11, 11/11 winners), so a further sealed claim needs races sealed by a metadata-only rule.

### Measurements (196 races: 145 + 26 silk-render, 25 numbered-disc)

- Held-out official boundary: §3. Unmasked product: every winner and finishing order reproduced, every official
  boundary within 0.02 s.
- Hand truth (video-only labels): 138 of 164 hand-labelled detections on 9 races, 84.1 %.
- Numbered-disc agreement on the 25 disc races: 99.13 % (video-only labels), 98.91 % (final labels).
- Withheld detections (391 stretches, 370 of them beside another horse; video-only paths): along-track RMS
  median 0.25 m, p90 0.84 m, 4.6 % above 2 m; lateral RMS median 0.19 m, p90 0.61 m; the path in the gap overlaps
  another horse in 5.9 % of stretches. With the same detections visible: 0.12 m, 0.27 m, 0 %; 0.12 m, 0.29 m;
  2.6 %. The errors above 2 m are mostly confident (posterior sd 0.2–0.4 m): the hidden horse's path follows
  another horse's detections.
- Withheld disc reads (25 disc races, alternate 10 s blocks): labels inside the hidden blocks equal the hidden
  read on 98.58 % of labelled, confidently read detections (video-only labels; final 98.45 %), 0.91 % left
  unlabelled (final 1.59 %); with every read visible, 99.19 % and 0.49 % (final 98.98 %, 1.12 %).
- Lateral yardstick on the final paths: AUC (WIDE > INSIDE) 0.961; MID > INSIDE 0.912, WIDE > MID 0.860 (INSIDE
  400, MID 262, WIDE 453 horse-runs).

## 8. Open problems

- **Shared metric error.** The held-out error is almost all common to the field (§3): the render metric between
  boundaries, interpolated from the other boundaries. Its steps between adjacent boundaries are close to
  independent, so midway between two visible boundaries the shared error is about the held-out figure / √2,
  falling to zero at each boundary.
- **Boundary speed kinks.** The fused path has a speed corner where it meets an official crossing (§3: median
  0.032 m/s, p95 0.216, against 0.004 / 0.023 video-only); the re-timing adds nothing to it.
- **Posterior calibration.** The stated sd is overconfident at intermediate boundaries (error relative to the
  race median well above the stated sd) but not at the finish; more detections do not shrink the gap, so a
  systematic term is missing. Candidates: effective sample size per run, or inflation fitted on held-out
  boundaries.
- **Identity at official crossings.** Most product glitches where the video-only path is smooth are identity
  conflicts: a wrong E-step label between boundaries pins the horse and the official time pulls it back. Relative
  acceleration separates them from genuine faders (faders ≤ 1.25 m/s², conflicts ≥ 2.4 m/s² on five traced cases).
- **Silk evidence.** Hand truth is 84.1 % (above). In `identity.assign_global` legend votes add per frame and silk
  once per tracklet (±3), so silk rarely overturns a legend label; a better silk likelihood is the open lever.
  Back markers get only elimination evidence and the weak silk term; draw order agrees with early lateral order
  (median ρ 0.75), and a heavy-tailed draw term is untested.
- **Finish-short horses.** Horses unobserved over the last metres (detections ending short of the line, the
  camera parked on the line after the winner) get no finish crossing. The duplicate test drops the real
  post-finish ticks (in the 3 s after the winner's crossing, 96 of 780 on-grid frames kept over 26 races), but
  keeping them gains no crossing: fusion does not label the stragglers' detections. The blind near-line cost also
  removes correctly placed unread back markers.
- **Render geometry.** The render draws the Sha Tin B fixed rail ~2.1 m further in than the course table (6.6 m
  vs 4.5 m), and the B racing rail ~6.5 m out from the A rail against the official 4.5 m, failing the
  lateral-scale gate there. Some races carry a common-mode render scale error between boundaries (HV 04-22 R9:
  +0.44 s at 800 m, −0.17 s at the line) that the film window is too short to measure.
- **Leader timing.** The render's odometer is late early in long Sha Tin races (up to ~1 s, both render eras)
  and early on others; in the 0–400 m zone the tick px/m is 1.3–4.8 % above the paint px/m; cause unknown. Beyond
  400 m no reference exists that is neither official nor the odometer. The straights' leader is ~0.2 s early from
  400 m (odometer/marker scale ratio 1.034–1.036).
- **Detector artefacts.** The avatar detector often keeps two boxes on one horse (IoU 0.32–0.50, under NMS's
  0.5), and the second box births a second track. The static-overlay filter can delete a real horse the camera
  holds at one panel cell (ST 06-07 R8, 242 frames) and misses disclaimer text below its bar (HV 05-20 R2, 31
  boxes, which the hard rank cost then treats as horses).
- **Tracklet exclusivity.** `assign_global` forbids one number on tracklets whose spans overlap, which can leave
  a tracklet unlabelled when its number's holder overlaps it by a few frames inside a gap; whether this is the
  sole blocker is not computed.
- **Orphan elimination** hands a leftover chain to the last missing horse, which can give a tailed-off horse a
  mid-field chain; turning it off costs paths and is not ensemble-tested.
- **Clock crops.** A sponsor box above the left-panel clock moves it about 45 px below the fixed crop, so no
  clock is read and `film` fails (2 races of the 200-race sample). Older seasons are at 1280 × 720 until 2023-05
  (the aerial feed exists back to 2015); the pipeline assumes 1920 × 1080.
- **Non-finishers in the results.** `historical._clean_race_data` drops every incident row (DISQ, DNF, FE, ML,
  PU, TNP, TO, UR, VOID, WR, WV, WV-A, WX, WX-A, WXNR); a code marking a horse that started would remove a
  rendered horse from the runners. Not observed: raw results pages show only WV and WV-A rows.
- **Resolution floors.** Official mid-race times sit on a 0.04 s grid relative to the leader (~0.2 m); the HUD
  odometer changes only every 0.4 s, so a raw tick is 0–0.4 s late.
- **Independent real-camera positions.** Drone and patrol are the only independent source for the start segment
  (the render starts at race clock 10.8–12.6 s) and for lateral validation; planned via segmentation seeded from
  fused positions. Also open: a non-overlap constraint between horses in fusion.
