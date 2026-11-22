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

CLI: `uv run --group vision python -m hkjc.recon 20260603 HV 8 [--force stage,...] [--root DIR]`. Stages are
cached under `data_cache/recon/<date>_<venue>_<no>/` (`--root` moves the run dirs); a stage is skipped when its outputs exist.
Runtime environment: `HKJC_DEVICE` (`cpu`, `cuda`, `intel:…` for OpenVINO), `HKJC_NPROC` (workers per
race), `HKJC_OCR_THREADS`, `HKJC_AVATAR_WEIGHTS` (detector weights), `HKJC_FILM_DEVICE` (the film
detector alone, e.g. `intel:gpu`; see the film section).

**Mask.** Every stage reads the race through `RaceRun.info`, a `hkjc.data.MaskedRace`: the pre-race facts
plus the official outcome records that the run's `hkjc.data.Mask` leaves visible, read only through
`official_boundaries()` (cumulative sectional times), `official_finish()` and `official_order()` (placings).
`validate` scores against the full record (`RaceRun.truth`). `Mask()` withholds nothing (the default);
`Mask(boundary=j)` withholds the cumulative official time at sectional boundary j for every runner (the last
index is the finish, which also withholds finish times and placings); `Mask(outcome=True)` withholds all
official outcome data (results, finish times, placings, sectionals, running positions). Select it with
`RaceRun(info, mask=...)`, CLI `--mask none|boundary=J|outcome`, or env `HKJC_MASK` (`HKJC_FINAL_HOLDOUT=j`
is an alias for `boundary=j`). A masked run writes `paths.parquet` and `validation.json` like any other, so
give it its own `root`: the run dir records a non-empty mask in `mask.txt`, and a run refuses a dir whose
`fuse_final`/`validate` artifacts were made under another mask.

Race identifiers below are `venue MM-DD Rn` (2026 meetings), e.g. ST 05-31 R1 is Sha Tin,
2026-05-31, race 1.

## 1. Scope

**Inputs.** Pre-race facts: distance, course and rail width, draw, runners, official silk images
(the draw enters only the final product).
Official outcome records (placings, finish and sectional times) enter only the final product and
validation. Four replay feeds per race:

| Feed | Content | Use |
|---|---|---|
| aerial | split screen: live race film (left, with the race clock) and the AVV 3D render (right) with a HUD odometer and a top-8 running-order legend | world frame, every in-race position, HUD, painted-marker crossings in the film |
| full | TV broadcast; after the race it replays a fixed finish-line camera | finish order and spatial gaps from saddlecloth OCR |
| patrol | head-on camera | synced viewer only |
| drone | one shot gate to finish | synced viewer only (not in the pipeline) |

**Two products.** `paths_video.parquet` is the VIDEO-ONLY posterior: no official outcome data
enters it; pre-race facts may. The fixed gates score it, and it stays the independent check.
`paths.parquet` is the FINAL product: it may use all information, including the official results
(owner ruling 2026-11-16). Acceptance is judged on the final product.

**Principles.** Every feed is an incomplete, noisy proposal of each horse's path; the path is a
posterior from Bayesian fusion under physics (bounded acceleration) and geometry (rails, course
width) priors; no feed is a master that the others correct. Computer vision is validated by
reprojecting the reconstruction onto raw frames, because aggregate numbers hide failures. Gates are fixed and never loosened
after seeing results; failures are reported as failures.

## 2. Pipeline

Stages run in this order; `hud` and `finish` read only the videos and run in background processes
until the first stage that depends on them. `pipeline.DEPS` declares each stage's inputs: forcing a
stage re-runs every requested stage downstream of it. A requested stage that has never run (none of its
outputs exist, e.g. a stage added after the cache was made) counts as forced, since its consumers' artifacts
predate it; a stage missing only some outputs (large files deleted for space) is regenerated without invalidating
its consumers (the pipeline is deterministic). Artifacts are
written to a temporary file and renamed, so a killed stage never reads as done; a failed stage
terminates the background stages. Each race appends `timings.jsonl` (wall s, CPU s, peak RSS).
ONNX Runtime sessions (avatar detector, OCR) run with intra-op spin-waiting off
(`ortsession.no_spin`): same threads and work split, so identical outputs; CPU about a third (avatar detector) and a half (OCR).

Accelerator switches (environment): `HKJC_OCR_THREADS=1` and `HKJC_MATCH_DEVICE=cuda` (the ORB brute-force
tables computed on the GPU in a helper interpreter: integer popcounts, OpenCV's tie order) give artifacts
identical to 8 OCR threads and OpenCV matching (3 races, every artifact). `HKJC_OCR_DEVICE=cuda` runs the OCR
recogniser on ONNX Runtime's CUDA provider (needs onnxruntime-gpu): not bit-exact (finish-camera reads, a legend
read, identity votes move), but on the 25 dev circle races no gate changed, 8 validations were identical and the
rest differed in the third decimal. With all three, per-race CPU falls several-fold (OCR dominated: on the circle
render the disc-number reads were most of `features`).

### frames

Two render styles: the avatar render with the silk legend (to 2025/26) and, from 2026/27, numbered blue
discs (radius ~25 px, the runner's number printed on each) with no legend. `video.layout` samples 12 frames
across the replay: a frame with ≥ max(3, 0.4 × runners) discs votes circle (tested first: discs crossing the
legend strip pass the silk gate), a frame passing the relaxed legend gate votes silk; ≥ 2 votes and a strict
plurality decide, else the race stops with "unknown render style" (213/213 replays classified correctly).
The style is stored in `frames_idx.npz` (silk caches keep their exact layout).

The panel box is a constant of the broadcast layout: the voting frames also give a median box, and the first
render frame keeps its own box unless it differs from that by more than 3 px (the antialiased edges jitter
1–2 px). The first accepted frame can be the crossfade from the full-screen live shot, whose letterbox is
not yet black: its box spanned the full height (rows 1–1076 instead of 271–806), the static-overlay mask
landed in the letterbox and the pose locked on the disclaimer and minimap (S1 pose lock; circle HV 10-14
R3: oval folded, 283 tracklets → 10 and all gates with the layout box). 196 of 209 cached races keep their
box exactly; 8 of the 13 that change are S1 silk races with full-height boxes (box, pose and tracklet
count clear on all 8, 4 pass every gate). Dev HV 05-13 R9 (first frame 7 crossfade rows lower) still passes every gate, with
sectional MAE 0.110 → 0.175 s and finish ρ 1.00 → 0.93 against Z12 (same with CPU and CUDA detectors).

Silk render: a frame is render-active when the legend band (0.70–0.86 of the
panel) shows a saturated box, the top strip is the dark HUD box (turf-green fraction ≤ 0.30) and the
track band is sharp (Laplacian variance ≥ 240). Once the strict test has passed, ≥ 5 boxes in a band
widened to 0.60 also pass: on ST 04-19 R4 and ST 04-26 R9 the legend sits higher and the strict gate
dropped up to 13 s of race. The render updates at 10 Hz in 25 fps video; a frame within mean |diff|
3.0 of the last kept frame is a duplicate. Panels are cached as JPEG with their video index as
timestamp. `video.grid_keep` drops off-grid re-encodes of a kept tick (HV 04-22 R1/R9 have a
keyframe every 100 frames that enters as a zero-motion tick).

Circle render: the render's yellow odometer text (fraction ≥ 0.05 of its box; live film and the dissolve
≤ 0.02, the render 0.10–0.33) plus ≥ 2 discs; the top-strip test stays and there is no texture floor (the
smoother 2026/27 render sits below the silk one's, median Laplacian variance ~230). Without the odometer test
the start gate's blue banner passed as discs. Once the render has started (`relaxed`), the odometer text is not
required: it leaves ~20 m out while the render runs on across the finish line to the end of the replay; requiring it
stopped every circle path ~20 m short (no render finish crossing on 24/25 dev races). With the relaxed gate the 25
dev circle races (commit 6f4d62e) score all five timing gates on 17 (68 %), winner 24 (96 %), lateral AUC
(stewards' WIDE vs INSIDE) 0.998, census 95.9 %. With the finish-camera misread rule and the number split (421ef9e; observe onward, local):
all five on 19 (76 %), winner 24, lateral AUC 1.000, census 96.8 %. Remaining: leader timing on four Sha Tin
races (the render's own odometer is late early in long races, by up to ~1 s, in both render eras; the film
read does not separate the failing races), HV 11-18 R7's first boundary (common to all horses; the render
odometer is early there too), HV 11-18 R6's winner (a short head with the winner's cloth hidden from the
camera), and the lateral-scale gate on Sha Tin B/B+2/A+3: the render draws the B racing rail ~6.5 m out from
the A rail (ridge profiles), against the official 4.5 m (B width 26 m, racing.hkjc.com course page).

### features

Circle render: disc centres replace the avatar detector (a disc-template filter on the closed blue mask, peaks
with fill ≥ 0.8 at least 18 px apart: discs 25 px apart, about 1.3 m, still split); the number inside each
disc is read with recognition-only OCR (`numbers.npz`, row-aligned with `yolo.npz`): 96–99 % of discs give a
valid read, 96.5–99.8 % agree with their tracklet's majority. `detect` uses the disc detections only.

The avatar detector runs once per panel here (`yolo.npz`: boxes + silk histograms, read by
`detect`); run dirs made before that recover `yolo.npz` from a YOLO-only `dets.npz`. ORB (3000 features) on a world mask: HUD removed, dark (V < 45) and vivid blobs dilated, and the
avatar detector's boxes, grown 1.6×, masked: the colour mask misses brown
and pastel avatars, which are static in the image because the camera follows the field, and step
matches locked onto them (ST 04-19 R4: camera "stopped" for 3 ticks, identities rotated). Over 26 races
the box mask cuts mid-race pose stalls 48 → 30, and with the frame grid 48 → 2 (the rest were HV 04-22
keyframes). Poses depend on the detector's device: the OpenVINO detector's boxes differ (ST 04-19 R4
9122 vs 9129) and change the ORB steps on 603/705 frames; CPU vs CUDA on HV 06-03 R8 moves the final
paths by ≤ 0.18 m (median 0.01 m). On ST 05-03 R7, HV 05-27 R9 and ST 05-24 R1 (CPU from scratch vs the CUDA reference) the frames are
byte-identical, the boxes match one to one (1 of 29,862 unmatched) within 1.4 px, and the features differ from there on; final paths agree to a
median 0.04–0.05 m (max 0.44 m; 1.6 m on HV 05-27 R9's video-only #12 where the posterior is loose) on the first two. On ST 05-24 R1
#2 and #3 exchange identities over most of the race (mirror-image 9 m / 4.5 m differences, tight in both runs) and #13 and #5 differ over
the last 20 s (sectional MAE 0.082 vs 0.103 s). A CPU result is therefore compared only with a CPU control. The reference roots are CUDA runs. Steps between frames ≤ 6 video frames apart
are RANSAC similarities (1.5 px) refitted by least squares (Umeyama): the camera is near-nadir with no
tilt and a slow zoom.

### pose

The pose graph has sequential edges (a failed step, < 60 inliers or
residual > 1.3 px, becomes an interpolated edge of weight 0.05), long-baseline edges at d = 2/5/12/24
frames and loop closures (frames > 120 apart, matched unguided because chain drift can exceed any
gate, kept if within 400 px of the chained prediction; weight 3). A similarity is z → Az + B in
complex numbers, so the probe residuals are linear: IRLS with Huber 3, gauge at the mid-race frame
(the first frame after the live-to-render dissolve is unreliable), edges > 10 px pruned twice.

### worldmap

The worldmap is the mean mosaic of all registered frames: horses average out, rails and painted
markers stay sharp.

### detect

The detections of a self-supervised YOLO avatar detector (conf 0.25, IoU 0.5), made in `features`
on the render panel; the legend icon band is dropped. Below a median of 0.6 N detections per frame (N runners) (all-weather dirt) a
map-difference blob detector is added. Inside the disclaimer band only conf ≥ 0.45 survives, and a
box recurring at one panel cell on ≥ 4 % of frames there is overlay text.

### silk

`detect.silk_feature` per detection drops turf and brown pixels and keeps 12 hue bins (pastel and
vivid), white and dark. Matching collapses pastel and vivid, because the render desaturates (correct
group in the top 3: 73/79 hand crops vs 63/79 split). `silkmap.SILK_MAP` maps each runner's official
silk prototype (`detect.silk_prototype`) into render appearance: a ridge fit on 187 legend-confident
tracklets, legend labels only (leave-one-race-out top-1 33 % → 47 %). Consumers: the appearance
split in `observe` and the silk likelihood ratio in `identify`.

### hud

Recognition-only OCR on fixed crops: the race clock on the film panel (every 10th frame), the render's
odometer and its 8 legend slots (re-read when the crop changes). Race-clock origin:
`clock_f0 = median(frame − 25·clock)`. On some Happy Valley broadcasts a LONGINES sponsor box sits above the
clock and pushes it ~45 px below the fixed crop, which then reads nothing (HV 2023-10-18 R8, HV 2024-10-16 R2).
When the fixed crop gives fewer than 3 clock reads, `ocr.read_clock_located` re-reads the same frames from the
video with the crop shifted vertically (−16…+96 px, step 4): each shift is scored by the reads that agree on the
clock zero (within 2 frames of their median) and the centre of the widest run of best shifts is kept. On HV
2023-10-18 R8 the shifts +24…+44 px read the whole clock and agree on the zero to ±0.25 frame. No race in the
196-race product has fewer than 3 fixed-crop reads, so the fallback changes none of them. With no clock read at
all, every stage that needs the clock zero stops with "no race-clock reads" (`start` records `none`).

The legend of a field of 8 or more shows 8 slots at a fixed pitch; a field of N < 8 spreads its N slots about the
same centre at a wider pitch (digit spacing 120 px for 7 runners, 133 px for 6, on the 945 px panel), so the 8-slot
crops read at most one slot there (ST 2024-04-14 R8: slot 8 only; ST 2025-04-13 R1: slot 4 only) and `identify` had
almost no legend votes. The slot crops follow the field size (`ocr._legend_slot_x`, a line through the two measured
pitches). Fields of 8+ read exactly as before. ST 2025-04-13 R1: 6 of 6 tracklets numbered instead of 2, winner
and finish-n now pass (leader timing 0.289 s still fails); ST 2024-04-14 R8: 10 of 17 instead of 6, every gate passes
as before. The product's one other field under 8 (ST 2026-03-01 R7, 6 runners; not used to fit the pitch) reads 96–98 %
of frames in each of its 6 slots instead of one slot: 12 of 18 tracklets numbered instead of 5, and the video-only
gates go from 3 to 8 of 9 (winner, finish ρ 0.20 → 1.00, sectional MAE 0.596 → 0.109 s, behind-winner, paths; the
lateral scale still fails).

### rails

Stations every 2 world px follow the camera path, then the field's inner edge (90th percentile of
inward horse offsets, smoothed by `rails.robust_offset`, a Huber Whittaker smoother: on HV 05-20 R9
an outlier run folded every ridge into a V). Thin bright ridges on the inward normals are linked by
a Viterbi DP (`rails.dp_track`) that scores +1 for any ridge above threshold and −1 for none, and
charges 0.002 per px inside the field edge. It rewards continuity, not brightness, so the DP stays on the
thin movable rail instead of the brighter fixed rail ~6.6 m inside. Then a world-space despike (> 0.55 m off a 26 m quadratic trend), sub-pixel refinement and
`rails.smooth_rail`, a support-weighted Huber Whittaker smoother with a third-difference penalty
(arcs are free; ~20 m cutoff; unsupported stations weight 0.1; end runs continue straight): 4 m-chord
kinks > 8° on ST 05-24 R1 81 → 0, marker grid found on all 26 races. `rails.RailFrame` takes
positions and normals from a 121-station (~12 m) cubic fit, because snap wobble (~0.1 m) makes
normals noisy at wide lanes; projection uses a time-ordered station hint for revisited ground.

The inner side is the measured turning sign when the camera path turns at least a quarter turn
(range of the cumulative turning, so a fold-back at one end cannot cancel a bend; ovals 2.8–9.1 rad,
all clockwise; straights ≤ 0.28), else the race's handedness (`rails.inner_side`). On straights
(the three Sha Tin 1000 m races) the mosaic shows the fence only at d ≈ 400–550, so
`rails.straight_datum` makes the field's inner-edge scaffold the rail and adds a measured per-race
fence offset `lat0_px` (median over the longest supported DP run: 111, 111, 61 px). Straight-course
lateral is pack-edge-relative plus that constant; pack-to-fence drift (up to ~1.8 m on ST 06-07 R8)
is not modelled. A "fence" measured more than 1 m (observe's inside-the-rail tolerance) on the field side of the
inner edge is another ridge, and `observe` treats it as unmeasured (`lat0_px` 0, as when no ridge is supported). On ST 2025-04-13 R1 the
ridge sat 27 px (1.4 m) inside the field, the real rail runs along the inner edge (pixels), and the gate dropped
the inner horse on 410 of 480 frames, so no frame tracked the whole field and `identify` numbered nothing. Over
the product's straights `lat0_px` ranges −2.1 (ST 2025-01-05 R6) to 114 px, so the rule changes none of them. On 3 straights × 6 seeds the datum with the constant lifts finish ρ 0.884 → 0.923.

`rails.movable_rail_check` moves the rail out when a supported ridge sits at the course-table gap
toward the field and the innermost horses are still outside it (the extracted ridge was the fixed
rail). Painted markers (FINISH/50…300 m) appear only near the end, so they are profiled on a mosaic
of the final 9000 world px. `rails.fixed_rail_gap` measures the lateral-gate gap on a padded window
(the ridge filter wraps at window edges; ST 06-07 R6 read 11.27 m instead of 9.73 m).

### film

The aerial's left half is a live race film on the broadcast clock; in the straight a front-on
camera sees the painted 400–50 m stripes. `film.crossings` tracks paint lines that sweep ≥ 80 px up
the image, takes the lowest COCO horse box as leader (the detector runs on a track's frames only up
to the crossing the track is read at: the first with a readable odometer), and names the marker from the HUD odometer (0
to 40 m behind). Crossings must imply 12–22 m/s to a neighbour and lie ≥ 20 m inside the home
straight (Happy Valley's 400 m marker is on the bend). Sha Tin 400 m crossings average −0.11 s
against official (early on 9/10 races); the bias is recorded, not corrected, since correcting it
would feed official data into the video-only product.
`HKJC_FILM_DEVICE=intel:gpu` (or `HKJC_DEVICE=intel:*`) runs this detector as the OpenVINO export on the
Intel iGPU, compiled at FP32 (a missing device raises). It is not bit-identical to the CPU default: on 26
races the marker sets match and every crossing is within 5.4e-4 s of the references; on the 4 CPU controls,
within 4e-7 s. The film stage takes 35–127 s instead of 98–484 s, and 47–93 CPU-s instead of
570–3,440. Each process holds about 2 GB of system RAM for it (1.1 GB of that shared iGPU buffers). The GPU plugin's FP16 default lost the 200 m crossing on ST 05-09 R1. The other `intel:` paths
(avatar detector, OCR recogniser, `horse_scale`) compile at the GPU default (FP16) and are untested at FP32.

### start

The race clock is the aerial's overlay clock, except where that clock starts late. Before the 2025/26 season the
overlay clock started about 0.4 s after the gates opened (up to ~3 s on some races), so every video-only time was
early by that much; from 2025/26 it starts with the gates. `start.lag` finds the gate opening in the live start
film that opens the aerial video: the closed stalls' doors are a band of dark pixels that drops when they swing
open. The door colour depends on the light: dark green in sun (OpenCV hue 35–95, saturation > 60, value < 110),
dark teal in shade (hue 96–107 on evening Happy Valley films). The teal band (hue 60–105) is tried only when the
green band finds no opening. It excludes the turf (hue ≤ 55), which the green band admits where the turf is shaded,
and the blue banner and night sky (hue ≥ 105). On an 8 × 6 grid of the quarter-resolution frames it takes the row
strip with the largest relative drop (≥ 25 %, ≥ 1 point) within 100 frames before the clock zero, provided the
drop persists over 40 frames (doors stay open) and is shared by ≥ 60 % of the strip's door-bearing columns (not a
person crossing). The onset is the 25 %
crossing of that step. The detector's convention lag (clock zero − onset where the clock is synced) is 1.4 frames at
Sha Tin and 2.55 at Happy Valley; a larger net lag of more than 4 frames re-anchors the clock (`RaceRun.clock_f0`
= overlay zero − lag, used by `observe` and every stage downstream). `film.npz` keeps overlay-clock times and
`observe` shifts them by the lag, so `film` does not depend on `start` and older caches stay valid. No detection,
or a lag within 4 frames, keeps the overlay clock unchanged. `start.npz` records the clock zero, onset, lag and status
(`late`, `synced`, `none`).

Measured (S1 pilot races, film 400 m read minus official): on 66 corrected pre-2025/26 races the residual after the
correction is −0.12 s median (the film read's own bias, as on synced races) with MAD 0.06 s and 1 race in 35 beyond
0.2 s; 6 of 74 pre-2025/26 races give no detection and keep the late clock. No correction on 54 synced-era races;
on the 26 development races one false correction (HV 06-03 R3, 0.34 s: its doors visibly open about 11 frames
before a clock zero that matches the official start).

On the 196 held-out races the green band finds no opening on 14. The teal band recovers 8 of them: 3 late (lags
7.4–10.4 frames) and 5 synced, each onset within about a frame of the first door motion seen in the frames. The
late lags (0.30, 0.29, 0.42 s) match those races' video-only whole-race errors before the fix (field-median
sectional error −0.51, −0.30, −0.39 s). The 5 synced races, all 2025/26 or later, already had errors of −0.01 to
+0.09 s. The other 6 stay `none`. On 5 the start film is not on the overlay clock's timeline: the clock zero extrapolates to
frames 0–2.5, but the doors open 25–60 frames later (on two of them the film after the opening is visibly slow
motion). The sixth (HV 2026-01-14 R8) is filmed from behind the stalls, so no door band is visible. On 50
held-out races where the green band works, the teal band alone also finds an opening on every race. Its onset is
a median 0.56 frames later (IQR 0.36–1.03), so a teal lag reads about 0.02 s short; 2 of the 50 differ by 12–13
frames (ST 2025-01-05 R6, HV 2023-10-11 R4).

### observe

Detections go to world px and to (s, lat) on the rail. The per-frame leader ignores a front
detection > 8 m clear only if the gap is not persistent over 11 frames (HV 05-20 R2's winner was
13 m clear).

**Distance metric** (`rails.calibrate_distance_ticks`). The odometer ticks V+10 → V as the leader
reaches V. Tick events give scale and finish through a robust line plus a cubic B-spline (150 m
knots) for map-scale drift; where ≥ 4 consecutive smoothed residuals exceed 10 m, knots densify to
25 m (the 1000 m straights, where the camera swings 180° at t ≈ 28–32 s and the pose steps ~2.5 % in
scale). Lateral keeps the base fit's px/m. The constant comes from the film (`film.anchor_offset`):
with ≥ 3 crossings the ticks are unpinned and shifted by the median offset (+2.4…+11 m) that puts the
render leader on the film's crossings. Against the film the render's painted FINISH pin is a step in
the last 50 m; film anchoring gave leader MAE 0.213 → 0.166 s and sectional 0.243 → 0.197 s on 26 races. With fewer
crossings, a tick-seeded marker grid (`rails.fit_marker_grid_seeded`: paint scale ≈ tick scale / 1.035
within ±8 %, FINISH within ±60 m, ≥ 4 markers including FINISH) pins d = 0 when its offset after
resolving whole-marker shifts is < 6 m and the scales agree within 8 %; a pinned grid also sets the
per-race tick convention (Sha Tin ≈ −3.2 m, Happy Valley ≈ 0). `rails.time_local_shift` subtracts a
per-frame along-track shift where ≥ 3 consecutive ticks keep |residual| > 10 m (a pose slip in the
straights' camera swing). Lateral = (lat_px + lat0_px) / local px/m; detections > 1 m inside the rail
are dropped.

**Timestamps.** The render updates at 10 Hz in 25 fps video, so new ticks start 2 and 3 frames apart
while their content advances a uniform 0.1 s. Each render frame is stamped with its tick time
(`video.tick_frame`): the residue followed by a 2-frame step moves −0.25 frame, the other +0.25
(mean zero; off-grid frames unchanged). All observation times (tracks, tracker, film anchor,
time-local shift) use it. Against video-index time the along-track residual on the fused path
alternated −0.17 / +0.18 m; with tick time it is +0.006 / +0.004 m. Over 26 × 6: twin tracks 208 → 48
pairs, hand-wrong final labels 175 → 153 (net: 57 corrected, 35 newly wrong; unread races 142 → 134, read 33 → 19), gate passes unchanged (paired over 156 runs: leader MAE −0.0006 s,
sectional MAE +0.003 s, SE 0.001).

**Tracking.** Constant-velocity prediction and per-tick Hungarian assignment in metres (gate 3σ,
σ = 0.9/0.45 m, coast 20 ticks), then kinematic stitching across gaps ≤ 6 s.
`tracking.split_by_appearance` cuts a tracklet where its mean silk signature changes (total-variation
distance > 0.30, ≥ 40 silk detections per side; legend-pure tracklets reach at most 0.28): hand
majority 58 → 62/79.

`HKJC_PERTURB_SEED=k` drops 1 % of detections here (seeded) for robustness ensembles.

### finish

The full feed is scanned from `distance/18` s for 70 s for the finish camera: a white vertical line
at a stable x, slow motion at a variable rate, so unique-frame index is not time. Saddlecloths are
OCR'd per unique frame of a run that reaches 40 unique frames (shorter runs are dropped unread): yellow on dark (Happy Valley, overhead) and, at Sha Tin only, dark digits on
white cloths inside a white ring. A COCO horse-box ruler (2.6 m nose to tail) gives camera px/m; the
box includes extended legs, so the ruler-based C reads ~×1.17 high, and official finish gaps must not
be used to calibrate it.

`finishcam.tracks` keeps, per number, the lateral cluster read at the most distinct places (a rail
post read 60 times at one spot outvoted the real cloth on ST 06-07 R8) and needs ≥ 4 reads and
≥ 100 px of motion. Digit drops ("1" on the "11" cloth) are removed when most of the shorter number's
reads lie on the longer number's track; coinciding numbers keep the longer one. On the overhead camera
(Happy Valley; `RaceRun.finish_sc` carries the geometry) reads of any number lying on a much better-read
track (≥ 10 reads, ≥ 3× the reads it claims, on its line within 60 px, its lane within 30 px) are that
cloth's misreads and are removed before tracks are built: on HV 2026-11-18 R6 '3' reads on the #7 cloth had
merged with the real #3's reads into a track that put #3 second (official: last). On 197 non-sealed races
camera-order inversions vs official fell 228 → 217 with no race worse; on the Sha Tin side-on camera, where
horses abreast share one image band, the same rule made 4–5 races worse at every setting tried, so it is not
applied there. `finishcam.finish_gaps`
gives spatial gaps at an instant: crossers are chained from the crosser that links the most, and other
horses get `gap_h = gap_r + x_h − x_r` from ≥ 2 shared frames, which recovers horses occluded at the
line (sd = robust spread / √n, floor 15 px).

### identify

`identity.legend_votes` uses frames where every runner is tracked by distinct tracklets: the
tracklet at rank r votes for legend slot r, and ranks 9+ get set-valued elimination evidence ("one of
the N − 8 unlisted numbers"). It takes the runners' saddlecloth numbers, so withdrawn numbers (gaps)
neither block the whole-field test nor enter the unlisted set: on HV 04-15 R6 (#2 and #4 withdrawn) the
previous field-size argument cast no legend votes; with the runner numbers it casts 3713–4255 slot
votes per seed, none for #2 or #4, numbers 11 of 13 tracklets instead of 6 (the 6 old names unchanged; of the new ones, #7 and #8 are
the reverse of the old EM's labels and match the silks on six seed-0 crops, read with the labels known), and changes no gate on any seed (sectional MAE
0.002–0.006 s lower on all six). One race chosen by a withdrawal rule, not a held-out claim.
A runner tailed off out of the render camera's view leaves no whole-field frame: on HV 2024-04-17 R7 #7 (first
section 43.9 s against ~24 s) is never rendered and never listed, at most 9 of 10 horses are tracked, and `identify`
numbered nothing (`fuse_video` stops with "identify numbered no tracklet" when that happens). When no frame casts a whole-field vote and the legend never
lists some runner, frames tracking all runners but one vote, the untracked runner taken as one behind the legend
(its ranks 9+ give the usual set-valued elimination evidence). On HV 2024-04-17 R7 that numbers 14 of 17 tracklets
(94 % of detections). Races with any whole-field vote are unchanged; in the 196-race product the
rule applies to one silk race, ST 2025-05-10 R6 (12 runners, #9 never listed): 11 of 14 tracklets numbered
instead of 3, horses without a path 3 → 0 and finish-short 3 → 1 (both gates now pass), sectional MAE 0.145 → 0.207 s
(coverage 93 → 83 %), finish ρ 0.98 → 0.99, behind-winner 0.051 → 0.025 s. The finish camera names tracklets by 1D registration
(`finishcam.identity_match`) of their line crossings to `t_w + gap/C`, scanning C ± 25 % around the
ruler prior; a name needs residual < 0.08 s with every other prediction ≥ max(2× residual, 0.15 s)
away, and adds 40 slot votes.

`identity.assign_global` solves the global maximum-likelihood assignment as a MILP: a vote is right
with probability ρ, else uniform, so evidence grows with the vote count; elimination frames score
log(ρₑN/m) for each unlisted number; ρ and ρₑ are re-estimated from labelled tracklets (camera votes
excluded); the silk LLR `2.24 − 6.37·TV` (clipped ±3) counts where the legend has spoken; one number
per tracklet and per frame; gain ≤ 0 stays unlabelled. Against the greedy labeller it raised identity
ρ 0.954 → 0.966 and hand majority 108 → 113/151. Unlabelled line-reaching tracklets are then matched
by elimination (`finishcam.eliminate`) to numbers with no line-reaching tracklet. Camera and
elimination names are hard constraints downstream.

Circle render: `observe` first splits tracklets where the disc number changes and stays changed
(`tracking.split_by_number`: ≥ 8 of the last 10 confident reads name one other runner number that is not a
dropped digit of the current one, and it holds for the next 20 reads, ~2 s). The motion tracker swaps
identities where discs touch, and the post-swap stretch kept the tracklet's majority label (ST 2026-10-19 R3:
#11 and #12 swap at ~60 s, #2's reads become '9' at ~90 s). On the 25 dev circle races: no gate changed for
the worse, ST 10-19 R3 finish ρ 0.88 → 1.00, ST 11-08 R7 ρ 0.73 → 1.00 and behind-winner 0.335 → 0.012 s.
Swaps between 1 and 11 (or 2 and 12, ...) look like dropped digits and are not caught.

Circle render: each confident disc-number read (OCR confidence ≥ 0.6, a runner's number) adds one vote for
that number to its tracklet, about one legend frame's worth; late-frame numbers count as shown for the
blind set in `fuse_final`.

### fuse_video, fuse_final

#### Video-only posterior (`fuse_video`)

`fuse_video` writes `paths_video.parquet` and persists its observations and paths
(`fuse_video.pkl`); `fuse_final` reads them, so final-product changes re-run only `fuse_final`.

**Model** (`fusion.smooth`). Per horse, s and lat on a 0.1 s grid with second-difference priors:
along-track acceleration sd 6 m/s² for t < 6 s, 1.2 m/s² after; lateral 0.25 m/s² plus a 2 m/s
lateral-velocity term; no along-track velocity term.
Observations enter at their own times by linear interpolation; render sd 0.8 m along, 0.35 m
lateral. Lateral is boxed to [0.3, width − 0.3] m. The MAP is a banded Cholesky solve with violating
nodes pinned by a stiff penalty (≤ 8 rounds); it matches an exact sparse LU to ~1e-8 m in 1/27 of the
time (declared non-exact: labels differ on 3/26 races by 0.1–3.2 %). Posterior sd comes from the
unconstrained normal equations at every 10th node, interpolated.
The grid ends at the race's latest observation. When `paths_video.parquet` is written, each path gets
a 0.5 s prior-only tail (`fusion.with_tail`): s continues at the end-node speed, lat is held, and sd_s
grows with the acceleration prior. The fusion, EM and `fuse_video.pkl` never see the tail. Without
it, a horse a few cm short of the line at the last observation has no finish crossing (HV 04-15 R1 #2
ended 4 cm past the line in one seed and 3 mm short in a corrected arm). Tail crossings come from
extrapolation: up to 7.6 m (ST 05-09 R2 #11, ST 05-09 R1 #8), most under 1.5 m.

**Association** (`fusion.fuse_race`). Anonymous tracklets are scored against each posterior by mean
χ² plus log-det ratio; the best must be < 9, beat the runner-up 2×, not overlap that horse's
same-feed tracklets in time and not be in the tracklet's forbidden set. Orphan tracklets are chained
and given to a path-less horse by its finish/official anchors, or by elimination when one horse is
missing. The anchor check demotes up to 3 render tracklets whose removal at least halves a
horse-specific anchor conflict above 4σ (field median residual subtracted; never below half the
horse's render data), then re-fuses once.

**Finish observations** (`RaceRun.finish_observations`, a second pass inside every fuse). Each
gapped horse crosses d = 0 at `t_w + gap/C`. `finishcam.anchor` fits t_w and C (800–4000 px/s) to
fused crossings with sd ≤ 3 m, with a soft prior C = line speed × ruler px/m (relative sd 0.30). If the
label-free registration disagrees by > 0.4 s or > 30 % in C, the registration is used (t_w sd 0.15 s,
its measured bias); if both fail, no finish observations are emitted. Per-horse sd is conditional on
the fitted clock.

**EM identity pass** (`RaceRun._em_identity`). Legend votes on frames with undetected horses, ranks
completed by the first posterior where every sd < 2 m, are added and `assign_global` re-run. It relabels
6.3 % of detections on long identified tracklets (26 × 6; ST 04-19 R4 42 %); on the unread hand races its
tracklet relabels help 20 distinct hand detections and hurt 20.

**Detection-level EM** (`assoc.assign_frames`). The tracker fragments are only the initialisation.
Per frame, one Hungarian assignment of all detections to all horses, clutter cost 16 (4σ). A
detection's rank is 1 + the number of detections more than 0.6 m ahead of it in the frame. Costs:
- Gaussian −2 log-likelihood against each posterior including the log-det ratio, so a vague
  out-of-view path pays for its vagueness.
- Hard legend rank: a detection with ≥ q detections more than 4.5 m ahead (4σ on a difference of two
  detections) cannot be the horse in legend slot q (+50, forbidden). At 0.6 m two
  detections 0.9 m apart hard-excluded correct horses (ST 04-26 R1); 4.5 m: hand 118 → 121/151.
- Soft legend rank: +9 for slot q where the detection's rank, confirmed by the count of tight
  posteriors ahead, differs from q.
- Negative legend evidence: a horse missing from a fully read top 8 is barred from detections surely
  inside the top 8 (detections ahead + undetected horses + 1 < 8) or > 8 m ahead of the rearmost tight
  listed horse.
- Finish-camera negative evidence: with recall r ≥ 0.5 (capped 0.95), detections in the last 15 m
  cost −2 log(1 − r) for unread horses; a horse unread and absent from the legend in the 8 s before
  the first crossing ("blind") pays CLUTTER, hard, in the last 50 m.
- Rank attraction: in the last 100 m a detection whose rank equals a listed horse's slot costs that
  horse at most 10. Slowing unread horses' real detections sat 7–13 m behind their extrapolated paths
  at χ² ≈ 22 (ST 04-26 R9 ρ 0.89 → 1.00 on 6/6 seeds).

`assoc.release_post_finish` makes anonymous any detection of a horse later than its first detection
past the line and > 3.2 m before it (ST 05-24 R1: two such detections dragged the camera clock and
lost the winner). `assoc.runs_to_obs` splits runs at gaps > 1 s and drops runs of
< 3. Unassigned detections stay anonymous per tracker fragment and carry the hard exclusions held on
≥ 50 % of their detections, or fusion's association reassigns them to an excluded horse. Up to 8 iterations, stopping
below 0.2 % label change.

`RaceRun._finish_seeds`: horses the camera read but still path-less after the EM get an E-step-only
seed path (camera crossing, field line speed, sd growing 1 m/s away from the line) and the EM runs
again. Shadow release: two tight paths within 0.8 m along and 0.5 m lateral for ≥ 0.5 s in the last
3 s lose the weaker horse's detections there (the unread one, else fewer detections).

#### Final product (`fuse_final`)

The final product reuses the video-only detections and adds the official records:
1. `RaceRun.official_observations`: the draw fixes lateral at t = 0 (stall 1 at 2.5 m, pitch 1.06 m,
   measured on HV 06-03 R8's drone start frame; sd 0.4 m) and official cumulative sectional times fix
   crossings at d = 400k with sd 0.1 m (their 0.01 s publication precision); runners without sectional
   times are held to their official finish time. At sd 0.4 m a mislabelled render tracklet outweighed the
   record (145 S1 silk races: official winner wrong on 7, finish-order inversions on 75).
2. `RaceRun._metric_bias`: at each boundary, b = median over ≥ 3 horses of (video-only crossing delay
   × speed); render s is shifted by b interpolated along s. The render metric error is shared by all
   horses, and hundreds of render observations otherwise outvote one official time (residual ~5× its sd).
3. Fuse with the grid running 0.5 s past the later of the last visible official time and the video-only
   grid's end (ending at the last observation left HV 04-22 R9 #11 0.8 m short of the line; with the
   finish withheld, the last visible official time is 400 m out). Held-out finish (196 races, finish
   withheld): median 0.135 → 0.125 s, p90 0.577 → 0.510 s, horses without a crossing 24 → 10.
4. Field-relative start prior: until the first render frame + 1 s there is no video, so acceleration
   has sd 0.63 m/s² per node about the field's median curve from step 3 (the clean races'
   horse-specific acceleration). The zero-mean 6 m/s² prior extrapolated a late-seen horse backwards
   out of the gate (ST 05-31 R1 #6 26 m out at 2 s, the field 8–15 m).
5. Final identity pass: `RaceRun._skeleton_merge` keeps the step-4 path except where it conflicts with
   the horse's official-only skeleton by > 4 combined σ (flat at 3–5); the E-step is re-run from there
   with the official observations in every fuse and bias-shifted detections.
6. `RaceRun._official_retime`: each fused path is re-timed so it passes every official boundary at the
   official time. The re-timing (`RaceRun._monotone_warp`) is a strictly increasing C1 cubic Hermite
   through the (fused crossing, official time) knots with Fritsch–Butland slopes, fixed at the path's
   first node and continued past the last boundary at the last section's rate, so it adds no speed jump
   at a boundary (5 races: speed jump at the boundaries the same as the un-re-timed fused path); horses sharing an official finish time keep the official placing, 2 ms apart. Shifts over
   0.05 s are logged (they mark identity errors left in the video evidence). `paths.parquet` reproduces
   the official result: on the 145 S1 silk races every winner, no finish-order inversion, every
   horse-boundary within 0.02 s.

### status

`run()` always writes `status.json` (not a stage artifact): `failed` with the stage and error when a stage raises,
else `unhealthy` with the failed video-only checks, else `ok`. Checks: render ticks per expected race second
(unique render frames / (distance / 16.5 m/s)) ≥ 4.0; median avatar detections per render frame per runner ≥ 0.4;
≥ 3 race-clock reads; the rails stage's course shape (straight writes `lat0_px`) matches the race facts (straight
only for Sha Tin 1000 m). The `start` status is recorded but not a failure. On S1 and the development set: all 26
development races and every completed silk-render S1 race are healthy; every completed 2026/27 race is unhealthy.

### validate

`recon.validate` writes `validation.json` (gates on `paths_video.parquet` plus the final product's
residual against official times). `recon.audit` writes `audit.json`: label-free invariants (rail off
the map, marker grid and spacing, kinks; field-to-rail gap, field-wide lateral slides; legend
negatives and ranks, ghost crossings, finish-order inversions). A contiguous off-map rail run over 100
stations blocks an ensemble (clean rails reach 44).

## 3. Feed synchronisation

`viz.sync_video` draws the four feeds beside the reconstructed field on a map built from the run's
own rail. Offsets are video seconds at race clock 0, a constant or `[(race t, offset), ...]`
interpolated.
- aerial: `clock_f0 / 25` (the on-screen race clock; 25.000 fps on every race).
- full: `viz.full_offset`, the lag maximising thumbnail correlation between the aerial's left panel
  (a scaled copy of the TV picture) and the full feed (correlation 0.66–0.92 on 26 races; two windows
  agree within 0.08 s); full sits at clock + 1.21–1.51 s.
- patrol and drone: gate opening read by eye from 0.2 s contact sheets, assuming the gates open at
  clock −0.12 s (measured on HV 06-03 R8 only). Patrol opens at 1.15–1.4 s, drone at 2.1–3.1 s, with
  −0.35 s for drone horses becoming visible clear of the gate. The drone feed loses time during the
  race: its offset at the finish board is ~1.5 s smaller than at the gate (HV 06-03 R8 0.96 vs 2.46 s;
  ST 04-19 R4 1.16 vs 2.67; ST 06-07 R6 1.16 vs 2.77), so it is anchored at both. Precision is a few
  tenths of a second. ST 05-03 R7 has no drone feed.

Display: a ring marks sd 1.5–3 m, a hollow marker 3–10 m; positions with sd > 10 m are omitted and
listed as "not placed"; the leader and window use sd < 3 m. A small sd does not flag a confident
error (ST 04-19 R4 #3 at sd 0.17 m on wrong detections).

## 4. Acceptance

### Fixed gates (`recon.validate`)

Scored on `paths_video.parquet`. A per-horse crossing is scored only where the posterior sd_s < 1.5 m
on both bracketing nodes. Boundaries are the official sectional ends, 400 m apart.

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

The last four were declared before held-out races. "Non-lateral gates" counts the first eight.

### Yardsticks for decisions

- **Perturbation ensembles.** 6 seeds of `HKJC_PERTURB_SEED` (1 % detection drop) per race, 26
  races; arms are compared paired by seed with SE from per-seed differences; the adoption rule
  (benefit and guards) is declared before reading results. Null: seeds 6–11 vs 0–5 give per-seed
  totals with sd 0.79, so 6-seed mean differences under ≈ 0.65 non-lateral gates are noise. A run never
  scores a directory another run writes into.
- **Hand truth.** Detection crops labelled at colour-group level against official silks; every set
  after the first 79 was labelled blind to pipeline labels. A detection counts for "hand majority" when
  its majority label over the seeds is in the true group. Scales are not comparable: 79 detections on
  4 races; 151 on 8 races (ST 04-19 R4,
  ST 05-24 R1, ST 05-31 R1, HV 05-27 R9, ST 05-03 R7, ST 04-19 R3, HV 04-15 R1, ST 04-26 R1);
  77 blind on ST 06-07 R6, HV 05-13 R9, HV 05-20 R9, ST 06-07 R8; 38 on ST 05-09 R1. ST 04-19 R3's
  original labels are quarantined (a red/yellow-check silk mislabelled); its 24 fresh labels are
  development only. Final-label counts (220/221) pool the sets; their denominator is not recorded.
- **Unanimity.** Share of detections whose E-step label agrees on all seeds (label-free stability).
- **Identity score.** Spearman of reconstructed vs official crossing times and the share of
  horse-boundaries within 0.25 s after removing each boundary's common mode (validation only).
- **Per-horse census** (what a viewer of the synced video sees). A horse-run is good when its
  posterior sd < 3 m on ≥ 90 % of the span from the first render detection to its crossing, it is scored at the line, and its largest boundary error
  with the boundary's race median removed is ≤ 0.25 s (~4 m). HV 06-03 R8, judged good by eye,
  scores 100 %. On the final product the residual is raw against the official records, which are
  then inputs, so it detects conflict only.
- **Glitch-seconds** (final product). Per horse, gap = d − the median d of horses still running;
  horse-seconds where the gap changes by > 3 m/s over a 1 s window, between the first render detection
  + 1 s and the first finisher − 1 s, with sd < 3 m at both ends; excluded vague seconds are reported
  alongside.
- **Held-out boundary.** `Mask(boundary=j)` (`--mask boundary=j`, alias `HKJC_FINAL_HOLDOUT=j`) withholds one
  official boundary from every stage (official observations, metric bias, re-timing); the final product's
  error there is independent of that record. `Mask(outcome=True)` withholds every official outcome record.
- **Pixels.** `viz.reproject` and the synced video; every claimed fix is checked on frames.

The census, glitch, identity-score and ensemble scripts live outside the package.

## 5. Current state

### Data

26 turf races, all development races (inspected). Sha Tin: 04-19 R3 (1000 m straight), 04-19 R4,
04-26 R1, 04-26 R9, 05-03 R7, 05-09 R1 (1000 m straight), 05-09 R2, 05-17 R1, 05-24 R1, 05-31 R1,
06-07 R6, 06-07 R8 (1000 m straight). Happy Valley: 04-15 R1/R9, 04-22 R1/R9, 04-29 R3/R9, 05-13 R1/R9,
05-20 R2/R9, 05-27 R3/R9, 06-03 R3/R8. Also inspected (not in the 26): HV 04-15 R6, the withdrawal test race.
The all-weather track is out of scope. The last sealed
held-out test (11 races, earlier frozen code, run once) passed all five timing gates on 9/11 with
11/11 winners; those races are in the development set, so a further held-out claim needs another
set of races sealed by a metadata-only rule.

### Baseline (tag `recon-baseline-Z12`)

Z11 plus the `start` stage (the race clock re-anchored to the gate opening when the overlay clock lags it), 26 × 6
via `scripts/perturb_ensemble.sh` (default stages now start, observe..validate): identical to Z11 on 150 of 156
runs; HV 06-03 R3 (all 6 seeds) takes a false 0.34 s correction. Gates per seed unchanged: non-lateral 201.67,
timing 126.67, all five on 23.67 races.

### Earlier baseline (tag `recon-baseline-Z11`)

Z10 with tick-time stamps in `observe` (observe onward on the Z10 stages, 26 × 6). Video-only: non-lateral gates
201.67 per seed, timing gates 126.67, all five on 23.67 races, lateral 10 pass / 3 fail / 13 n/a (unchanged from Z10),
finish-short 26–27 per seed. Final product: hand truth (seeds 1–5) 1117 correct, 153 wrong, 51 blank of 1321
(Z10: 1095 / 175 / 51; net of 57 corrected and 35 newly wrong; unread races 142 → 134 wrong, read 33 → 19); glitch-seconds 21.9 per seed (Z10: 25.6); seed unanimity 90.2 % (89.2 %); census 97.5 % of horse-runs good; interior boundaries
withheld in turn (seeds 1–5, also from the shared metric bias): median error 0.112 s (1.97 m) at 400 m from the nearest official
time (≈ 0.08 s midway between boundaries in the product, see §6); relative to the field at that boundary 0.0322 s (0.56 m), 61 % beyond
2 posterior sd.

### Earlier baseline (tag `recon-baseline-Z9`)

Commit 35c7c72 (withdrawn numbers in `legend_votes`) leaves `labels.npz` identical on 26 × 6.

From-scratch CUDA run of all 26 races on the current code (pose stalls mid-race: 2, both HV 04-15 at the
render start). Video-only, 26 × 6: non-lateral gates 200.67 per seed (200–202), timing gates 126.67, all
five on 23.67 races, lateral 10 pass / 3 fail / 13 n/a, finish-short 28–29 horses per seed. With the
written tail (tag `recon-baseline-Z10`, fuse/validate rerun on the same stages): finish-short 26–27,
ST 05-09 R2 finish-n passes on all 6 seeds, no other gate changes. The box mask
and frame grid failed their own adoption rule on the video-only gates against the unmasked poses
(non-lateral 201.33 → 200.67 per seed, SE 0.33; timing −0.50; all five −0.67), reported as a failure;
they are the code because the pixels are right: mid-race pose stalls 48 → 2, ST 04-19 R4's wholesale
identity rotation gone, ST 05-09 R1 blind labels 13/38 → 37/38, seed unanimity ST 04-19 R3 49 → 88 %.

**Final product** (26 × 6): census 97.4 % of horse-runs good, glitch-seconds 25.6 per seed (without the
final identity pass, measured on the previous base: 93.1 %, 56.4; the start prior removes every
implausible pre-render race); held-out boundary median error relative to the field 0.032 s (all interior boundaries, seeds
1–5, previous base, whose outputs differ only on HV 04-22 R1/R9); hand truth on the final labels (seeds 1–5) 1095 correct, 175 wrong, 51 blank of 1321; video-only
paths byte-identical with and without the final chain.

### Persistent failures

Gates failed on ≥ 5 of 6 seeds:

- HV 04-22 R9 leader and sectional: a common-mode scale error, +0.44 s late at 800 m, +0.37 s at
  400 m, −0.17 s at the line (median of 12 horses); the film window is too short to measure the slope.
- ST 04-19 R4 sectional (0.28 s on 5/6 seeds): #3, tailed off and out of view, sits on a lookalike
  tracklet (worst boundary error 4.55 s); #5, #12 and #14 reach 0.5–0.8 s. Its finish camera is a
  slow-motion post camera: the winner #11 is read only sparsely and #10's "1" is hidden by the boot at
  the line, so neither gets a gap.
- Finish-n, video-limited (every horse the blind near-line cost applies to, 24 race-horses over 26 races, is finish-short on all 6 seeds,
  against 26–27 finish-short horses per seed): ST 05-03 R7 (#4 8 m short at the last render frame, #10 3 m short, #11's
  detections end 200 m out), ST 05-09 R2 (#9 3 m, #11 9 m short at the last observation; passes since
  the tail, with #11's crossing extrapolated over 5.9–7.6 m),
  HV 05-27 R9 (no detections over t 70.56–72.28; #7 unobserved for its last 16 m, sd 1.78 m at the
  line; #9 and #10 lose detections 73–87 m out). ST 05-24 R1: #8 and #12 have no detections after
  125/90 m; #4 has 4 sparse detections in the last 25 m that the blind cost bars.
- Lateral: ST 05-03 R7 and ST 05-31 R1 measure the B-course fixed-rail gap at 6.6 m vs the table's
  4.5 m (one clean ridge at 6.5–7.0 m on 98 % of stations; the official course diagram confirms the
  table, so the render draws the B fixed rail ~2.1 m further in); HV 04-29 R3 (weak ridge, 9.21 m).
- Occasional: ST 04-26 R1 winner (2/6, a 0.05 s finish), HV 05-20 R2 sectional (1/6).

## 6. Open problems

- **Sealed pilot S1 (200 turf races, 2023-09 to 2026-11, frozen tag `recon-pilot-S1`, run once): failed.** 164
  races completed, 36 crashed; all five timing gates on 35 of 200 (2023/24 3, 2024/25 0, 2025/26 32, 2026/27 0;
  development set 91 %); winner on 121 of 200. S1 is now development data. Causes found: (1) the overlay clock's
  late start before 2025/26, fixed by `start`: on 51 pre-2025/26 S1 races outside the 30 used to tune the
  detector (its rules were chosen while viewing film-minus-official on all pulled races, so these are not
  held out) leader-timing passes go 1 → 39 and all five 1 → 29; on the 49 S1 races pulled after the detector was
  fixed (held out), pre-2025/26 (19): leader timing 0 → 14, all five 0 → 12; 2025/26 on (30, one correction on the
  season's opening weeks): 16 → 17 and 13 → 14; (2) the 2026/27 render draws horses as blue
  numbered circles with no silk avatars or legend, so the avatar detector and silk identity do not apply (24 of
  the 36 crashes); (3) a sponsor box above the left-panel clock moves it about 45 px below the fixed crop, so no
  clock is read and `film` fails (2 races; the clock is now located, see hud); (4) one race (ST 2024-04-07 R5) with pose residuals of 0.05 px and 494
  loop closures reads as a straight course and gets no paths: its panel box came out full height (letterbox
  bars included), so the HUD mask missed the disclaimer text and pose locked onto it (verified on the inliers). Further crashes are listed in
  the S1 failure log and not yet traced. On the frozen code the lateral check against stewards' comments passes on
  the silk render (AUC 0.931 on 91 pre-2025/26 races, 0.947 on 47 from 2025/26) and fails on the new render
  (0.604, 26 races); the final-product census is 88.2 % / 93.4 % / 21.4 % good horse-runs (threshold 95 %). After the
  `start` fix the silk races pass all five on 41 of 70 (pre-2025/26) and 28 of 40 (2025/26); the largest remaining
  classes are leader timing (missed clock lags and 4 gross failures) and finish-only failures (15).
- **Older seasons.** The aerial feed exists back to 2015, at 1280 × 720 until 2023-05; the pipeline assumes
  1920 × 1080.

- **Final-product calibration.** On held-out interior boundaries (Z11, seeds 1–5, n = 4295) the median error is
  0.56 m and 61 % of errors exceed 2 posterior sd (median z 3.05); counting the official crossing's own sd (0.4 m) the median z is
  1.25 and 32 % exceed 2 sd. The video-only product shows the same pattern at intermediate boundaries (error relative to the race
  median 0.53 m against a stated sd of 0.17 m) but not at the finish (0.27 m, stated 0.28 m; median z 0.88, or 0.51 with 0.4 m for the
  official). Those held-out figures are relative to the field at the boundary: until d4812bb `HKJC_FINAL_HOLDOUT` did not withhold boundary j
  from `_metric_bias`, which removed the race's common mode at j using the withheld times. Withholding it too, the median held-out
  error is 0.112 s (1.97 m; median z against path ⊕ official sd 4.4): the shared metric between boundaries (median 1.8 m from an
  interpolation over the other boundaries) dominates that figure. It is the error 400 m from the nearest official time with the
  neighbours 800 m apart; in the product every point is within 200 m of one. The shared bias's steps between adjacent boundaries
  are close to independent (median second difference / median step 0.77, 45 triples), so the product's shared error midway between
  boundaries is about the held-out figure / √2, ≈ 1.4 m (0.08 s), falling to zero at each boundary. Verdicts made with the old holdout (0157, 0160) were paired on the same
  measure and stand as relative-error verdicts. Measured map (held-out rows within 1 s of the boundary leader, path sd plus
  f = 0.71 m, official 0.2 m, fitted leaving one race out): median z 0.68, but 11 % beyond 2 sd (heavy tail); 0.2–0.5 m stated sd
  0.93 / 18 %; rows > 1 s behind the leader 1.29 / 31 %. Boundary leaders are no better (0.555 m) than horses within 0.3 s of them (0.453 m), which does not locate the floor. Where the stated sd is ~0.17 m and
  5–15 observations lie within ±0.5 s, the median error is still 0.55 m: an error floor that more
  detections do not reduce, so a systematic term, not only correlated detections. Candidates:
  effective sample size per run, or empirical inflation from held-out boundaries, under their own rule.
- **Remaining glitches.** Most final-product glitch-seconds (relative speed > 3 m/s) where the
  video-only path is smooth come from identity conflicts at an official crossing: a wrong E-step label
  between boundaries pins the horse, the official time pulls it back (HV 05-20 R2: one tracker fragment that runs the whole race and crosses the boundaries at #2's
  official times is labelled 2, 5, 6, 2, 5, 4, 2 along its length; ST 05-31 R1's #6 runs wide and unlabelled until the last 50 m). Genuinely
  fading horses also exceed 3 m/s relative (HV 04-29 R9 #8, ST 05-03 R7 #11); relative acceleration
  separates them (faders ≤ 1.25 m/s², conflicts ≥ 2.4 m/s² on five traced cases). Boundary-crossing
  runs already agree with their official times in the final labels (1 of 783 off by > 0.3 s).
- **Speed kinks at official crossings come from the fuse.** The fused final path, before re-timing, already
  has a speed corner where it meets an official crossing: on 5 offA races, the step-to-step speed change
  within 0.15 s of a boundary averages 3.5× its value elsewhere (0.073 vs 0.021 m/s per 0.1 s). The
  median jump is 0.09 m/s (p95 0.23) net of local acceleration, against 0.005 m/s (p95 0.025) for the
  video-only path at the same instants. The re-timing adds nothing to it. Speed and acceleration
  features derived near a boundary carry this artefact.
- **Back-marker identity.** Back markers get only elimination evidence and a weak silk LLR: HV 05-20
  R2 #5/#12 are swapped at 400 m (~1 s, every seed), and at 1200 and 800 m #5 and #12 cross 0.6–1.4 s early on all six seeds (video-only;
  #2 0.6–0.75 s at 1200 m on six seeds and at 800 m on three; #6 and #8 0.3–1.0 s where they have a crossing); in one seed-0 frame at 1200 m the avatar the
  legend lists sixth (#2) carries label 5; ST 04-19 R4 #3 (tailed off, out of view) lands on a
  lookalike tracklet. Draw order agrees with early lateral order (median ρ 0.75), but a Gaussian draw
  term was overconfident (it broke HV 05-13 R1 and ST 06-07 R6); a heavy-tailed form with per-race
  parameters is untested. The blind near-line cost also removes correctly placed unread back markers
  (HV 05-27 R9 #7 at d 49 m).
- **Identity evidence (parked).** Of the final product's 153 hand-wrong labels (Z11, seeds 1–5; 45 distinct hand detections),
  139 are already wrong in the video-only product. By stage, in distinct detections: 23 wrong in `identify` and never changed;
  7 blank in `identify` and given a wrong number by fusion's E-step; 5 turned wrong by the EM identity pass; the rest
  (EM iterations, `fuse_final`) at most a few each. Identity work is parked at the hand set's resolution: 45 distinct
  wrong detections split into mechanisms of at most 8 cannot steer an arm, and every aggregate of the set has been read. `identify` gave the tracklet a wrong label on 91 (the tracklet silk ranks a hand candidate first on 55, 28 of them
  in ST 04-19 R3) and a right label or none on 62, where fusion's per-detection label is the wrong one (55 outside ST 04-19 R3; all 19
  in the read races). Silk's first rank disagrees with `identify` on about half of all labelled tracklets of ≥ 100 detections, even
  where the legend margin exceeds 150 votes (353 of 741, 26 × 6). On Z10 the hand-wrong error rate was 8.8 % where every hand candidate is in the legend's top 8, 12.3 % where none is,
  and 21.1 % where the hand set straddles it. A blind silk read by a second reader (avatar crop and official silks only) agreed with the hand sets on 35 of 36
  controls and on 28 of 36 detections that are wrong on ≥ 3 of 5 seeds (20 of 21 outside ST 04-19 R3 and ST 04-26 R1); both
  are silk reads, so this shows the reading repeats, not independent truth. The pipeline's own tracklet silk ranks a hand
  candidate first on 30 of 36 and 21 of 36 (seed 1), but in `identify.assign_global` legend votes add per frame and silk once per
  tracklet (±3): on the 84 wrong instances already wrong in identify the label's vote margin has median 46 votes. As a flag, silk does not separate wrong from right
  tracklets well enough: at the margin flagging 5 % of 571 right seed-tracklets it flags 60 of 128 wrong ones (AUC 0.87). A better
  silk likelihood is the open lever.
- **Drone configuration contradictions.** A drone horse with no path horse within 5 m for ≥ 4
  consecutive drone frames (drone points mapped through the frame's render detections; the final
  product compared after removing its per-frame metric shift): 25 drone tracks in 14 races video-only
  and 27 in 15 races final, counting tracks present in ≥ 4 of 6 seeds; 19 tracks in every seed of both
  products. By eye on 30 single frames, two labellers agreed on a real configuration error in 16
  (rear or wide horses without a path, two paths on one horse, order reversed); the 4-frame filter
  is untested (too few independent tracks). On up to 11 of 26 tracks (seed 0) two observed paths
  sit on one drone horse while the render tracklet's labels alternate between the two horses and the
  drone shows one horse throughout (76 of 78 label changes), often the legend's 7th/8th horse and an
  unlisted one. The soft legend-rank term switches at about half of those label changes; widening its
  margin through the full EM cuts the alternation by two thirds but leaves the contradictions; cause not established. HV 04-15 R1: a wide drone horse has no path horse in the 9 seed-product
  runs that place #10 in the pack and in none of the 3 that place it wide.
- **Double boxes on one avatar.** The detector often keeps two boxes on one horse (one on the jockey and forequarters, one on the jockey
  and hindquarters; IoU 0.32–0.50, under NMS's 0.5; intersection over the smaller box ≥ 0.5 on 486 of 554 such pairs, 14 of 4308 other
  close pairs, seed 0). The second box births a second track; about half of the long mid-race tracklet starts are on a double. A gate in
  `observe` (keep the more confident box when intersection over the smaller ≥ 0.5) was run 26 × 6 under a declared rule and FAILED its
  primary bar: hand truth on the 10 races not used to build it 884 / 142 / 42 → 887 / 143 / 38 (wrong needed ≤ 137); gates +0.33 per seed
  (SE 0.33), long tracklets minus runners 604 → 370, glitch-seconds 25.6 → 15.1 per seed, but the glitch drop is three races (HV 05-20 R2,
  and ST 04-19 R4 and ST 05-31 R1, which it was developed on). It also lets `stitch` join two horses (ST 04-19 R4 #6 → #14 across 5.4 s);
  the control's `stitch` joins two horses on about a third of a small sample of long links, but such links do not mark wrong labels
  (tracklets with a link > 1 s: 20 % wrong vs 16 % right). Seed unanimity 89.2 → 96.5 %; persistent drone contradictions (seed 0) 26 → 21
  video-only, 27 → 21 final. Not adopted.
- **Yardstick and HUD resolution.** Official per-horse times at intermediate boundaries sit on a 0.04 s grid relative to the leader
  (871 of 871 mid-race gaps over 26 races; 36 % at the finish): about 0.2 m of quantisation in every mid-race official crossing. The render's
  HUD odometer changes only every 0.4 s (intervals 0.4 s on 1833 changes, 0.8 s on 1443), so a raw tick is 0–0.4 s late.
- **Post-finish ticks.** After the winner crosses, the render camera parks on the line, and the duplicate test (mean |diff| < 3.0 to the
  last kept panel) drops the real ticks on which the trailing field crosses: over 26 races, 780 on-grid frames in the 3 s after the winner's
  crossing, 96 kept. In the races checked by eye, the finish-short horses cross on dropped frames. Keeping those frames, with or without
  copying the parking pose onto them, gained no finish crossing. With the pose copied (ST 05-03 R7, ST 05-24 R1), fusion's per-detection
  labels on the kept frames never name one of the seven finish-short horses. On ST 05-24 R1 the three stragglers are tracked across the line
  on those frames, and identify labels one of them correctly (#12, crossing 0.08 s from its official time), but fusion leaves their detections
  unlabelled or gives them other numbers. Why existing crossings got worse when the pose was copied (pose and map unchanged; rail and metric
  changed) is untraced.
- **Static-overlay filter deletes a real horse.** In `world_dets` a box recurring in one 8 px panel cell on ≥ 4 % of frames in the top band is
  treated as overlay text. On ST 06-07 R8 the camera holds one horse at panel (460, 115) from 450 m to the line, and the filter removes it on
  242 frames (confidence 0.85–0.90); on HV 05-20 R2 it misses 31 boxes on the disclaimer text (at most 30 per cell against a bar of 35). Restoring
  the 242 rows after the seeded draw (6 seeds) changes no gate and no other horse's hand label, but moves #4's path onto the horse (median
  lateral offset 3.6 → 0.45 m over the last 450 m); removing HV 05-20 R2's text boxes failed its own rule (worklist). No replacement filter yet.
- **Non-finishers in the results.** `historical._clean_race_data` drops every incident row (DISQ, DNF,
  FE, ML, PU, TNP, TO, UR, VOID, WR, WV, WV-A, WX, WX-A, WXNR), which is right for withdrawals; if a code
  marks a horse that started (the meanings are not documented), that horse would vanish from the
  runners while still rendered, and the race would pass the whole-field legend test with one more avatar
  than runners. No such race has been observed: 33 raw results pages show only WV and WV-A rows, and none
  of the 26 races has an incident row. Separately, 80 of 10551 legend-vote frames (seed 0, 26 races) hold more detections than
  runners and still vote; requiring exactly N would drop those frames' votes.
- **Worklist.** HV 04-15 R1 gained 7 hand-wrong final labels in Z11 (10 wrong, 9 of them right video-only; untraced). Off-grid render frames (the first frame after a render dropout) keep their video index as tick time, 1–2 frames late. 31 disclaimer-text boxes survive the static-overlay gate in HV 05-20 R2 (pixel-verified; max 30 in one 8 px cell vs the 4 % bar of 35); they are never labelled, sit in 1–2 whole-field legend-vote frames per seed, and are > 4.5 m ahead of real detections in 31 E-step frames, where the hard rank cost has no false-positive allowance (labels there differ from the tracklet's neighbouring frames on 14–27 % of detections vs 1.2–1.6 % elsewhere). Removing the 31 pixel-verified text boxes (oracle, seeds 0 and 1) changes 1.9 % and 0.9 % of video-only labels; #5's and #12's boundary errors are unchanged, seed 0's sectional MAE crosses its gate (0.261 → 0.247 s; seed 1 passes in both, 0.222), and glitch-seconds move both ways. HV 05-27 R3 #5 crosses 1200 m ~13 m off its official time on every seed (untraced); HV 05-20 R2 sectional (+0.05 s every seed; render crossing times
  inconsistent between horse groups by ~0.15 s, residual sd 0.078 s vs 0.03 s typical), ST 05-31 R1
  tracklets splitting at t 34–36 (untraced), ST 04-19 R4 hand count, straights' leader ~0.2 s early
  from 400 m (odometer/marker scale ratio 1.034–1.036 on all three straights).
- **Tracklet exclusivity in identify.** `assign_global` forbids one number on tracklets whose
  first-to-last spans overlap. Over 26 × 6, 71 unlabelled tracklets (≥ 30 detections, ≥ 10 slot votes
  for one number) have that number held by a tracklet whose span overlaps by 1–12 frames, and 147 by one
  overlapping more (72 sharing ≤ 12 frames, i.e. inside a gap; 69 sharing most frames, i.e. side by side;
  6 between); whether the overlap is the sole blocker was not computed. The ST 05-09 R1 #3 hole this
  entry described is gone (#3 labelled in 96–97 % of frames over t 10–31 on every seed). The end-overlap
  case carries hand truth on only 3 seed-tracklets (none wrong), too few to judge.
- **Orphan elimination** in fusion hands a leftover chain to the last missing horse; with ST 04-19
  R4's #10 read it gives tailed-off #3 a mid-field chain. Turning it off is gate-neutral on single runs
  but costs 8 races a path; not ensemble-tested.
- **ST 04-26 R1 camera fit** is broken (C 6727 px/s, residual sd 0.125 s).
- **Leader timing shape.** Identity-free leader error is late at early boundaries on some races;
  clock rate, rail configuration, render frame holes, camera change near the line and false leaders
  are ruled out. Beyond 400 m no reference exists that is neither official nor the odometer. In the
  0–400 m zone the tick px/m is 1.3–4.8 % above the paint px/m (the HUD counts fewer metres); cause
  unknown.
- **Independent real-camera positions.** Drone and patrol are the only independent source for the
  start segment (the render starts at race clock 10.8–12.6 s), lateral validation and races where the
  render disagrees with the record; planned via segmentation seeded from fused positions. Also open:
  a non-overlap constraint between horses in fusion; retraining the avatar detector under overlays. Sizing (seed 0): an overlay dropout
  of a true horse precedes 8 of 36 persistent wrong labels (16 % of correct ones) and 3 of 25 finish-short horses, so retraining is parked.
