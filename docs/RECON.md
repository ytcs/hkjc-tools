# Race reconstruction

`hkjc.recon` turns one race's facts (`hkjc.data.race_info`) and its four HKJC replay videos into
per-horse paths on the race clock (`t = 0` when the gates open), each with a posterior sd: `d(t)` metres to the
finish along the racing rail, `s = distance − d` metres run, and `lat(t)` outward of the racing (movable) rail in
the render's lateral units (§8; metres in `lat_m` where a validated map exists, §4).

```python
from hkjc.data import race_info
from hkjc.recon import RaceRun
run = RaceRun(race_info("20260603", "HV", 8)); run.run(); run.paths()
```

CLI: `uv run --group vision python -m hkjc.recon DATE VENUE RACE_NO [...] [--force stage,...] [--stages stage,...]
[--mask SPEC] [--root DIR]`. Stages are cached under `<root>/<date>_<venue>_<no>/` (default root
`data_cache/recon`); a stage is skipped when its outputs exist. Environment variables: §7.

## 1. What the library produces

The reconstruction is one posterior of what happened in the race, built from all the information
available: the four replay videos, the pre-race facts, the official records and the track geometry.
Each video is an incomplete, noisy proposal of each horse's path; the path is the posterior of a
Bayesian fusion under physics (bounded acceleration) and geometry (rails, course width) priors, with
no feed acting as a master that the others correct.

| Artifact | Content |
|---|---|
| `paths.parquet` (`run.paths()`) | **The product**: per horse `horse_no, t, s, d, lat, sd_s, sd_lat, n_obs, lat_m, sd_lat_m` on a 0.1 s grid, from every source, including the official records the run's mask leaves visible, then the contract columns (§4) |
| `paths_video.parquet` | The reconstruction from the videos and pre-race facts only (no official outcome record); the fixed gates score it, and it is the reference for what the videos alone determine |
| `validation.json` | fixed gates on `paths_video.parquet` against the full official record, plus the product's residual against the official times (`final_official_resid_mae`) |
| `audit.json` | label-free invariants (§4, validate) |
| `status.json` | `ok`, `unhealthy` or `failed` (§4, status) |
| `timings.jsonl` | per stage: wall s, CPU s (the stage's process and its own children; background stages count only in their own record), peak RSS |
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
| draw | stall per runner | `fuse_final` (position at t = 0) |
| official records | cumulative sectional times, finish times, placings | `fuse_final` only (`pipeline.READS_OUTCOME`): official observations (interior times as noisy measurements, the finish near-hard), shared metric bias, finish-camera crossings replaced by the official finish, final identity pass, finish re-timing, placing ties |
| track geometry | `hkjc.data.track_for`: handedness, course widths, home-straight length | `rails` (inner side, movable-rail check, expected fixed-rail gap), `film` (markers inside the home straight), fusion's lateral box, `lat_m` scope |

## 3. The mask and held-out evaluation

Every stage reads the race through `RaceRun.info`, a `hkjc.data.MaskedRace`: the pre-race facts plus the
official outcome records that the run's `hkjc.data.Mask` leaves visible, read only through
`official_boundaries()` (cumulative sectional times), `official_subsplits()` (cumulative times at the 200 m
sub-split points), `official_finish()`, `official_order()`, `official_placing()` (placings; a dead heat shares
one) and `official_nonfinish()` (the placing code of each runner without a finish time; withheld with the finish).
`validate` scores against the full record (`RaceRun.truth`).

HKJC prints each horse's two 200 m halves of the race's last two 400 m sections; `official_subsplits()` returns
their points (`[600.0, 200.0]` m to go) and the cumulative time at each (section-start time plus the first half),
separate from `official_boundaries()`, whose indices the mask keys on. A sectional cache without sub-split columns
gives no points. No stage reads them.

The runners are every horse that started; withdrawn horses (WV, WV-A, WX, WX-A, WXNR, WR) are not runners. A horse
that started and did not finish (FE, UR, PU, DNF, TNP) is a runner with no finish time and no placing; its
cumulative sectional times run to its last complete section and are NaN after. A disqualified horse (DISQ) keeps
its finish time and sectionals and has no placing.

Outcome part (read by `fuse_final`):
- `Mask()` withholds nothing (the default).
- `Mask(boundary=j)` withholds the cumulative official time at sectional boundary j for every runner (0 is the
  first section end; the last index is the finish, which also withholds finish times and placings), and the
  sub-split points within 200 m of it (a sub-split time is the previous boundary's time plus a 200 m half).
- `Mask(boundary=(j, k, ...))` (spec `boundary=J+K`) withholds several boundaries at once, each as above: an
  evaluation tool that puts withheld points farther from every visible boundary. A single index stays an int, so
  its spec (and every existing run dir's `mask.txt`) is `boundary=J`.
- `Mask(subsplit=k)` (spec `subsplit=K`) withholds the cumulative official time at sub-split point k (an index into
  `official_subsplits()`' points, race order) for every runner; exclusive of `boundary` and `outcome`. No stage
  reads sub-splits, so it changes no output: an evaluation tool for when one does.
- `Mask(outcome=True)` withholds all official outcome data (results, finish times, placings, sectionals,
  running positions); pre-race facts, the draw included, stay visible.

Evidence part (from `observe` on, `pipeline.READS_EVIDENCE`), for held-out scoring where the videos are silent:
- `Mask(dets={i, ...})` removes those render detections (`dets.npz` row indices) in `observe`, as if the
  detector had missed them: tracking, identity and every fusion pass run without them. `rails` (upstream)
  still sees them.
- `Mask(discs=P)` counts every disc-number read (circle render) on the race clock's odd `P`-second blocks
  `[kP, (k+1)P)` as unread (`discs_phase=0`: the even blocks), in all three consumers: `observe`'s number
  split, `identify`'s votes and the fusion's blind set (`RaceRun._number_reads`).
- `Mask(legend=P)` counts every legend read (silk render: the 8 running-order slots of a render tick) on the
  race clock's odd `P`-second blocks as unread (`legend_phase=0`: the even blocks), its slots read as on the circle
  render, in all three consumers: `identify`'s legend votes, the EM identity pass and the detection-level E-step
  (rank costs, negative legend evidence, blind set), all through `RaceRun.legend_rows`.

Select it with `RaceRun(info, mask=...)`, CLI `--mask SPEC`, or env `HKJC_MASK`; a spec is `none` or
`;`-joined parts `boundary=J[+K...]` | `subsplit=K` | `outcome`, `dets=I,J,...`, `discs=P` | `discs=P@PHASE`, `legend=P` | `legend=P@PHASE`
(`HKJC_FINAL_HOLDOUT=j` is an alias for `boundary=j`; setting both is an error). A masked run writes
`paths.parquet` and `validation.json` like any other, so give it its own `root`: the run dir records a
non-empty mask in `mask.txt`, and a run refuses a dir whose mask-dependent artifacts were made under another
mask (the outcome part governs `fuse_final`/`validate`, the evidence part `observe` and everything downstream;
an evidence-masked run therefore starts from a dir without `observe`'s outputs). A withheld boundary is absent
from the official observations, sets no metric bias and is not re-timed, so the product's error there is
independent of that record. The yardsticks built on the mask are in §6.

## 4. Pipeline

Stages run in this order: `frames → features → pose → worldmap → detect → silk → hud → rails → start → film →
observe → finish → identify → fuse_video → fuse_final → validate`. `hud` and `finish` read only the videos and
run in background processes until the first stage that depends on them. `pipeline.DEPS` declares each stage's
inputs: forcing a stage re-runs every requested stage downstream of it. A requested stage that has never run
(none of its outputs exist) counts as forced, since its consumers' artifacts predate it; a stage missing only
some outputs (large files deleted for space) is regenerated without invalidating its consumers (the pipeline
is deterministic). Artifacts are written to a temporary file and renamed, so a killed stage never reads as
done; a failed stage terminates the background stages.
Stages that read a later part of a replay (film, finish, silk, the shifted clock search) seek to the IDR keyframe
before the first frame they need (`video.decode_from`; same frame indices and pixels as decoding from frame 0, else
it decodes from frame 0). ONNX Runtime sessions (avatar detector, OCR) run with intra-op spin-waiting off
(`ortsession.no_spin`): same threads and work split, identical outputs. Devices and scheduling: §7.

### frames

Two render styles: the avatar render with the silk legend (to 2025/26) and, from 2026/27, numbered blue discs
(radius ~25 px, the runner's number printed on each) with no legend. `video.layout` samples 12 frames across
the replay: a frame with ≥ max(3, 0.4 × runners) discs votes circle (tested first: discs crossing the legend
strip pass the silk gate), a frame passing the relaxed legend gate votes silk; ≥ 2 votes and a strict
plurality decide, else the race stops with "unknown render style". The style is stored in `frames_idx.npz`.

The panel box is a constant of the broadcast layout: the voting frames give a median box, and the first render
frame keeps its own box unless it differs from that by more than 3 px (antialiased edges jitter 1–2 px; the
first accepted frame can be the crossfade from the full-screen live shot, whose box spans the full height).

Silk render: a frame is render-active when the legend band (0.70–0.86 of the panel) shows a saturated box,
the top strip is the dark HUD box (turf-green fraction ≤ 0.30) and the track band is sharp (Laplacian variance
≥ 240). Once the strict test has passed, ≥ 5 boxes in a band widened to 0.60 also pass (the legend sometimes sits
higher). Circle render: the render's yellow odometer text (fraction ≥ 0.05 of its box; without it the start gate's
blue banner passes as discs) plus ≥ 2 discs, the top-strip test, and no texture floor (the disc render is
smoother); once the render has started the odometer text is not required (it disappears near the finish).

The render updates at 10 Hz in 25 fps video, so new ticks start 2 and 3 frames apart; a frame within mean |diff|
3.0 of the last kept frame is a duplicate. Panels are cached as JPEG with their video index as timestamp.
- `video.film_keep` drops live start film that passes the render gate before the render. Film changes on every
  frame (static gate shots thin to 2–3 frame gaps); the render never changes on two consecutive frame pairs. The
  render starts at its first 10 inter-frame gaps alternating 2 and 3 frames, if a run of ≥ 10 consecutive 1-frame
  gaps precedes it; every frame up to the end of the last run of ≥ 2 consecutive 1-frame gaps before it is dropped
  (slow film and the dissolve included).
- `video.grid_keep` drops off-grid re-encodes of a kept tick (a periodic keyframe would enter as a zero-motion
  tick). The tick grid (`video.tick_grid`) is the two residues of `frame % 5` holding ≥ 90 % of the frames; when that
  vote fails, only frames 2–3 frames after the previous one vote.

### features

Circle render: disc centres replace the avatar detector (a disc-template filter on the closed blue mask, peaks
with fill ≥ 0.8 at least 18 px apart, so discs about 1.3 m apart still split); the number inside each disc is read
with recognition-only OCR (`numbers.npz`, row-aligned with `yolo.npz`). `detect` uses the disc detections only.

Silk render: the avatar detector runs once per panel (`yolo.npz`: boxes + silk histograms, read by `detect`;
a run dir with only a YOLO-only `dets.npz` recovers `yolo.npz` from it). ORB (3000 features) runs on a world
mask: HUD removed, dark (V < 45) and vivid blobs dilated, and the avatar detector's boxes, grown 1.6×, masked
(brown and pastel avatars escape the colour mask, sit still in the image because the camera follows the field,
and would lock the step matches). Steps between frames ≤ 6 video frames apart are RANSAC similarities (1.5 px)
refitted by least squares (Umeyama): the camera is near-nadir with no tilt and a slow zoom.

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
render panel; the legend icon band is dropped. On the silk render, below a median of 0.6 N detections per frame
(N runners; all-weather dirt) a map-difference blob detector is added. When detections are read
(`RaceRun.world_dets`), inside the disclaimer band only conf ≥ 0.45 survives, and a box recurring at one panel cell
on ≥ 4 % of frames there is overlay text.

### silk

`detect.silk_feature` per detection drops turf and brown pixels and keeps 12 hue bins (pastel and vivid),
white and dark. Matching collapses pastel and vivid, because the render desaturates. `silkmap.SILK_MAP` maps each
runner's official silk prototype (`detect.silk_prototype`) into render appearance: a ridge fit on legend-confident
tracklets, legend labels only. Consumers: the appearance split in `observe` and the silk likelihood ratio in
`identify`.

### hud

Recognition-only OCR on fixed crops: the race clock on the film panel (every 10th frame), the render's
odometer and its 8 legend slots (every frame). No read is skipped by a pixel-change test: the legend's number boxes
are not opaque, so a changed slot can differ less from its last read than an unchanged one differs between frames.
Race-clock origin: `clock_f0 = median(frame − 25·clock)`. When the fixed crop gives fewer than 3 clock reads (a
sponsor box above the clock pushes it down on some Happy Valley broadcasts), `ocr.read_clock_located` re-reads the
same frames with the crop shifted vertically (−16…+96 px, step 4) and keeps the centre of the widest run of shifts
whose reads agree on the clock zero (within 2 frames). With no clock read at all, the stages that need the clock
zero stop with "no race-clock reads". A field of N < 8 spreads its N legend slots about the same centre at a wider
pitch; the slot crops follow the field size (`ocr._legend_slot_x`).

### rails

Stations every 2 world px follow the camera path, then the field's inner edge (90th percentile of inward horse
offsets, smoothed by `rails.robust_offset`, a Huber Whittaker smoother, so an outlier run cannot fold every
ridge into a V). Thin bright ridges on the inward normals are linked by a Viterbi DP (`rails.dp_track`) that
scores +1 for any ridge above threshold and −1 for none, and charges 0.002 per px inside the field edge: it
rewards continuity, not brightness, so it stays on the thin movable rail instead of the brighter fixed rail inside
it. Then a world-space despike (> 0.55 m off a 26 m quadratic trend), sub-pixel refinement and `rails.smooth_rail`,
a support-weighted Huber Whittaker smoother with a third-difference penalty (arcs are free; ~20 m cutoff;
unsupported stations weight 0.1; end runs continue straight). The refinement snaps to the ridge nearest the DP
rail, not the strongest in its ±1.4 m window (`refine_rail(nearest=True)`): on A courses the render draws a second
rail line just inside the racing rail, and the strongest ridge alternates between the two.
`rails.RailFrame` takes positions and normals from a 121-station (~12 m) cubic fit, because snap wobble makes
normals noisy at wide lanes; projection uses a time-ordered station hint for revisited ground.

The inner side is the measured turning sign when the camera path turns at least a quarter turn (range of the
cumulative turning, so a fold-back at one end cannot cancel a bend), else the race's handedness
(`rails.inner_side`). On straights (the Sha Tin 1000 m) the mosaic shows the fence only over a short stretch, so
`rails.straight_datum` makes the field's inner-edge scaffold the rail and adds a measured per-race fence offset
`lat0_px` (median over the longest supported DP run); straight-course lateral is pack-edge-relative plus that
constant. A fence measured more than 1 m (observe's inside-the-rail tolerance) on the field side of the inner edge
is another ridge and counts as unmeasured (`lat0_px` 0).

`rails.movable_rail_check` moves the rail out when a supported ridge sits at the course-table gap toward the
field and the innermost horses are still outside it (the extracted ridge was the fixed rail). Painted markers
(FINISH/50…300 m) appear only near the end, so they are profiled on a mosaic of the final 9000 world px.
`rails.fixed_rail_gap` measures the lateral-gate gap on a padded window (the ridge filter wraps at window edges).

### start

The race clock is the aerial's overlay clock, except where that clock starts late (before the 2025/26 season it
typically started a fraction of a second after the gates opened; from 2025/26 it starts with the gates).
`start.lag` finds the gate opening in the live start film that opens the aerial video: the closed stalls' doors
are a band of dark pixels that drops when they swing open. Doors are dark green in sun (OpenCV hue 35–95,
saturation > 60, value < 110) and dark teal in shade; the teal band (hue 60–105) is tried only when the green band
finds no opening, and excludes the turf (hue ≤ 55, admitted by the green band where shaded) and the blue banner
and night sky (hue ≥ 105). On an 8 × 6 grid of the quarter-resolution frames it takes the row strip with the
largest relative drop (≥ 25 %, ≥ 1 point) within 100 frames before the clock zero, provided the drop persists
over 40 frames and is shared by ≥ 60 % of the strip's door-bearing columns (not a person crossing). The onset is
the 25 % crossing of that step. The detector's convention lag (clock zero − onset where the clock is synced,
`start.CONVENTION`) is 1.4 frames at Sha Tin and 2.55 at Happy Valley; a net lag of more than 4 frames re-anchors
the clock (`RaceRun.clock_f0` = overlay zero − lag, used by `observe` and every stage downstream); no detection,
or a lag within 4 frames, keeps the overlay clock. `film.npz` keeps overlay-clock times and `observe` shifts them
by the lag, so `film` does not depend on `start`. `start.npz` records the clock zero, onset, lag and status
(`late`, `synced`, `none`).

### film

The aerial's left half is a live race film on the broadcast clock; in the straight a front-on camera sees the
painted 400–50 m stripes. `film.crossings` tracks paint lines that sweep ≥ 80 px up the image, takes the
lowest COCO horse box as leader (the detector runs on a track's frames only up to the crossing the track is
read at: the first with a readable odometer), and names the marker from the HUD odometer (0 to 40 m behind).
Crossings must imply 12–22 m/s to a neighbour and lie ≥ 20 m inside the home straight (Happy Valley's 400 m
marker is on the bend). The film crossings' small early bias is not corrected.

### observe

Detections go to world px and to (s, lat) on the rail. The per-frame leader ignores a front detection > 8 m
clear only if the gap is not persistent over 11 frames.

**Distance metric** (`rails.calibrate_distance_ticks`). The odometer ticks V+10 → V as the leader reaches V.
Tick events give scale and finish through a robust line plus a cubic B-spline (150 m knots) for map-scale
drift; where ≥ 4 consecutive smoothed residuals exceed 10 m, knots densify to 25 m (the 1000 m straights, where
the camera swings 180° and the pose steps in scale). Lateral keeps the base fit's px/m.
The constant comes from the film (`film.anchor_offset`): with ≥ 3 crossings the ticks are unpinned and shifted
by the median offset that puts the render leader on the film's crossings (against the film the render's painted
FINISH pin is a step in the last 50 m). With fewer crossings, a tick-seeded marker grid
(`rails.fit_marker_grid_seeded`: paint scale ≈ tick scale / 1.035 within ±8 %, FINISH within ±60 m, ≥ 4
markers including FINISH) pins d = 0 when its offset after resolving whole-marker shifts is < 6 m and the
scales agree within 8 %; a pinned grid also sets the per-race tick convention (default 0).
`rails.time_local_shift` subtracts a per-frame along-track shift where ≥ 3 consecutive ticks keep
|residual| > 10 m (a pose slip in the straights' camera swing). Lateral = (lat_px + lat0_px) / local px/m;
detections > 1 m inside the rail are dropped (`RaceRun.LAT_GATE`).

**Metric line.** The HUD's metres per rail px step where the rail turns (about 4 % at the Happy Valley home-bend
entry, 600 m to go), as if the odometer measured arc length along a line w metres outside the extracted rail; the
150 m spline in rail s smooths over the step and cuts the corner. `RaceRun.tick_metric` can fit the ticks against
s' = s + w·ppm·Θ(s) (`rails.offset_arc`: Θ the rail's cumulative inward turning, ppm the rail-s fit's tick-line scale;
the marker grid stays in rail s, its finish mapped to s'). `observe` always fits in rail s (w = 0); the s' metric is
used only by the product (§4 fuse_final).

**Timestamps.** Each render frame is stamped with its tick time (`video.tick_frame`), since tick content advances a
uniform 0.1 s while ticks start 2 and 3 frames apart: the residue followed by a 2-frame step moves −0.25 frame, the
other +0.25 (mean zero; off-grid frames unchanged). All observation times (tracks, tracker, film anchor, time-local
shift) use it.

**Tracking.** Constant-velocity prediction and per-tick Hungarian assignment in metres (gate 3σ,
σ = 0.9/0.45 m, coast 20 ticks), then kinematic stitching across gaps ≤ 6 s. `tracking.split_by_appearance`
cuts a tracklet where its mean silk signature changes (total-variation distance > 0.30, ≥ 40 silk detections
per side; legend-pure tracklets stay below). Circle render: `tracking.split_by_number` splits tracklets where the
disc number changes and stays changed (≥ 8 of the last 10 confident reads name one other runner number that is
not a dropped digit of the current one, and it holds for the next 20 reads, ~2 s), because the motion tracker
swaps identities where discs touch. Swaps between 1 and 11 (or 2 and 12, ...) look like dropped digits and are
not caught.

`HKJC_PERTURB_SEED=k` drops 1 % of detections here (seeded) for robustness ensembles.

### finish

The full feed is scanned from `distance/18` s for 70 s for the finish camera: a white vertical line at a stable
x, slow motion at a variable rate, so unique-frame index is not time. Saddlecloths are OCR'd per unique frame
of a run that reaches 40 unique frames (shorter runs are dropped unread): yellow on dark (Happy Valley,
overhead) and, at Sha Tin only, dark digits on white cloths inside a white ring. A COCO horse-box ruler (2.6 m
nose to tail) gives camera px/m; the box includes extended legs, so the ruler-based C reads high. C is not
calibrated on official finish gaps.

`finishcam.tracks` keeps, per number, the lateral cluster read at the most distinct places (a static object read
many times at one spot must not outvote the real cloth) and needs ≥ 4 reads and ≥ 100 px of motion. Digit
drops ("1" on the "11" cloth) are removed when most of the shorter number's reads lie on the longer number's
track; coinciding numbers keep the longer one. On the overhead camera (Happy Valley; `RaceRun.finish_sc`
carries the geometry) reads of any number lying on a much better-read track (≥ 10 reads, ≥ 3× the reads it
claims, on its line within 60 px, its lane within 30 px) are that cloth's misreads and are removed before
tracks are built; not on the Sha Tin side-on camera, where horses abreast share one image band.

`finishcam.finish_gaps` gives spatial gaps at an instant: crossers are chained from the crosser that links the
most, and other horses get `gap_h = gap_r + x_h − x_r` from ≥ 2 shared frames, which recovers horses occluded
at the line (sd = robust spread / √n, floor 15 px).

### The core: identify, fuse_video, fuse_final (`recon.core`)

Design memo (Stage A of the core redesign: render-only evidence). The three stages share one model and one objective.
The video-only product and the final product run the same code; they differ only in which observations the mask
leaves visible (the final product adds the draw, the official times and the metric bias). Nothing is decided
before the output: labels stay soft inside the solve, and the argmax is taken only when the label files are written.

**Unit of identity: segments.** `core.segments` cuts each tracker tracklet (observe's `members`) into segments at:
- stitched gaps > `CUT_GAP` (1 s);
- lateral jumps > `CUT_JUMP_LAT` (0.6 units) within `CUT_JUMP_T` (0.2 s);
- encounters: another tracklet starts or ends within `ENC_T`/`ENC_S`/`ENC_LAT` (0.2 s, 3 m, 1.5 units);
- render camera transitions (`core.camera_transitions`: the sequential pose step into the frame is missing or has a
  residual > `CAM_RESID` px);
- an occluder's edge: the box enters or leaves `EDGE_PX` of the list band (silk render) or the disclaimer band.

Tracker continuity across a cut is a link with a reward in the objective: `LINK_TRACK` (4) at gaps and encounters,
`LINK_WEAK` (1) at jumps, camera transitions and occluder edges, `LINK_CROSS` (2) for an end-to-start continuation
between segments (≤ 1 s, ≤ 3 m at the segment's speed, ≤ 1 unit across). Exclusivity is per frame: one segment per
horse per frame. A segment's detections are summarised by their means per `CHUNK` (0.5 s), sd
√(0.8²/n + `SD_S_COMMON`²) along and √(0.35²/n + `SD_L_COMMON`²) across, plus `CAM_SD_S`/`CAM_SD_L` within `CAM_WIN`
of a camera transition. The within-chunk scatter is a label-independent cost of a labelled chunk. The hard
appearance and number splits of `observe` still cut tracklets; in the core they are only extra cuts with no link.

**Per-box P(horse).** `core.p_horse` gives `P_GHOST` (0.01) to a short-lived box (tracklet < 50 detections) in an
overlay zone: the list's icon band or the dark corner beside it (confidence < 0.75, silk render), the disclaimer band
at glyph size, or the left edge far outside its frame's field. Any other box overlapping an occluder band gets
`P_EDGE` (0.5); every other box gets `P_HORSE0` (0.97). A labelled detection costs −2 log(P/`P_HORSE0`). An
unlabelled one costs `CLUTTER` − 2 log((1 − P)/(1 − `P_HORSE0`)). Rank counts weight each detection by P.

**Detection term.** Per render frame, a local affine map from (s, lat) to panel px is fitted to the frame's horse
boxes (`core.view_maps`: ≥ 4 boxes with P ≥ 0.5, else the nearest frame's map). A runner's P(in view) is the
probability that its posterior projects inside the visible panel, between the disclaimer text and the list band
(silk render), with `VIEW_PX` px added to the projected sd (`core.in_view`). Occluded or off-panel runners are
unobserved, so their missing box is no evidence.

The frame's detection probability is estimated from its own count: p_f = (boxes + 1)/(runners in view + 2), clipped
to [`P_DET_MIN`, `P_DET_MAX`]. The ceiling is the detector's recall of visible horses: silk 0.95, 0.86 on bends, disc
0.985. The detector's recall varies by frame (camera, zoom), so a frame that boxes half the
field makes a miss cheap. A runner labelled in the frame costs −2 log p_f. An unlabelled one costs
−2 log(1 − p_f) × P(in view) (`core.miss_terms`). This keeps a path from floating unobserved through a part of the
panel where the detector boxes the field.

**State.** Every runner has s and lat at every node of the 0.1 s grid, seen or not. Along-track positions form one
joint banded Gaussian over all runners, ordered node-major with lower bandwidth 2H. Its terms are:
- the physics prior of `fusion.Prior` (6 / 1.2 m/s²);
- a proper start prior: s at the first node, sd 1 m at the gate, else 100 m; speed sd 8 m/s;
- the chunk observations;
- the extra observations: finish camera, draw, official times;
- the list's pairwise order factors.

The mode comes from Newton with line search. The fault-state posteriors are the factors' weights (EM). Lateral
uses one box-bounded banded smoother per runner. An unseen runner carries the posterior that the order factors
shape, behind the horses the list puts ahead of it, with its own sd. The assignment step sees that sd.

**Order evidence.** `core.statements` turns each list row into pairwise facts "a ahead of b" at the row's render
time − `LAG` (0.3 s): adjacent read slots, and the last listed horse ahead of every unlisted runner on fully read
rows. Each fact has probability (1 − ε)Φ(Δ/`SIG_AB`) + ε/2, with ε = `EPS_LIST` (0.02) for misreads and `SIG_AB`
(1 m) the order noise of a pair abreast. In the fault chains a fact's likelihood is taken relative to a clearly kept
order. Two persistent fault chains act on top (`core.list_chains`, forward-backward):
- a list-level frozen state, in which the row says nothing (each adjacent fact a coin flip). Only a row identical to
  the previous full row can be frozen (`P_FROZEN_IN`, `P_FROZEN_STAY` per row);
- per runner, an omission state, in which its unlisted facts say nothing (`P_OMIT_IN`, `P_OMIT_STAY`).

The fact API is source-agnostic: (time, ordered pair, source fault model).

**Objective.** J(z) = −2 log p(z, observations, list), with the paths integrated out (Laplace at the joint mode).
It is the sum of:
- the along-track term: the mode's misfit, physics and order factors, plus the log det of the Laplace Hessian minus
  the prior's;
- the lateral term;
- the list's chain log-likelihood;
- per detection, the horse-prior or clutter cost;
- the labelled chunks' scatter;
- the detection term;
- the rewards of broken links;
- disc reads: per detection, error `READ_ERR`, `READ_ERR_TOUCH` with another disc within 50 px, `READ_ERR_DROP`
  between dropped-digit partners;
- silk: −2 × the clipped `silkmap` LLR of each segment with ≥ 10 features;
- hard exclusions (stopped non-finishers after their last section).

**Inference (`core.solve`).**
1. Start-free solve: one MILP over segments × runners on the label-free evidence. This is the list's rank-count vote
   with undetected horses ahead at `P_AHEAD_UNSEEN`, plus reads and silk. It keeps only segments with a positive
   gain.
2. Repeat:
   - Assignment step: a MILP over segments × runners minimises J's increments, linearised at the current state.
     Each increment combines the block predictive of the segment's chunks against each runner's posterior (leave
     the segment out for its owner), the rank-count vote, the horse/clutter prior, reads, silk and links.
   - State step: J is evaluated at the proposal. The proposal is kept if J falls. Otherwise its changes are tried
     one segment at a time, in order of predicted gain, and the first that lowers J is kept.
3. When neither the proposal nor any of its five best single-segment changes lowers J, window proposals follow
   (`WINDOWS`, 10 s then 30 s). Each is a MILP whose kinematic term leaves out each horse's own observations within
   the window around the segment. Its evidence farther away, the extra observations and the order curvature stay.
   The window MILP leaves out the rank votes. A list omission shifts every horse behind the omitted one at once, and
   only such a move reaches that labelling.
4. Then permutation moves: at a cut where tracker links are broken between labelled segments, or where the
   segments ending and starting at a camera cut pair up by along-track rank, the label permutation is carried over a
   whole stretch, back to an earlier or on to a later camera cut (a time where ≥ 3 labelled segments start together).
   The move lowering J most is kept (`core.permutation_moves`).
5. Then omission insertions (`core.insertion_moves`): over a stretch between camera cuts where the omission chain
   gives horse u a mean P(omitted) > `INSERT_OM`, u is put at rank r of the field and every horse from r to u's old
   rank moves one avatar back. Per horse, the `INSERT_WINDOWS` stretches of ≥ `INSERT_MIN_T` s with the largest
   length × mean P(omitted) are tried, at every rank ahead of u.
6. The loop ends when no proposal lowers J. There is no iteration cap. Every accepted step lowers J; the log prints J
   after each.

Each product solves twice: without the finish camera, then with its crossings (`finish_camera` from the first
state), the first solve's labels being a candidate start. `fuse_final` also offers the video labels as a start. The
lowest J among the starts is kept.

**Outputs.**
- `segs.npz` (identify): segments, links, P(horse), camera-transition detections.
- `det_labels.npz` / `det_labels_final.npz`: per tracks row, `labels` (argmax, 0 = clutter), `prob` (its probability)
  and `marg` (P per runner, last column clutter). `core.marginals` computes the marginals per segment, from its
  conditional given the other labels, with swaps against overlapping holders.
- `core_video.npz` / `core_final.npz`: segment labels, marginals, the final unaries, J per accepted step.
- `det_margins_final.npz`: 2 log(p/(1 − p)) of the kept label.
- `paths_video.parquet`, `fuse_video.pkl` (segment labels, finish observations, paths), `paths.parquet` and the
  contract, as before.

`fuse_final` then:
1. sets the render metric bias at the visible boundaries from the video paths (`_metric_bias`), as a shift of chunk s;
2. runs the core with the draw, the official times and the stopped horses' exclusions;
3. re-evaluates the state once on the metric-line shape (HV), labels fixed;
4. ends stopped horses at their last labelled detection;
5. re-times onto the official finish (`_official_retime`).

**Decision points of 0316 §6, status in this slice.**
- **Gone:**
  - 1: atomic tracklets;
  - 2: argmax between stages;
  - 3: whole-field gating;
  - 7: anchor demotion;
  - 9: the iteration cap;
  - 10: greedy association and orphans;
  - 11: shadow and post-finish release;
  - 13: the RANK_ATTR cap;
  - 8: skeleton replacement (official times are observations in J).
- **Pending:**
  - 4: the silk term is still the per-segment clipped histogram LLR (Stage B);
  - 5: observe's appearance and number splits;
  - 6: finish-camera handling: the anchor-or-registration clock inside `finish_camera`, camera crossings computed
    once from the first state. The camera's `fixed` names, line elimination and blind veto are gone;
  - 12: the metric bias and HV shape as point estimates;
  - 14: top-1 OCR integers;
  - the unseated rule (UR) is not ported.

### Lateral in metres

Both parquets carry `lat_m = a + b(d) × lat` and `sd_lat_m = b(d) × sd_lat` after the render-unit columns
(`RaceRun.lat_metres`).
- One map for both venues and both drone-era render styles, continuous in the distance to go d (`RaceRun.lat_b`).
  With x = d − the home straight's length, b = `LAT_MAP`'s 0.645 for x ≤ 0 (the home straight), `LAT_B_FAR` for
  x ≥ 400 m, linear between (`LAT_RAMP`); a = −0.020 m throughout. In the home straight it is the Sha Tin
  home-straight map. The render spreads a compact field more than a fanned-out one; b follows the field's typical
  fan-out through the race, not the local spacing of abreast horses.
- `sd_lat_m` is the mapped posterior sd only. The map's own error is in the contract columns `sd_lat_m_horse`,
  `sd_lat_m_node`, `sd_lat_m_scale` (race-level, shared per race and segment) and `sd_lat_m_total` (provisional;
  Output contract below).
- Defined only where the map passed its drone test (`RaceRun.lat_m_scope`), NaN elsewhere: turf oval races (not
  the 1000 m straight) with dates in `LAT_M_DATES` (2026-01-07, when drone feeds begin, to the end of the 2026/27
  season; the drone test covered races to 2026-11-23); from the first render frame to the post, in the segments of `LAT_M_SEGMENTS` for the venue
  and render style (disc from 2026-09-01, silk before). Segments (`RaceRun.lat_m_segment`): start (the first 6 s
  after the first render frame), bend (`is_bend`), home straight, other straight.
  - Sha Tin disc: straight, bend, home straight.
  - Sha Tin silk: start, straight, bend, home straight.
  - Happy Valley disc: start, straight.
  - Happy Valley silk: straight, bend.
- The slope changes continuously along a path, so `lat_m` has no step anywhere it is defined. Where a path leaves
  the scope (an untested segment), `lat_m` is NaN, never render units.

### Output contract

For modelling on the paths (`recon.contract`). `fuse_final` appends these columns to `paths.parquet` after the path
and metre columns and writes two artifacts. They are computed from the finished paths and change none of them, and
none of the additions is a stage output or cache key.
- `valid`: the node lies in a validity-flagged stretch of the horse (`contract.flag_stretches`). A detection is
  flagged when its video-only and final labels agree, it is at least 1 s after the first render detection and before
  the line, the horse's final path is firm there (sd_s < 1.5 m) and overlaps no other firm path (2.0 m along, 0.7
  across), and, on the disc render, its tracklet's majority confident disc read is the label. Flagged detections bridged
  over gaps ≤ 0.5 s form stretches of ≥ 10 detections and ≥ 2 s; on the silk render a stretch also needs the legend to
  confirm the horse's rank on ≥ 3 and ≥ 80 % of its legend-read frames. Masked reads count as unread. So do the
  frames of a frozen running-order list (`contract.legend_frozen`, from the list reads alone): the same near-cloth read
  (every slot the field fills read, at least 6, and at most one inversion against ascending order, e.g. 2,1,3,…,8)
  held for ≥ 5 s of race time, from its first to its last read; reads with an unread slot neither break nor extend
  the hold. No stretch spans such a window: its detections are not flagged, so the stretches on either side must
  each meet the criteria on their own. A list frozen in any other order cannot be told from a strung-out field by its
  reads and is not caught.
  The flag is validated against disc reads only (circle render); on the silk render no per-detection truth exists.
- `pos_ok`: the position flag. A crossing of a horse at an official boundary the run sees passes when the horse is
  observed there (a render detection with its final label within 0.5 s of the final path's crossing time), the final
  path is firm (sd_s < 1.5 m), and the video-only path agrees with the official time within 1.25 m field-relative: its
  crossing error (t − t_official) × its speed, minus that boundary's median over ≥ 3 horses. A node is `pos_ok` when
  the horse is observed (as above, at the node's t) and firm at the node, and the identity check (the 1.25 m test)
  holds at both bounding visible boundaries (the last at or behind the node and the first ahead of it; the single
  neighbouring one before the first and after the last; no visible boundary: false). A withheld boundary is not a
  bounding boundary, so masked runs read the next visible ones.
- `free`: no other horse's path node at the same t within one length along (2.4 m) and one render abreast spacing
  across (2.6 render units).
- Along-track uncertainty (`contract.uncertainty_columns`; terms the posterior `sd_s` omits, added at write time, the
  paths unchanged):
  - `sd_s_axis`: sd of the shared distance-axis error, a race-level latent b(s) common to every horse at the same s.
    Its core is a stationary OU process along s (`AXIS_S2`, correlation length `AXIS_ELL`) conditioned on its values
    at the visible anchors (the boundaries `_metric_bias` set a bias at); a scale mixture (weight `AXIS_PI` at `AXIS_K`
    × the core sd) gives it a heavy tail, and the column is the mixture sd. NaN without an anchor.
  - `sd_s_horse`: a per-horse persistent term (`HORSE_VAR` by render style; correlation 1 along the path, independent
    across horses).
  - `sd_s_gap`: a long-gap term. From its last labelled detection (Δt the time since it; before its first, the time
    to it) an unseen horse coasts with white acceleration of the detection-level scale `GAP_SA` (0.67 m/s²/√s), an
    integrated random walk v = `GAP_SA`² Δt³ / 3; the field holds it near its neighbours, so the term saturates:
    var = v / (1 + v / `GAP_CAP`²), `GAP_CAP` 12.5 m. NaN for a horse never labelled.
  - `sd_s_ident`: an identity term. With probability `IDENT_EPS` the label at a node belongs to one of the
    `IDENT_K` = 2 nearest other paths along s within 2.6 render units across, and the error is then minus the offset
    to it; the column is that mixture's sd, √(`IDENT_EPS` × the mean squared offset), 0 with no such neighbour.
  - `sd_s_rel` = √(sd_s² + sd_s_horse² + sd_s_gap² + sd_s_ident²), the field-relative sd; `sd_s_total` =
    √(sd_s_rel² + sd_s_axis²).
- `id_robust` (`contract.id_robust`, appended after the along-track terms): the node's window is identity-robust, i.e. robust to how much
  of the identity was decided by the final pass rather than the video alone. Two tests per window:
  - relabel share: for the horse and every horse within 5 m along and 2.5 render units across of it at any window node,
    take the share of its window detections labelled it by exactly one of the video-only (`det_labels`) and final
    (`det_labels_final`) labels. The maximum must be ≤ tR;
  - read support: the share of the horse's final-labelled window detections with a confident disc read naming it
    (disc render), or of the legend rows in the window listing it (silk render), must be ≥ tP. Without a read source
    it is not tested.
  - windows: run nodes are those with finite s, before the line and at or after the field's render start (the
    median over horses of each horse's first observed node after 0.5 s).
    - The column uses the node's section unit: the horse's run nodes between two official section ends, at least 20
      nodes. Its detections are those within 0.05 s of the unit's first and last node. A node outside every unit is
      false.
    - For descriptors over bends, contract.json gives a per-horse bend verdict: all its run nodes on the race's bends
      pooled, the bend detections at or after the render start, and the legend rows between its first and last bend
      node.
  - thresholds: tR and tP are set per render style and window type (`ID_THRESHOLDS`). Each is the median of the two
    folds' selected thresholds from one cross-fit on masked reruns. contract.json holds the run's per-section and
    per-horse bend signals and verdicts and its yield (units, nodes and bend horses robust).
- Lateral-metre error (`contract.lat_m_err`, appended last; **provisional**): the `lat_m` map's own error, which
  `sd_lat_m` omits. By segment (`RaceRun.lat_m_segment`), `LAT_M_ERR` gives the sd of a per-horse part persistent
  along the segment (`sd_lat_m_horse`) and of a per-node remainder about it (`sd_lat_m_node`). `sd_lat_m_scale` is the
  race-level lateral scale; `sd_lat_m_total` = √(sd_lat_m² + sd_lat_m_horse² + sd_lat_m_node² + sd_lat_m_scale²).
  - Sha Tin bend: persistent 0.60 m (the p90 of the per-horse bend-mean error against the drone / 1.645), per-node
    0.25 m.
  - Sha Tin home straight: one unsplit 0.58 m term (the map's leave-race-out residual p90 / 1.645), carried as the
    persistent part; its per-node 0 means not split.
  - NaN everywhere else, including every Happy Valley segment, even where `lat_m` is defined; NaN where `lat_m` is.
  - These are design values, not a fit: Sha Tin only, from a 7–9-race drone comparison. On the Sha Tin bend,
    `sd_lat_m` alone is badly overconfident and `sd_lat_m_total` is close to calibrated (refit leaving each race out:
    z RMS 15.1 → 1.4, 90 % coverage 0.18 → 0.86), but still narrow at the tail out of sample. The home-straight term
    has not been scored.
  - Race-level scale (`LAT_M_SCALE_*`, `contract.lat_m_scale_sd`): the render's per-race lateral scale varies race
    to race, and no tested observable predicts it (field size, the abreast-gap ruler, the drawn width, the drone box
    ruler). It is carried as variance: error = z × σ × (lat_m − a), with z ~ N(0, 1) ONE per race and segment, shared
    by every horse and node of that race and segment. Consumers must treat it as one shared error per race and phase,
    not as independent per node: a race or field mean keeps all of it. `sd_lat_m_scale` = σ × |lat_m − a|, defined
    wherever `lat_m` is (Happy Valley included); the total stays NaN where the horse/node terms have no value.
    - σ (log-scale sd): 0.15 in the start and straights; in bend and home a ramp in x = d − home straight length,
      0.11 at x ≤ 0, 0.26 at x ≥ 400 m, linear between. These are the noise-free race-to-race sds of the render/drone
      lateral ratio within venue × render style (drone-era q3d frames); contract.json `lat_m_err.race_scale`.
    - Against the drone (Sha Tin, 42 races per segment, rank-paired kept frames), with the scale term vs without:
      per race (mean error over the segment) z SD 4.8 → 1.4 on the bend and 3.8 → 1.1 on the home straight, 90 %
      coverage 0.43 → 0.79 and 0.45 → 0.86; per node z SD 1.44 → 0.99 and 1.30 → 0.92, coverage 0.83 → 0.91 and 0.82
      → 0.92. On the 7-race identity-linked bend means (whose horse term was set from those same rows) the total with
      the scale term is too wide (z SD 0.51, coverage 1.00).
- Lane (`contract.lane_columns`, appended after the lateral-metre error): `lane_units` counts lateral in the race's own
  abreast pitch from its rail runner, and `lane` is its category: 1 rail (< 1.5), 2 two-wide (< 2.5), 3 three-plus.
  The count is scale-free, so it needs no metres and is defined in every season and render style.
  - Per race and segment class (`RaceRun.lat_m_segment`'s bend, home straight and other straight; start nodes use the
    straight class), on firm nodes (sd_lat < 1.5) at whole multiples of 0.5 s:
    - pitch: the median nearest-abreast lateral gap, to another firm node at the same time within 1 m along;
    - fallback: with fewer than 20 gaps, the pitch is `LANE_PITCH_POOL`, the median race pitch for venue × period ×
      class (`LANE_PERIODS`; none before 2023-09-01, so NaN there);
    - origin: the median over time of the innermost firm node's `lat`;
    - pitch3: the median inner gap of the depth-3 nodes (third from the inside among the class's firm nodes at the
      same time within 1 m along), or the pitch where the class has fewer than 20 such gaps or a pooled pitch. The
      lane beyond the second is about 0.94 pitch wide.
  - `lane_units` = 1 + (lat − origin) / pitch up to 2, then 2 + (lat − origin − pitch) / pitch3, on firm in-race nodes.
    It is NaN where sd_lat ≥ 1.5 or outside the race (before the first render frame, past the post), and `lane` is null
    there.
  - contract.json `lane` holds the run's pitch, pitch3, origin, gap counts and sources per class, with the rule and the
    cuts.
- Enclosed (`contract.box_columns`, appended after `lane`): `enclosed` is true where the horse is walled in ahead and
  on both sides, with lateral counted in the race's abreast pitch, so it needs no metres and is defined in every season.
  - At nodes with finite `lane_units` (neighbours count only where theirs is finite), with dS the along-track gap (m)
    and dL the lateral gap divided by the pitch of the horse's segment class:
    - ahead: a horse with 0 < dS ≤ 4.8 m (two lengths) and |dL| < 1;
    - outside: a horse with |dS| < 2.4 m and 0 < dL < 2;
    - inside: the same on the inside, or the rail (`lane_units` < 1.5).
  - Runs of enclosed nodes are joined over gaps under 0.5 s and kept when they last ≥ 1 s.
  - One pitch is about 1.2–1.4 m of `lat_m` off the home straight and 1.8 m in it (derived: drone-era race pitch ×
    the `lat_m` slope), so this is a broader state than the same rule with 1 m lanes in metres: 30 % of drone-era
    `lat_m` nodes against 16 %, and 29 % of in-race horse-time over 458 turf-oval races (2023-09 to 2026-11).
  - Both `enclosed` and `boxed_in` are false wherever `lane_units` is NaN, so false can mean undefined.
    contract.json `box` holds the rules of both columns.
- Boxed in (`contract.box_columns`, after `enclosed`): `boxed_in` marks a horse held up behind in the final 600 m. It
  applies to an enclosed run with at least 0.5 s at d ≤ 600 m (its late part). The late window W is the nodes with
  finite `lane_units` and d ≤ 600 m.
  - Closing node: a node t such that t and the node 1 s earlier are both in W, and t is in the late part or the 1 s
    before it. Over that second, the gap to the nearest horse ahead (0 < dS ≤ 4.8 m, |dL| < 1) shrank by at least
    0.3 m.
  - Held up: from the first closing node, the horse spends at least 1 s of the next 3 s within one length behind that
    horse, with |dL| < 1 and both horses in W.
  - The column is true on the run's late part.
- Enclosed late (`contract.box_columns`, after `boxed_in`): `enclosed_late` is the regressor of hypothesis B-1 (the
  modelling framework's late box-in test): enclosed in the final 600 m while in contention.
  - Node condition: the raw enclosed condition (as `enclosed`, before runs are joined), finite `lane_units`,
    d ≤ 600 m, and within four lengths (9.6 m) of the leader. The leader is the furthest-forward horse at the node,
    capped at the post, so the rule needs no labels.
  - Runs are joined over gaps under 0.5 s and kept when they last ≥ 1 s, as for `enclosed`.
- `det_margins_final.npz`: per tracks row, the final identity pass's assignment margin: the frame's optimal E-step
  cost with that detection forbidden from its horse minus the optimal cost (−2 log-likelihood units; NaN where
  unlabelled, `assoc.assign_frames(margin_out=)`). A margin, not a calibrated probability.
- `contract.json`: the definitions, the run's flagged stretches, its frozen-list windows, every visible crossing's
  position verdict (observed, firm, r_v, identity check, pass), the uncertainty constants with the run's axis anchors,
  the provisional `LAT_M_ERR` values, the run's lane pitch and origin per segment class, and `NOISE_TABLE`, the
  per-detection noise of the
  detection-only state-space model by feed and region (white sd and a frame-common AR(1) term); a row whose split
  between the two is not identified carries `split_identified` false.

### status

`run()` always writes `status.json` (not a stage artifact): `failed` with the stage and error when a stage
raises, else `unhealthy` with the failed video-only checks, else `ok`. Checks (`RaceRun.health`): render ticks per
expected race second (unique render frames / (distance / 16.5 m/s)) ≥ 4.0; median detections per render frame per
runner ≥ 0.4; ≥ 3 race-clock reads; the rails stage's course shape (straight writes `lat0_px`) matches the race
facts (straight only for Sha Tin 1000 m). The `start` status is recorded (`clock_anchor`) but not a failure.

### validate

`recon.validate` writes `validation.json` (gates on `paths_video.parquet` plus the product's residual against
the official times). `recon.audit` writes `audit.json`: label-free invariants (rail off the map, marker grid and
spacing, kinks; field-to-rail gap, field-wide lateral slides; legend negatives and ranks, ghost crossings,
finish-order inversions). `audit.rail_blocking` flags the rail as unusable on a contiguous off-map run over 100
stations, fewer than 3 markers, or marker spacing off by more than 10 %; it is logged, and the audit never fails or
stops a run.

## 5. Feed synchronisation

`viz.sync_video` draws the four feeds beside the reconstructed field on a map built from the run's own rail.
Offsets, supplied by the caller, are video seconds at race clock 0: a constant or `[(race t, offset), ...]`
interpolated linearly (`None`: the feed is drawn blank).
- aerial: `run.clock_f0() / 25`, the race-clock zero (after `start` re-anchoring) at 25 fps.
- full: `viz.full_offset`, the lag maximising thumbnail correlation between the aerial's left panel (a scaled
  copy of the TV picture) and the full feed, on two windows, added to the aerial's offset.
- patrol and drone: no offset is computed; the caller reads the gate opening on the feed. The drone feed loses
  time during the race, so it is anchored at both the gate and the finish board where the board is visible.

Display: a ring marks sd 1.5–3 m, a hollow marker 3–10 m; positions with sd > 10 m are omitted and listed as
"not placed"; the leader and window use sd < 3 m. A small sd does not flag a confident error.

## 6. Acceptance

### Fixed gates (`recon.validate`)

Scored on `paths_video.parquet` against the full official record, unchanged by the mask. A per-horse crossing
is scored only where the posterior sd_s < 1.5 m (`SD_MAX`) on both bracketing nodes. Boundaries are the
official sectional ends, 400 m apart. The thresholds are `validate.GATES`.

| Gate | Threshold | Definition |
|---|---|---|
| `leader_timing_mae` | ≤ 0.25 s | identity-free: running minimum of raw detection d per frame, crossing time at each boundary vs the official leader |
| `sectional_mae` | ≤ 0.25 s | per-horse crossing times at every boundary vs official |
| `finish_rho` | ≥ 0.90 | Spearman of reconstructed vs official finish times (≥ 3 scored) |
| `winner_ok` | == official | earliest scored crossing is an official winner (either horse of a dead heat) |
| `behind_winner_mae` | ≤ 0.20 s | time behind the winner at the line |
| `sectional_coverage` | ≥ 0.80 | share of (horse, boundary) crossings with an official time that are scored |
| `horses_with_path_short` | ≤ 1 | runners without a fused path |
| `finish_n_short` | ≤ 2 | runners with an official finish time but no scored finish |
| `lateral_scale_err` | ≤ 0.75 m | \|racing-to-fixed rail gap on the map − course table\|; n/a where unmeasurable (A course, no supported ridge) |

The last four were declared before any held-out race was run. "Timing gates" are the first five; "non-lateral
gates" the first eight.

### Yardsticks

What each measures and how; the scripts beyond `validate`/`audit` live outside the package.
- **Held-out boundary** (§3, `Mask(boundary=j)`): one official boundary withheld per race, rotating over the
  boundaries; the product's absolute crossing-time error there, also relative to the race's shared shift, and
  order inversions among pairs with an official gap > 0.05 s.
- **Interior residual** (product): the horse-specific residual at interior boundaries (product − official crossing
  × speed, race-boundary field median removed; robust sd = 1.4826 MAD) against the video-only product's on the same
  horse-boundaries; at the finish, distance from the official times, winners and finish-order inversions.
- **Hand truth.** Detections hand-labelled at colour-group level against the official silks; a detection counts
  as correct when its label is in the labelled set.
- **Numbered-disc agreement.** On the disc render, detections with a confident disc read and a label: the share
  whose label equals the read number.
- **Withheld detections** (§3, `Mask(dets=...)`). Per race, two horses × one 3 s stretch in the middle third of
  the race, chosen by a fixed seeded rule from an unmasked reference run: ≥ 80 % of the stretch's render frames carry
  a detection both the video-only and the final E-step label as the horse, the horse is never the frame's
  frontmost detection, and stretches with another detection within 3 m along and 2 m across in at least half the
  frames are preferred. Every detection either E-step labels as the horse in the stretch is withheld (plus
  unlabelled duplicate boxes within 0.6 m / 0.4 m). Scores: along-track and lateral RMS and max error against the
  withheld detections, and whether the path in the gap overlaps another horse's firm path (|Δs| < 2.0 m and
  |Δlat| < 0.7 m); the unmasked run gives the floor.
- **Withheld disc reads** (§3, `Mask(discs=10)` and `Mask(discs=10, discs_phase=0)`). Each confident disc
  read is scored once while hidden: among detections with a confident read inside the withheld blocks, the
  share of labelled ones whose label equals the read, and the share left unlabelled.
- **Withheld legend reads** (§3, `Mask(legend=10)` and `Mask(legend=10, legend_phase=0)`; silk render). Each legend
  read is scored once while hidden, at least 0.5 s inside its block (the consumers take the nearest legend read
  within 6 frames). In a frame where every runner is detected (boxes within 0.6 m along and 0.4 m across merged),
  slot r names the unique detection of rank r (none within 0.6 m of it) and agrees when that detection's label is
  the listed number. A horse's hidden 10 s block (≥ 5 scored frames) is confirmed when ≥ 80 % of its slots agree;
  the unmasked run is the reference.
- **Lateral yardstick.** AUC of final-path lateral position against the stewards' comments (WIDE > INSIDE, with
  MID between).
- **Perturbation ensembles.** Seeds of `HKJC_PERTURB_SEED` per race; arms are compared paired by seed with SE from
  per-seed differences, under an adoption rule declared before reading results. A run never scores a directory
  another run writes into.
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

## 7. Environment

- `HKJC_DEVICE`: `cpu`, `cuda`, `intel:…` (OpenVINO). `HKJC_NPROC`: workers per race. `HKJC_AVATAR_WEIGHTS`:
  detector weights.
- `HKJC_OCR_THREADS`, `HKJC_OCR_DEVICE`: OCR threads and device. `HKJC_MATCH_DEVICE=cuda`: ORB brute-force tables
  on the GPU in a helper interpreter (integer popcounts, OpenCV's tie order).
- `HKJC_FILM_DEVICE`: the film detector alone, e.g. `intel:gpu`.
- `HKJC_MASK`, `HKJC_FINAL_HOLDOUT`: the mask (§3). `HKJC_PERTURB_SEED`: seeded 1 % detection drop in `observe`.

Scheduling: one matcher helper interpreter serves the race from features to pose, and features runs the avatar
detector in its own process feeding up to 4 feature workers (a CUDA context per race from features to pose).

Exactness: `HKJC_OCR_THREADS=1` and `HKJC_MATCH_DEVICE=cuda` leave every artifact
identical. Not bit-exact:
- `HKJC_OCR_DEVICE=cuda` (ONNX Runtime's CUDA provider, needs onnxruntime-gpu): finish-camera reads, legend reads
  and identity votes can move.
- The film detector on `intel:` is the OpenVINO export compiled at FP32 (the GPU plugin's FP16 default loses
  crossings; a missing device raises). The other `intel:` paths (avatar detector, OCR recogniser, `horse_scale`)
  compile at the GPU default (FP16).
- Poses depend on the avatar detector's device (CPU, CUDA and OpenVINO boxes differ slightly and change ORB steps),
  so results are compared only within one device.

## 8. Known limitations

- **All-weather track.** Not supported: on the brown dirt the avatar detector misses horses, so races on the all-weather
  track are not reconstructed reliably.
- **Lateral scale.** `lat` is in render units; the render spreads the field about 1.4–2× (most in compact fields,
  least in the fanned-out home straight), on both render styles and both venues.
- **Rail offset.** Horses on the rail sit about 2 m out in `lat`.
- **Wide-running distance.** Distance run wide on the bends is overstated by the same spread.
- **Lateral metres.** Metres exist only in `lat_m` (§4), for drone-era turf races in the tested segments; lateral
  order is usable everywhere, lateral metres nowhere else. Not covered: the Sha Tin disc start, the Happy Valley
  disc bends and home straight, the Happy Valley silk start and home straight, and every pre-2026 season. The
  home-straight scale holds to about ±7 % across course settings (Sha Tin A reads about +0.3 m wide); bend metres
  are calibrated against a linear rail-to-rail drone ruler, which does not model perspective or rail height on
  oblique bend views. The map's own error (`sd_lat_m_*`, §4) is provisional: design values for the Sha Tin bend
  and home straight only, from a 7–9-race drone comparison; NaN at Happy Valley and in every other segment. Its
  race-level scale term (`sd_lat_m_scale`) is shared per race and segment and defined wherever `lat_m` is.
- **Lane category.** Against the official running comments' explicit lane statements, `lane` agrees about 0.83 of the
  time (0.90 in 2026/27), and stated three-plus horses read low (recall 0.78). No season reaches 90 %.
- **Boxed in.** Scored per run against the stewards' late blocked-run statements (566 statements on 458 turf-oval
  races, 2023-09 to 2026-11), `boxed_in` finds 0.52 of them, 1.9 times the share chance would find. Its flagged runs
  overlap a statement 2.0 times as often as chance (precision 0.125).
  - The closing term was chosen on these same statements, so the figures are not confirmed on a fresh set.
  - Node timing inside the late part is not validated.
  - It fires for about a third of runners. The stewards report about one horse per race, so most of its flags have no
    statement to confirm or refute them. Across masked reruns its horse × race κ is 0.69 on the silk
  render (0.88 on `id_robust` units) and about 0.96 on the disc render, so on silk use it beside `id_robust`.
- **Enclosed late.** Across masked reruns (both arms at one commit; 176 silk and 153 disc races) its horse × race κ is
  0.85 on the silk render (0.93 on `id_robust` units) and 0.99 on the disc render, at a base rate of about half the
  runners with late nodes. No calibrated probability goes with it: a per-node probability built from the identity
  margins understates the rerun disagreement on both renders.
- **Enclosed is not a costly box-in.** Enclosure is common and often cheap before the home turn. In the final 600 m it
  agrees with the stewards' blocked-run statements only 1.4 times as often as chance would give. Across masked
  reruns, its horse × section reliability (κ) is 0.86 on the silk render and 0.97 on the disc render.
- **Draw units.** `fuse_final` carries the spread unchanged: the draw prior (`STALL1_LAT`, `STALL_PITCH`, from a drone
  frame, approximately metres) acts only at t = 0, and the lateral box never binds.
- **Shared metric error.** Between visible boundaries the product's distance metric is the render's, shifted by the
  bias interpolated from the visible boundaries; its error is common to the field and zero at each visible boundary.
- **Official interior spread.** The official interior times compress the field behind the leader, relative to the
  render, by a factor that varies by race-boundary and season; it is not modelled.
- **Posterior calibration.** The stated sd is overconfident at intermediate boundaries but not at the finish; a
  systematic error term is missing.
- **Uncertainty columns.** The uncertainty columns (§4) do not yet meet the calibration targets; the gap term is
  too narrow for gaps of 1-6 s; Happy Valley 600 m carries a known distance bias, and a withheld first boundary with
  no visible boundary behind it runs 3-7 m late.
- **Identity-robust flag.** The `id_robust` thresholds come from one cross-fit and are not yet confirmed on a fresh
  set. The flag drops box-in units disproportionately, because box-in lives where identity is unstable. So box-in
  should ship with the flag attached, not filtered by it.
- **Render-vs-official conflicts.** A horse whose render path and official interior times disagree by more than a
  few metres keeps its render path; the video alone cannot say which side is wrong.
- **Silk evidence.** Legend votes add per frame and silk once per tracklet (±3), so silk rarely overturns a legend
  label; back markers get only elimination evidence and the weak silk term; the draw does not enter identity.
- **Finish-short horses.** A horse unobserved over the last metres has a path to the grid end, extrapolated with
  growing sd; once sd_s exceeds `SD_MAX`, or the grid ends first, it has no scored finish crossing.
- **Blind near-line cost.** It can remove correctly placed unread back markers. More generally, near the line a horse's
  own tracker fragment can be cut from it: the blind and camera-unread costs bar it, and its path runs on at race speed
  while the horse slows, so the fragment's last detections stay anonymous and the path crosses early. With the finish
  withheld this is one cause of tail horses without a label at the line (others: no detection there, or detections
  labelled to another horse).
- **Fixed-rail drawing.** On some Sha Tin course settings (A+3, B, B+2) the render draws the fixed rail at a
  constant offset from the course table's gap, so `lateral_scale_err` fails there while the lateral scale is
  consistent.
- **Rail snap at junctions.** The nearest-ridge snap can follow a crossing or chute marking where it breaks the
  rail line (a Happy Valley junction).
- **Render scale between boundaries.** Some races carry a common-mode render scale error between boundaries that
  the film window is too short to measure.
- **Leader timing.** The odometer can run late early in long Sha Tin races and early on others; beyond 400 m no
  reference exists that is neither official nor the odometer.
- **Straight-course leader.** On the 1000 m straights the leader runs slightly early from 400 m (odometer/marker
  scale ratio about 1.035).
- **Pack-to-fence drift.** On straights, lateral is pack-edge-relative plus a constant fence offset; drift between
  pack and fence is not modelled.
- **Legend misreads.** A legend digit d is occasionally read as "1d" (3 → 13) when a vertical edge (a distance post,
  the rail, a box edge) lies at the left of the slot crop.
- **Duplicate boxes.** The avatar detector can keep two boxes on one horse (IoU under NMS's 0.5); the second box
  births a second track.
- **Static-overlay filter.** It can delete a real horse the camera holds at one panel cell in the disclaimer band,
  and misses disclaimer text below its bar, which the hard rank cost then treats as horses.
- **Tracklet exclusivity.** `assign_global` forbids one number on tracklets whose spans overlap, which can leave a
  tracklet unlabelled when its number's holder overlaps it by a few frames inside a gap.
- **Orphan elimination.** The last uncovered horse takes the largest leftover chain without an anchor check, which
  can give a tailed-off horse a mid-field chain (§4 fuse_video).
- **Video resolution.** The pipeline assumes a 1920 × 1080 aerial feed; seasons before 2023-05 are 1280 × 720.
- **Non-finishers.** A stopped horse's path ends at its last observation, not where it stopped, and before the last
  50 m its label can still ride another horse's avatar (ST 2025-07-05 R5 #2 from 385 m to 68 m out). An unseated
  horse's path runs to the grid end (constant-speed extrapolation, growing sd); it can still take another runner's
  detections where `identify` left the fragment unlabelled, and where `identify` put its number on another runner's
  avatar the rule holds it there (ST 2023-10-15 R3 #14 on #12's avatar). With the finish withheld (evaluation masks) no placing
  code is visible, so neither rule applies and a non-finisher's label can ride a finisher's avatar to the line. In the
  video-only product nothing marks a non-finisher. Its completed sections count for coverage and `sectional_mae`, but
  it is excluded from the finish gates.
- **Post-start non-runners.** A horse declared a non-runner after the start (WXNR) is not a runner and gets no path,
  even if it raced.
- **Resolution floors.** Official mid-race times sit on a 0.04 s grid relative to the leader (~0.2 m); the HUD
  odometer changes only every 0.4 s, so a raw tick is 0–0.4 s late.
- **No real-camera positions.** Drone and patrol give no positions; before the first render frame there are no
  video positions, and nothing independent of the render checks lateral position in-race.
- **No non-overlap constraint.** Fusion does not keep horses from occupying the same place.
