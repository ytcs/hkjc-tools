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

CLI: `uv run --group vision python -m hkjc.recon 20260603 HV 8 [--force stage,...]`. Stages are
cached under `data_cache/recon/<date>_<venue>_<no>/`; a stage is skipped when its outputs exist.
Runtime environment: `HKJC_DEVICE` (`cpu`, `cuda`, `intel:…` for OpenVINO), `HKJC_NPROC` (workers per
race), `HKJC_OCR_THREADS`, `HKJC_AVATAR_WEIGHTS` (detector weights), `HKJC_FILM_DEVICE` (the film
detector alone, e.g. `intel:gpu`; see the film section).

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
| drone | one shot gate to finish | synced viewer only (`recon.drone` fits a camera to the rails but is not in the pipeline) |

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
stage re-runs every requested stage downstream of it, and a stage whose artifact is merely missing
is regenerated without invalidating its consumers (the pipeline is deterministic). Artifacts are
written to a temporary file and renamed, so a killed stage never reads as done; a failed stage
terminates the background stages. Each race appends `timings.jsonl` (wall s, CPU s, peak RSS).
ONNX Runtime sessions (avatar detector, OCR) run with intra-op spin-waiting off
(`ortsession.no_spin`): same threads and work split, so identical outputs; CPU about a third (avatar detector) and a half (OCR).

### frames

Unique frames of the render panel. A frame is render-active when the legend band (0.70–0.86 of the
panel) shows a saturated box, the top strip is the dark HUD box (turf-green fraction ≤ 0.30) and the
track band is sharp (Laplacian variance ≥ 240). Once the strict test has passed, ≥ 5 boxes in a band
widened to 0.60 also pass: on ST 04-19 R4 and ST 04-26 R9 the legend sits higher and the strict gate
dropped up to 13 s of race. The render updates at 10 Hz in 25 fps video; a frame within mean |diff|
3.0 of the last kept frame is a duplicate. Panels are cached as JPEG with their video index as
timestamp. `video.grid_keep` drops off-grid re-encodes of a kept tick (HV 04-22 R1/R9 have a
keyframe every 100 frames that enters as a zero-motion tick).

### features

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
`clock_f0 = median(frame − 25·clock)`.

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
is not modelled. On 3 straights × 6 seeds the datum with the constant lifts finish ρ 0.884 → 0.923.

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
pairs, hand-wrong final labels 175 → 153, gates unchanged.

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
reads lie on the longer number's track; coinciding numbers keep the longer one. `finishcam.finish_gaps`
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
0.002–0.006 s lower on all six). One race chosen by a withdrawal rule, not a held-out claim. The finish camera names tracklets by 1D registration
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
completed by the first posterior where every sd < 2 m, are added and `assign_global` re-run.

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
   crossings at d = 400k (sd 0.4 m).
2. `RaceRun._metric_bias`: at each boundary, b = median over ≥ 3 horses of (video-only crossing delay
   × speed); render s is shifted by b interpolated along s. The render metric error is shared by all
   horses, and hundreds of render observations otherwise outvote one official time (residual ~5× its sd).
3. Fuse with the grid running 0.5 s past the last official finish (ending at the last observation left
   HV 04-22 R9 #11 0.8 m short of the line).
4. Field-relative start prior: until the first render frame + 1 s there is no video, so acceleration
   has sd 0.63 m/s² per node about the field's median curve from step 3 (the clean races'
   horse-specific acceleration). The zero-mean 6 m/s² prior extrapolated a late-seen horse backwards
   out of the gate (ST 05-31 R1 #6 26 m out at 2 s, the field 8–15 m).
5. Final identity pass: `RaceRun._skeleton_merge` keeps the step-4 path except where it conflicts with
   the horse's official-only skeleton by > 4 combined σ (flat at 3–5); the E-step is re-run from there
   with the official observations in every fuse and bias-shifted detections; the last fuse writes
   `paths.parquet`.

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
- **Held-out boundary.** `HKJC_FINAL_HOLDOUT=j` withholds one official boundary; the final product's
  error there is independent of that record, except that the metric bias still reads it, so the
  error is also reported with race medians removed.
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

### Baseline (tag `recon-baseline-Z11`)

Z10 with tick-time stamps in `observe` (observe onward on the Z10 stages, 26 × 6). Video-only: non-lateral gates
201.67 per seed, timing gates 126.67, all five on 23.67 races, lateral 10 pass / 3 fail / 13 n/a (unchanged from Z10),
finish-short 26–27 per seed. Final product: hand truth (seeds 1–5) 1117 correct, 153 wrong, 51 blank of 1321
(Z10: 1095 / 175 / 51); glitch-seconds 21.9 per seed (Z10: 25.6); seed unanimity 90.2 % (89.2 %).

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
implausible pre-render race); held-out boundary median error 0.032 s (all interior boundaries, seeds
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
- Finish-n, video-limited: ST 05-03 R7 (#4 8 m short at the last render frame, #10 3 m short, #11's
  detections end 200 m out), ST 05-09 R2 (#9 3 m, #11 9 m short at the last observation; passes since
  the tail, with #11's crossing extrapolated over 5.9–7.6 m),
  HV 05-27 R9 (no detections over t 70.56–72.28; #7 unobserved for its last 16 m, sd 1.78 m at the
  line; #9 and #10 lose detections 73–87 m out). ST 05-24 R1: #8 and #12 have no detections after
  125/90 m; #4 has 4 sparse detections in the last 25 m that the blind cost bars.
- Lateral: ST 05-03 R7 and ST 05-31 R1 measure the B-course fixed-rail gap at 6.6 m vs the table's
  4.5 m (one clean ridge at 6.5–7.0 m on 98 % of stations; the official course diagram confirms the
  table, so the render draws the B fixed rail ~2.1 m further in); HV 04-29 R3 (weak ridge, 9.21 m).
- Occasional: HV 05-20 R2 sectional (2/6), ST 04-26 R1 winner (1/6, a 0.05 s finish).

## 6. Open problems

- **Final-product calibration.** On held-out boundaries (one seed, n = 320) the median error is
  0.58 m and 63 % of errors exceed 2 posterior sd (median z 3.1). Where the stated sd is ~0.17 m and
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
- **Back-marker identity.** Back markers get only elimination evidence and a weak silk LLR: HV 05-20
  R2 #5/#12 are swapped at 400 m (~1 s, every seed), and at 1200 and 800 m #5 and #12 cross 0.6–1.4 s early on all six seeds (video-only;
  #2 0.6–0.75 s at 1200 m on six seeds and at 800 m on three; #6 and #8 0.3–1.0 s where they have a crossing); in one seed-0 frame at 1200 m the avatar the
  legend lists sixth (#2) carries label 5; ST 04-19 R4 #3 (tailed off, out of view) lands on a
  lookalike tracklet. Draw order agrees with early lateral order (median ρ 0.75), but a Gaussian draw
  term was overconfident (it broke HV 05-13 R1 and ST 06-07 R6); a heavy-tailed form with per-race
  parameters is untested. The blind near-line cost also removes correctly placed unread back markers
  (HV 05-27 R9 #7 at d 49 m).
- **Identity evidence (parked).** Of the final product's 175 hand-wrong labels (seeds 1–5), 164 already carry the same
  label in the video-only product; their error rate is 8.8 % where every hand candidate is in the legend's top 8, 12.3 % where none is,
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
  margin through the full EM cuts the alternation by two thirds but leaves the contradictions (§7); cause not established. HV 04-15 R1: a wide drone horse has no path horse in the 9 seed-product
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
  copying the parking pose onto them, gained no finish crossing (§7). With the pose copied (ST 05-03 R7, ST 05-24 R1), fusion's per-detection
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
- **Worklist.** 31 disclaimer-text boxes survive the static-overlay gate in HV 05-20 R2 (pixel-verified; max 30 in one 8 px cell vs the 4 % bar of 35); they are never labelled, sit in 1–2 whole-field legend-vote frames per seed, and are > 4.5 m ahead of real detections in 31 E-step frames, where the hard rank cost has no false-positive allowance (labels there differ from the tracklet's neighbouring frames on 14–27 % of detections vs 1.2–1.6 % elsewhere). Removing the 31 pixel-verified text boxes (oracle, seeds 0 and 1) changes 1.9 % and 0.9 % of video-only labels; #5's and #12's boundary errors are unchanged, seed 0's sectional MAE crosses its gate (0.261 → 0.247 s; seed 1 passes in both, 0.222), and glitch-seconds move both ways. HV 05-27 R3 #5 crosses 1200 m ~13 m off its official time on every seed (untraced); HV 05-20 R2 sectional (+0.05 s every seed; render crossing times
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

## 7. Rejected (do not retry)

Each line: the approach, then the measurement that decided it. "Removed untested" means removed
without a verdict.

Frames, features, pose
- Dropping every off-grid frame: removed the only copy of ticks after finish-zone dropouts; finish-short 34 → 39, ST 05-24 R1 winner flipped, hand 15 → 10/19.
- Dropping a frame one after the last kept, keeping the old duplicate reference: zero-motion ticks stayed (14 per race) and later ticks were stamped 1 frame late.
- Keeping on-grid frames the duplicate test drops after the finish (the camera parks; real ticks differ 1.5–2.1 from the last kept panel vs tolerance 3.0): 0 of 10 finish-short horses gained a finish crossing on ST 05-03 R7, HV 05-27 R9 and ST 05-24 R1; the winner flipped on 2 of 3. Detections were unchanged; shared frames' course position near the line moved by up to 3.4 m. With the parking pose copied onto the kept frames and the pose solved on the other frames only (their poses bit-identical): 0 of 7 gained on ST 05-03 R7 and ST 05-24 R1, ST 05-24 R1 lost `finish_rho` (1.0 → 0.88), HV 05-27 R9 stopped in `observe` (its rail tested as straight).
- Render gate with the widened strip and ≥ 1 box: admitted 11–65 pre-race live frames per race.
- Relaxed legend fallback before the first strict frame: admitted a dissolve tail on HV 05-27 R3, which went 5/5 → 2/5 gates.
- Colour-whitelist ORB mask: removed the stall but dropped posts and markers; HV 04-29 R3 loop closures 335 → 0.
- Half-scale ORB (1500 features): 5× faster, but 2 vs 5 of 10 dev races pass every timing gate; HV 06-03 R8 loops 171 → 0.
- Nonlinear (trf) pose solve: 3 vs 5 of 10 dev races pass every timing gate, 33× slower; it satisfies its world-px objective by collapsing frame scales.
- Rigid pose (no scale): ~5× worse on long-baseline edges; the render zooms slowly.
- ECC refinement: the mowing-stripe aperture causes scale drift.
- LMEDS refit: integer FAST keypoints snap steps to whole pixels, a speed-dependent accumulating bias.
- Projection-guided matching: no gain (~30 keypoints per 40 px gate cost as much as brute force).
- XFeat features: 4× slower extraction, matching not faster, d = 24 pairs fail (24 vs 90 inliers).

Detection
- Adding background-subtraction blobs on turf: duplicates and false positives shattered HV 06-03 R8's tracking; used only where YOLO is blind.
- Isotropic merge radius for blob fragments: too small for head-to-tail splits, too large for horses abreast.

Rails and metric
- Brightness-scored rail DP: left the movable rail for the fixed rail or apron edge via diagonals (ST 05-31 R1 bend ~4700 stations).
- Continuity score alone (without nearness cost, robust scaffold, gap weight): non-lateral 198.0 → 193.8 per seed.
- Rail smoother alone (brightness line choice, no silk ties): non-lateral 197.8 per seed, hand 53/79.
- Brightness score plus the robust scaffold: hand 62 → 53/79; wrong-line segments unchanged (15/347).
- Continuity score, nearness and scaffold without the 0.1 gap weight: 198.5 per seed; ST 05-24 R1's rail bulged off the map and lost its marker grid.
- Unseen offsets scored 0 on straights: rail bridged off the map for 1800–2800 stations, marker grids lost, blocking audit failed.
- DP slope bound on straights: the lateral step becomes a ramp.
- Whole-course coverage band for the DP: ST 04-19 R4's rail clears it by ~2 px.
- Robust straight line for straights: the mosaic bends (30 % inliers, end errors up to 10 m on ST 06-07 R8).
- Straight datum without the measured fence constant: ~10 % of detections inside the rail; ρ 0.845 vs 0.923.
- Bare turning sign on straights: flipped between pose variants; ST 05-09 R1's field slid ~3.5 m and #4 ended 3.4 m short.
- Offset-space gap interpolation: humps where the scaffold jogs; HV 06-03 R8 adjacent tracklets swapped.
- Gaussian despike trend: flattened curvature at the ends and shifted the marker grid by one marker.
- Gaussian normals (σ = 3 stations): wobble amplified at wide lanes.
- Exponential nearest-to-field DP preference: pulled the rail into the field on the home straight.
- Lateral gate counting only stations imaged to the expected rail + 1 m: corrected HV 06-03 R8, broke HV 04-29 R3, left ST 05-03 R7; noise at low support.
- Unseeded marker grid: took a coverage-edge artefact as FINISH (HV 04-29 R3).
- Geometry-only metric (marker scale, no ticks): far worse on every race; the map scale drifts along the trip.
- Fixed tick convention c = 0 at Sha Tin: a constant +0.3 s at every boundary.
- Measured tick convention applied with an unpinned marker grid: ST 04-01 R5 and R6 got worse.
- 51-station median + Gaussian for the inner-edge scaffold: a 107-station outlier run on HV 05-20 R9 (up to 247 px vs ~30 px) survived it, bent the scaffold and folded every rail ridge into a V the DP could not follow.
- Dense knots wherever any tick is flagged: an isolated start tick gave 170 m residuals.
- Local refit's derivative for lateral: ST 04-19 R3 p99 lateral 30 m vs 19.4 m.
- Painted FINISH pin as the anchor when the film has ≥ 3 crossings: step in the last 50 m (+0.2 s late, then −0.16 s early).
- Two-parameter film anchor (offset + scale): film-marker slope correlates with the official slope at r = 0.26.
- Film single-frame longest line: static stripes won (104 of 351 frames on ST 04-19 R4).
- Film markers outside the home straight: HV 400 m −0.32/+0.10/+0.68 s; HV course A 300 m +2.32 s.
- Film constant-speed extrapolation to the line as an observation: biased early by 0.28–0.61 s.
- Correcting the film's Sha Tin −0.11 s bias with official times: would put outcome data into the video-only product.

Observe and tracking
- Per-frame 8 m leader cut alone: dropped HV 05-20 R2's 13 m clear winner; finish 1.06 s early.
- Common along-track shift in the tracker: −4 gates on 26 races (ST 04-19 R3 ρ 0.98 → 0.88).
- Silk appearance cost in the tracker: −2 gates; seed unanimity 54.5 → 49.9 %.
- Merging tracker seam duplicates: gates +0.17 per seed but hand majority 57 → 48/79.

Finish camera
- Temporal gaps from the unique-frame index: the playback rate varies near the winner.
- Segmenting the whole white cloth: merges with white breeches and grey horses.
- Fixed V < 100 digit cut or black-hat: fragments blurred strokes or lights up dark horse bodies.
- White-ring minimum 0.55: lost far-side cloths (ST 05-03 R7 winner read on 2 frames).
- White-cloth reader at Happy Valley: false reads (HV 05-27: a "#7" at 0.5 s, the real #7 2.36 s back).
- Requiring ≥ 2 gap estimates for line crossers: dropped the first crosser (ST 06-07 R8 order [2, 7, 1]).
- Digit-drop by gap proximity: dropped ST 06-07 R8's real #2 beside #12.
- Spread test after the running median: drops genuine short tracks (ST 04-19 R4 winner spans 95 px).
- Robust line x(u) plus motion-rate test: restores ST 04-19 R4's #11 and #10 but timing gates 126 → 123 on 26 races.
- Motion-rate test alone for short slow-motion tracks: timing gates 126 → 124.
- Digits-only CTC decode: P("0") ≈ 0.01 where the OCR reads "U".
- Integrating cloth speed for gapless back markers: the post camera pans with the leaders, so the integral saturates.
- Pairwise neighbour test in elimination: #1/#5 predicted 0.16 s apart blocked an unambiguous joint match.
- Weak-track sd inflation (< 10 reads disagreeing > 0.3 s): inflated HV 04-29 R3's winner, giving a wrong winner; blanket inflation lost HV 06-03 R3's winner.
- Ruler calibration ×0.90 on the C prior and registration window (fitted on 26 races; 0.88 and 0.82 also tried): passed the gate rule but ST 04-19 R4 lost #5's camera name on 3/6 seeds; registration-only and anchor-only splits each cost hand truth or a race.
- Tight C prior (relative sd 0.12): 197.33 vs 198.0 non-lateral per seed on both rail variants.
- Rejecting an anchor pinned at its C bound, or fixing C at the prior: ST 04-26 R1 ρ stays ≈ 0.83.
- Unconditional (marginal) finish-observation sd: HV 06-03 R8 ρ 0.93 → 0.55.
- Anchor fitted to crossings without the sd check: ST 04-19 R4 C 800 px/s, clock 2.2 s early.
- Fixed 75 % recall cutoff for negative evidence: missed a race at 71 % recall.
- Finish-camera read set = every read cloth for the E-step: removed untested (one run: ST 05-09 R1 finish-n 7 → 7).

Identity
- Official silk images as raw prototypes: top-1 1/8 (avatar vs jersey domain gap).
- Per-detection silk cost in the E-step: hand 49–58/79 across weights (53 without), gates 205 = 205, per-race changes in both directions.
- Silk cost relative to the best positionally plausible candidate: hand 50/79.
- Silk as a veto: vetoes the true horse on 25 % of 53 camera-named tracklets.
- Tracklet-averaged official-silk matching: top-1 34/79 vs 40/79 per detection; central crops and legend icons no help.
- Greedy longest-first labelling with a margin, plus a silk tie-break for tied back-marker groups: turned 72–100 % unanimous labels into 38–99 %; replaced by the MILP with its silk LLR.
- Elimination as fractional votes in the MILP: back markers never numbered; non-lateral 200.3 → 197.5, hand 68 → 60/79.
- Draw tie-break for silk ties: 5/15, the same as the EM.
- Silk tie-break on the brightness-scored rails: non-lateral 197.7 per seed, hand 48/79.
- Finish-camera registration by relative gaps alone on a packed finish: renamed legend-#2 as #10 on HV 06-03 R8.
- Cloth linker (identity carried from read runs onto unread cloth boxes, ST 04-19 R4): parked; its self-test was confounded (reads hidden by frame index, constant-velocity gate on a bimodal track).
- Gaussian draw term in identify: overconfident; changed 24 tracklets and broke HV 05-13 R1 and ST 06-07 R6.
- Ranking legend votes by own-lane distance: vote consistency 0.85 → 0.52.
- Set-valued near-level legend votes: 27–33 right / 7–12 wrong vs 31 / 8, not flat in the margin.
- Leaderboard lag correction (0.3–0.45 s): no hand-truth change.

E-step and association
- Rank attraction over the whole race: ST 04-19 R3 sectional 0.174 → 0.266; for non-tight horses only: never fired.
- Negative-legend floor margin 0.6 m: HV 06-03 R3 lost #10/#5 (4, 8 and 16 m equivalent).
- Hard rank exclusions at 0.6 m: correct labels left unassigned (ST 04-26 R1).
- 4σ margin on the negative-legend count bound: flips 35 % of its claims; non-lateral −0.17, ρ −0.004 on 6/6 seeds.
- Union instead of majority of a fragment's exclusions: indistinguishable on ensembles.
- Camera negative evidence only for horses predicted inside its coverage: non-lateral −1.83 per seed; ST 05-03 R7 #11 crossed 3.66 s early.
- 3 s grid tail past the data after the EM: finish counts gained on three races, all five −1.33 per seed (SE 0.21).
- Prior-only grid nodes past the data inside the EM: labels change on most races (up to 4–6 %); a 2 s tail lost a crossing a 1 s tail kept.
- Waiving the blind cost on a blind horse's own fragment: safe but no gate benefit; EM 2-cycle on ST 05-24 R1.
- The same waiver on E-step majority alone (no identify label): would have exempted ST 04-19 R4 #9's stolen fragments on 4/6 seeds.
- Keeping runs shorter than 3 detections: ST 05-24 R1 timing 5/5 → 3/5; singletons near the line drag the camera clock.
- Tightness cut (sd < 2 m) for the blind cost: does not separate rendered back markers from ST 04-19 R4 #9 (0.81 s early when exempted).
- General monotone-progress rule: field-wide pose slips are not identity errors.
- Shadow release mid-race: released 1323 detections of a real horse on HV 04-29 R3 (coverage 94 → 90 %).
- EM to convergence (30 iterations): identical timing gates on 36 runs, spreads unchanged, 2–3× time.
- Finish seeds from the first EM iteration: changed races without targets (ST 04-19 R4 common-pair sectional 0.409 → 0.467).
- Forbidding orphan elimination onto a blind horse: elimination hands the next chain instead.
- "Rank" association mode: removed; its motivation was a contaminated reading.

Fusion
- Iterative least squares (lsq_linear, lsmr): 1.5 m error; the acceleration prior makes the system stiff.
- Takahashi selected inverse for variances: exact, no faster than LAPACK's multi-RHS solve.
- Along-track velocity prior: decelerated paths toward a stop across end-of-race gaps.
- χ² association without the log-det term: vague paths accept anything (HV 05-27 R3 #9 1.4–2.0 s early).
- Log-det association alone on the brightness-scored rails: non-lateral 199.5 per seed, missed its rule.

Final product
- Final identity pass comparing raw detections with bias-shifted paths: census 87.3 %, hand −4.
- Official sd 0.1 m: glitch-seconds 55.8 → 85.4.
- Final identity pass started from the official-only skeleton: hand 220 → 164 (no lateral detail; bunched field mislabelled).
- Field-relative prior over the whole final race: share of held-out errors beyond 2 sd 0.628 → 0.694 (guard +5 points).
- Field-curve centre without the stiffer width after the start: glitch-seconds 24.3 → 42.4, no held-out gain.
- Fragments verified by an official crossing restricted to their horse in the final E-step (forbid only): 26 races × 5 seeds, jerk-seconds 7.1 → 4.0 per seed and hand wrong 175 → 170, but on a blind pixel test of 43 disagreements the arm is right on 11 vs 10; 15 of 20 others turn a wrong label into a blank (the verified horse's path is elsewhere, so its detections go to clutter).
- Keeping the detection path's lateral sd where the final pass falls back to the official skeleton: 26 races × 5 seeds, jerk-seconds +0.7 per seed (one seed 6.8 → 15.6), hand wrong 175 → 180, held-out error on the 24 changed horse-boundaries 0.52 → 0.80 s.
- Official crossing times as hard label exclusions in the final product: no headroom; 1 of 783 boundary-crossing runs is > 0.3 s off its own horse (the errors sit between boundaries).
- Leave-one-run-out demotion in the final product (a render run > 4σ and > 2 m from its horse's path refit without it, official crossings kept, made anonymous): one seed, glitch-seconds 24.3 → 36.4; demoted fragments re-associate to other horses and a tailed-off horse loses its only constraint.
- Stiffer along-track prior in the final product (a_sd_s 1.2 → 0.72 m/s² after the start window; 26 races × 5 seeds, all interior boundaries held out): held-out median 0.0322 → 0.0318 s (bootstrap CI spans 0), faders at the last boundary 0.040 → 0.042 s, hand wrong 175 → 177; glitch-seconds 25.8 → 19.6. At 0.63, unperturbed seed, end of section 2 held out: 0.033 → 0.036 s.

Synchronisation and other evidence
- Frame-difference threshold or change point for gate opening: cameras pan after the start (ST 04-26 R9 drone fired at 0.04 s).
- Matching the shared finish-camera replay across feeds: it sits at a different edit position per feed.
- Drone offset from formation matching: 1.47 s vs 2.46 s measured on HV 06-03 R8.
- Drone camera fitted to the rails plus horse ICP for positions: the rails register, but along-track tilt is unobservable on straights and the ICP does not lock.
- Double-box gate as an identity fix (keep the more confident of two boxes with intersection over the smaller ≥ 0.5): wrong hand labels on the 10 races not used to build it 142 → 143 (26 × 6; §6).
- Lateral projection correction (rail-normal skew or camera tilt): painted cross-track markings map square through pose and rail on 6 home straights; the slope of video-only boundary error against lateral offset (−0.159 m per m pooled, race-bootstrap SE 0.118) comes from 56 errors over 10 m on 13 horses (44 in HV 05-20 R2 and ST 04-19 R4, races with recorded identity errors; 12 untraced), and with race-clustered intervals every robust per-stratum slope contains 0 (finish line −0.005 HV, +0.007 ST; ST bends negative in 5 of 9 races).
- Free 8-DOF homography for the drone camera: collapsed onto degenerate solutions (every rail pixel snapped to one rail); replaced by a 7-DOF look-at pinhole.
- TV-feed cloth reads as an identity yardstick: failed its declared criteria (HV 06-03 R3 order 62.5 %); the TV camera follows the leaders, so back markers are rarely read.
- TV running-order bar: 8 fixed slots, a delayed copy of the render legend.
- Post-gate patrol silk templates: top-1 0.61 vs 0.63 for in-race TV templates; the colour histogram is the ceiling.
- Identity-free per-frame registration of drone horse detections to the render detections as an acceptance test: the same median residual gate (1.2 m) passes 23 % of fields from other races and 48 % of the same field 8 s later.
- Drone horse tracks in place of interpolation across render gaps, for position accuracy when the horse's identity across the gap is given (linked to render tracklets where the render frame is complete): simulated 10 s dropouts on 25 races, drone 0.92 m vs interpolation 0.64 m median (extrapolation 1.85 m); coverage collapses on the worst render races (HV 05-20 R9 0 %), where complete frames are rare.
- Drone observations (render-gap stretches of drone tracks, labelled by each whole track's majority video-only detection label) in the video-only fusion: on 25 races the unseen-at-boundary timing error fell 11 % (0.128 → 0.114 s, 20 % required), 4 gates flipped to FAIL and jerk-seconds rose 2.7 → 5.0; one horse received two simultaneous drone tracks 13 m apart (the drone tracker switches horses inside long tracks).
- Drone bridge observations labelled by render coincidence on the same track (within 10 s): in the EM they flip gates and raise jerk; injected after the EM they help where they reach (12 unseen-at-boundary cases 0.126 → 0.085 s, blind pixel A/B 11 of 12 decided) but break where a drone track switches horses; a kinematic junction gate removes the breaks and with them the benefit (0.126 → 0.149 s) — identity across the gap needs evidence from the drone picture, not from the control's labels.
- Official-silk colour histograms (the render's `silk_feature`) on the drone detector box: chance level (465 units on 25 races, top-1 0.084 and top-3 0.247 against 0.60 / 0.85 required; the render's crops of the same units give 0.41 / 0.68). The box covers the horse's body; the jockey's cap and back lie outside it, ahead in the direction of travel.
- TV running-order bar as an extra ranking source: it repeats the aerial legend about 2 s late, in 1.2 s pages (runner sets match 0.98–1.00 after the shift; 3 races read), and is removed before the finish.
- Drone silk colour histograms from a fixed jockey region ahead of the detector box (course direction from the registration): top-1 0.112 and top-3 0.344 on the same 465 units (render 0.41 / 0.68).
- Hand-built drone jockey descriptors (oriented crop ahead of the box; hue histogram + Lab grid): official-silk match top-1 0.137 (474 units); drone-to-drone enrolment across tracks 0.249 top-1 (3242 windows, chance 0.088, required 0.60).
- Drone jockey embedding trained with batch-hard triplet loss alone (from scratch, Adam 1e-3, single-race batches): the embedding collapses (loss at the 0.3 margin from ~100 iterations; pairwise spread 2e-3). Enrolment across tracks reached 0.585 top-1 anyway (3245 windows, required 0.60; top-3 0.810, chance 0.088). The idea of a learned embedding is not rejected; only this loss is.
- Fusion grid extended to cover the latest observation (one extra sample): removes grid-end shorts (0–3 fewer per seed) but flips `winner_ok` in ST 04-26 R1 (seeds 0 and 3; the control's winning margin is 4 ms), because the extra sample moves EM decisions. The defect is fixed by the write-only tail (§2, fusion model); it had also failed the short guard of the drone-correction rules on HV 04-15 R1 #2 (4 cm).
- Identity-free drone corrections (unmatched drone points assigned to horses without their own render detection by least displacement, injected after the EM): held-out drone agreement improves in every seed (unexplained 0.565 → 0.503), but 7 gate flips — a tailed-off horse (ST 04-19 R4 #3, 7.5 s behind) received pack drone points and got a confident, 5.9 s early finish.
- The same corrections kept only where the drone track confirms the horse (track purity ≥ 0.9 on control labels, majority = the assigned horse; optionally a learned drone jockey embedding): every gate and guard holds and held-out drone agreement improves (0.565 → 0.510), but official boundary timing shows no gain where the corrections act (25 distinct horse-boundaries, seed-mean median 0.090 → 0.092 s; 5 improved and 6 worsened by > 0.02 s; HV 05-27 R9 #10 0.02 → 0.16 s in every seed).
- Drone configuration check as a predictor of official boundary-timing error (not tested as a configuration check; a path horse in drone view with no drone point within 2.5 m, one-to-one, in ≥ 50 % of frames within ±1 s of its boundary crossing): AUC 0.630 for official relative boundary error > 0.3 s (0.65 required; 792 horse-boundaries, 19 bad, precision 0.12 against a 0.024 base). Errors at boundaries are rare where the drone sees the field, so the test has little power, and it cannot see configuration errors between markers.
- Identity by elimination for persistent drone contradictions (a drone horse with no path horse within 5 m for ≥ 4 frames takes the one in-view path horse left over with no video observation): under the current labels, the candidate is determined in ≥ 80 % of frames on 5 of 26 persistent tracks (13 required; seed 0, video-only). In most frames the leftover horse has observations and its nearest drone point within 5 m is held by another observed horse: the conflict is in the video observations themselves, which elimination cannot settle.
- Soft legend-rank margin 0.6 → 1.13 m (the sd of a difference of two detections; hard part unchanged), against label alternation on one render track (screen on drone outcomes only, not on hand labels or identity score): full EM on the 8 affected races (seed 0) cuts the alternation by two thirds (188 → 62 label changes), but persistent drone contradictions stay (video-only 17 → 17, final 18 → 19), elimination-determined tracks 1 → 2 and 4 → 2, no gate change; on the 4 most affected tracks a blind silk read of the render avatar agrees with the track's new holder on 1 in the video-only labels (HV 05-20 R9's track goes to a third horse) and 2 in the final labels.
- Film detector (YOLOv8x, torch CPU) in a process pool plus painted-line detection in a process pool: `film.npz` identical on 4 of 4 CPU controls, but CPU 15.2 → 19.1 min on ST 05-09 R1 (wall 149 → 114 s); one measurement script hung with the pool workers idle after a fork (cause not traced). The workers meant to run 3 threads ran 8 (ultralytics resets torch to 8 threads at predictor setup, after the cap), so this does not reject a pool capped after the first predict (untested). The detector's boxes depend on the torch thread count: 1 thread differs from ≥ 2 by up to 9e-5 in box coordinates or confidence (2–8 threads bit-identical), so `film.npz` depends on the thread environment. At equal cores a capped pool gains at most the 13–16 % CPU that capping torch threads saved (see below).
- ONNX Runtime thread caps for the avatar detector: 1 and 2 intra-op threads agree with each other but differ from 4 and from the default by up to 1.2e-4 in box values (49 panels), so a cap changes `yolo.npz`. Torch thread cap on the YOLOv8x ruler/film detector (ultralytics sets 8 threads at predictor setup): bit-identical at 2 and 4 threads, CPU 150 → 131 / 126 s on 8 frames (−13 / −16 %) for wall 20 → 37 / 72 s.

The off-switches of adopted rules (negative evidence, blind cost, shadow release, anchor check,
detection EM, paint-pin override, fragment-forbid fraction, rail smoother, local refit) were removed.
Two off-states are measured: negative legend evidence off raised HV 05-20 R2's sectional 0.147 → 0.187 s;
the local refit off on the masked pose moved no race by more than 1e-4 s (kept for the straights'
residual after the camera swing). The others carry no recorded verdict as alternatives.
- Film detector as the OpenVINO export on CPU (FP32): CPU per call 18.7 → 17.0 s (−9 %) on ST 05-09 R1, leads identical; not worth a second backend. On the iGPU at the plugin's FP16 default, one lead moved 280 px and the 200 m crossing was lost (FP32 is the opt-in above).
