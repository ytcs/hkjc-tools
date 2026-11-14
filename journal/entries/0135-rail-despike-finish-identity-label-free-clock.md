---
title: Rail false-ridge despike fixes a #4/#8 identity swap; clear-leader odometer bug; finish-camera identity (registration + elimination); label-free clock sanity check
id: 0135-rail-despike-finish-identity-label-free-clock
author: recon
kind: finding
tags: [cv, recon, rails, identity, finish-camera, odometer, fusion]
created: 2026-11-13
---
## Rail (look at the pixels)
- **False ridge captured by the Viterbi rail (R08 home straight, ~200 m):** a faint line ~1.7 m inside
  the track scored as a *supported* ridge for ~40 stations -> the magenta rail dipped into the track,
  the normals bent, (d, lat) jumped for horses there, and two adjacent tracklets (#4/#8) effectively
  swapped (render tracklet labelled #8 crossed the line 0.32 s early; the finish camera's own gap for
  #8 was right to 0.06 s). Poses were checked clean (posecheck blends single-lined). FIX
  `rails.despike_world`: supported points > 10 px (~0.55 m) off a local QUADRATIC trend
  (Savitzky-Golay, 241 stations ~ 26 m) are demoted and bridged. REJECTED: Gaussian trend
  (mode='nearest' flattens curvature at the array ends -> demoted the good rail at the finish and
  shifted the marker grid by one marker, finish 0.41 s off).
- **Unsupported gaps** were interpolated in station-OFFSET space; where the station scaffold jogs this
  makes a world-space hump. FIX `rails.fill_world` (bridge in world coordinates).
- **High-frequency wobble** (R02A ~0.10 m rms, p99 0.35 m, ~9 m wavelength; R08 0.05 m): per-station
  sub-px snaps with only ~0.4 m smoothing; normals from differentiating it were noisy (cyan 5/10 m
  guides wobbled, lateral/d noise for wide runners). FIX `RailFrame.from_points`: position + normals
  from a local cubic fit over ~12 m. Before/after reprojection sent to the owner.

## Distance metric
- **Clear-leader bug:** the per-frame leader position dropped the front detection whenever it was
  > 8 m clear (meant to reject false positives). A genuinely clear winner (R02A, 13 m) was dropped ->
  odometer ticks lagged 15 m -> finish 1.06 s early. FIX: drop only if the gap is NOT persistent
  (median over 11 frames). Leader timing R02A 0.352 -> 0.110-0.137 s.
- **Marker pin gate:** the zone tick scale vs marker scale. MEASURED on R08/R03/R02A: over the final
  200 m the HUD odometer counts 2.7-4.3 % more metres than the painted markers, while the global tick
  scale and the zone offset agree (< 1.5 m). Same-span check done (marker ppm 0-200 m = full-grid ppm),
  so a span/zoom mismatch is RULED OUT; a property of the HUD. Mis-assigned grids are off by tens of
  metres / > 20 %, so the offset is the discriminating test; scale tolerance 8 %.

## Finish camera
- `anchor()` accepted the physical C prior but **never used it** (C drifted 2395 vs prior 1942). Now a
  residual; per-horse sd is method-derived CONDITIONAL on the fitted clock (scale term + gap spread).
  The marginal sd (incl. the shared clock error) is wrong for per-horse-independent fusion: it buried
  the precise relative order (R08 rho 0.93 -> 0.55). sd_s is in METRES (time sd x line speed) — the old
  0.05 was 0.05 m (~0.003 s).
- The physical C prior (COCO horse box 2.6 m ruler) is biased ~15-20 % HIGH (true ~1700-1750 px/s on HV).
- **Label-free registration** (`finishcam.identity_match`): 1D registration of anonymous tracklet
  line-crossings to the camera gap profile. MEASURED on 4 HV races: C within ~15 %, clock 0.10-0.24 s
  EARLY (the render leads near the line). Used as a sanity check/fallback for the labelled anchor
  (> 0.4 s or > 30 % disagreement -> registration; both failing -> no finish obs). R02A behind-winner
  0.74 -> 0.08-0.14 s.
- Near the line the render updates at only ~2-4 Hz and the virtual camera jumps ~500 px between unique
  frames; poses remain clean. Crossings must be bracketed by samples < 0.3 s apart (else extrapolate
  from the last pre-line sample) — R02A crossings were quantized at the first post-gap frame.
- **Finish identity:** (a) unambiguous well-separated matches (absolute separation >= 0.15 s; the
  relative test alone renamed legend-#2 as #10 on packed R08); (b) ELIMINATION: unlabelled
  line-reaching tracklets vs numbers with no line-reaching tracklet, accepted when forbidding the pair
  raises the optimal assignment cost by >= 0.15 s (pairwise test REJECTED: blocked #1/#5 0.16 s
  apart). Finish names are hard constraints through the EM identity pass (EM had silently dropped
  them). R08: back-markers #1/#5 now have paths; 11/11.
- Fusion `_anchor_conflicts`: a numbered render tracklet whose removal halves a > 4 sigma conflict with
  an independent anchor is demoted to anonymous. CAVEAT (advisor): assumes the anchor is right; a
  common-mode render bias vs official anchors can strip correct tracklets — needs an on/off sweep on
  all races and a horse-specific (median-subtracted) residual before trusting it.

## State (dev races, video-only posterior, fixed gates)
R08 HV C 1800: all PASS except lateral-scale (premise invalid) — leader 0.161, sectional 0.148 @ 89 %,
rho 0.99 (n=11), behind 0.019 s, 11/11 paths. R03: all timing PASS, 2 path-less. HV 0527 R3: timing
PASS, coverage/finish-short fail. R02A: leader/rho/winner/behind PASS; sectional 0.40 and 4 path-less
(legend identity 48 %: the clear leader is often unnumbered). Run C (pre-fix code) numbers are stale
for rails->validate; a clean all-10 rerun + on/off toggles is next. The ST/HV22 races are now dev-2
(fixes were made after seeing failures); a fresh never-inspected batch is needed for a held-out claim.

## Addendum: fast loop + baseline D; Sha Tin render lag
- `scripts/dev_loop.sh`: re-run only the changed stages, races in parallel (JOBS, default 2 — the
  15 GB machine is shared). 10-race rails->validate in 95 s at JOBS=4 (was ~15-20 min sequential).
- Baseline D (10 races): R08 all timing PASS; R03 timing PASS except rho 0.87; R02A/HV0527 fail on
  identity (sectional, path-less horses); all Sha Tin races FAIL the identity-free leader gate.
- **[CORRECTED below — this was the tick convention, not a render lag]** Sha Tin render lag: ST6/ST8/ST7 leader errors are a near-CONSTANT +0.3 s at every
  boundary (ST6 [.35 .18 .30 .37], ST8 [.27 .36 .41], ST7 [.48 .41 .25 .31]); the render clock reads
  cleanly (25.000 frames per clock-second, zero offset spread), so this is the render's POSITIONS
  lagging official timing by ~0.3 s (~5 m) at ST — not a clock-OCR or distance-scale bug. HV ~0.
  Calibrating it from official times would break the validation-only rule; the independent fix is a
  real-video timing reference (drone / side-on footage). Open.

## Addendum: detection-level EM association (`recon/assoc.py`) — the main identity lever
Frames are a continuous motion, so identity is no longer fixed on tracker fragments. Each iteration
(<= 8, stop at < 0.2 % changes): per frame, ONE joint Hungarian of all detections to all horses (+ a
clutter option, cost 16) with cost = full Gaussian -2 log-likelihood vs each horse's fused posterior
(chi^2 + log-det ratio: a VAGUE out-of-view path must pay, or it steals nearby detections — HV0527
#9) + a ONE-SIDED legend rank constraint (>= r detections clearly ahead => cannot be the legend's
rank-r horse; undetected horses only push ranks down); then re-fuse. Fragments = initialisation only.
Finish-camera C prior loosened to +-30 % (the horse ruler is 15-25 % high; at 10 % it compressed
tail-of-field gaps: R02A #9 -1.35 s).
10-race result (vs baseline D): R08, R03 ALL timing PASS; R02A sect 0.39 -> 0.23, rho 0.99; HV0527
sect 0.25, rho 0.90; HV0429 rho 0.58 -> 0.99; ST8 rho 0.43 -> 1.00, behind 0.07; ST7 rho -0.6 -> 0.1.
Remaining failures are mostly UPSTREAM of identity: ST render lag (0.3 s), HV0429 metric (leader
0.40 s), AWT detection, ST finish-camera recall, tailed-off out-of-view horses (path-less).

## Addendum: markers on multi-pass trips; map-scale drift (HV 0429, 2200 m)
- `fill_world` clamped unsupported stations at the array ENDS to a constant point (np.interp) ->
  zero tangents / NaN normals near the finish. Now interior gaps only.
- Painted markers are drawn only near the END of the race; on HV 2200 the finish area is also
  covered in lap 1, so the all-frames mean mosaic faded them below detection. Markers are now
  profiled on a FINAL-PASS mosaic (frames whose time-ordered rail station lies in the last 9000 px),
  cropped to those frames' bounds (a full-extent float mosaic OOM'd R03 under its 3 GB cap).
- With crisp markers HV 0429 still fails the grid fit: marker spacing implies ~23-24 px/m locally
  (inconsistent between markers) vs 21.2 px/m from ticks over the trip -> MAP-SCALE DRIFT over a
  1.5-lap trip. Not a tolerance problem: the parametric rail fit with the published lap length and
  exact 50 m marker spacing (owner's point: HKJC geometry is known, rails do not wobble) is the fix.
- ST 1000 m straight: racing-rail support only 18 % -> its gates move with any rail change.


## Addendum: CORRECTION + metric/rail fixes (tick convention, seeded grid, movable rail)
- **The "Sha Tin render lag" was WRONG.** It was the odometer tick convention: c = 0 had only been
  measured on two HV races. Measured per race on its own painted markers (in-video, HUD vs paint):
  Sha Tin c ~ -3.2 m (ST6 -3.2, ST7 -3.3), HV ~ 0. Using it (only with a fully consistent, PINNED
  grid): ST6 leader 0.30 -> 0.11 s, sectional 0.32 -> 0.16; ST7 leader 0.36 -> 0.15, sect 0.39 -> 0.24.
  A grid shifted by n markers shows c_m ~ -50 n + c (ST8 -52.9 -> n=-1, c=-2.9): resolved, not
  rejected -> ST8 leader 0.35 -> 0.20. On AWT the grid is unreliable; applying its offset unpinned
  made ST5/ST6-0401 worse, hence pin-gated.
- **The HV 0429 "map-scale drift" claim was WRONG**: the unseeded grid took a coverage-edge artefact
  (-159 px) as FINISH; at low thresholds the true markers sit at a consistent ~20.8 px/m. The grid is
  now tick-SEEDED (scale prior = tick-zone scale / 1.035 measured HUD/paint ratio, sd 3 %; finish
  within +-60 m of the ticks') and accepted only with FINISH + >= 3 more markers. HV 0429 remains
  ambiguous (ticks ~22 px/m locally vs ~20.8 markers, implied convention +9 m) -> ticks only.
- **Wrong rail on B course (HV 0527):** the strongest ridge was the FIXED inner rail, ~3.5 m inside
  the movable rail -> every lateral inflated ~3.5 m (invisible to timing gates; seen in pixels).
  Course-table prior: `rails.movable_rail_check` — if the innermost horses sit >= one course gap
  outside the extracted rail AND a supported ridge exists at that gap toward the field, move out and
  re-snap. HV 0527 -> ALL timing gates PASS (sect 0.18, rho 0.98, behind 0.06). Fires nowhere else
  (ST7 B: innermost horses ~2 m outside -> already the movable rail). REJECTED: an exponential
  nearest-to-field preference in the DP (pulled the rail through the field on the home straight).
- Parametric HKJC-geometry rail DEFERRED: the render's own geometry does not match the course table
  (fixed-rail gap measured 5.9 m vs ~8 m expected on R08), so a hard geometry prior is unvalidated;
  smoothness + course-gap prior are in.

## Addendum: EM legend term two-sided; Sha Tin finish reader v2
- E-step legend term is now TWO-SIDED where a detection's rank is confident (detections ahead ==
  horses whose tight posterior is ahead): taking a legend horse of a different rank costs 9 (3 sigma).
  The one-sided form only forbade, never attracted (HV 0527 '#9' sat in 5th where the legend said #8).
  R02A sectional 0.296 -> 0.247 (PASS); R08/R03/HV0527 stay all-timing-PASS.
- Sha Tin side-on finish camera, white cloths: digits = dark strokes INSIDE the closed white cloth
  (closing fills the digit holes). REJECTED: fixed v<100 (fragments blurred strokes) and black-hat
  (lights up whole dark horse bodies on green turf; ST7's wide shot gave 0 candidates). Digit height
  floor 16 -> 9 px (ST7 wide shot).
- Finish scan cost: OCR of ~20 candidates per unique frame ~12 s/frame under load -> OCR only
  cloths within +-700 px of the line; batched recognition call.
- Verified in pixels (advisor): EM labels follow the legend order on HV0527/R02A at 600 m; final
  product honours the official records to 0.12-0.14 s (final_official_resid_mae).

## Addendum: camera misreads; checkpoint E
- ST finish camera misreads (1/4/14-type) put a number ~1.4 s from where it finished (ST6 "#14").
  In-camera coincidence test (two numbers' tracks overlapping in x,y) added but does not catch it
  (the true number is often unread). Cross-feed rule: a camera track with < 10 reads AND a > 0.3 s
  disagreement with the render-fused crossing AND a well-determined anchor (>= 5 horses) gets its sd
  inflated to the disagreement. REJECTED: blanket inflation on any disagreement (on HV the camera is
  right: R03 lost its winner); inflation with a 3-horse anchor (ST8 lost its winner).
- Checkpoint E (10 races, video-only posterior): ALL timing gates PASS on R08 (HV C 1800 night), R03
  (HV C 1200 night), HV0527 (HV B 1650), ST6 (ST C 1600 day), ST8 (ST 1000 straight); R02A (HV A) 4/5
  (behind 0.27); HV0429 (HV 2200) 3/5 (metric ambiguous); ST7 (ST B soft) 1/5 (camera misses winner);
  AWT ST5 1/5, ST6-0401 0/5 (AWT detection + camera). Code: data_cache/recon/checkpoint_E_code.tgz.

## Addendum: speed — Phase 1 (EXACT changes only, artifact-verified)
Idle profile of one race (R08, 970 unique render frames): frames 54 s, features 86, pose 190,
worldmap 24, detect 152, hud 99, finish ~300-600, rails->validate ~25 => ~15-20 min idle (the 35 min
seen earlier was contention). The current code is fully DETERMINISTIC (re-run noise floor = 0), so
"no accuracy loss" is tested as bit-identical artifacts (`scripts/compare_artifacts.py`).
NOTE: the dev caches' poses came from an OLDER pose version; a reference re-run (features->validate,
frozen current code) is the new baseline for all speed work.
Exact changes (each verified bit-identical on R08):
- features: per-frame ORB in a fork pool + step matches in the pool (OpenCV RANSAC seeds its RNG per
  call, so worker order cannot change results).
- pose: sequential guided matches precomputed in the pool (only chain-guided steps stay sequential);
  pool sized to the cores (was 6); guided-match gate check vectorised (same first-hit rule).
- detect: frames in a pool (one ONNX session per worker).
- hud: two passes — the re-read rules depend on pixels only, so crops are chosen first and OCR'd in
  large same-size batches (identical hud/legend).
- run(): hud and finish run as BACKGROUND processes concurrent with the render chain, joined before
  their first consumer (observe / identify).
- video decode frame-threaded (same pixels).
Under 60 s per race is NOT reachable exactly on this CPU (detect alone is 81 s of ONNX inference);
Phase 2 (approximate: half-res ORB, fewer graph edges, sparse solve, coarser mosaic, GPU) only after
batch F, validated on dev + F artifacts.
