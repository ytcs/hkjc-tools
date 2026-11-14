# Race reconstruction — method and state

`hkjc.recon` turns one race's facts (`hkjc.data.race_info`) plus its four HKJC replay videos into
per-horse paths: `d(t)` metres to the finish along the racing rail and `lat(t)` metres outward of
the racing (movable) rail, on the race clock, with posterior uncertainty.

```python
from hkjc.data import race_info
from hkjc.recon import RaceRun
run = RaceRun(race_info("20260603", "HV", 8)); run.run(); run.paths()
```
CLI: `uv run --group vision python -m hkjc.recon 20260603 HV 8`
(stages cached under `data_cache/recon/<date>_<venue>_<no>/`; `--force stage,...` to re-run).

## 1. The feeds are proposals, the path is a posterior

Each video is an incomplete, noisy proposal of each horse's path; the path is reconstructed by
Bayesian fusion (`recon/fusion.py`) under physics + geometry priors, not by correcting one master
feed. Latent per horse on a 0.1 s grid: `s_k` (m run) and `l_k` (m off the racing rail). Prior:
acceleration ~ N(0, 1.2 m/s²) along, N(0, 0.25 m/s²) laterally. Likelihood: every observation at
its own timestamp and noise (linear interpolation between nodes). Geometry: `l` boxed to
`[0.3, width-0.3]` (outside the racing rail, inside the outer rail; width from the course). MAP is
an exact sparse solve (the stiff prior defeats iterative LS solvers) with an active set for the
bounds; posterior sd from the normal equations. Anonymous tracklets are associated by posterior
chi² (Hungarian-style, no same-feed time overlap) and the paths re-fused.

| Feed | Role | State |
|---|---|---|
| **aerial (AVV render)** | world frame + most observations | ✅ end-to-end |
| full (RF) — **finish-line camera** | finish ORDER + spatial GAPS, saddlecloth identity | ✅ in fusion |
| drone (real, one shot gate→finish, RF timeline) | start segment, bend coverage, independent lateral | 🟡 rails register; along-track open |
| patrol (head-on) | lateral on straights | ⬜ |

## 2. AVV render inversion (the world frame)

The render is a static textured 3D world seen by a smooth near-nadir virtual camera, so the camera
is recovered from the turf texture alone and the race is mapped once:

1. **features/pose** — unique frames (the render updates at **10 Hz** inside a 25 fps MP4); ORB on
   HUD/horse-masked turf; RANSAC similarity + **least-squares (Umeyama) refit**; pose graph of
   sequential, long-baseline (d=2/5/12/24) and loop-closure edges, robust similarity solve, mid-race
   gauge, outlier pruning. The solve is LINEAR: a 2D similarity is z -> A z + B (complex A, B), so the
   probe residual in frame-j pixels, (A_i x + B_i - A_j S x - B_j)/A_j, is linear given A_j (fixed
   point) — IRLS with Huber weights and a weak chain prior (weakly connected blocks otherwise collapse
   to zero scale). 0.7 s vs 23 s for the capped nonlinear solve, and better gates on the dev set.
2. **worldmap** — mosaic of all frames: horses average out; rails and painted markers are crisp.
3. **rails** — racing rail extracted once: stations along the field's inner edge, thin-ridge
   profiles (median-filtered along s against painted markers), stiff Viterbi DP, sub-px refine;
   then a robust world-space **despike** (points > 0.55 m off a local quadratic trend are false
   ridges — one captured R08's home straight and caused a #4/#8 tracklet swap), gaps bridged in
   WORLD coordinates (offset-space fill humps where the station scaffold jogs), and the rail frame's
   position + normals from a ~12 m local cubic fit (per-station snaps wobble ~0.1 m; noisy normals
   are amplified at wide lanes). Painted FINISH/50/100/150/200/300 markers → px/m + finish anchor.
4. **observe** — YOLO avatars (full panel; legend icon band, low-confidence HUD-band and static
   HUD-glyph detections rejected) → world → `(d, lat)` with the **odometer-tick distance
   calibration**: the display is a round-up readout (ticks V+10→V as the leader reaches V; convention
   c = 0, measured on two races), ticks alone give scale + finish (robust line + smooth spline for map
   scale drift); the painted FINISH pins d = 0 only when the marker grid is consistent with the ticks
   (|offset| < 3 m, ppm within 3 %) — the grid mis-assigned markers on two races → world-frame Hungarian
   tracking + stitching.
4b. **metric refinements** — the tick convention c is MEASURED per race on its own painted
   markers (Sha Tin ~ −3.2 m, HV ~ 0; a fixed c = 0 had produced a constant +0.3 s at every Sha Tin
   boundary), used only with a fully consistent pinned grid; a grid shifted by n markers
   (c_m ≈ −50n + c) is resolved; the grid is tick-SEEDED (scale prior = tick-zone scale / 1.035,
   the measured HUD/paint ratio) and accepted only with FINISH + ≥ 3 more markers; markers are
   profiled on a FINAL-PASS mosaic (they are drawn only near the end; multi-pass trips faded them).
   **Movable vs fixed rail**: course-table prior — if the innermost horses sit ≥ one course gap
   outside the extracted rail and a ridge exists at that gap toward the field, the extracted ridge
   was the fixed rail (HV B course) and the rail is moved out.
5. **identify** — legend running order (top 8) votes per rank on complete-field frames, plus
   elimination votes for ranks 9+, plus learned silk prototypes; no-overlap constraint; plus the
   **finish camera**: tracklets reaching the line are named by 1D registration of their crossing
   times to the camera's gap profile (well-separated finishers only) and by ELIMINATION (unlabelled
   line-reaching tracklets vs numbers with no line-reaching tracklet; assignment-level ambiguity
   test). Finish names are hard constraints through the EM pass.
6. **detection-level EM association** (`recon/assoc.py`) — frames are a continuous motion: every
   detection is re-assigned per frame by one joint Hungarian against all horses' fused posteriors
   (full Gaussian −2 log-likelihood incl. the log-det term, so vague out-of-view paths cannot steal
   detections; clutter option; legend rank constraint, two-sided where the rank is confident), then
   the paths are re-fused; the tracker fragments are only the initialisation. Two finish-specific
   rules: **negative evidence** — with camera recall r, detections in the last 15 m cost
   −2·log(1 − r) for horses the finish camera never read; **shadow release** — two tight paths within
   0.8 m along / 0.5 m lateral for ≥ 0.5 s in the last 3 s (a horse riding a neighbour's
   detections) lose the weaker horse's render observations there. **Negative legend evidence** — a
   horse missing from a fully read top-8 legend ranks below 8th: it cannot take a detection that is
   surely inside the top 8 (detections ahead + every undetected horse + 1 < 8) or that is more than
   8 m ahead of every listed horse's tight posterior (rail-projected position and the legend's
   ranking disagree by a few metres; 4/8/16 m equivalent on the ensemble, 0.6 m failed). These HARD exclusions (and the blind-horse
   near-line cost) travel with the anonymous fragments the E-step leaves unassigned (`Obs.forbid`,
   majority of the fragment's detections); otherwise fusion's χ² / orphan association hands the
   fragment straight back (HV 0520 #8).

Hard-won facts (do not re-learn): camera is near-nadir with **no tilt** but **slow zoom** (similarity,
not rigid); **LMEDS refits are biased** (integer FAST keypoints); **ECC refinement rejected**
(mowing-stripe aperture → scale drift); painted markers are **dynamic** (drawn only near the
finish); the first render frame after the live→render dissolve is unreliable; loop candidates must
be matched **unguided**; projection must be **time-consistent** on multi-pass trips; HUD-band
detections need a confidence gate; the official-silk-image prior fails (1/8; domain gap); the
odometer is a good *relative* distance but its convention must be measured, not assumed.

## 2b. Finish-line camera (RF/drone shared finish replay)

A fixed overhead camera on the line, in slow motion with a VARIABLE playback rate (unique-frame
index is not linear in time near the winner — temporal gaps rejected). Saddlecloth numbers (yellow on
dark) are OCR'd per unique frame; gaps are SPATIAL at an instant (`gap_h = gap_r + x_h − x_r` over
frames shared with referenced horses — horses occluded exactly at the line are recovered). Orders match
official on the HV dev races (10/11, 9/12, 12/12 with one 0.0x s swap). Into fusion: each horse crosses
d = 0 at `t_w + gap/C`; `t_w` and `C` (px/s) fitted jointly against the render-fused arrivals with a
physical prior C = fused speed × px/m (isolated COCO horse boxes, nose-to-tail 2.6 m; measured
~15-20 % high). Per-horse sd is method-derived and CONDITIONAL on the fitted clock (the shared clock
error must not be added per horse). The label-based anchor is sanity-checked against a LABEL-FREE
registration of anonymous tracklet crossings (C within ~15 %, clock 0.10-0.24 s early: the render
leads near the line); on disagreement the registration is used, and if both fail no finish
observations are emitted. Near the line the render updates at only ~2-4 Hz with ~500 px camera jumps
(poses stay clean); crossings must be bracketed by samples < 0.3 s apart.
**Sha Tin** uses a SIDE-ON finish camera with WHITE cloths and dark digits (HV: overhead, yellow on
dark): digits are found by black-hat contrast with a white ring, grouped for two-digit numbers;
horses seen crossing the line are always kept (the first crosser used to be dropped), and the
white-cloth reader runs only at Sha Tin (it added false reads on HV's overhead camera).

## 2c. Drone (open)

Physical look-at camera (C, P, heading, f) fitted to labelled world rails (racing; outer = racing +
course width; fixed inner = racing − (A width − course width)) with bidirectional truncated residuals,
coarse-to-fine, multi-start: the rails register correctly (all three lines land on real rails), but on
straights the along-track tilt is unobservable from parallel rails and horse ICP does not lock
(render horses sit outside the drone jockeys; lateral disagreement unresolved). The render pipeline
itself is faithful (rail-hugger body centres 2.3 m off the rail in the render pixels) — the render's
own lateral offset vs reality is the open question the drone must answer.

## 3. Acceptance gates (fixed; validation-only use of official data)

leader timing MAE ≤ 0.25 s · per-horse sectional MAE ≤ 0.25 s (only where posterior sd < 1.5 m) ·
finish Spearman ≥ 0.90 · winner correct · time-behind-winner MAE ≤ 0.20 s; declared before the
held-out races: sectional coverage ≥ 80 % · fused path for ≥ N−1 runners · finish ρ over ≥ N−2 ·
lateral scale (racing→fixed rail gap on the map vs the course table, where visible) ≤ 0.75 m.
Development races (tuned on): R08, R03, R02A, HVB. Held-out: HV 2200 + the five Sha Tin races.
`recon/validate.py`; results per race in `data_cache/recon/<key>/validation.json`.

Validation set: HV C 1800/1200 night, HV A 1650, HV B 1650,
HV C 2200 good-to-yielding, ST C 1600 day, ST AWT 1800 wet-slow, ST 1000 straight, ST B 1800 soft,
ST AWT 1200 night.

## 3b. Held-out validation (batch F, 2026-11-13)

Eight races never inspected during development, chosen by a fixed rule from racecard metadata
(earliest race per condition category on meetings not used in development), run once on frozen
code. **Happy Valley: 2 of 3 fresh races pass every timing gate** (HV B 1200, HV C+3 1650); HV A
1000 passes the identity-free leader gate but not sectional/order — a small sample. **Sha Tin: 0 of
5** (A 1200, A 2000, C+3 1400, AWT 1800, 1000 m straight) pass the identity-free leader gate. A
HUD-only check (odometer tick frame on the render clock vs the official leader time) shows two
causes [AMENDED 2026-11-14: the HUD-only number also contains each race's odometer->paint convention
offset (-3..-6 m, ~0.2-0.35 s), and on ST4/ST9 our painted-marker calibration FAILED (render frames
lost to the legend gate, 3d) -> ticks-only metric; with markers found, ST4's calibrated leader error is
0.13/0.08/0.17 s vs HUD-only 0.66/0.51/0.31 s. Remaining leader failures: ST9 0.5-0.7 s, ST1 0.69 s at
its first boundary, ST3 1.66 s at 400 m (straight chute)]: (a) on some races the render ITSELF disagrees with the official record by 0.3-1.1 s,
DECLINING along the race (e.g. 2026-04-26 R9: 0.86 -> 1.11 -> 0.82 -> 0.50 s) — not a constant clock
offset; a geometry-only metric (marker scale, no ticks) is far worse on every race (the map scale
drifts along the trip), so the ticks stay the metric and no render-side CV can remove this; it needs an
independent reference from the real-camera feeds; and (b) our own errors on top (AWT
detection/rails, the straight chute's weak rail). Lateral position has no working gate yet
and is unvalidated.

## 3c. Clean frozen run (tag `recon-freeze-1`, 2026-11-14) — turf scope

Scope: **turf only** (owner, 2026-11-14); the all-weather (dirt) track is out of scope for now — its
failures are render-side (detections lost for seconds, the rail framed under the legend band,
unreliable markers). One end-to-end run of the tagged commit on a pod (18 races in 37 min; CUDA
detection), 15 turf races: all five timing gates pass on **9/15** — 7 of 8 Happy Valley races (incl.
the three fresh ones) and Sha Tin ST6/ST8. The final product matches the official times to
0.03-0.19 s on every race. Failures: Sha Tin fresh races fail the identity-free leader gate where the
render's own HUD disagrees with the record (0.3-1.1 s, declining along the race); finish identity is
FRAGILE — on fresh caches HV 0429's winner flipped and ST7's order collapsed (rho 0.53) though both
passed on the development caches; the path-count gate fails on 5 races (tailed-off horses out of
view). Lateral sanity (innermost horses 1.6-2.8 m off the rail) holds on every oval race; both
straight-course races are wrong (0.2-0.6 m). The sealed held-out batches stay sealed.

## 3d. Render-gate and finish-anchor fixes (tag `recon-baseline-O`, 2026-11-14) — turf, development races

Two bugs found by checking the race film (left half of the aerial replay) against our paths:
- **Render-active legend gate.** The legend sits higher on some layouts (~0.67 of the frame on ST
  0419 R4, 0426 R9); only its boxes' bottom pixels entered the 0.70-0.86 test strip, so whole
  stretches of render were rejected: ST9 lost ~13 s of race, ST4 2 s at 432 m out (every tracklet
  broke; fusion extrapolated them). Fallback: >= 5 legend boxes in a strip widened to 0.60, allowed
  only after the strict test has passed once in the replay (pre-span frames are dissolve tails).
- **Finish anchor fitted to extrapolated paths.** Only crossings where the fused path is tight
  (sd <= 3 m) now fit the finish clock and C; the line speed for the C prior likewise.

15 turf races, full pipeline on a pod (RTX 4090, 760 s): all five timing gates on **10/15** (before:
10/15 on the freeze-1 caches with the anchor fix, 9/15 at freeze-1); timing gates passed **65/75**
(61); winner correct **15/15** (13). ST4 0 -> 3/5 and ST9 1 -> 2/5 (both winners now right; ST4's
identity-free leader gate passes). **HV 0520 5 -> 4/5** (sectional 0.205 -> 0.264 s): on the common
(horse, boundary) pairs v2 is no worse (0.169 vs 0.154; median 0.127 vs 0.140) — the jump is newly
scored crossings of #8, mislabelled in both runs (fixed in 3e). The old frames' result was itself
fragile: their legend-gate hole (39.5-42.5 s) is weakly tied in the pose graph, and moving the
fixed frame shifts it by up to 2.4 m (sectional 0.283 s). HV 0527 is bit-identical to a
control run without the gate change.

Race film: in the straight it is a front-on camera that sees the painted 400m..50m stripes. Leader
crossings (`recon/film.py`, line tracks that sweep up the image) match official 400 m times within
-0.12..+0.09 s on 6 races; the render HUD lags them 0.4-0.9 s at Sha Tin. Not wired into fusion: it
covers only the last 400 m, and replacing that boundary's error with the film's ~0.1 s would flip at
most one race's leader gate. The early-race render lag dominates.

## 3e. Identity: negative legend evidence (tag `recon-baseline-P`, 2026-11-14) — turf, development races

Judged on a **perturbation ensemble** (drop 1 % of detections at observe, 5 seeds + unperturbed;
HV 0520, ST9, HV 0527, ST6): single-run gate moves inside a race's spread are not evidence (ST9's
rho spans 0.86-1.00 across seeds). HV 0520 sectional 0.269 [0.252-0.314] -> **0.147 [0.137-0.162]**
(all five timing gates on every seed; #8 no longer rides a front-runner's fragment); ST6
behind-winner 0.037 -> 0.019; ST9 / HV 0527 unchanged. The 15 turf races (fuse+validate on the
`recon-baseline-O` stage artifacts): all five timing gates on **11/15**, timing gates **66/75**, winners
15/15; declared gates: no regressions, +2 (ST4 path count, HV 0520 sectional). Ensemble on races the
rule was not designed on: ST4 sectional 0.475 -> 0.413 (rho median 0.94 -> 0.90, range narrower:
[0.79, 0.95] -> [0.90, 0.94]); ST7 0.221 -> 0.178 (worst seed 3/5 -> 5/5). ST9 has 8 runners: the
top-8 legend is the whole field and the rule never applies. All 15 races are now DEVELOPMENT races
(inspected); held-out evidence must come from the sealed batch H. Rejected on the
way: the 0.6 m floor margin (HV 0603 R3 lost #10/#5, closing from 9th-10th). Majority vs union of a
fragment's exclusions: indistinguishable (ensemble).

## 3f. Leader-timing error shape (diagnostic, 2026-11-14) — no code change

Signed identity-free leader error (recon − official) on the 15 turf races: the SAME shape on every
race, HV included. It is late at the early boundaries and early at the line (pooled mean by
distance-to-go: 1600 m +0.43, 1200 +0.24, 800 +0.12, 400 +0.21, 0 m −0.16 s; 13/15 negative at the
line). Ruled out:
- **Clock rate:** the HUD clock is 25.000 fps on all 15.
- **Rail configuration:** ST1 and ST9, the worst, are both course A.
- **Render frame holes:** none > 0.4 s inside any race; the render starts at race clock 10.8–12.6 s.
- **Render camera change near the line:** box size and aspect and pose scale are flat over the last 400 m.
- **False positives in the raw leader:** the trace is smooth through the line.

Against the race film (non-official, non-HUD), recon is **flat at ~+0.2 s from 400 m to 100 m**
(pooled medians +0.22/+0.24/+0.19/+0.14/+0.18; film − official at 400 m within ±0.12 s), then
steps to −0.16 s at the line, mostly inside the last 50 m. Beyond 400 m there is no reference that
is neither official nor the odometer. The odometer-only leader error is late too (ST9
+0.86/+1.11/+0.82/+0.50 s), but that comparison is circular for the rendered horses: recon distance
is fitted to the same ticks.

**The paint pin causes the step.** With the ticks unpinned (no paint FINISH pin, no paint-derived
convention offset), recon − film becomes one flat offset per race that continues through the line
(e.g. ST9 +0.52…+0.63 s in the straight, +0.53 at the line; one exception, HV 0415). The pin
stretched the last ~50 m: ST4 leader speed near the line is 17.0 m/s pinned vs 15.2 unpinned, and
the film's 100 m → line average is 15.7. So the film supports the tick scale in the straight, and
the render's painted FINISH is the wrong anchor.

**Film-anchored metric, built but OFF by default** (`HKJC_FILM=1`; `film` stage +
`film.anchor_offset`): ticks unpinned, minus one constant per race (+2.4…+11 m) that puts the render
leader on the film's marker crossings, with the paint pin as fallback below 3 crossings. Against a
clean control (15 races; 6-seed ensembles on ST4/ST3/ST1/ST9/HV 0520):
- Robust gains: ST9 timing 2 → 4/5, sectional 0.455 → 0.232 s; ST1 sectional 0.36 → 0.25 s, leader
  0.393 → 0.253 s (still a failure); ST6 0.157 → 0.080; ST8 0.236 → 0.105. Sectional better on
  10/15; all-five timing 11 → 11/15.
- Consistent regression on a declared gate: ST4 loses 2–3 horses' paths on every seed and ST1 4–5
  (control 0–2). ST4 sectional 0.41 → 0.49 s.

Those horses have no labelled tracklets, so they hang on finish-camera registration and
orphan-chain matching. Both flip when the line speed changes, which moves the C prior and the
anchor clock. A fixed-grid registration changes the control's ST4 naming too: ST4 registration is
ill-posed because of the "3"/#7 misread. Not fixed, so the film anchor stays off. **Diagnostic
only:** official finish gaps put the true camera C at 12.3–15.7 m/s × camera px/m (median ~13.4) on
14 races. The horse-ruler prior (line speed × px/m) therefore overstates C by ~15–20%. This must
not be used to set the ruler or any prior.

**Declared before batch H (known failures, not fixed):**
- The odometer-derived metric disagrees with official beyond the marker zone (leader late by
  0.2–0.9 s at early boundaries on some races), cause not established.
- The leader is ~0.16 s early at the line, caused by the render-paint FINISH pin (above). The fix
  exists but is off by default, because it costs horses' paths on ST4/ST1.
- ST 1000 m straight: weak rail/metric (ST3 1.66 s at 400 m).
- Finish-camera misreads of tailed-off horses (ST4 "3" is #7; ST7 #11).
- ST4 finish ρ sits on the 0.90 gate line (ensemble median 0.90).
- Path-count gate fails where tailed-off horses leave the render.

## 3g. Batch H protocol (declared BEFORE the run, 2026-11-14)

- **Code:** tag `recon-freeze-2` (this commit). Default configuration: film anchor OFF.
- **Races (sealed, never inspected):** 11 turf races chosen earlier by a fixed rule (journal 0135).
  - Sha Tin 2026-05-24 R1 (A 1600), 05-31 R1 (B 1200), 05-09 R2 (C 1200), 05-17 R1 (C+3 1800),
    05-09 R1 (1000 m straight). The AWT race is out of scope.
  - Happy Valley R9 on 2026-04-15, 04-22, 04-29, 05-13, 05-20, 05-27.
- **Run:** once, the full pipeline from scratch on a RunPod GPU pod. Code is fetched by tag; videos and
  data are re-downloaded. No change after seeing results; failures are reported as failures.
- **Primary result:** the fixed gates in `recon/validate.py` (five timing gates plus the declared
  coverage, path-count, finish-n and lateral gates), per race and as counts.
- **Secondary arm (declared now):** `HKJC_FILM=1` observe → validate on the same stage artifacts,
  scored the same way. It answers whether the film anchor's dev-set gains and losses transfer.
- **Declared diagnostics:**
  - Signed leader error per boundary.
  - HUD-only leader error.
  - Innermost-horse lateral p5.
  - Final-product residual vs official.
  - Finish-camera C vs its physical prior.
  - Recon − film at the markers (film arm).
- **Expected from the development set (not a gate):**
  - Sha Tin leader late at early boundaries.
  - The 1000 m straight fails (weak rail/metric).
  - Path-count failures where horses tail off out of the render.

## 4. Open / next

- **Robustness yardstick** — judge changes on perturbation ensembles, not single runs (ST9's rho
  spans 0.86-1.00 across 1 %-detection-drop seeds). The HV 0520 flip under a gauge shift was a
  LOCALIZED pose instability in a legend-gate hole (fixed in 3d), not global chaos.
- **Drone** — resolve along-track (seed on bend frames, track forward from the previous camera; or a
  two-scalar field centroid/length fit), then use it for the start segment, draw-seeded identity
  and the independent lateral check.
- **Identity coverage** — fragmentation on some races; legend votes only count complete-field frames
  (could use fused positions to fill missing horses, EM-style).
- **Start segment** — the render begins ~150 m in; early sectional boundaries lack tight posteriors
  (coverage gate) until the drone covers the start.
- **Non-overlap** between horses in fusion (sparse QP, e.g. OSQP + sequential convex programming).
- **Detection under overlays** (disclaimer text, legend band) — retrain the avatar YOLO on
  track-interpolated pseudo-labels (GPU via RunPod if needed); SAHI-style slicing for the drone.
- **ST 1000 m straight**: no turning → the inner side must come from the course, not the path.
- **Legend identity** is the dominant remaining error (long tracklets carry one wrong label for a
  whole race: HV 0527 k3 "#8" is #12). Fusion's anchor-conflict demotion helps order but must stay
  horse-specific (median-subtracted) and never strip a horse below half its render data.
- **Odometer HUD vs painted markers**: in the 0-400 m zone the odometer tick scale is 1.3-4.8 % above
  the paint scale in px/m (all 15 turf races), i.e. the odometer counts FEWER metres than the paint
  over the same span (earlier wording "more metres" was inverted); cause unknown. Recon d sits between
  the two (1-2.4 % short of the paint at the 400 m marker). The race film supports the TICK scale in
  the straight (unpinned recon − film is flat; 3f).
- **Film-anchored metric** (3f): turn it on once finish-camera identity no longer hinges on marginal
  orphan-chain/registration matches. Candidates: give finish-read horses with no tracklets a path
  seed so the EM association can place them, or make registration robust to tailed-off misreads.
- **AWT** — OUT OF SCOPE for now (turf only). Known: head/shadow duplicates fixed by an anisotropic
  merge; far-side avatars on bends missed; the back-straight rail is framed under the legend band.
- **Real-camera feeds (drone, patrol) + SAM 2** — the only independent source of positions: needed for
  races where the render disagrees with the record, for lateral validation and for the start segment.
  SAM 2 video segmentation seeded from fused render positions (GPU/pod) is the planned approach.
- **Finish-identity robustness** — results flip with small detection differences (HV 0429, ST7).
