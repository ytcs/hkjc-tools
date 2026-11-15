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

## 3h. Batch H result (held-out, `recon-freeze-2`, run once on 2026-11-14)

The run followed protocol 3g on a RunPod RTX 4090: 11 races, 1310 s for the primary run and 1415 s
total. Nothing was changed afterwards.

**Primary (default code): all five timing gates pass on 9/11 races.** That is 51/55 timing gates,
with the winner correct on 11/11. Sha Tin passes 4/5; batch F's five fresh Sha Tin races (one of them
AWT) passed the leader gate 0/5. Happy Valley passes 5/6. **Scope caveat:** the six HV races are R9
of meetings that each already have a development race, so they are unseen RACES, not unseen
meetings. Only the five Sha Tin races are meeting-disjoint.

| race | leader s | sectional s (cov) | ρ | behind s | paths short | finish short | lateral m |
|---|---|---|---|---|---|---|---|
| ST 05-24 R1 A 1600 | 0.217 | 0.211 (89 %) | 0.96 | 0.090 | 0 | 4 ✗ | n/a |
| ST 05-31 R1 B 1200 | 0.134 | 0.205 (97 %) | 0.96 | 0.054 | 2 ✗ | 3 ✗ | 2.15 ✗ |
| ST 05-09 R2 C 1200 | 0.168 | 0.155 (92 %) | 1.00 | 0.030 | 0 | 3 ✗ | 0.43 |
| ST 05-17 R1 C+3 1800 | 0.108 | 0.167 (99 %) | 0.98 | 0.030 | 0 | 1 | 0.07 |
| ST 05-09 R1 1000 straight | **0.388 ✗** | **0.369 ✗** (90 %) | 1.00 | 0.016 | 0 | 2 | n/a |
| HV 04-15 R9 A 1200 | 0.105 | 0.171 (100 %) | 0.98 | 0.037 | 2 ✗ | 2 | n/a |
| HV 04-22 R9 B 1200 | **0.421 ✗** | **0.431 ✗** (97 %) | 1.00 | 0.040 | 0 | 1 | 0.47 |
| HV 04-29 R9 C 1650 | 0.124 | 0.109 (98 %) | 1.00 | 0.015 | 0 | 1 | 2.10 ✗ |
| HV 05-13 R9 C+3 1650 | 0.127 | 0.201 (98 %) | 1.00 | 0.013 | 0 | 0 | 1.28 ✗ |
| HV 05-20 R9 A 1200 | 0.191 | 0.248 (92 %) | 0.99 | 0.050 | 0 | 0 | n/a |
| HV 05-27 R9 B 1200 | 0.206 | 0.179 (92 %) | 1.00 | 0.072 | 0 | 3 ✗ | 0.41 |

Both timing failures match failures declared in 3f:
- **HV 04-22 R9:** the leader is late (+0.57 s at 800 m, +0.53 s at 400 m, i.e. partly inside the
  marker zone). It matches the declared class by shape; cause not established. The odometer-only
  error, +0.36/+0.75 s, is circular evidence, as 3f notes.
- **The 1000 m straight:** +0.75 s at 400 m.

3g's expectation of Sha Tin leaders late at early boundaries did NOT show up at Sha Tin. First
boundaries were +0.11 to +0.23 s on the four oval races.

The other declared gates:
- Coverage passes 11/11.
- Path count fails 2/11.
- Finish-n fails 4/11.
- Lateral scale fails 3 of the 7 races where it can be measured.

Declared diagnostics:
- The final product's residual against official is 0.02–0.10 s.
- The innermost horse sits 1.9–2.8 m off the rail (median p5) on every oval race. The straight
  shows 1.55 m median and −0.36 m minimum, which is wrong, as expected.
- The video-only finish-camera C (the last pass before the official records are fused) is 4–33 %
  below its physical prior on all 11 races, consistent with the dev diagnostic in 3f.
- The signed leader error has the dev shape: early at the line on 11/11 (−0.02 to −0.33 s).

**Secondary arm (`HKJC_FILM=1`, declared): 7/11, 50/55 timing gates.** Leader error is lower on
8/11 (HV 05-20 0.191 → 0.036 s; the straight 0.388 → 0.249 s) and sectional error on 8/11. But
identity breaks:
- Two winners are wrong (ST 05-24, HV 05-27).
- Path count fails on 5 races (2 in the primary).

This matches the dev-set finding: the film anchor improves the metric but destabilises the
finish-camera/orphan-chain identity. It stays off.

**The film reference is weaker on H than on dev.** Film − official at 400 m:
- Sha Tin −0.12, −0.14, −0.24, −0.25 s, all early, against ±0.12 s on dev.
- Happy Valley −0.32, +0.10, +0.68 s.

Recon − film at the markers is not an independent check, because the anchor is fitted to the film.
It is mostly within ±0.15 s, with these exceptions:
- The straight: +0.24/−0.21 s.
- HV 04-22: −0.33 s.
- HV 04-15: −0.61 s at 300 m.
- HV 05-20: +2.32 s at 300 m, a mismatched crossing that passed the speed filter at 12.5 m/s.

**Batch H is now SPENT** (development set from here on). The next held-out claim needs a new batch,
sealed by a metadata-only rule before any further development.

## 3i. In-sample fixes on all 26 turf races (2026-11-14) — all races are development races now

Yardstick: observe → validate on the 15 dev races plus the 11 batch-H races, run locally. A local
re-run of the H races matches the pod to 1e-6 m, with identical tracklets and gate numbers. Baseline:
all five timing gates on 20/26, timing gates 117/130, all gates 188.

**Census of horses that are path-less or vague at the line (46).**
- 10 were READ by the finish camera, crossed while the render was still running, finished 5th–12th,
  and still had no path. None had a labelled tracklet, and a finish observation alone cannot make
  a path (no lateral, one instant).
- 17 were unread and in view, but vague or path-less (a render identity failure; includes ST4's
  runner-up).
- About 19 were tailed off and still running when the render ended (structural).

**Finish seeds (`RaceRun._finish_seeds`).** After the unseeded EM, each finish-read horse that is
still path-less gets an E-step-only seed path: its camera crossing, the field's line speed, sd
growing 1 m/s away from the line, lateral at mid-course ± half the width. The EM then iterates
again; detection runs the seed wins become ordinary render observations.
- Only 5 races change; the other 21 are byte-identical. All gates 188 → 196, timing gates 117 →
  117.
- Path count fixed on HV0415-9, ST1, HV0520, HV0527-3 and ST8; finish-n fixed on 3 of them.
- Sectional on common (horse, boundary) pairs is unchanged or better; the average rises only
  through newly scored horses.
- 6-seed ensembles on the 5 races: paths short 0 on all 30 runs (unseeded: 2–3 on most); finish
  short down on every race; timing gates unchanged. ST8 finish ρ 0.99 → 0.94 [0.90, 0.97].
- REJECTED: seeding from the first iteration. It also fired for horses the EM places anyway,
  changed races with no target (ST4 common-pair sectional 0.409 → 0.467), and failed HV0422-1
  behind-winner.

**Film markers restricted to the home straight (`film.STRAIGHT_MARGIN`).** A marker counts only if
it is at least 20 m inside the course's home straight. Happy Valley's 400 m marker is on the bend
(−0.32/+0.10/+0.68 s vs official), and HV course A's 300 m gave a +2.32 s mismatch. Only used with
`HKJC_FILM=1`.

**Film-anchored metric is now the DEFAULT** (`HKJC_FILM=0` disables; tag `recon-baseline-Q`). With
the seeds, it no longer costs paths. 26 races, deterministic, seeds on, film off → on:
- All gates 196 → 198; timing gates 117 → 120; all five timing gates 20 → 19 (the photo finish
  below).
- Leader mean 0.213 → 0.166 s (Sha Tin 0.259 → 0.182, Happy Valley 0.173 → 0.152); sectional mean
  0.243 → 0.197 s; signed leader bias +0.089 → −0.018 s.
- ST9 2 → 4/5 and ST 05-09 R1 3 → 4/5 (its leader error of 0.249 s sits on the 0.25 gate, so it is
  not a robust pass); ST1 passes the sectional gate.

6-seed ensembles on ST4, ST1, ST 05-24, ST3, HV 05-27 R9:
- ST1 timing 3 → 4/5 on every seed, sectional 0.350 → 0.220.
- Sectional also better on ST3 (0.632 → 0.584) and HV 05-27 R9 (0.181 → 0.127).

Declared losses:
- **ST 05-24's winner** is wrong on 6/6 seeds (film off: 5/6 right). Official margin: a short head,
  0.01 s ≈ 0.16 m, inside both paths' sd at the line. The runner-up #14 is unread by the camera.
- **ST3's two tail horses** stop 0.3–0.8 m short of the line at the last render frame (structural:
  the render ends ~1.2 s after the winner).
- **ST4** loses 2 horses on 3/6 seeds. Its finish anchor is ill-posed (C drifts to 1209 px/s; the
  "3"/#7 misread), so a seeded horse's crossing lands too late to win detections.
- Film data for the dev races here comes from filtered cached scans; for HV0422-1 and HV0513-1 the
  `film` stage ran locally.

**Local map distortion on the 1000 m straights (`rails.LOCAL_RES_M`; tag `recon-baseline-R`).** The
smoothed residual of the calibrated leader at the odometer ticks is ≤ 6.9 m on 24 races. On both
failing straights it is 14 m (ST 05-09 R1) and 28 m (ST3) at ~410 m to go. There the leader
"stalls" 40 m in 4 s and then jumps. MEASURED cause: on all three straights the render camera
swings 180° between t ≈ 28 and 32 s. On these two the pose solve takes a ~2.5 % scale step during
the swing (0.963 → 0.938 and 1.003 → 0.978); ST8 does not (0.993 → 0.992). The 150 m spline under a
huber loss treated the result as outliers.

Fix: where ≥ 4 consecutive ticks keep a smoothed residual > 10 m, refit with 25 m knots over that run
± 50 m, inside the tick-covered range. REJECTED: dense knots wherever any tick is flagged; one
isolated tick at the race start sent knots into the sparse start (170 m residual).

The local term applies to distance only. Lateral keeps the base fit's px/m: letting the refit's
derivative rescale lateral put ST3's p99 lateral at 30 m, against a course width of 30.5 m and
19.4 m from the base fit.

Only the two straights change. Both now pass all five timing gates on every seed of a 6-seed
ensemble:
- ST3: leader 0.527 → 0.214 s (boundaries +0.03/+0.53/−0.08, none unscored), sectional 0.593 →
  0.174.
- ST 05-09 R1: leader 0.249 → 0.130 s, sectional 0.280 → 0.171. Totals: all five timing gates 19 → 21/26, timing gates 120 → 123,
all gates 198 → 202. A from-scratch 26-race pod run of the default pipeline at `recon-baseline-R`
reproduces these exactly (the `film` stage ran end to end). Canonical baseline-S, built from those
artifacts: 22/26, timing 124/130, gates 204.

**Finish-camera digit drops (`finishcam.tracks` / `finish_gaps`; tag `recon-baseline-U`).**
Pixels of the camera crops:
- ST4's "3" reads are cloth #13 with the "1" dropped, 9 of 10 (one at confidence 0.99).
- The winner #11's cloth is often read as "1".

Two rules then misfired:
- The digit-drop test required same-frame coincidence. One cloth gives one read per frame, so it
  never fired, and the tailed-off #3 took #13's crossing, 6.2 s early.
- The misread rule kept the number with more reads, so "1" (pooling the "11" cloth with the real #1)
  dropped the WINNER #11.

Fixes:
- Compare a shorter number's reads with the longer number's interpolated track, and with a robust
  line x(u) of its raw reads. Remove those reads and rebuild the shorter track from the rest.
- When two numbers coincide, prefer the longer one.

Results:
- Only three races' finish orders change.
- ST4 passes all five timing gates on 6/6 seeds (was 3/5): sectional 0.392 → 0.173 s,
  behind-winner 0.614 → 0.040 s.
- ST 05-31 regains #1 (path count and finish-n pass, 3/6 seeds).
- ST 05-17 loses a mixed "1" read (no gate change).
- 26 races: timing gates 124 → 126, gates 205 → 207.

Still open:
- ST4's #11 is read only in an earlier pass of the full feed (u 25–86; the line crossings are at
  u 130–200), so it gets no gap.
- ST4 #10 (2nd) has no path; #3 (tailed off) honestly has none.

**Time-local pose-slip shift (`rails.time_local_shift`; tag `recon-baseline-T`).** The pixels show
the cause on the straights: the render camera reverses (180°) at t ≈ 28–32 s. After the d(s)
refit, the calibrated leader still departs from the odometer per FRAME: on ST3, −29 m at 30.2 s →
+26 m at 33.4 s. Labelled detections fall to 0–6 % in that window. No position-indexed mapping can
absorb an error indexed by time.

Rule: where ≥ 3 consecutive ticks keep |residual| > 10 m (relative to the median), every
detection in those frames is shifted by the time-interpolated tick residual, tapered over one tick.
- It triggers only on ST3. ST 05-09 R1's runs are 2 ticks long, and its d(s) refit stays necessary:
  the time shift alone loses its sectional gate.
- ST3 leader 0.214 → 0.072 s on 6/6 seeds (0.072–0.079); in the window the per-frame residual is
  flat and labelled detections are 44–64 %.
- Pixels: some labels in the window are still wrong (the green/white leader is "#3" at 31 s, "#11"
  at 33 s).
- Gates on 26 races unchanged (205); no other race changes.

**Final-stretch rank attraction (`assoc.RANK_ATTR`; tag `recon-baseline-S`).** Census of the 15
unread horses that are in view but vague at the line: most have zero own detections in the last
3 s, while 1–2.8 unassigned detections sit in nearly every near-line frame. These are not
duplicates (3.5–10 m from any assigned detection); they lie 7–13 m BEHIND the horse's path. The path
is extrapolated at constant speed from its last own detection while the horse slows, and it is
overconfident (sd ~1.3 m), so its real detections cost χ² ~22, above the clutter cost of 16.

Rule: in the last 100 m, a detection whose ahead-count equals a listed horse's legend rank costs
that horse at most 10.
- 26 races: all five timing gates 21 → 22/26, timing gates 123 → 124, all gates 202 → 205; no gate
  regresses.
- 6-seed ensembles: ST9 8/8 gates on every seed, ρ 0.89 [0.69, 0.95] → 1.00 [0.98, 1.00]. ST4
  sectional 0.476 → 0.441, more gates on most seeds. ST3 and ST7 (controls) unchanged in gates.
- Common-pair sectionals unchanged; changes come from newly scored horses.
- Known bad: HV 05-27 R9 #7 is newly scored 0.88 s early at the line. Its race still passes, with
  behind-winner 0.072 → 0.138 s.
- REJECTED: over the whole race (ST3 sectional 0.174 → 0.266, a failure); only for non-tight horses,
  sd ≥ 2 m (never fired for ST9 #5).

**Silk appearance (pixels → official silks; machinery committed, E-step term OFF by default).**

The pixels showed that the E-step uses position and legend rank only, while the render avatars'
silks are distinct. I hand-labelled 79 near-line crops on 4 races at the colour-GROUP level, against
the official silk GIFs, a pre-race input. That exposed:
- a "3" label on a white silk (HV 05-27 R9);
- heavily scrambled labels on ST4;
- #1 and #14 swapped near the line on ST 05-24. That race's crossings are still within 0.05 s of
  official, a genuine near dead heat.

Feature: the v1 `silk_hist` is dominated by the brown horse body and counts pastel silks as white.
v2 `detect.silk_feature` drops turf and brown pixels and keeps white and dark bins, matching on hue
with pastel and vivid collapsed, because the render desaturates. Against prototypes from the official
GIFs (`detect.silk_prototype`):
- the correct group is in the top 3 for 73/79 crops;
- where the EM label is wrong (19 cases), the silk prefers the right group 19/19.

As a bounded per-detection E-step cost (`assoc.SILK_W`, a `silk` stage, `det_idx` in tracks.npz),
the result is not distinguishable from noise. The run-to-run spread of the hand-set score was never
measured, so "worse" and "better" per race are unestablished:
- hand-set label accuracy ranges from 49 to 58/79 across weights and normalisations (53 without);
- ST4 goes 9 → 15/16 and ST 05-24 13 → 18/19, while ST 05-31 goes 12 → 7–9 and HV 05-27 R9 19 → 16;
- gates on 26 races are unchanged (205 → 205).

It is off by default, and the stage is skipped unless `HKJC_SILK_W > 0`.

**PARKED (stop rule).**
- Every hand-set EM error starts at identify: 9 tracklets labelled wrong there and 17 left
  unlabelled. Of those 17, the EM later labels 10 wrong.
- Averaging over tracklets does not help: aggregate top-1 is 34/79 against 40/79 per detection, so
  the render-vs-GIF gap is a bias, not noise. Central crops do not help either.
- The legend icons are flat drawings in the GIFs' style, so they do not bridge the gap.
- Silk as a VETO failed on independent names: on 53 finish-camera-named tracklets across the 26
  races, a 0.3 margin vetoes the true horse in 25 % (17 % of pure ones). On the hand set it looked
  like 4 %, a per-crop, best-of-group lower bound.
- Revisit only with render-domain prototypes.

**The film reference's own bias (recorded, NOT corrected).** The film now sets every race's distance
constant. Film − official at Sha Tin's 400 m marker is early on 9 of 10 races, mean −0.11 s:
- dev: −0.12, −0.05, −0.08, −0.10, −0.12, +0.09;
- H: −0.14, −0.24, −0.25, −0.12.

Happy Valley's 400 m marker is on the bend, so it is excluded. Using official times to correct
this would make the outcome data an input, so it stays a recorded property of the reference.

**Lateral-scale gate failures (6/26), not fixed; the gate is unchanged.** The rails-stage ppm equals
the tick-seeded ppm within 1 %, so lateral and along-track share one scale. On ST7 (pixels) the
map ends 4.0–4.4 m inside the racing rail for the whole trip: the fixed rail at 4.5 m is never
imaged, and the ridge picks a faint line at 2.3 m (48–51 px on HV0422-1, ST7, ST 05-31 alike).
REJECTED: counting only stations imaged to the expected rail + 1 m. It fixed HV0603-8 (5.89 → 8.03)
but broke HV0429-3 (7.12 → 9.36) and left ST7 at 6.09: noise at low support. An independent lateral
reference is needed.

**Finish-camera C (finding, not fixed).** The C prior does not bind: the anchor fit with and
without it agrees within a few % on all 26. On 5 races (HV0415-1, HV0422-1, HV0513-1 and the two
1000 m straights ST3/ST8) the render crossings fit the camera gaps to ≤ 0.1 s per horse, and
leave-one-out C is stable, yet official finish gaps are 20–30 % larger. Render and camera agree with
each other against official there; within 4 % on 7 other races. Cause not established.

**Canonical in-sample baseline (`recon-baseline-U` = 1b018d4).** A from-scratch 26-race pod run of
the default pipeline at d3be006 (differs from the tag only in comments; every stage, including `film`, run end to end) reproduces the
local observe → validate run exactly: every metric is identical on all 26 races.
- All five timing gates 23/26; timing gates 126/130; all gates 207 of 222 applicable (15 failures).
  The lateral gate is n/a (`None`) on the 12 races where the fixed-rail gap is unmeasurable.
- Yardstick: local observe → validate, the same as the from-scratch run at U. This settles the
  earlier 204/205 split at baseline S. The 204 came from fusing pod-R artifacts; 205 came from
  re-running observe locally. They differed by one fragile horse, HV 05-27 R9 #7: rank attraction
  did not fire on the pod artifacts, so that race failed finish-n there instead of placing #7
  0.88 s early.
- The 15 failures: 6 lateral (measured, see above), 4 timing, 4 finish-n, 1 path count.

Remaining failures (declared; ST4 #10 and ST 05-24's finish-n are real and open):
- Timing:
  - HV 04-22 R9, leader and sectional: early-boundary lateness (3f).
  - ST1, leader 0.253 s: 0.003 s over the gate.
  - ST 05-24, winner: a 0.01 s short head, inside both paths' sd at the line.
- Finish-n is scored on `paths_video`, whose time grid ends at the last render frame. The census on
  this run gives, for each short horse, the last render frame minus its official finish time.
  - **ST7 (4 short) is structural.** #3, #4, #10 and #11 finish 0.2–3.6 s after the render ends.
  - **ST 05-09 R2 (3 short) is structural for the gate as defined.** #9 and #11 finish 0.45 and
    0.89 s after the render ends. #10 is camera-read and its finish observation does reach fusion:
    in `paths.parquet` it crosses with sd 0.5 m. But it crosses 0.28 s after the render ends, and at
    the last render frame its `paths_video` path is 0.4 m short (sd 0.6), the same case as ST3's
    tail horses.
  - **ST4 (3 short) is partly real.** #3 (tailed off, −5.7 s) and #9 (−0.4 s) are render-ended. But
    #10, the runner-up, has no path while the render runs 1.3 s past it. Its 6 raw camera reads
    (u 86–92) lie at x 1185–1244, right of the line (x 960), and move only 59 px, so the
    ≥ 100 px movement rule drops the track. Pixels: the finish segment is a slow-motion post
    camera, and through the crossing (frames 2835–2875) the jockey's boot hides the "1" of the cloth.
    The OCR reads the visible "0" as "U" (conf 0.5–0.94) and returns "0" once.
    REJECTED: a digits-only CTC decode, because P("0") ≈ 0.01 where the OCR says "U".
  - **ST 05-24 (3 short) is real.** #4, #8 and #12 (places 12–14) are vague at the line (sd 3–6.5 m),
    and the render runs 0.6–0.7 s past #4 and #12.
- Path count: ST4 (#3 render-ended, #10 as above).

**ST4 finish-camera ceiling (diagnostic, scratch copies only; linker PARKED).** The question was
whether reading #10 (and the winner #11) at the line could pay. Their crossings were hand-placed
from the frames as synthetic reads, then identify → fuse → validate re-ran. An untouched control copy
reproduces U exactly.
- With #10 placed: its own boundary errors are −0.13/−0.19/−0.04/+0.10 s, path count and finish-n
  pass (paths short 2 → 0, finish short 3 → 2), but ST4 drops to 4/5 timing gates. Sectional is
  0.173 → 0.302 s.
- The loss is all #3, the tailed-off horse. It was path-less and now gets a path 0.9–4.6 s early.
  The injection relabels 2863 of 8665 detections back to 1240 m to go. #3 takes 330 detections
  (143 from #6, 127 previously unassigned); no tracklet label involves #3.
- The seed pass is NOT the cause: with `HKJC_SEED=0` the result is identical, because #10 gets its
  path in the first pass. What the injection changes is the first EM pass's finish-camera inputs:
  the read fraction behind the near-line cost (2.06 → 2.51), the blind set, and the finish
  observations in fusion.
- Non-convergence was tested as the cause and RULED OUT (see "EM iterations to convergence"
  below). The pass does stop unconverged (1.2 % of labels still changing at the 8th, last,
  iteration), but at a cap of 30 the injection still relabels 2912 detections. Across the 26 races
  at U, 10 first passes converge (< 0.2 %) within 8 iterations and 16 hit the cap.
- The split happens at iteration 0: the injected run has paths for 14/14 horses after it0's fuse,
  the control for 10/14.
- Adding #11 too gives no gap for #11. It wins by a wide margin in the slow-motion post camera, so
  its cloth shares too few frames with the field. That exposed a latent bug: a first crosser that
  links nobody wiped every other gap. Fixed: the chain now starts at the crosser that links the
  most (no-op on all 26 races, verified bit-identical).
- A reader fix that trades a timing gate for two count gates is not adoptable, the standard that
  rejected the robust-line variant.

Linker prototype (identity carried from confident read runs onto unread cloth boxes, ST4 only):
the self-test was confounded. It hid reads by u rather than px and gated on a constant-velocity
prediction while x per u is bimodal, so "boxes too sparse" is NOT established. It is parked,
because the ceiling above says even a perfect read costs ST4 a timing gate until fusion's orphan
elimination stops handing #3 a mid-field chain (traced below).

**Finish-camera digit drops must not delete a real short number (`finish_gaps`).** Both
coincidence tests counted only frames shared with the longer number. On HV 06-03 R3, 3 dropped-digit
"2" reads on the "12" cloth deleted the REAL #2: 37 reads, crossing after #12 had left; pixels show a
different silk, black-and-white spots against #12's pink and cyan. The same happened on HV 04-15 R1.
Now a number is dropped only if most of ALL its reads sit on the longer number's cloth.
- ST4's "3" (9 of 10 on #13) and ST7's "4" are still dropped.
- Three horses regain gaps, each in its correct relative order: HV 04-15 R1 #2, ST 05-17 #1 and
  HV 06-03 R3 #2. No other race's gaps change (bit-identical).
- Gates unchanged. HV 04-15 R1: ρ 0.94 → 0.97, behind-winner 0.120 → 0.105 s. ST 05-17: finish
  short 2 → 1.

**EM iterations to convergence (diagnostic `HKJC_EM_ITERS`; default still 8).** With a cap of 30,
24/26 races converge (ST 05-24 and HV 05-27 R3 still hit the cap). Gates are unchanged (207).
Sectional improves on ST 05-24 0.162 → 0.138, ST3 0.188 → 0.176, ST4 0.173 → 0.162 and
ST9 0.221 → 0.215, and worsens on ST8 0.096 → 0.106. Convergence does NOT remove the ST4 ceiling
sensitivity: with #10 injected at a cap of 30, 2912 detections are still relabelled and #3 takes
385, so the two inputs reach different fixed points.
NOT ADOPTED. 6-seed perturbation ensembles on ST 05-24, ST4, ST8, HV 06-03 R3, ST3 and ST9 (cap 8
vs cap 30, 36 runs each) give identical timing gates on every run. Median sectionals move within
the seed spread (ST3 0.181 → 0.170, ST 05-24 0.154 → 0.149, the others ±0.003), and the spreads do not
narrow (ST4 ρ range 0.95–0.99 → 0.90–0.99). Convergence buys no robustness at 2–3× the EM time.

**The ST4 ceiling's #3 mechanism: orphan ELIMINATION in fusion (traced; parked).** With #10
placed, four horses lack a path when fusion runs its orphan step (#3, #6, #12, #14). #6, #12 and #14
each take one anonymous chain through their finish-camera anchors. The rule "give a chain to the
last missing horse when it has no anchor score" then hands #3 a mid-field chain crossing at 82.4 s
(#3 finished at 88.54 s). The chain fits none of the other horses' anchors (χ² 7400–10400 against
9). In the control, two horses stay missing, so elimination never fires.
- Elimination OFF fixes the ceiling: ST4 stays 5/5 timing, path count and finish-n pass, gates
  7 → 9. On the 26 races at V (single runs) it is gate-neutral (207), but 8 races each lose a path
  (paths short 0 → 1) and sectionals move both ways (HV 05-20 R2 0.161 → 0.203, ST3 0.188 → 0.165).
- The 8 horses with a path at V and none with elimination off (a diff, not a trace of who was
  eliminated; HV 04-15 R1 #11 is camera-read and may not be an elimination case). Worst boundary
  error per horse:
  - small: HV 04-15 R9 #8 +0.19, HV 04-22 R1 #6 −0.20, ST7 #10 −0.29, ST 05-09 R2 #9 −0.06 s;
  - 0.4–0.65 s: HV 04-15 R1 #11 −0.56, HV 05-27 R9 #10 −0.65, ST 05-31 #3 +0.62, ST3 #14 +0.42 s.
    These four are the path-losing races whose sectional IMPROVED with elimination off.
  - HV 04-15 R1 #11 (−0.15/−0.24/−0.56) and HV 05-27 R9 #10 (−0.05/−0.65) drift early toward the
    line, the same shape as ST4 #3.
  The rule is kept for now because removing it is gate-neutral but costs 8 paths, and it has not
  been ensemble-tested. It is not shown to be accurate.
- REJECTED: forbidding elimination of a final-stretch chain onto a BLIND horse (the E-step's rule
  applied in fusion). It blocked the first chain, then elimination handed #3 the next one (ending at
  s 1327). Matched horses keep extra fragments, so "every leftover chain is the last horse's" is
  false.
- Parked: no production gate depends on it. ST4's two count gates need a #10 reader AND an
  elimination fix together.

**HV 04-22 R9 (two of the four timing failures): a shared scale error that the film can't
measure reliably.** The race is film-anchored (4 marker crossings, not on the paint pin). The error
is common-mode: the median over all 12 horses is +0.45 s late at 800 m, +0.39 s at 400 m and
−0.20 s early at the line, so the last 400 m runs ~2.5 % short in time. Within the film window the
anchored leader minus the film goes +0.15 s at 200 m to −0.11 s at 50 m, a slope of
+0.16 s/100 m, matching the official +0.15.
- REJECTED: a two-parameter film anchor (offset + scale). Across the 26 races the film-marker
  slope correlates with the official error slope at only r = 0.26. Examples: HV 05-13 R1 +0.145
  against −0.04, HV 05-27 R9 −0.137 against +0.004, ST6 +0.104 against −0.006. Per-marker noise
  (~0.1 s) over a 150 m window swamps the slope, so the term would fix this race and break others.
- Declared. A scale reference spanning more of the straight is needed (the drone, or the
  400 m marker, which sits on the bend at Happy Valley).

**Identity fragility and the tracker (2026-11-14; code committed OFF by default).**
Label-free fragility yardstick: per-detection label unanimity across 6 perturbation seeds (1 %
detection drop). At V: ST3 25 %, ST4 45 %, ST 05-24 50 %, ST8 48 %, ST9 80 %, HV 06-03 R3 79 %; mean
54.5 %. Hand-set spread across seeds: ST4 9–12/16, ST 05-24 9–15/19, so the earlier single-run silk
"gains" sat inside the noise.
- PIXELS: render avatars are distinguishable by eye and consistent within a race. Most tracklets
  are IMPURE: the silk changes mid-tracklet. On ST4 many tracks switch together at ~54 s, and the
  labels rotate (#7 → #13 → #12 → #5) while the silks stay unchanged.
- Symptom, from per-tick field speed: the whole field's world speed drops to ~0 for 1–3 ticks,
  then catches up at 23–29 m/s. Anomalous ticks: ST3 77, ST 05-31 32, ST 05-24 25, ST4 17,
  HV 05-27 R9 6, HV 06-03 R3 1. The switches are made in `track`, not `stitch`.
- CAUSE (pixels): POSE STALLS, not stale render positions; the pixel boxes move 4–6 px on those
  ticks.
  - On ST4 frames 1368–1373 the solved image centre moves < 1 px per tick (normally ~36) and gets no
    catch-up afterwards. The features stage's stored step is already (2, −5) px with a spurious
    0.5° rotation.
  - Its RANSAC inliers sit on the HORSE AVATARS. The camera follows them, so they are static in the
    image, and they outvote the weak turf.
  - `register.world_mask` (a blacklist: V < 45 or vivid hue) misses brown-bodied and pastel-silk
    avatars.
  - A turf/rail WHITELIST mask gives (34–36, −11 to −13) px steps there with ~0 rotation.
  - Stalled cached steps per race: 0–19 (ST3 19, HV 04-22 R1 14).
- `tracking.common_shift` (HKJC_TRACK_SHIFT=1) estimates the per-tick shared along-track shift by a
  robust vote. On 26 races it costs 4 gates (ST3 ρ 0.98 → 0.88, behind-winner 0.056 → 0.211;
  HV 05-27 R3 ρ; HV 05-27 R9 finish-n). Not adopted.
- `tracking.calibrate_llr` + appearance cost (HKJC_TRACK_APP=1): −2 log(p_same/p_diff) of the TV
  distance between a detection's silk and the track's running mean, learned per race. Feature AUC
  is 0.96 at 1-detection lag. With the shift, on 26 races it costs 2 gates (ST1 winner, a 0.05 s near
  dead heat; HV 05-27 R9 finish-n); hand set 52 → 53/79.
  - By eye, tracklets are purer: ST4 has ~9 switches → ~3–4, and HV 05-27 R9's yellow survives the
    48 s camera event.
  - 6-seed ensembles on six races: timing identical, but unanimity 54.5 % → 49.9 % (ST4 45 → 57 %,
    ST8 48 → 17 %, ST3 25 → 19 %). Not adopted.
- Pose-stall fixes (code committed OFF by default; pose-only local check on 5 races):
  - `register.mask_boxes` (HKJC_WORLD_MASK=yolo): the features stage also masks the avatar
    detector's boxes, grown 1.6× to cover shadows. Mid-race stalls go to zero; what remains is the
    render start, after the finish, and keyframes. HV 04-29 R3 keeps its loop closures (336) and
    long-baseline d24 edges rise on every HV race.
    REJECTED: a colour WHITELIST mask (`world_mask_v2`). It fixed the stall but dropped small static
    world features (posts, markers): HV 04-29 R3 loop closures 335 → 0.
  - `video.skip_gap1` (HKJC_FRAME_GAP1=1): HV 04-22 R1/R9's video has a keyframe every 100 frames.
    It re-encodes the same render state (mean |diff| 3.8 > the 3.0 duplicate tolerance), so frames
    400, 500, … entered as new ticks with zero motion (14 per race). At a 10 Hz render in 25 fps
    video a one-frame gap is impossible: gaps alternate 2/3 on all 26 races.
    FAILED in pod run W: dropping the gap-1 frame kept the stale duplicate reference, so the frame
    after each keyframe differed by the keyframe's codec noise and entered as a zero-motion tick, and
    the next ticks were stamped 1 frame late (HV 04-22 index 399, 401, 403, 405 instead of 399, 402,
    404). Stalls stayed at 14 per race; 7 races with a single gap-1 frame were shifted the same way.
    Replaced by `video.grid_keep` (HKJC_FRAME_GRID=1). The render sits on two fixed residues of
    frame % 5 for the whole replay (26/26 races, 99.9–100 %). Off-grid yields are dropped AFTER the
    de-dup, which is exact: an HV 04-22 R1 decode equals the V stream minus off-grid frames,
    byte-identical. Outside HV 04-22, off-grid frames are only the first render frame or follow a
    render-gate dropout near the finish (none mid-race).
  - Pod run W (both on, from scratch, commit 9073964), against V:
    - All five timing gates 23 → 23, timing gates 126 → 124, all gates 207 → 201 (a failure).
    - Frames are identical to V on 17 races; there W is a clean mask-only test. The other 9 races
      were shifted by `skip_gap1`.
    - Prediction (1) HOLDS for the mask. Pose stalls between the first render tick + 1 s and the
      winner's finish fall from 48 to 30. Every mid-race horse-lock stall is gone (ST3 6, ST4 5,
      ST 05-09 R1 4, ST8 3 → 0). What remains: the 28 HV 04-22 keyframes (the `skip_gap1`
      failure) and one early stall each on HV 04-15 R1/R9.
  - Pod run X (both on with the first grid rule, from scratch, commit 36e4d48, 28 min), against V:
    - All five timing gates 23 → 24, timing gates 126 → 127, all gates 207 → 205. Lost:
      ST 05-09 R1 and HV 05-27 R9 finish-n, ST 06-07 R6 lateral scale (2.07 m, the same in W:
      a real mask regression from the rails stage). Gained: the ST 05-24 winner.
    - Mid-race pose stalls 48 → 2 (one early stall each on HV 04-15 R1/R9).
    - Prediction (2) HOLDS (eye strips). V's ST4 rotates identities wholesale at 54–56 s: six
      tracklets change silks and five new ones start. In X every long tracklet is pure from 12 to
      82 s. The switches that remain are in the 36–44 s camera swing; V has them too.
    - Prediction (3) HOLDS mechanically. The time-local shift fired only on ST3 at U (2 windows,
      28.3 m) and fires nowhere in X. Turning the local refit off changes no race beyond 1e-4 s.
      KEPT anyway: with the mask, ST3 and ST 05-09 R1 leader errors are about −0.2 s from 400 m
      (V +0.10 / 0.00). That is a real residual after the swing which the thresholds no longer
      catch.
    - 6-seed ensembles on six races (V → X): final unanimity 54.5 → 59.6 % (ST3 25 → 53, ST4
      45 → 66, ST8 48 → 23). Whole-race identity score 0.924/86.6 % → 0.928/90.1 %.
      The identity score is Spearman of reconstructed vs official crossing times and the share of
      horse-boundaries within 0.25 s after removing the common mode, intermediate boundaries
      only (validation-only diagnostic). ST3 within-0.25 s 71 → 97 %.
    - The first grid rule dropped EVERY off-grid frame, including the only copy of a tick after a
      finish-zone render dropout. Finish-short went 31 (V) → 34 (W) → 39 (X). ST 05-24 differs from
      W only by three dropped frames: its winner flipped and the hand set fell 15 → 10/19. FIXED
      (5087fb9): only re-encodes of an already-kept tick are dropped (the 28 HV 04-22 keyframes).
    - Mask only (W's artifacts, V frames), 6 seeds: ST 05-24 hand set mean 15.3/19 (V 12.3),
      identity 0.949/88.3 % (V 0.909/81.3 %). ST8 unanimity 48 → 32 % with an unchanged identity
      score (0.921/93 % vs 0.920/95 %): less stable, not less correct. Open.
  - Declared before pod run Y (da14cc1: mask + corrected grid ON by default, from scratch, cuda):
    - The 17 races whose W frames equal V's reproduce W bit-identically.
    - HV 04-22 R1/R9 lose only their 14 keyframes each. The 7 races `skip_gap1` shifted in W get
      V's frames back.
    - Finish-short returns to W's level or better.
    - Baselines with the avatar mask come from CUDA. The OpenVINO iGPU detector differs (ST4:
      9122 vs 9129 detections), and its masks change the ORB steps on 603/705 frames.
  - Pod run Y (da14cc1, defaults ON, from scratch, 46 min): every declared prediction HOLDS.
    - The 17 races whose W frames equal V's reproduce W bit-identically.
    - The 7 shifted races get V's frames back. HV 04-22 R1/R9 lose only their 14 keyframes each.
    - Finish-short 31, the same as V.
    - Gates vs V: all five timing gates 23 → 22, timing gates 126 → 125, all gates 207 → 202 (a
      FAILURE on single runs):
      - Mask, also in W: HV 04-15 R9 paths-short; ST 05-09 R1 and HV 05-27 R9 finish-n;
        ST 06-07 R6 lateral scale.
      - ST 04-26 R1: winner lost (a 0.05 s near dead heat), leader gate gained.
      - HV 05-20 R9 behind-winner 0.287 s. The finish-camera registration aliased, below.
    - 6-seed ensembles on all 26 races (observe → validate, 1 % detection-drop seeds), V → Y:
      - All five timing gates mean 22.83 → 22.17; timing gates 125.8 → 124.5; all gates 206.3 →
        202.0, seed range [205, 207] → [199, 206]; finish-short 31.3 → 32.2.
      - Identity score 0.930/89.8 % → 0.931/90.0 %; unanimity 64.9 → 64.5 %; hand set mean ~50.7 →
        ~53.0/79 (inside the per-seed spread).
      - NOT ADOPTED: both defaults back OFF, so `main` behaves as tagged V. The mask is right in
        pixels (stalls 48 → 2, ST4 rotation gone) but costs gates downstream.
      - Races losing on every seed: HV 04-15 R9, HV 05-20 R9, ST 06-07 R6. Also lower: ST 04-26
        R1, ST 05-09 R1, HV 05-27 R9. Gained: ST 05-31 R1.
  - Finish-camera scale prior (`finishcam.RULER_CAL`, HKJC_FC_CAL=1|reg|anchor, off by default):
    - The COCO horse box includes the extended legs, so the ruler (2.6 m nose-to-tail) reads px/m
      high and the physical prior on C is too large.
    - The registration searches a ±25 % window around the prior. HV 05-20 R9 (pod run Y) locked onto
      an aliased C = 1.25 × prior: no tracklet named, #2/#3/#9 swapped.
    - Clean registrations (not at an edge, ≥ 80 % matched) span fitted C / prior 0.73–1.12. The
      constant 0.90 centres the window on that range (window 0.68–1.13) and was FITTED to these 26
      races. Edge-pinned races: Y 7 → 1. REJECTED: 0.88, the first median, taken over edge-excluded
      fits only and biased high. REJECTED: 0.82, the median fixed point, which pins the clean high
      fits.
    - Adoption rule, declared before reading: mean all gates not below the base arm's, and no race
      losing a gate on all six seeds.
    - 26-race 6-seed ensembles:
      - On V's artifacts: all gates 206.3 → 206.3, all five timing gates 22.83 → 23.00. It PASSES the
        gate rule, but it is NOT ADOPTED because of hand truth. ST4 loses #5's finish-camera name on
        3/6 seeds (11.0 → 9.0/16, on Y's artifacts too): its weakly determined registration (4–6 of
        13 matched) only named #5 correctly while pinned at the old window edge. The hand set
        belongs in the next rule.
      - On Y's artifacts: 202.0 → 205.2, all five timing gates 22.17 → 23.50, the best timing arm.
        Against plain V still −1.1 gates (ST 06-07 R6 lateral on every seed, finish-n).
    - Split by prior (6 seeds, the six affected races). Neither half is clean:
      - Registration-only recovers ST 04-26 R1 and HV 05-20 R9, but costs ST4's hand truth.
      - Anchor-only recovers HV 04-15 R9 and keeps ST4, but costs HV 05-27 R3 and ST 05-24's hand
        set on V.
      - The hard registration window is the fragile part, not the constant. NOT ADOPTED in either
        form.
  - Lateral-scale gate MEASUREMENT bug (`rails.fixed_rail_gap`, fixed for every arm; the threshold
    is unchanged):
    - `ridge_response` compares each offset with its sides via `np.roll`, which wraps around. The
      gate sampled only the ±2.5 m band, so offsets near the band edge compared against the far end
      and fired.
    - ST 06-07 R6, the mask's one all-seeds blocker, in pixels: maps rebuilt locally from V's and Y's
      poses reproduce the pod values exactly (9.68 / 11.27 m). On a wide window both maps show a
      single fixed-rail ridge, at 9.59 / 9.73 m. The 11.27 m was the artefact (3755 stations vs 232
      at the real rail).
    - Fix: sample a padded window and take the argmax inside the band only. Unit test fails on the
      old code.
    - Re-scored on all 14 lateral races, maps rebuilt from each arm's pose:
      - V: 8 pass / 6 fail → 10 pass / 1 fail / 3 n/a.
      - Y: 7 pass / 7 fail → 10 pass / 0 fail / 4 n/a.
      - Newly passing on both arms: HV 04-22 R1, HV 04-29 R9, HV 06-03 R8; on Y also ST 06-07 R6.
      - ST 05-03 R7, HV 05-13 R9, ST 05-31 R1 → n/a: no supported ridge anywhere near the
        expected gap (support ≤ 5 %). Their old ~2.3 m gaps were edge artefacts.
      - HV 04-29 R3 is the same case: V's 8.20 m "pass" was the artefact. Now V fails (9.21 m,
        weak) and Y is n/a.
  - Declared before the from-scratch run with both on:
    (1) mid-race stalls and field-speed anomalies go to ~0;
    (2) ST4's 54 s identity rotation disappears with the OLD tracker (eye strips);
    (3) ST3's camera swing may no longer need HKJC_TIME_SHIFT or the local d(s) refit (tested with
    both off afterwards).
    Gates may move both ways (downstream constants were tuned on stalled poses); judged on
    ensembles and unanimity.
- Open:
  - break the cross-seed disagreement down by stage (tracklet partition, identify, EM) before
    redesigning anything downstream.

**Baseline V (`recon-baseline-V` = e4ee137): U plus the chain-start guard and the digit-drop
fix.** Nothing upstream of `finish_gaps` changed, and `finish_gaps` output is bit-identical to U on
23 races (verified with final src; exactly 3 races differ). So the 26-race root
(`data_cache/recon_V`) reuses U for those 23 and re-runs identify → fuse → validate for HV 04-15 R1,
ST 05-17 and HV 06-03 R3. No pod run is needed. All five timing gates 23/26, timing gates 126/130,
all gates 207 of 222 applicable (unchanged from U). 6-seed ensembles of the digit-drop fix (U code
vs V code): timing 5/5 on every seed in both arms. HV 04-15 R1 ρ 0.94 [0.94, 0.95] → 0.97
[0.96, 0.97] and behind-winner median 0.119 → 0.102 s; ST 05-17 is unchanged.


**Straight-course rail side (2026-11-15; adopted).** `rails.inner_sign` took the inner side from
the sign of the camera path's summed turning. Ovals turn |Σ| ≥ 2.7 (all +1: HK racing is clockwise).
The three Sha Tin 1000 m straights turn |Σ| < 0.2, so the sign was noise. ST 04-19 R3 and ST 05-09
R1 flipped between V and Y (whose poses differ only by the avatar mask); ST8 was −1 in both. With
−1 the "rail" is the other band edge.
- Pixels (ST 05-09 R1, Y): the −1 rail traced the mosaic's warped coverage boundary mid-race and
  dipped 15–25 m before the line. The whole field slid ~3.5 m laterally at 56–57 s, association
  scrambled (#9 onto the wide leader's detections, #4 onto a back-marker at 56.7 s) and #4's path
  ended 3.4 m short.
  - The +1 rail follows the lower band edge, which mid-race is a visible straight rail line (the
    oval's curved rail joins below it).
  - The draw prior also lands on the wrong side with −1: draw-vs-lateral ρ is +0.73 with +1 and
    −0.21 with −1.
- Fix: `rails.inner_side` returns +1 (right of travel; on the straight, stall 1 is on the right as
  on the ovals). The old inference stays as the diagnostic `HKJC_INNER=auto`.
- Declared before reading: all gates over 3 straights × {V, Y} × 6 seeds must not drop, and the rail
  must sit on a real edge in pixels.
- Result: 285 → 286.
  - Y: 141 → 144 (ST 05-09 R1 45 → 48 with ρ 1.00 on every seed; ST3 ρ 0.91 → 0.98).
  - V: 144 → 142. ST3 and ST 05-09 R1 are bit-identical (already +1). ST8 ρ 0.92–0.95 → 0.98–1.00
    and behind-winner ≈0.055 → 0.025 s, but sectional 0.09 → 0.12 s and its close #2/#12 finish
    flipped on 1 of 6 seeds (winner and finish-n).
  - Ovals are unchanged (all already +1). No hand-truth race is a straight.
- Also recorded: `HKJC_FC_READS=reads` (`finishcam.read_numbers`: count read-but-gapless cloths as
  read for the E-step's negative evidence). It is opt-in and was not assessed on ensembles; it did not
  fix ST 05-09 R1.

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
- **Film-anchored metric**: DEFAULT since baseline Q (3i), after finish seeds stopped it costing
  paths. Open: the film's own −0.11 s bias at Sha Tin's 400 m marker (recorded, not corrected).
- **ST4 finish camera** (slow-motion post camera): #11, the winner, is read sparsely (u 25, 27, 56,
  then 69–86), so the running-median filter drops the reads near the line and it gets no gap. #10,
  2nd, has its "1" hidden by the boot at the line (3i). The count gates need BOTH a reader (e.g.
  identity carried from confident reads onto unread cloth boxes) AND a fix for fusion's orphan
  elimination (3i ceiling). Elimination test to try: count the leftover orphan chains when it fires
  (≥ 2 for one horse in the ST4 ceiling). Do not loosen the filters
  blindly (the robust-line and motion-rate variants cost 2–3 timing gates).
- **AWT** — OUT OF SCOPE for now (turf only). Known: head/shadow duplicates fixed by an anisotropic
  merge; far-side avatars on bends missed; the back-straight rail is framed under the legend band.
- **Real-camera feeds (drone, patrol) + SAM 2** — the only independent source of positions: needed for
  races where the render disagrees with the record, for lateral validation and for the start segment.
  SAM 2 video segmentation seeded from fused render positions (GPU/pod) is the planned approach.
- **Finish-identity robustness** — results flip with small detection differences (HV 0429, ST7).
