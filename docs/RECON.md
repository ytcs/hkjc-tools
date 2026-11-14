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
   gauge, outlier pruning.
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
   detections; clutter option; one-sided legend rank constraint), then the paths are re-fused; the
   tracker fragments are only the initialisation.

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
recall per frame is partial (orders come out correct but incomplete — open).

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

## 4. Open / next

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
- **Odometer HUD vs painted markers**: over the final 200 m the HUD counts 2.7-4.3 % more metres
  than the markers on three HV races (same span; cause unknown) — the markers are the metric there.
- **AWT** (bgsub detection): head/shadow duplicates fixed by an anisotropic merge along the body
  axis; far-side avatars on bends still missed (map coverage?).
