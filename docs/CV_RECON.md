# CV race reconstruction — 4-feed fusion (methods + validated state)

> Consolidated write-up of the CV trajectory-reconstruction subsystem (journals 0067–0092,
> wave-2 CONTROL dimension). The reconstruction turns HKJC's auth-free replay feeds into a
> per-horse `(longitudinal, lateral, t)` trajectory, to supply the **control/lateral trip**
> (ground saved/lost) that the longitudinal sectionals and form cannot see. Official data
> (finish order, margins, running positions, stewards' comments) is **VALIDATION-ONLY, never a
> model input**; the **draw** is the one exception, allowed because it is a pre-race INITIAL
> CONDITION (same class as distance/weight), not an outcome.

## 0. Working discipline (CV) — VISUALLY SCRUTINIZE, don't trust numbers

**The governing CV discipline: visually inspect the raw frames AND the diagnostic overlays — every
time — to surface issues and identify fixes. Numbers lie by omission.** Hard-won examples: a "11
horses detected" count hid k-means placing 2-3 dots on the infield hedge; a green rail fit that
"ran" was locked on a far boundary, not the running rail; a rail fit that looked solid on 4 frames
failed on f1760 (the band was cut off). None were visible in the summary stats — only in the
pixels. So: (1) overlay every intermediate (boxes, masks, rail, candidates) on the real frame;
(2) zoom in — thumbnails hide mis-placement; (3) sample BROADLY across the race (bends, straights,
bunched, strung, distant) — failures hide in the un-checked phase; (4) scrutinize EACH sampled
frame, not a spot-check. Fixes follow from looking, not from metrics.

## 1. What it delivers (validated) vs what it does not

Validated across a 3-race distance spread (HV 2026-06-03 — **R01 1650m, R03 1200m, R06 1200m,
R08 1800m**, "C" course; R01/R03/R08 fully validated, R08 held out during hardening):

| Deliverable | Status | Evidence (R01 / R03 / R08) |
|---|---|---|
| Per-horse **longitudinal ORDER** (running positions) | ✅ solid | per-pole ρ vs official RP: body +0.8–0.99; R08 (route) is the corpus best +0.96/0.95/0.98/0.94 |
| **Finish MARGINS** (the S3 surface) | ✅ lead-group | RF photo-finish, MAE 0.41 len (R01) / 0.72 fused; deep tail compresses |
| Winner recovery | ✅ 3/3 | R01 #3, R03 #7, R08 #6 all recovered |
| **Lateral CLASSIFICATION** (rail-vs-wide / ground saved-lost) — the S8 control signal | ✅ sign 3/3 | fused rail-vs-wide gap +4.29 / +5.81 / +3.97m, correct sign all three |
| **Absolute mid-race lateral METRES** (the S9/drag level) | ⚠️ partial | only pinned at the start (draw) + the patrol window; un-metred on bends — see §7 |

**One-line state:** we can reliably say *who* saved ground and *who* went wide, everywhere,
across distances (enough for the **S8 control-cause** test); we can say *how many metres* wide
only at the start and in the home straight (the **S9 drag** level — the deferred gap).

## 2. The feeds — four partial sensors of one latent race

HKJC serves ≥4 auth-free deterministic MP4 replay feeds (URL pattern + tokens in `HKJC_DATA.md`).
Each measures a different slice of the same latent `(long, lat, t)` state:

| Feed (`{TOKEN}`) | Camera | Best for | Limit |
|---|---|---|---|
| **Aerial** `replay-aerial` | synthetic 3D AVV render (split-screen: real cam left + render right) | full-race lateral ORDERING + odometer arclen + legend identity | oblique render → lateral METRIC degrades on bends (§7) |
| **Patrol** `replay-patrol` | fixed high mount, head-on early | **cleanest lateral METRIC** (fixed cam ⇒ constant homography) on the straight entry | narrow band; PANS on routes; head-on identity needs a bridge |
| **Drone** `replay-drone` | real, high, moving/zooming | real positions; lateral ORDERING cross-check; finish-zoom numbers | moving cam ⇒ metric nulls (ordering only) |
| **RF** `replay-full` | real, side-on tracking | per-horse longitudinal ORDER + finish MARGINS; crisp saddlecloth numbers | lateral weak (foreshortened); margins resolve the lead group |

The legend strip (aerial/RF) is a broadcaster DATA OVERLAY (≈ the official running order) — a cheap
dense order feed + identity anchor, **not** the novel signal. The novel signal is the lateral trip.

## 3. Shared clock (the sync spine)

One race-time axis `race_t` ties all feeds, anchored on the official sections (sum = winner time):
- **Aerial** → `race_t` via its on-render odometer arclen through the official arclen→time curve.
- **Patrol** → kinematic anchor (winner's finish-line cross) → `race_t = patrol_vid − offset`.
- **RF** → its on-screen race clock (`race_t = RF_vid − offset`, manual/verified).
- **Drone** → affine `drone_vid → race_t` from the gate-jump + the photo-finish cut.

Confirmed 3-way consistent (journal 0077). All per-race anchors are now AUTO-SCANNED (the section
count adapts 3/4/5; clock anchors frame-verified), so the clock is race-agnostic.

## 4. Identity — a single fixed CONSENSUS (race-agnostic)

`studies/cv_identity.py::Identity.consensus_assign` (journals 0089→0090). Per feed, a one-to-one
Hungarian assignment of tracks→runner numbers minimising **4 weighted independent votes** (the
SAME weights on every race — no per-race tuning):

```
cost = 1.00 · own_finish_cam_silk_LUT(distinctness-weighted)   # each feed's OWN crisp finish-cam
     + 0.45 · RF_anchor_silk_LUT                               # cross-feed shared backbone
     + 0.60 · reliability_gated_clock_ORDER                    # two horses can't share a place at a time
     + 0.70 · DRAW_order                                       # race-independent START order
```

Key lessons banked:
- **Solve once, propagate** — read numbers from the crispest source (RF side-on), build a silk-
  colour **hue-histogram** signature (a LAB/BGR median collapses overlapping silks — do not use it),
  propagate by silk + clock. This killed the per-feed OCR bottleneck (aerial **2.4 hr → <1 s** cached).
- **The clock-order vote breaks silk-degeneracy** — the red/pink silk family is genuinely
  silk-ambiguous; "can't be two places at once" disambiguates it (it flipped R03 patrol −5.7→+4.4m).
- **The draw is the cleanest order vote** — known exactly, race-independent; it fixed the R01
  regression that an RF-only LUT + weak head-on-depth proxy had caused.
- Validated **no-regression on R01 + fix-held on R03** with one fixed scheme; **R08 held out** then PASS.

## 5. The draw → `y_lat(0)` anchor (initial condition)

The barrier draw is a known pre-race physical fact. At the gate, horses sit in evenly-spaced stalls:
`y_lat(draw d) = 0.5·w + (d−1)·w`, stall width `w = 21.3 m / 14 ≈ 1.52 m` (cross-checked vs the
patrol metric span). Added to the smoother as a **low-sd (1.5 m) lateral observation at `race_t=0`**.
Two roles: (1) a SECOND lateral metric anchor (start + patrol-mid) that restored the fused magnitude
(R03 +1.95 → +5.81m); (2) a third identity vote. The bounded-lateral-accel prior between the gate
anchor and the feeds correctly bridges **in/out shifters** (R01 #8 drawn gate-12 raced the rail →
fused 5.2 m off rail, correct, though patrol-alone mis-read 14.5 m).

## 6. The fusion smoother

`studies/cv_fusion_4feed.py` — a per-horse numpy **Kalman forward-filter + RTS backward-smoother**
on the shared `race_t` grid (5 Hz). Latent `z = [s, ṡ, l, l̇]` (arclen, long-vel, lateral, lat-vel).
The first-principles motion prior is applied **jointly in the smoother, not as a per-feed filter**:
bounded longitudinal accel (σ≈0.8 m/s², near-constant cruise) + bounded lateral accel (σ≈0.35 m/s²,
slow sideways drift = the "no teleport" prior). Each feed enters as an observation over its coverage
window with its own noise; PATROL anchors the lateral METRIC, AERIAL+DRONE carry lateral ORDERING
(inflated metric variance), RF carries longitudinal order + finish-margin metric, DRAW anchors `t=0`.
State covariance is propagated, so the uncertainty bands honestly widen where coverage is thin.

## 7. The one remaining limit — absolute lateral metres on bends

The pixels→metres conversion needs a per-frame ground homography. On a **straight** the single drawn
inner rail + known track width gives a workable map (aerial lateral autocorr +0.9). On a **bend** the
rail curves away from the camera, foreshortening varies non-linearly, and a single rail is
geometrically **under-determined** (no outer rail; markers don't OCR; minimap is a 1-D progress bar).
So the aerial lateral metric becomes per-frame noise on bends (R08 per-bend autocorr **+0.9 → −0.16**;
aerial-alone gap flips to **−3.2 m wrong**). No other feed rescues the bend metric (patrol pans on
routes, drone is metric-null, RF is longitudinal). Hence absolute lateral metres exist only at the
**start (draw)** and the **patrol window**; mid-route the fused `y_lat` sd balloons (12→52 m on R08).

- **Does NOT block S8** (control re-emergence) — that needs the lateral CLASSIFICATION (rail vs wide),
  which is validated 3/3.
- **Blocks S9** (drag/ground-loss energetics, C_dA) — that needs absolute metres throughout.
- **Candidate fix (deferred):** a per-frame **bend homography from the rail ARC** (an arc carries a
  centre + radius; HV "C" bend radius is a known surveyed constant) — hard, uncertain payoff, worth it
  only when S9 demands it.

## 8. Library layout (promoted)

The reconstruction lives in two packages with one-way deps
(`core ◄ data ◄ reconstruction`, `core ◄ vision ◄ reconstruction`):

**`hkjc.vision` — pure perception** (never imports `data`; heavy CV deps load lazily):
- `silk.py` — silk-colour signature (`silk_signature`, `silk_distance`, `silk_distinctness`).
- `geometry_cv.py` — rail primitives (white-rail fit, fixed-rail homography, `RailSpline`, stable-window rail).
- `ground_contact.py` — `shadow_base` / `plumb_to_hoof`.
- `ocr.py` — side-on/finish-cam saddlecloth OCR (AVV-legend OCR stays in `mode_b`).
- `identity.py` — `Identity` consensus (`consensus_assign`, `build_rf_anchor`, `augment_silk_lut_from_patrol`).
- `feeds/{aerial,patrol,drone,rf}.py` + `feeds.FeedContext` — the four observation models (per-race scalars threaded, not baked).
- `schema.py` — the canonical output (`RaceTrajectory`/`RunnerTrajectory`/`TrajPoint`; `d`=from-finish, `off`=from-rail).
- `mode_b/` — the AVV-render classical-CV machinery the aerial feed reuses (legacy name; slated to fold up into the modules above once the studies retire).

**`hkjc.reconstruction` — orchestration on top of `vision` + `core`/`data`:**
- `geometry.py` — `RaceGeom(core.Race)` sources width/bend/draw/start-lateral from `core` (track facts are never re-encoded in CV code); `draw_to_start_lateral`.
- `clock.py` — the shared `race_t` (`ClockAnchors`, per-feed anchors, section curve).
- `fusion.py` — ONE smoother v2 (`smooth_field`, `Obs`): phase-gated EOM longitudinal kernel + bounds-only lateral + the truncated-Gaussian box / PAVA ordering / geometry bend-downweight constraints.
- `reconstruct.py` — `Reconstructor(race, recipe).run() -> RaceTrajectory`; `RaceRecipe` holds the per-race sync/calibration inputs (video paths, sections, clock offsets, windows, name→number).

Physical constants (`THOROUGHBRED_LENGTH_M`, `HORSE_LANE_WIDTH_M`, `BARRIER_STALLS`) live in `hkjc.core.track`.

**Open seam:** `Reconstructor.extract_feeds()` (raw 4×mp4 → numbered per-feed rows + identity) is not yet
wired — the validated, tested tail `from_feeds(...)` runs the clock→fusion→assemble path from supplied rows.
Until `extract_feeds` lands, end-to-end runs still go through the `studies/cv_r0*` adapters, which remain the
source-of-truth and are not yet archived.

Data panels: `data_cache/cv_fusion_trajectories_{R01,R03,R08}_physkernel.parquet` (the v2 trajectories).
Walkthrough: `data_cache/figs/cv_recon_walkthrough_{R01,R08}.mp4`. Official validation cache:
`data_cache/race_official/20260603R{1,3,6,8}_results.*`.

## 9. Journal trail

0073 (feed map) · 0075 (aerial render-inversion) · 0076 (patrol) · 0077 (fusion spine + shared clock)
· 0078 (minimap = dead) · 0080 (drone, first cross-feed agreement ρ+0.93) · 0082 (patrol silk identity)
· 0086 (RF order+margins) · 0084/0087 (fusion v1 capstone) · 0088 (R03 generalization) · 0089 (identity
v1, the honest regression) · 0090 (identity v2 consensus + draw) · 0091/0092 (R08 held-out PASS + the
bend-metric limit quantified) · 0093 (lateral physical constraints: hard box win, ordering inert on routes)
· 0094 (phase-gated physics longitudinal kernel; squiggle ≡ order) · 0096 (walkthrough video) · 0095/0097
(library promotion: vision perception + reconstruction halves).
