# CV race reconstruction — 4-feed fusion (methods + validated state)

> Consolidated write-up of the CV trajectory-reconstruction subsystem (journals 0067–0092,
> wave-2 CONTROL dimension). The reconstruction turns HKJC's auth-free replay feeds into a
> per-horse `(longitudinal, lateral, t)` trajectory, to supply the **control/lateral trip**
> (ground saved/lost) that the longitudinal sectionals and form cannot see. Official data
> (finish order, margins, running positions, stewards' comments) is **VALIDATION-ONLY, never a
> model input**; the **draw** is the one exception, allowed because it is a pre-race INITIAL
> CONDITION (same class as distance/weight), not an outcome.

## Current state & roadmap (read this first)

Two layers of CV recon exist:

1. **4-feed fusion recon (built, §1–§9 below).** Fuses aerial+patrol+drone+RF into per-horse
   `(arclen, y_lat, t)`. Validated R01/R03/R08; promoted to `hkjc.reconstruction`. Its one open limit
   was **absolute lateral metres on bends** (§7) — a single aerial rail under-determines the bend
   homography.

2. **Aerial-feed deep revamp (current frontier, §10).** A from-scratch rebuild of the AERIAL feed's
   perception to fix §7 at the source — make the aerial feed alone yield clean 2D paths through bends.
   The v3 rail tracker was falsified at the rotating bend (journal 0115); the redesign was built, validated
   on R08, and **PROMOTED to `hkjc.vision.feeds.aerial_rail`** (journals 0116→0117). It is FIELD-FREE,
   unified on the correct two-rail venue model (fixed inner + movable racing rail), and rail IDENTITY is
   now **GLOBAL** (`thread_rails`: tracklets + co-occurrence bipartition + one naming vote per component,
   journal 0122) — see §10 "CURRENT STATE" for the live state, the hard constraints (no field, no circle
   fits, visual-gate-only, no-upstream-trust), and the next-wave plan (verification + the non-circular
   global smooth-curve second pass). Work is on branch `cv-rail-identity-threads`; R08 is the gold
   standard (pixel-level regression tripwire). Then wire into the 4-feed fusion.

**Where things live now:** the aerial pipeline is `studies/cv_aerial_track_id.py` (detection → rail →
odometer-anchored field-consensus recon); the rail tracker is `studies/cv_rail_track{3,4}.py`
(+`cv_rail_prep.py`, `cv_rail_audit_dump.py`); detection R&D is `studies/cv_{legend_detect,avatar_shape}.py`;
odometer obs is `studies/cv_build_obs.py`. Library primitives: `hkjc.vision.{aerial_layout,feeds/aerial_rail,
feeds/aerial_detect}`. Caches/overlays in `data_cache/` (git-ignored).

**Roadmap (priority order):**
- **Verify + freeze the rail identity rework** (§10, journal 0122): re-render R08+R02A under the
  tracklet build, §0b red-team audit both, resolve the remaining per-component naming bits, then freeze
  the R08 pixel-level regression tripwire. (The 0115 rail-tracker redesign itself is DONE and promoted.)
- **Phase 2 — global smooth-curve second pass** (non-circular, odometer-anchored "smooth in s"):
  spanning + far-rail fill + the §7 bend metric; also the principled anchor if image-space naming
  evidence stays thin on fragmented config-A components.
- **Generalize detection**: promote a silk-anchored *shape* detector (kills shadow false-positives;
  recall 75–100% on R08/R03/R02A) over the current `detect_avatars` count-forcing (journals 0112/0113).
- **Then integrate** the revamped aerial perception into the library + the 4-feed fusion, and **ablate**
  the now-redundant per-frame rail machinery (`select_running_rail` carry/override, geometry `is_bend`).
- **Studies hygiene:** the `studies/cv_*` set is a tightly-coupled fusion+aerial web (e.g. `cv_r08_aerial`
  imports `cv_avatar_edgefill`) with the library's `extract_feeds` glue not yet wired, so it can't be
  bulk-trimmed safely — trim per workstream with care.

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

**And the seam discipline (PHILOSOPHY §7b, owner-directed): no downstream step may assume upstream
output is correct.** The per-frame fit emits plausible garbage (hedge/margin slivers, a fabricated
914 px arc through the disclaimer text that passed every extent gate — journal 0122); every later
pass must gate its evidence independently, contain a wrong input to its own segment, and decide
global quantities ONCE from pooled evidence rather than chaining per-step decisions.

### 0b. RED-TEAM VISION AUDIT — fan out parallel subagents to scrutinise the overlays

The single reviewer (me) misses things: I closed the f694–710 "weave" as an overlay-fill artifact and
never re-checked f760/840/900, where the two rail threads genuinely CROSS. A fan-out of independent
red-team reviewers caught it. This is now the standing audit gate, and the loop that lets the CV work
**iterate end-to-end without the human in the detection step**:

```
re-render the diag overlays  →  fan out N red-team subagents (parallel, small batches, native vision)
   →  AGGREGATE their per-frame verdicts  →  VERIFY each checkable claim myself by eye (the safeguard;
   subagents are calibrated but make interpretive near/far calls)  →  fix the confirmed issues at root
   →  re-render  →  re-audit.  Done only when every batch returns clean.
```

How to run it (proven on R02A, journal 0119/0120):
- Spawn **6-ish `general-purpose` Agents in ONE message** (so they run concurrently), each a *small batch*
  (~6 frames) spanning one race phase (start / early bend / back straight / late bend / turn / run-in).
- Each agent gets: the colour legend (GREEN=running rail / BLUE=other / CYAN=clean px), the **venue model**
  (fixed far + movable near/racing rail; racing rail = nearest the field; config-A close-on-straight,
  diverge-on-bend), the **NOT-rails list** (grandstand/apron, disclaimer text, odometer minimap, silks,
  shadows), and **what CORRECT looks like** (every curve ON a real rail; short/single-rail is allowed).
- Tell them to **crop+zoom** with `studies`-adjacent helper (`/tmp/crop.py IN OUT x0 y0 x1 y1 scale`, or any
  NEAREST-upscale) — rails are thin, the full thumbnail hides everything; read the full frame first, then
  zoom the curve region (upper-mid band Y~300-560, or the bottom edge Y~520-720 on zoomed bends).
- Demand a **compact per-frame verdict** (`fNNNNN: OK` | `ISSUE [hi|med|lo] — which curve, what, where`)
  + a one-line worst-issue summary, every claim backed by a crop they viewed.
- **Redundancy is the point**: overlapping batches that independently flag the same frame = high-confidence
  real issue; a lone interpretive call = verify before acting. Trust the literal-geometry calls (a crossing,
  a floating curve, a curve on text) over the interpretive ones (near/far identity from camera reasoning).

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
  only when S9 demands it. **Update:** the aerial-feed revamp (§10) attacks exactly this, but the v3 rail
  tracker was falsified at the bend (journal 0115) — it does NOT yet recover the concentric rails through
  the rotating bend. The R3 redesign (a tracked ground→image homography) is the route to the bend metric;
  the §7 limit therefore stands until R1–R5 land.

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

## 10. Aerial-feed deep revamp (current frontier — journals 0108–0114)

A from-scratch rebuild of the AERIAL feed so it alone yields clean per-horse 2D paths, including
through bends (the §7 limit). Pipeline `studies/cv_aerial_track_id.py`: detect blobs → fit the running
rail → project to `(s, perp)` → associate into tracklets → **odometer-anchored field-consensus recon**.

- **Frame classification (0111, promoted to `aerial_layout.render_active`).** The live→render
  cross-dissolve leaks blended frames past a legend-only check. Three required gates: legend strip +
  dark top-strip (`TOP_GREEN_MAX`) + render-level track texture (`TRACK_LAP_MIN`). Validated R08/R03/R02A.
- **Field-consensus factorisation recon (`field_consensus_recon`).** The two residual artifacts —
  box-car lateral flips and longitudinal step jumps — both came from trusting ONE fragile per-frame
  reference (a single outer-rail line; the p96 leader). Fix: replace each with an N-horse **consensus**
  nuisance + one absolute anchor. Lateral `perp = b(f) + lane·scale(f)` (b = common-mode rail-baseline
  jitter; scale trusted-outer where the zoom-gate believes it, else field-filled, clamped near-constant).
  Longitudinal `A = s·scale + C(f)` (C = consensus origin + smooth odometer front-anchor). Sound idea,
  but its "validated at bends" claim is void: the recon rides on the (broken) rail scale at the bend —
  re-validate after the rail redesign (journal 0115).
- **Rail tracker v3 (`studies/cv_rail_track3.py`) — FALSIFIED at the bend (journal 0115).** v3 tracks the
  rail in IMAGE space with a K=5 sliding window that assumes the rail is ~stationary in-panel. That holds
  on straights but breaks under camera ROTATION: at BEND1 the rail swings 35° (up to 12°/frame) and
  translates up to 201 px/frame, so the 18 px assignment gate rejects the moved pixels → the buffer
  starves → the curves collapse to ~0.16–0.19× frame width and the inner/outer labels flip (R08 f734–860).
  The candidate pixels are PRESENT and span the rail throughout — the failure is the tracker, not pixtag.
  Late bend (f2158) fails differently: outer-rail pixel STARVATION (dark boundary + bottom-legend cut).
  KEEP from v3: the orientation-free `fit_curve_pca` representation and the pixel-tagging physics audit
  (top-hat is a dead end; brightness+whiteness+thinness+length is the right discriminator). DROP: the
  stationary-window assumption, independent (non-concentric) two-rail fitting, per-frame nearest-field
  inner/outer labeling, and the fit-over-support-only span. **Redesign R1–R5 below.**
- **Detection R&D (0112/0113, `cv_legend_detect.py`/`cv_avatar_shape.py`).** The current
  `detect_avatars` count-forcing fabricates positions when the ROI clips the field; a **silk-anchored
  shape** detector is shadow-immune and recalls 75–100% (the legend lists only ~8 runners, so it's a
  colour/identity prior, not the count authority).
- **First-principles redesign — BUILT + R08-validated (journals 0115→0116).** Reframe: the rails are
  STATIC rigid world structures; the CAMERA POSE is the smooth thing — track that, enforce the geometry.
  What was actually needed and now works (studies `cv_rail_exp_pixtag.py` + `cv_rail_exp_fit.py`):
  - **Pixtag (R4):** a LONG-SMOOTH-CURVE gate — group thin components into smooth curves and keep only
    pixels on a curve spanning ≥0.30 W with perp spread ≤4 px. Replaces the per-pixel tagger AND the
    bottom-legend cut (outer rail now survives the legend band → fixes f840). Drops silks/text/minimap/
    FINISH. (Dead end: a static-overlay persistence mask — the rail is the MOST persistent pixel, not the HUD.)
  - **Rail fit (R1–R3):** fit each rail INDEPENDENTLY (the constant-pixel-offset coupled model is an
    epicycle that misplaces at the bend); per-rail fit+gather+refit on clean pixels (bridges fragmentation);
    NO extrapolation (clean pixels already span; box-extension shoots diagonals); partner accepted only if it
    does not CROSS the running rail (the one bend-robust discriminator — parallelism tests fail on arcs);
    and **temporal SEEDING** — carry the prev frame's curve as the gather seed but REFIT on the current
    frame's pixels (bridges occluded bend frames with no smearing; single-rail frames 65→1/1227).
  - **Validated on R08** across all phases (incl. the BEND1 catastrophe) by VISUAL inspection of the full-res
    overlays `data_cache/diag/exp_railfit/` (metric used only to triage which frames to look at).
  - **PROMOTED to the library (journal 0117).** The reconstruction now lives in `hkjc.vision.feeds.aerial_rail`
    (one rail module — no sprawl): `clean_pixels` (LOOP 1), `track_sequence`/`fit_rails` (LOOP 2), `build_kappa`/
    `backproject_fixed_r`/`radius_at` (global geometry). The `studies/cv_rail_exp_{pixtag,fit,global}.py` are now
    thin I/O+overlay drivers over it. The port is verified BYTE-IDENTICAL on R08 (LOOP 1 cache identical to the
    pre-port baseline; R08 still 0-rail 0 / 2-rail 1226 / f768=2; f768+f2158 visually re-confirmed). The legacy
    per-frame `select_running_rail` is retained (demarcated) only for the not-yet-ablated full aerial pipeline
    (`cv_aerial_track_id.py`/`cv_pose_backbone.py`) — remove once the new perception is wired in (integrate-then-ablate).
  - **R02A field-proximity swap (journal 0118) — SUPERSEDED + reverted.** It re-asserted RUNNING = the
    rail nearest the FIELD each frame. It helped R02A but **REGRESSED R08 (f796–818)**: there the bunch runs
    WIDE, so the nearest rail is the WRONG one. Lesson banked: **field proximity is not just noisy, it is the
    wrong signal** for near/far. Removed.

### CURRENT STATE — field-free two-rail model (branch `cv-aerial-r02a-recon`)
The CORRECT venue model (owner + HKJC_SPECS §1 + raw-frame inspection): the track has a **FIXED far rail
and a MOVABLE near/RACING rail**. The horses race just **OUTSIDE** the racing rail, which is the rail of
**WIDER radius** (the fixed rail is the narrower, infield boundary). **Config C (R08)** separates the pair
cleanly; **config A (R02A)** sits them ~15–20 px apart on straights (joined by V cross-braces, easily fit
as ONE band) and they **diverge on bends**. The **grandstand apron/tiers below the rails are NOT a rail.**

The pipeline is now **FIELD-FREE end-to-end** (the field detector is not dialled in). `track_sequence`:
  - **Acquisition (field-free):** primary rail = the curve of largest HORIZONTAL (track-direction) span —
    a vertical blob (bunched-start field, gate post, shadow column) is "long" only via height and is rejected.
  - **Tracking by CONTINUITY** (temporal gather-seed), not field; per-frame tracking only maintains the
    curves — identity/naming is the global second pass (`thread_rails` below).
  - **Two-rail pair handling:** no-cross gate lowered 20 px → 6 px so a legitimately CLOSE parallel pair
    (~15 px) is accepted while coincident/crossing (gap≈0) is still rejected; **`_split_band`** (ONE
    mechanism — partition the pixel pool by SIDE of its joint robust midline, refit each side, accept iff
    both sides have rail-grade support and genuine separation) resolves the brace-joined config-A band,
    the weave, and the smooth cross. Its TRIGGER (`WEAVE_FAC`/`CLOSE_FAC`) is the domain-of-validity
    condition, not a bandaid: on widely-separated rails the joint midline collapses onto the denser rail.
  - **NEAR/FAR identity + naming (field-free, GLOBAL — `thread_rails`, journal 0122):** confident curves
    (extent ≥ 0.30·W AND support-occupancy ≥ 0.55 — extent alone is not rail evidence) chain into
    TRACKLETS (relative 2×2 set-distance linking when both rails are seen — rotation-proof; lone-curve
    links capped below the config-A pair gap, so a broken continuity ENDS the tracklet instead of
    guessing); a lone curve matching BOTH alive tracklets = the config-A pair MERGED into one band →
    labeled racing, identity restarts at the split; co-occurrence edges 2-COLOR each component of the
    tracklet graph and ONE wider-radius vote per component (side-of-common-CENTER — projection-robust;
    owner rule, HKJC_SPECS §1: racing = the wider/further rail) names which color is racing. A sustained
    mid-video racing↔fixed swap is impossible by construction. Non-confident slivers inherit a name by a
    DIRECTED nearest-locus cost and carry no evidence. Replaces `_assign_racing_by_radius` + the
    windowed `couple_rails` (0121, falsified: a swap sustained past the window became its own majority).
  - **Rail curve = the DIRECT support fit** (fit to the frame's own pixels); the global-κ `backproject_fixed_r`
    is invoked ONLY to fill a genuinely short sliver and only if consistent with the pixels.
  - **PIXTAG continuity gate (Phase 1, journal 0119 — DONE).** `clean_pixels` keeps a spanning line iff it is
    **THIN** (`spread ≤ S_THIN`, ~1.2 px — a razor-thin line IS a rail even when a bend cuts it gappy) **OR
    CONTINUOUS** (`occ ≥ OCC_HI = 0.80` — one near-complete curve; HUD/disclaimer text scatters to occ ≤ 0.7),
    capped at the 2D-block ceiling. This REPLACES the old coverage+adaptive-thr keep (which let thin text leak
    AND let a thin text line knife-edge the real rail out). `_occupancy()` lives in `enumerate_lines`.
  - Metrics (triage only; visual is the gate): **R08 gold HELD** — f768/f2158/f734/f800/f860 all two-rail,
    zero-rail 0 (5 frames slipped two→one: 3 are text-artifact removals, 2 are a gappy far rail → Phase-2 fill).
    **R02A zero-rail 22 → 0**; rails/frame 0:0 / 1:651 / 2:475 (singles are the genuine config-A reality).

### HARD CONSTRAINTS / lessons (do NOT violate)
  - **NO field detector, anywhere** — not dialled in, and the field runs wide so proximity mislabels near/far.
  - **NO circle fitting** — proven disastrous historically. The rails are GENERAL smooth curves, not arcs.
    The current `backproject_fixed_r` (a fixed-radius CIRCLE) is therefore suspect and slated for removal.
  - **VISUAL inspection on full-res frames is the only gate**; counts/spans are for TRIAGE (selecting frames).
  - **Continuity** cleanly separates rails from HUD text: a real rail's along-length support occupancy is
    ~0.96–1.0; disclaimer-text / text-contaminated curves are ~0.25–0.54 (word/glyph gaps). The same
    principle at the CURVE level (`_occ_frac`) gates identity evidence — a long fitted arc with patchy
    support (the R02A f690 fabricated 914 px sweep through the disclaimer text) is NOT a rail observation.
  - **No upstream trust (PHILOSOPHY §7b)** — every downstream pass gates its own evidence, contains a
    wrong input to its own segment, and decides global quantities once from pooled evidence.

### REMAINING PATHOLOGIES (visually identified, owner-confirmed)
  1. ~~**Green-on-artifacts** (R02A f340, f2052–2088)~~ — **RESOLVED by the Phase-1 pixtag continuity gate
     (journal 0119).** f340/f2052/f2070 now show only the correct racing rail; the disclaimer-text / grandstand /
     minimap band is dropped. The far/blue rail stays correct.
  2. ~~**Rail-separation weave/crossing** (R02A f694–710)~~ — **was an OVERLAY ARTIFACT, not a library defect.**
     The library returns a correct short outer SEGMENT there (fit-to-pixel med_dist ~0.6 px); the driver's
     circle-based `backproject_fixed_r` fill EXTRAPOLATED that sliver across the box → a straight line shooting
     off the curving rail (the "floating outer"). Removed the fill: the driver now draws the DIRECT support fit
     (commit `4b91784`); f694/702/710 verified clean (short segments ON the pixels), R08 gold spans edge-to-edge
     unchanged. Advances the NO-circles directive (`backproject_fixed_r`/`build_kappa`/`radius_at` now dead in
     the active path — kept only until Phase 2 lands their non-circular replacement).
  3. R02A single-rail FILL (651 frames) + R08 f816/f956 — the far rail is faint/gappy / out-of-band but smooth
     in odometer-s and present in neighbours. **(→ Phase 2 fill; not chased with a gate bandaid — that re-admits text.)**

### NEXT WAVE
  - **Phase 1 — PIXTAG precision — DONE (journal 0119).** Continuity (occ) OR thinness (spread) keep rule;
    killed the green-on-artifacts class; R08 gold held, R02A zero-rail 22→0. See CURRENT STATE above.
  - **Phase 2 — GLOBAL geometry + physics second pass (GENERAL SMOOTH CURVES, NO circles).** The per-frame
    LIBRARY output is now correct (racing rail everywhere; outer = correct where seen, short/absent in config-A
    reality), so Phase 2 is no longer a pathology fix — it is the SPANNING + FILL + metric layer: model each
    rail as a smooth curve evolving smoothly with odometer arclength `s`; COUPLE the pair (NON-CROSSING,
    separation a smooth positive function of `s`); robust global fit to the Phase-1 per-frame curves (jumps /
    text residue get zero weight, re-derived from the model). Fills the far rail on single-rail frames (it is
    smooth in `s` + present in neighbours: R08 f816/f956, the 651 R02A singles) and provides principled
    edge-to-edge spanning to replace the removed circle fill. Image-space first; ground-space (odometer-
    anchored, non-circular) is the eventual home and also yields the §7 absolute lateral metre. **Owner to
    review the global-model design before build (per the standing request).**
  - Then: wire the promoted perception into the 4-feed fusion + ablate the legacy `select_running_rail`.

**Diag sets (current, regenerated from HEAD):** `data_cache/diag/exp_railfit{,_R02A}/` (rail fit) +
`exp_pixtag{,_R02A}/` (kept=green/dropped=red). Caches `data_cache/_railclean_{R08,R02A}.npz`. Studies are
thin drivers `studies/cv_rail_exp_{pixtag,fit,global}.py`; algorithm lives in `hkjc.vision.feeds.aerial_rail`.

### IDENTITY REWORK STATUS (journals 0122→0123, branch `cv-rail-identity-threads`)
The `thread_rails` global identity (design in CURRENT STATE above) replaced `_assign_racing_by_radius`
+ `couple_rails` after the owner found racing↔fixed swaps SURVIVING mid-video on R08.
- **R08: RE-VERIFIED under the tracklet build** (gold + ex-swap frames + 2-batch red-team fan-out;
  identity correct everywhere) and **FROZEN as the pixel-level regression tripwire** (owner request):
  `studies/cv_rail_freeze_r08.py` → `data_cache/rail_tripwire_R08.npz` + `tests/test_rail_tripwire.py`
  (median ≤2 px / p95 ≤15 px / exact labels; skips without data_cache; PASSES). Re-freeze only after
  a re-verified visual audit. Residual R08 defects frozen as-is (upstream classes, not identity):
  LOOP 1 dropouts (f564 far rail, f956 near rail) + ~50-250 px end under-coverage on rotation frames.
- **R02A: 6-batch §0b fan-out (36 frames) found ONE root cause for every naming failure (0123):
  fabricated disclaimer-text arcs** (occ 0.50-0.85 — the 0.55 occupancy gate cannot separate them
  from R08's gappy bend rails) **become confident tracklets, and the side-of-center vote then
  CORRECTLY ranks them wider → the text steals the racing name** (comps 6/7/10; comp 7 additionally
  stitched a real rail onto a text tracklet via the thresholdless 2×2). Comp 8/13/14 + start/tail
  are correct. **Next: reject the fabricated-arc class UPSTREAM in the fit (support COMPOSITION —
  few-long-components vs dozens-of-glyph-blobs), then re-audit; the tripwire is the safety net.**

## 9. Journal trail

0073 (feed map) · 0075 (aerial render-inversion) · 0076 (patrol) · 0077 (fusion spine + shared clock)
· 0078 (minimap = dead) · 0080 (drone, first cross-feed agreement ρ+0.93) · 0082 (patrol silk identity)
· 0086 (RF order+margins) · 0084/0087 (fusion v1 capstone) · 0088 (R03 generalization) · 0089 (identity
v1, the honest regression) · 0090 (identity v2 consensus + draw) · 0091/0092 (R08 held-out PASS + the
bend-metric limit quantified) · 0093 (lateral physical constraints: hard box win, ordering inert on routes)
· 0094 (phase-gated physics longitudinal kernel; squiggle ≡ order) · 0096 (walkthrough video) · 0095/0097
(library promotion: vision perception + reconstruction halves).

**Aerial-feed revamp (§10):** 0108 (green-box odometer + R03 generalize) · 0109 (A-config R02A caveat)
· 0111 (render_active clean-render gates) · 0112 (legend-templated silk detection) · 0113 (shape-template
avatar detection) · **0115 (rail tracker v3 FALSIFIED at bends — root causes + R1–R5 redesign;
supersedes the removed 0110 field-consensus recon + 0114 temporal rail tracker).** (0104–0106 were
earlier folded into 0110/0114; all four removed.)
