# CV Trajectory Pipeline — Architecture & Plan

Extract, **per horse**, a dense time series of track position and speed from race
**video pixels only** (worst case: no positional data feed), to feed the
physics-based energetics model (`v_c`, `W'`, efficiency, pace preference) that today
is fit from only 2–4 sectional split times.

This document is the **design**; the scaffold lives in
`experiments/trajectory/`. It does **not** modify anything under `src/`.

> Context: read this together with `docs/TRAJECTORY_RESEARCH_PLAN.md` (Tier-2
> section). That plan's V1 finding stands and **dominates this whole document**:
> HKJC's Aerial Virtual View is rendered from an RF/GPS positioning system, so an
> upstream `(x, y)` feed *probably exists*. **Always exhaust the feed hunt (V1)
> before building any CV.** Everything below is the fallback for when there is no
> feed — i.e. "we only have pixels."

---

## 0. TL;DR recommendation

1. **First, don't do CV at all** — re-run V1 (network/websocket recon of the
   Aerial Virtual View) to find the raw position feed. Order of magnitude cheaper
   and more accurate than any pixel pipeline.
2. **If pixels only: start with Mode B (the virtual render), not Mode A (broadcast).**
   Mode B is near-rectified, has stable distinct markers, legible saddlecloth
   numbers, and on-screen distance markers — it removes ~80% of the hard CV
   (calibration, occlusion, re-ID). Treat Mode B as the way to *re-derive* the
   trajectory the render is drawing.
3. **Mode A (moving broadcast camera) is a research project**, not a deliverable.
   Gate it behind a confirmed signal from Mode-B/sectional trajectories and accept a
   materially lower accuracy ceiling. See §7 for the honest comparison.

The whole pipeline is **only worth building if the energetics residual signal is
confirmed worth sharpening** (TRAJECTORY_RESEARCH_PLAN gate). Don't build a
trajectory rig for a ~zero signal.

> **UPDATE 2026-06-04 — "Mode B" is now the DRONE replay, and it's a different beast.**
> The Trakus *virtual render* this doc calls "Mode B" is **dead** (404s on every race; see
> `FEED_RECON.md` §UPDATE). The live overhead source is the **`replay-drone`** angle of the
> Multi-Angle player — **real drone-camera footage**, not a render. Implications vs the
> assumptions above:
> - ✅ **Homography is easy** as hoped: near-top-down oblique, **both rails crisp & continuous**
>   → register to `hkjc.track` geometry. This is the genuine reason to prefer it over side-pan.
> - ❌ **Detection is HARDER, not easier** (the render's "rendered consistent markers" don't
>   exist — these are real horses). In a wide overhead frame the field is a **tiny cluster
>   (~15–30 px/horse)**: stock **YOLOv8n @ imgsz=640 detects ZERO** (verified, PoC). Needs
>   **high-res inference (imgsz≈1920) and/or SAHI tiling, and a small-object fine-tune.**
> - ❌ **Identity = real-silk OCR**, not a rendered glyph — same difficulty as Mode A.
> So the drone is a **"Mode B′": Mode-B-easy geometry + Mode-A-hard detection/ID.** Net it's
> still the best target (homography is the part that actually caps `off`/lateral accuracy), but
> budget for small-object detection. PoC: `experiments/trajectory/poc_visualize.py`
> (`POC_IMGSZ`, `POC_MODEL`, `POC_CONF` env knobs). **Data-licensing is OUT (too expensive),
> so step 1 above is moot — drone CV is the only path.**

---

## 1. World model we map INTO (from `src/hkjc/track.py`)

The output coordinate system is defined by `TrackGeometry` (do not modify it):

- `circumference` — one lap (m); HV 1450, ST turf 1900, ST AWT 1560, ST straight 1000.
- `courses[course] -> (home_straight_m, usable_width_m)` — the **movable rail**
  shifts both per race; width is 18–31 m depending on course.
- `bend_segments` — `((start_d, end_d), ...)` in metres-from-finish (`d`), the TURNS.
- `handedness` — right-handed (clockwise) for all HK ovals; inner rail favours low draws.

Native coordinates, matching the model's convention:

- **`d`** = metres-from-finish (arc-length; a horse runs from `d=distance` down to `d=0`).
  This is `circumference - s` modulo a lap, where `s` is arc-length from a chosen origin.
- **`off`** = lateral metres outward from the **current inner rail** (0 = on the rail,
  up to `width(course)`). On a bend, being `off` wider adds `off · turn_angle` of arc —
  this is the *ground-lost* mechanism the traffic model integrates, and the single
  highest-value hidden signal.
- Optionally world **`(x, y)`** in a venue-fixed metric frame, from which `(d, off)`
  is derived by projecting onto a centreline spline. `(x, y)` is the convenient
  intermediate; `(d, off)` is what the model consumes.

**Reference landmarks** available for calibration (these are the knowns we anchor to):
- Finish line (`d = 0`), winning post.
- Distance markers / sectional sampling points: HKJC paints/segments the track at the
  sectional points (HV bend boundaries ~300, 770, 920 m; ST ~430, 925, 1405 m); the
  virtual view overlays numeric markers (e.g. "420M", "920M").
- The two rails (inner running rail at the current course offset; outer rail).
- Known per-course `home_straight` and `width` give a metric scale on the straight.

---

## 2. Architecture (ASCII)

```
                        ┌─────────────────────────────────────────────────┐
                        │  PRE-STEP (do FIRST): hunt for the position FEED │
                        │  network/ws recon of Aerial Virtual View.        │
                        │  If found → SKIP all CV, parse (x,y) ticks.      │
                        └───────────────┬─────────────────────────────────┘
                                        │ no feed → pixels only
                                        ▼
 video file ─► [1 INGEST] ─► frames @ target fps, metadata, time base, race id
                                        │
                                        ▼
              [2 DETECT]  per-frame horse/marker boxes (+ optional keypoints)
                                        │
                                        ▼
              [3 TRACK]   tracklets: stable per-object IDs across frames
                                        │
                                        ▼
              [4 IDENTIFY] map track_id ─► saddlecloth number / silks ─► runner
                                        │
            ┌───────────────────────────┴───────────────────────────┐
            ▼ (Mode A: moving cam)                  (Mode B: render) ▼
   [5A CALIBRATE/HOMOGRAPHY]                  [5B CALIBRATE (lighter)]
   per-frame pixel→world H from               near-rectified; fit one
   rail/marker landmarks + temporal           homography per shot / slow
   smoothing; hardest stage.                  drift; markers are exact.
            └───────────────────────────┬───────────────────────────┘
                                        ▼
              [6 PROJECT]  pixel (u,v) ─► world (x,y) ─► (d, off) via
                           centreline spline from TrackGeometry; scale recovery
                                        │
                                        ▼
              [7 TRAJECTORY]  per-runner resampled series: t, d, off, speed,
                              accel; smoothing (Kalman/RTS); gap-fill; traffic events
                                        │
                                        ▼
              [8 EXPORT]   typed records (Parquet/JSONL) → energetics model
                                        │
                                        ▼
              [9 QA]   cross-check recovered section times vs HKJC published
                       sectionals; finish order vs official; per-horse residuals
```

---

## 3. Per-stage plan, tool choices, rationale

### Stage 1 — INGEST (`ingest.py`)
- **Job:** open the video, emit frames at a target rate with a monotonic timestamp,
  attach race identity (`venue, date, race_no, course, distance, surface`) so we can
  resolve `TrackGeometry` via `track_for(...)`.
- **Tools:** `pyav` (preferred — exact PTS timestamps, no frame-drop ambiguity) or
  OpenCV `VideoCapture` (simpler, but seek/timestamp is flaky). FFmpeg for any
  transcode/keyframe work.
- **Frame rate:** see §4. Decode native (usually 25/30 fps); *downsample in the
  pipeline*, not at decode, so timestamps stay exact.

### Stage 2 — DETECT (`detect.py`)
- **Job:** per-frame bounding boxes for horses (and, in Mode B, the silk/number
  markers and distance-marker glyphs).
- **Choice:** **YOLO (Ultralytics, v11)** as the default detector — best
  speed/accuracy/tooling trade-off, trivial custom-class fine-tuning, exports to
  ONNX/TensorRT. **RT-DETR** as the upgrade path when small, clustered, occluding
  horses (Mode A) hurt YOLO — DETR-family handles dense overlap and needs no NMS
  tuning, at higher compute. Don't start on RT-DETR; start on YOLO, escalate if the
  occlusion metrics demand it.
- **Mode B specific:** the markers are *rendered and consistent* → a tiny fine-tune
  (or even classical template/colour matching) detects them near-perfectly. This is
  why Mode B is easy.
- **Training data:** bootstrap with a pretrained COCO `horse` class, then fine-tune
  on a few hundred HK frames auto-labelled and hand-corrected (Mode B labels are
  cheap because appearance is synthetic and uniform).

### Stage 3 — TRACK (`track.py`)
- **Job:** link detections into tracklets with persistent IDs through occlusion.
- **Choice:** **BoT-SORT** (default) — motion (Kalman) + camera-motion
  compensation + appearance Re-ID, which matters under HK's heavy mid-pack
  occlusion; ships inside Ultralytics so it pairs with the YOLO detector with no glue.
  **ByteTrack** as the fast baseline (associates low-confidence boxes well, no Re-ID —
  fine for Mode B where appearance is clean and crossings are rare). **OC-SORT** if
  long full occlusions cause ID switches (better motion recovery after gaps).
- **Reality:** trackers give *temporary* IDs. They **will** switch identity at
  bunching, the rail, and the line. Identity is fixed by Stage 4, not here — the
  tracker only needs to be locally consistent between re-ID anchors.

### Stage 4 — IDENTIFY (`identify.py`) — the crux of "per-horse"
- **Job:** bind each tracklet to a **runner** (saddlecloth number → horse), and keep
  that binding stable across ID switches.
- **Saddlecloth OCR:** **PARSeq** (state-of-the-art scene-text recognition, strong on
  short numeric strings, robust to blur) as the engine; **EasyOCR** as the
  quick-start baseline. Run OCR only on the saddlecloth ROI (crop top of the box),
  not the full frame.
- **Silks colour classification:** a small CNN / colour-histogram classifier on the
  jacket ROI, trained from the race's **declared silks** (HKJC publishes per-runner
  silks) — a strong *prior* that disambiguates when OCR fails. In **Mode B** the
  markers are synthetic and per-runner-distinct, so silks/number ID is near-trivial
  (often a fixed colour+number glyph) — again why Mode B is recommended.
- **Fusion / stabilisation:** accumulate OCR+silks votes per tracklet over its whole
  life; assign by **constrained global assignment** (Hungarian) under the
  one-number-per-runner constraint and the known field size, so the final mapping is
  a permutation. Re-anchor after every occlusion. Treat per-frame OCR as noisy votes,
  never as a hard per-frame label.

### Stage 5A — CALIBRATE / HOMOGRAPHY (Mode A only) (`homography.py`)
The hard stage; the whole reason Mode A is a research project.
- **Job:** recover, per frame, the pixel→world map `H` (planar homography; the track
  is ~flat) for a pan/tilt/zoom camera.
- **Approaches (in increasing robustness/cost):**
  1. **Landmark / rail detection** — segment the inner/outer rail lines and detect
     painted distance markers / the finish line; solve `H` from ≥4 correspondences to
     known world landmarks from `TrackGeometry` (rail position from `course`, marker
     `d` values, `home_straight`/`width` for scale). Cheapest; brittle when few
     landmarks are visible (tight zoom on the pack mid-bend).
  2. **Keypoint matching / frame-to-frame** — track features (SIFT/ORB or learned,
     e.g. SuperPoint+SuperGlue / LoFTR) between consecutive frames to get inter-frame
     homography, and *chain* it from anchor frames where absolute landmarks are
     visible. Handles long stretches with no landmarks but **drifts** — must be
     re-anchored whenever a landmark reappears.
  3. **Learned field/track registration** — the sports-broadcast "field
     registration" approach (à la soccer-pitch homography nets: TVCalib / robust
     keypoint-to-template models): train a network to regress the track template pose
     directly. Highest ceiling, needs a per-venue annotated template and training
     effort. The right *eventual* answer for Mode A; not the starting point.
- **Plan:** start with (1)+(2) hybrid — absolute landmark solve when ≥4 references are
  in view, ORB/LoFTR chaining between, RTS-smooth the homography time series. Escalate
  to (3) only if the section-time QA (§6) won't close.

### Stage 5B — CALIBRATE (Mode B) (`homography.py`, lighter path)
- The render is **near-top-down and near-rectified**, with a 3D camera that pans/zooms
  but exposes **exact** rail lines and numeric distance markers.
- One homography per shot (or a slowly-drifting one) suffices; markers give *exact*
  world references (no painted-line ambiguity). Often the view is piecewise-static
  enough that a handful of homographies cover a whole race.
- This collapses Stage 5 from "research project" to "fit a few homographies from
  labelled markers" — the core reason to start with Mode B.

### Stage 6 — PROJECT (`project.py`)
- **Job:** pixel `(u, v)` → world `(x, y)` via `H`, then `(x, y)` → `(d, off)`.
- **Centreline model:** build a centreline + rail spline for the venue/course from
  `TrackGeometry` (`circumference`, `bend_segments`, `home_straight`, per-course
  `width`/rail offset). `(x, y) → (d, off)` is the nearest-point projection onto the
  spline: arc-length along it gives `d` (metres-from-finish), signed normal distance
  gives `off` (outward from the current inner rail).
- **Foot point, not box centre:** project the **ground-contact point** (bottom-centre
  of the box, ideally the hooves) so height-above-plane doesn't bias position. Box
  centre projects the horse's *body height* onto the track and adds a systematic,
  zoom-dependent error.
- **Straights vs bends (see §5).**

### Stage 7 — TRAJECTORY (build inside `project.py`/`export.py`)
- Resample each runner to a uniform grid (e.g. 5–10 Hz), **Kalman + RTS smoother** on
  `(d, off)` to get clean `speed`/`accel`; constrain to monotone-ish `d` (a horse
  doesn't run backwards in `d`). Gap-fill short occlusions via the motion model;
  flag long gaps. Derive **traffic events**: blocked/steadied (sharp negative accel
  with a runner directly ahead in `d` within a small `off` band), wide trip
  (sustained high `off` on a bend → ground lost), checked, switched.

### Stage 8 — EXPORT (`export.py`)
- Emit the typed schema (§ schema in code) to **Parquet** (columnar, plays with the
  existing pandas/feature-store stack) plus a JSONL/`RaceTrajectory` summary.
  See `experiments/trajectory/export.py` and the `TrajPoint`/`RunnerTrajectory`
  dataclasses.

### Stage 9 — QA / VALIDATION (`qa`, lives in `export.py` for now)
- **Primary check — recovered section times vs HKJC published sectionals.** Integrate
  recovered speed to get the time to cross each sectional `d`-boundary; compare to the
  scraped `get_sectional_times(date, race_no)` (already built per the research plan).
  Target: per-section time error within published rounding (~0.1–0.3 s) for Mode B.
- **Finish order / margins** vs the official result (cheap, strong sanity check).
- **Total distance** ≈ race distance (per runner; wide trips legitimately exceed it).
- **Cross-runner consistency:** the field's pace curve should match the known
  early/late split structure.
- **Scale sanity:** speeds in a plausible band (~14–19 m/s gallop); reject frames
  whose `H` implies absurd scale.

---

## 4. Frame rate & sampling

- **Decode native** (25/30 fps). The model needs only **~5–10 Hz** of *smoothed*
  position, but detect/track benefit from native fps for association through fast
  motion and occlusion. Run detect+track at native fps where compute allows;
  otherwise detect every 2nd–3rd frame and let the tracker's motion model interpolate.
- A 1000–2400 m race is ~57–145 s; at 25 fps that's ~1.4–3.6k frames — small. Compute
  is not the bottleneck per race; **labelling and calibration** are.
- Smoothing window must respect the physics: don't over-smooth the **acceleration**
  out of the trace — `W'` depletion lives in the accel/high-speed phase.

---

## 5. Scale recovery, straights vs bends

- **Scale** comes from world references: per-course `width` across the track and the
  spacing of distance markers / sectional boundaries along `d`. On the straight, the
  metric `home_straight` length and known rail-to-rail `width` pin the scale directly.
- **Straight (finish straight, and ST 1000 chute):** `bend_segments` empty or far; the
  map is closest to a clean planar homography, scale is most reliable, `off` is a
  simple cross-track distance. Calibrate and validate here first.
- **Bends:** `(d, off)` is curvilinear — `d` is arc-length along the curved centreline
  and `off · turn_angle` is the ground-lost term. Two hazards: (a) homography is
  weakest mid-bend (fewest landmarks, sharpest perspective), (b) the centreline spline
  must match HKJC's actual bend radius, which `track.py` only *approximates* for the ST
  oval/AWT (bend boundaries reconstructed, not surveyed). So **bend `off`/ground-lost
  has a structurally lower accuracy ceiling than straight-line speed** — state this in
  any downstream feature's error budget. HV (firm bend map from sectional points) is
  more trustworthy on bends than the ST reconstructions.

---

## 6. Mode A vs Mode B — effort / risk / accuracy

| Dimension | **Mode A — broadcast/aerial camera** | **Mode B — Aerial Virtual View (render)** |
|---|---|---|
| Camera | Moving pan/tilt/zoom, unknown intrinsics | Moving 3D cam but synthetic, near-top-down, exposes exact rails/markers |
| Detection | Hard: small, similar, occluding horses | Easy: uniform synthetic markers |
| Tracking | ID switches at every bunch | Few crossings, clean appearance |
| Identity | OCR under motion blur + occlusion (hard) | Legible numbers / distinct silks (easy) |
| Calibration | **Per-frame homography — the blocker** | A few homographies from exact markers |
| Accuracy ceiling | Coarse; lateral/ground-loss noisy, esp. mid-bend | High for `d`/speed; good for `off` (but *render itself is RF-lossy*) |
| Effort | Research project (months) | Engineering task (weeks) |
| Risk | High (calibration may never close to sectional tolerance) | Moderate; main risk is parsing the render reliably |
| Data caveat | "Ground truth" pixels of the real race | Render is **derived from a lossy RF feed** — good for lane/ground-loss calibration, **not** sub-metre truth |

**Recommendation: Mode B first.** It is far cheaper and accurate enough for the
energetics use-case (speed curve + lane/ground-loss), and — crucially — if the feed
hunt (V1) succeeds, Mode B is moot anyway because you read the numbers directly.

---

## 7. Phased roadmap

- **Phase 0 — Feed hunt (always first).** Re-run V1 network/websocket recon of the
  Aerial Virtual View for the raw `(x, y)` tick feed. If found, parse it (no CV) and
  jump straight to Stage 7/8. Highest leverage; cheapest.
- **Phase 1 — Mode B PoC, single race, straight only.** Ingest one ST 1000 m straight
  race; detect+ID markers; fit homography from on-screen distance markers; project to
  `(d, off)`; export; **QA against published sectionals**. Decision gate: do recovered
  section times land within sectional rounding?
- **Phase 2 — Mode B full lap (bends).** Add the centreline spline + bend handling;
  validate `off`/ground-loss on HV (firm bend map) and ST. Produce full
  `RunnerTrajectory` for a handful of races; feed the energetics fit; check whether
  dense `v_c`/`W'` beat the 2–4-split fit and the market.
- **Phase 3 — scale Mode B.** Batch many races; robustify the render parser; build the
  `DataSource` that joins on `(date, race_no, runner)` to the feature store.
- **Phase 4 — Mode A, only if justified.** Start broadcast CV (landmark+chaining
  homography, escalate to learned registration) **only** if a real signal is
  confirmed *and* Mode B/feed can't supply it (e.g. older races with no virtual view).
  Accept the lower accuracy ceiling.

Each phase is gated by the energetics-signal check — don't advance a phase for a
~zero downstream edge.

---

## 8. Known failure modes & mitigations

| Failure | Where | Mitigation |
|---|---|---|
| ID switch at bunching / the rail / the line | Track | Re-ID anchors (Stage 4) + global one-number-per-runner assignment; BoT-SORT/OC-SORT over ByteTrack |
| OCR fails on blur/occlusion/obscured cloth | Identify | Vote over the whole tracklet; fuse with declared-silks colour prior; Hungarian permutation constraint |
| Homography drift with no landmarks mid-bend (Mode A) | Calibrate | Chain frame-to-frame + re-anchor on landmark reappearance; RTS-smooth; escalate to learned registration |
| Wrong scale → absurd speeds | Project | Cross-track `width` + marker-spacing scale checks; reject frames with implausible `H`; speed-band gate |
| Box-centre vs foot-point bias (zoom-dependent) | Project | Always project ground-contact (hooves/box bottom-centre) |
| Bend geometry mismatch (ST/AWT bends only approximate in `track.py`) | Project | Treat bend `off`/ground-loss as lower-confidence; prefer HV for bend validation; quote a wider error bar |
| Over-smoothing kills the accel signal | Trajectory | Tune smoother to preserve high-speed/accel phase (where `W'` lives) |
| Render is RF-lossy "truth" (Mode B) | Whole | Use Mode B for lane/ground-loss calibration, not sub-metre ground truth; cross-check vs sectionals, not vs itself |
| Section times won't match published sectionals | QA | This is the **kill criterion** — if Mode B can't hit sectional tolerance, the pixel ceiling is too low; fall back to Tier-1 sectional features |
| Occlusion gaps too long to motion-fill | Track/Traj | Flag gaps explicitly in the schema (`gap` mask); never silently interpolate across long dropouts |
| Movable rail / course not accounted for | Project | Resolve `course` per race and offset the rail spline; never assume course "A" |

---

## 9. What this is NOT

- Not a real-time / in-running system — this is post-race batch trajectory extraction.
- Not a replacement for the feed hunt — if the RF feed is reachable, use it.
- Not sub-metre surveying — the accuracy target is "good enough to fit a stable
  per-horse speed curve and a useful ground-loss term," validated against published
  sectionals, not a total station.
</content>
</invoke>
