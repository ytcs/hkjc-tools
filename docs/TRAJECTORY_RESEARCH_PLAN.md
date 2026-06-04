# Dense Trajectory / Physics Model — Research Plan (parallel line)

A **secondary, parallel** research line (carryover remains the main line — it's a
simpler, structural strategy). The thesis: today we record one number per horse
per race (finish time). Sectional splits and race video can turn that **sparse**
signal into a **dense** one (a velocity/position curve per horse), from which a
**physics-inspired energetics model** can recover deep, stable, per-horse latent
features — possibly an edge the market (which sees only the chart) underprices.

It is **orthogonal but synergistic**: it doesn't compete with carryover, it
strengthens carryover's weakest dependency — a better per-leg Trio model (RQ6) →
more efficient Flexi cover — and it reopens the model-edge exotics that journal
0011 closed off with crude features.

> **DATA IS THE BLOCKER (current priority).** Modelling below is **design notes,
> not a build** — we solve data acquisition first. Status: ⬜ todo · 🟡 partial.

---

## 1. Data acquisition plan (do this first)

### Tier 1 — Sectional splits + running positions  *(low effort, do first)*
What we want: per-section split **times** (HK publishes ~per-400m/section splits)
and **running positions** at each call, per horse per race. ~6 dense points/horse —
an immediate density jump, structured (no CV needed).

What we already have: result pages give `RunningPosition` (sectional *positions*)
and total `FinishTime`; SpeedPro JSON carries per-run energy/sectional figures.
The missing piece is the per-section **split times**.

Data-source problem (🟡): the "Sectional Time & Position" page is a **JS SPA**
(`racing.hkjc.com/en-us/.../displaysectionaltime`); the legacy `.aspx` 302-redirects
to it and serves no server-side table. No obvious data XHR fired on first load
(only analytics), and some races show "released shortly".
- **Action S1 ✅ DONE (journal 0020):** the "released shortly" failure was a
  URL-format bug. Correct format `displaysectionaltime?racedate=DD/MM/YYYY&RaceNo=N`
  (lowercase, day-first, no `Racecourse`) is **server-rendered** — scraped with
  plain `requests`. Built `hkjc.data.sectional.get_sectional_times(date, race_no)`
  → tidy per-horse, per-section position + split time, keyed to
  (date, race_no, runner). Verified live.
- **Action S2 ⬜ next:** wrap as a `SectionalSource` (`DataSource`) and join to the
  feature store; also pull `racecard` (27 cols — see HKJC_DATA_CATALOG.md) for
  ratings/gear/body-weight.
- **Action S3 ⬜ next:** derive `v_c`/finishing-speed features and run the cheap
  harness kill/confirm vs the market baseline.
- **Action S2:** build a `SectionalSource` (`DataSource`) → per-(race, runner)
  split times + positions, joined on our canonical keys.
- **Action S3 (cheap kill/confirm):** derive 1–2 sectional features (finishing-
  speed, sustained-speed split, a crude `v_c`) and run through the existing harness
  — *do sectional features beat the market baseline where draw+weight couldn't?*

### Tier 2 — Video trajectory extraction  *(high ceiling — but the data may already exist upstream)*
What we want: per-frame `(x, y)` track position per horse → full kinematics
(velocity, acceleration, **actual distance travelled**, lateral path/lane, drafting).

**Key finding (2026-06, from inspecting `aerial_sample.mp4`):** HKJC publishes an
**"Aerial Virtual View"** — a *synthetic 3D render* of the race, not drone footage.
Its own disclaimer ("accuracy may be affected by mobile-phone signal interference
at the racecourses; refer to Real Replay for actual results") reveals it is rendered
from an **RF/GPS horse-positioning system**. So the dense per-horse `(x, y)`
trajectory data **already exists at HKJC** and the hard real-footage CV problem
(moving broadcast camera, occlusion, re-ID) is already solved upstream — we just need
the positions. This reorders the plan:

- **Action V1 (do first, cheap):** browser/network recon of the Aerial-Virtual-View
  page — look for the **underlying position feed** (JSON/websocket of `(x,y)` per
  horse per tick, analogous to the live-odds GraphQL we already tap). If it exists,
  we get dense, ~25 Hz, lateral+longitudinal trajectories with **no CV at all** — the
  maximal-granularity outcome. This is the single highest-leverage data check.
- **Action V2 (fallback CV, only if no feed):** the virtual view is far more tractable
  than broadcast footage — clean rendered silk-icons (distinct colours), a schematic
  track with rail lines and **distance markers (e.g. 420M, 920M)** for calibration,
  and a saddlecloth/silks legend for ID. The remaining hard part is per-frame
  **camera→track homography** (the render uses a *moving 3D camera* that pans/zooms),
  recoverable from the rail/markers. Gate behind a single-race PoC. Note the virtual
  view is *lossy* (RF noise) — good enough for lateral-lane / ground-loss calibration,
  not for sub-metre ground truth.
- **Gate:** build either only after the energetics residual (the one live signal —
  faint `v_c` edge, strongest at middle distances; journal 0035) is confirmed worth
  sharpening. Don't build a trajectory pipeline for a ~zero signal.

---

## 2. Modelling ideas  *(design notes — pending Tier-1 data)*

### Energetics / critical-power model (the core)
Borrow the **CP–W′** framework from running physiology: each horse has a
sustainable aerobic speed `v_c` and a finite anaerobic reserve `W′` that depletes
when running above `v_c` (and partially recovers below it). Fit the velocity
profile `v(t)` (from sectional splits, later full trajectory) per run to recover
latent **`(v_c, W′, efficiency, peak acceleration, preferred early pace)`**.
- These parameters are **intrinsic to the horse and stable across races** — unlike
  noisy finish positions — so they are the "true" features.
- Fit **hierarchically / Bayesian** across all of a horse's runs (partial pooling),
  which **regularizes the ~800-races/year small-sample problem**: each race is now
  information-rich, and a few latent params explain many observations.

### Path/geometry features (Tier 2)
- **Ground lost**: actual distance travelled vs race distance — a horse wide on the
  turns ran further for the same time → genuinely faster than the chart shows. The
  single highest-value hidden signal.
- **Drafting/slipstream** and traffic: energy saved behind cover; trouble lines.

### Counterfactual pace prediction (the edge mechanism)
Performance is pace-dependent. With per-horse energetics you can **simulate how a
horse runs under a *different* pace** — specifically the upcoming race's likely pace
(from the field's front-runner mix; cf. `features.FRPct`). The market handicaps
pace crudely; a physiological counterfactual is exactly the structural feature that
could beat an efficient win market where crude features failed (journal 0011).

### Outputs / how it plugs in
Per-horse latent params + per-run features become a `DataSource` feeding the
existing model layer: a results-trained win model (`ConditionalLogit`), a better
**per-leg Trio model for the carryover cover (RQ6)**, and — with the forward odds
capture — a renewed shot at model-edge exotics.

---

## 3. Open questions
- **OQ1** Are published sectionals accurate/granular enough to fit `v_c`/`W′` (only
  ~6 points)? Or does a useful energetics fit need Tier-2 trajectories?
- **OQ2** Do energetics latents actually predict *out-of-sample* results better than
  the market — or is the market already efficient w.r.t. sectionals too?
- **OQ3** Video homography accuracy per HK track (ST/HV geometry, camera setups) —
  is ground-lost measurable to useful precision?
- **OQ4** Compute/storage envelope for the video pipeline (the expensive axis).
- **OQ5** Identity linking at scale (saddlecloth OCR under motion blur/occlusion).

## 4. Sequencing
1. **S1–S3 (Tier-1 sectional)** — solve the data source, build `SectionalSource`,
   run the cheap kill/confirm through the harness. *Decides whether to go further.*
2. If signal: build the **energetics model** + hierarchical fit; feed the carryover
   leg model and a win model.
3. **V1 (replay URLs)** in parallel; **V2 CV** only if Tier-1 confirms signal *and*
   ground-lost is judged worth the cost (single-race PoC first).

*A research workflow can be spun up to (a) design the energetics model + sectional
feature set and (b) architect the CV pipeline — once Tier-1 data is in hand.*
