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
- **Action S1 (attempted — blocked here, journal 0019):** browser spike found the
  sectional page is a JS SPA that renders "Information will be released shortly" and
  fires **no data XHR** (only analytics) across multiple races/dates; the legacy
  `.aspx` 302s to it with no server-side table. So the split-time source is not
  accessible in this environment. **Fallback signal available now:** result pages'
  `RunningPosition` (sectional *positions* per call) + `FinishTime`, already
  scraped. Revisit the sectional API later (deferred GraphQL call / may be
  unprovisioned here); otherwise prioritise the video/CV path for true density.
- **Action S2:** build a `SectionalSource` (`DataSource`) → per-(race, runner)
  split times + positions, joined on our canonical keys.
- **Action S3 (cheap kill/confirm):** derive 1–2 sectional features (finishing-
  speed, sustained-speed split, a crude `v_c`) and run through the existing harness
  — *do sectional features beat the market baseline where draw+weight couldn't?*

### Tier 2 — Video trajectory extraction  *(high effort, high ceiling)*
What we want: per-frame `(x, y)` track position per horse → full kinematics
(velocity, acceleration, **actual distance travelled**, path width, drafting).

Data-source problem (⬜): race replays are archived (the old form table had a
`VideoReplay` column we drop; morning track-work has the known `twstarter` mp4
URL pattern). Need the **race-replay stream URLs** (find via the replay page's
network panel) and a download path.
- **Action V1:** browser recon of the replay page → capture the video stream URL
  pattern (HLS/mp4); confirm archive depth.
- **Action V2:** CV pipeline scope — detect horses → multi-object tracking (handle
  occlusion) → **camera→track homography** (broadcast pan/zoom to ground
  coordinates; the hard part) → runner identification (saddlecloth/silks). A real
  CV project; gate it behind a **single-race proof-of-concept** before committing.

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
