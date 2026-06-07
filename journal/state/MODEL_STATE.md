# Model State — the causal-physics generative win model

> The modelling agent's working memory. Self-contained snapshot of where the CAUSAL rebuild
> stands: the architecture, the identified physical constants, the open queue. The full trail
> is the journal (entries 0046–0050); deleted scratch can be rebuilt from it.
> Last updated: 2026-06-06 (the lean refactor / namespace fold).

## Objective & method — see `docs/PHILOSOPHY.md`

The governing principles live in **`docs/PHILOSOPHY.md`** (read first) and the architecture in
**`docs/MODEL_DESIGN.md`** / **`docs/CAUSAL_MODEL.md`**. In one line: latent physical CAUSES →
generative PATH → finishing ORDER → pool prices, built from **first-principles physics, not
regression**; meta-signals (form/jockey/collateral) are CONSEQUENCES that must **RE-EMERGE**,
never inputs; identify each constant from its **dedicated surface** and demand **cross-surface
consistency**; path-fit + re-emergence + consistency are PRIMARY; **% of market is the parked
final yardstick**, never the fitting target.

> **History (2026-06).** An acausal additive-regression win model (`hkjc.model.fit` +
> `features` meta-signal tables + baked `constants` gains) reached **ST 65% / HV 60%** of market
> but was **DELETED** in the lean refactor as "regression dressed up as physics." The
> `hkjc.statespace` package that replaced it was **folded into `hkjc.model`** — it IS the model
> now. There is **no frozen fallback** — the rebuild is the only model; the journal is the trail.
> Do NOT revive meta-signal feature tables or fitted-gain assemblies (ANTI-REGRESSION).

## Architecture (three generative layers — `hkjc.model`)

1. **LATENT STATE** — `latent.LatentState` (θ: v_c, w_prime, ρ recovery, a control), inferred
   from the PATH. `dynamics` is the first-principles EOM core (Newton power-balance + critical-
   power energetics + bang-CP-bang optimal control); `identify` pins constants per surface
   (Hill critical-velocity); `field` does the mean-field pace/draft de-confounding.
2. **GENERATIVE MODEL** — `generative.GenerativeRaceModel` (θ+conditions → Wald first-passage law
   → sampled paths → `OrderDistribution`), priced by `pricer.PoolPricer` (win/place/exotics).
   Identified via `likelihood.PathLikelihood` (the path-reconstruction loss surface).
3. **TWO-OBJECTIVE LOOP** — IDENTIFY honestly from the path; CALIBRATE coupling + VALIDATE on the
   pool book (`coupling`, `gate`). Coupling is gate-calibrated, not honest-strength (the #40 lesson).

`pure_mechanism.pure_mechanism_theta_builder` is the meta-signal-free build (journal 0050): θ from
Hill CP → de-confounded cruise → train-only population prior, every meta-signal STRIPPED.

## Identified physical constants (per-surface, with units — NOT fitted gains)

| Constant | Value (ST / HV) | Surface it's pinned from | Status |
|---|---|---|---|
| **Critical power CP** | ~54 / 53 W/kg | cruise-in-routes AND Hill `d = CP·t + W'` | identified, cross-surface consistent (journal 0048–0049) |
| **Critical velocity v_c** | ~17.4 / 17.2 m/s | Hill slope over prior distinct-distance trials | the robust de-confounded anchor |
| **W'_0 (anaerobic reserve)** | ~225 / 246 kJ | Hill intercept D′ | identified |
| **P_max** | P_max/CP ~1.85 / 2.18 | opening transient | aliased ~2× in-race (weakly identified) |
| **τ (recovery time-const)** | — | sit-and-kick | unidentifiable in-race (needs the recovery surface) |
| **C_dA (drag area)** | — | — | needs CV lateral paths (#34) |

All match equine physiology (literature-plausible). Over-determination is the test: CP from the
apple (cruise) and the Moon (Hill) must agree — disagreement = missing physics, not free fit.
The opening-ramp is captured by mechanism (R²→0.77); the mean-field coupling de-confounds cruise
toward the Hill CP and **collateral-form RE-EMERGES** from it (ST −0.02→+0.33, HV +0.20→+0.40, 0049).

Provisional acausal couplings still in the pricer (`wald_constants.py`): `DRAW_*` (bend ground-loss),
`FIGVAR_*`/`DRAW_VAR_*` (→ diffusion D), `GATE_BREAK_*` (start phase). These are **debt, not physics**
— slated for mechanistic replacement (draw→track geometry, figvar→D, gate-break→EOM transient).

## WAVE-1 RESULT + RE-ORIENTATION (journals 0050→0060): one causal dimension built; WIN is the diagnostic

Wave 1 built and stress-tested the FIRST causal dimension — physiology, read from the longitudinal
sectional path. The investigation (S1/S1b/S1c/S1d + S4) is complete; the NUMBERS are these, the
INTERPRETATION is corrected per journal 0060:

- **Single-dimension (physiology) diagnostic coverage (0057):** the best path-ability extraction
  converts to **~27% (ST) / 32% (HV) of market**; mechanistic reads ~14%/27%. Form-alone is 60%/45%,
  the deleted multi-signal stack 65%/60%.
- **What that MEANS (corrected):** WIN% is a **DIAGNOSTIC of how much of the causal network is built**,
  NOT a verdict on the philosophy. One dimension (physiology) underperforming the multi-dimension
  form aggregate is EXPECTED — form integrates physiology + control + prep + condition + deployment
  (THEORY doc). The ~27% is the **floor to build UP from by adding causal dimensions**, never by
  re-admitting the proxy. **WIN is not authorized to reject first-principles causal modelling.**
- **REJECTED:** form-as-measurement (the 0059 Route 2) — re-admitting form/jockey/collateral even as
  a "measurement" is regressing onto the correlation PROXY of the causes we are here to model. The
  acausal shortcut stays deleted.
- **The chain (for the record):** 0052 de-confounded level ≈ Hill slope, both weak; 0049's strong
  re-emergence was route-only-subsample. 0054 path is UNDER-extracted (oracle +0.23 = 74% of ceiling),
  signal in the career velocity-PROFILE SHAPE. 0056 forward-EOM index is a scalar collapse. 0053 (S4)
  uniform ~10–13% across the pool book.

## WAVE 2 — EXPAND THE CAUSAL NETWORK (journal 0060; each dimension from its OWN mechanistic surface)

WIN diagnoses cumulative coverage; each dimension also earns its OWN re-emergence test; NONE re-admit
a meta-signal as input.

1. **CONTROL cause (jockey) ← CV LATERAL path** — the frontier. Rail-offset position, ground-saved,
   traffic, draft: orthogonal to the longitudinal sectionals and invisible to form's outcome-aggregate.
   OCR solved (RapidOCR, 0058). Route: **S0b** (full re-id + backfill) → **S8** (control latent →
   jockey re-emergence) → **S9** (C_dA / draft / geometry×control).
   **STATUS — GATED on a VALIDATED LATERAL read (journals 0067→0068→0069, S0d).** Key distinction
   (0069): the aerial *legend strip* is a DATA OVERLAY (the broadcaster's rendered running-order, ≈ the
   official chart) — OCR-ing it (0068's ρ+0.94) is a cheap dense order-feed + a re-id ANCHOR, NOT a
   physics reconstruction and NOT the novel signal. The NOVEL CV deliverable is the LATERAL/CONTROL
   trip (rail-relative y_lat: ground saved/lost, traffic), and it is **UNVALIDATED** for both views:
   the genuine geometric/arc-length recon is noise (spurious −0.50), aerial sprites collapse into a
   bunch at the run-in (winner #3's surge never captured — the tracked paths even stop ~140m before the
   true t≈100s finish), and drone y_lat is mis-scaled ~3-4× + anonymous. FRAME SCRUTINY (0069)
   corrected 0068's "drone numbers illegible": the drone ZOOMS to legible saddlecloth numbers at the
   finish → a non-circular silk→number bridge → drone (REAL horse positions) is the PRIMARY path to a
   genuine numbered lateral read. Also unexploited: the aerial SPLIT-SCREEN left real-camera + clock,
   the minimap, and the start-gate number anchor. **S8 (control latent) + S0c (scale-out) stay GATED
   until a VALIDATED lateral reconstruction exists** (order-data alone, from legend/chart, does NOT
   unlock the control cause). Real geometry gains banked in 0068: green-render detect, scale_from_
   sprites, ocr_every, "C"-width 21.3m. This is a §4b extraction mis-spec, NOT a CONTROL-dimension null.
   **CONVERGED RESULT after improving BOTH views (0070 drone + 0071 aerial).** Longitudinal/order/
   finish/sections are SOLVED but NON-NOVEL (≈ the official chart): the aerial legend drift-robust
   reader reads the official top-8 through the finish, winner #3's surge now TRACKED to 1640m, sections
   to 0.125s MAE; the drone number-bridge binds 10/12 non-circularly (0068's "illegible" falsified).
   The NOVEL lateral/control y_lat trip WAS a §4b MIS-SPEC NULL on both substrates (per-frame rail too
   noisy → within-runner y_lat dispersion EXCEEDED between-runner trip spread: aerial 3.8 vs 3.3m ratio
   0.87<1; drone unstable turf-boundary rail + low-contrast altitude silks). **0071's AERIAL NULL is now
   OVERTURNED by the render-inversion revamp (#82, journal 0075).** Treating the AVV as deterministic 3D
   GRAPHICS (not a camera) and fitting a STABLE RAIL SPLINE (frame-to-frame rail-midpoint shift median
   3.3px, valid 454/454) + GROUND-CONTACT (shadow base) recovers a BETWEEN-RUNNER lateral that is
   SUBSTANTIVELY FAITHFUL: lag-1 autocorr +0.90 (smooth, graphics-like, NOT 0071's high-freq noise),
   corr(median y_lat, mean official RP) +0.45 (Spearman +0.44), and a clean ~6m TIGHT-vs-WIDE gap that
   MATCHES the official stewards' Comments-on-Running (rail horses #12/#8/#4/#11 read 2.6-4.1m; wide
   horses #2/#6/#9/#1 read 9.3-9.8m). The residual limit is NOT a null: inverting an OBLIQUE render to
   absolute metres needs the ground-plane homography, and a SINGLE drawn rail is UNDER-DETERMINED for it
   (all correspondences on the y_lat=0 line; no outer rail; painted distance markers render as
   non-OCRable stylized outlines) → a single global m/px over/under-reads with depth, worst at the
   foreshortened run-in (within/between 1.29 on the clean <1300m body vs 1.6-1.8 full-race). So the
   aerial render is now a VIABLE lateral source at the runner-ORDERING / rail-vs-wide CLASSIFICATION
   level; absolute per-runner TRIP metres await a 2nd ground fiducial (marker pixel-positions or the
   minimap inset → proper per-frame homography) + run-in down-weighting. The drone (0070) remains
   anonymous + no global longitudinal axis (pan/zoom/cut camera). **The PATROL feed (#83, journal 0076)
   is now the 2nd VALIDATED lateral model — and the geometrically cleanest.** Fixed stewards' mount ⇒
   CONSTANT homography (white rail fits x=-0.3934·y+393.5, slope CV 3.0% across the t=83-96s head-on
   window, solved ONCE so 0071's per-frame-refit root cause cannot occur); head-on ⇒ lateral ≈ horizontal
   image coord after rectification, not the oblique projection that nulled. Headline within/between
   lateral dispersion ratio 0.06 (frame jitter) / 0.16 (1.5s window) / 0.35 (full-track, inflated by
   anonymous cross-linking) — ALL beat aerial's 0.87 and drone's >1. 12/12 field resolved on the run-in;
   a clean rail~5-7m / mid~9-11m / wide~13-15m fan matching the stewards' words (#2 shifts IN from a wide
   barrier, #6 caught wide). Honest limits: single race, single visible rail (scale anchored not
   two-rail-solved; the ratio is scale-free), anonymous head-on identity inflates the full-track
   estimator (next: bridge from the t≥105 finish-camera saddlecloth numbers or time-sync to RF).
   **NET: 2 of 4 feeds now carry a validated lateral (AERIAL render between-runner ordering #82/0075 +
   PATROL home-straight metric #83/0076) — the ≥2-model gate for FUSION (#80) is MET, fusion is UNBLOCKED.**
   **DRONE FOLDED IN AS THE 3rd FEED (#87, journal 0080).** 0070 nulled the drone with the OLD
   per-frame-rail; #87 retried with the 0075 toolkit (stable per-window rail-line pooled over a
   near-static window + ground-contact/shadow-base, NOT per-frame turf-boundary refit) and — the
   structural win — TIME-ANCHORED it on the shared clock (affine drone_vid→race_t from the gate-jump
   vid 3.0s→race_t 0 and the photo-finish CUT vid 113.68s→winner 98.85s, NOT the field-relative
   odometer 0070 used). RESULT: the drone covers race_t −2.7→98s == arclen 0→1640m — a genuine 3rd
   FULL-RACE feed, with a CLEAN home-straight band (arclen ~1270-1566m, straightness 0.9-1.0) that
   co-locates with the aerial's clean zone. It delivers the FIRST per-horse REDUNDANT cross-feed
   lateral AGREEMENT (drone vs aerial render Spearman ρ +0.88..+0.93, cls 100%, leave-one-out
   +0.83..+0.91) on the arclen 773-1000m band — the coherence 0077's bend overlap (patrol identity
   bottleneck, ρ −0.03) could not earn. HONEST MIXED: the per-frame within/between DISPERSION test
   STILL NULLS (ratio 1.04 run-in / 2.6 home-str / 5.2 bend, all ≥1) — the panning-camera single-line
   rail recovers ORDERING + cross-feed REDUNDANCY but NOT absolute per-runner TRIP metres (same §4b
   metric mis-spec class as the aerial). And the cross-feed prize sits on a BEND, not the clean
   straight: the genuinely-clean home-straight overlap NULLS (−0.58) because the AERIAL itself
   degrades there (aerial-vs-stewards +0.04, the 0075 foreshortening) — reaffirming 0077's "no arclen
   band is clean for BOTH at once" structural finding, now from 3 feeds. **SYNTHESIS (0075⊕0080, the
   actionable reframe): the +0.93 drone↔aerial agreement sits at 773-1000m, which is INSIDE aerial's
   validated <1300m zone — so the whole arclen <1300m band (~80% of the race) now has a CROSS-VALIDATED
   lateral ORDERING (two independent feeds agree, LOO-robust). The genuine remaining GAP is the RUN-IN
   (>1300m, final ~350m): aerial foreshortens, patrol occludes/cuts, drone's metric nulls and its
   ordering weakens (ρ+0.32). That single band — not the whole race — is what stands between us and full
   coverage. NOTE: what is validated everywhere is rank/ORDERING (tight-vs-wide), NOT absolute metres
   (all three feeds §4b-null on the metric) — enough for the CONTROL CLASSIFICATION layer (S8: who saved
   ground / went wide), not yet for precise ground-loss energetics (S9/C_dA).** NEXT, on the run-in gap
   (the only band not yet conceded-or-covered): a proper per-frame DRONE homography on the run-in (real
   fanned positions patrol+aerial lose), and the PATROL FINISH-CAM (near-head-on at the line, t≥105) as
   a 4th look at the last ~100m. Then FUSE the cross-validated <1300m segment + the drone numbered
   home-straight + the patrol head-on metric (0076) on the shared clock. RF margins remain the 4th feed.
   **NORTH STAR — the 4-FEED FUSION holy grail (task #80, journal 0073).** HKJC exposes ≥4 auth-free
   replay feeds, each a COMPLEMENTARY PARTIAL MEASUREMENT of the same latent race: AERIAL render
   (full-race long+lat, synthetic-fidelity-bounded, #82), DRONE (real lateral, back-straight, #81), RF
   /race-film (longitudinal gaps→MARGINS + velocity readout + finish identity, #60), PATROL (clean
   lateral in the home straight via a FIXED head-on camera + interference events, #83). The grail is to
   FUSE them into a single high-resolution latent per-horse (longitudinal, lateral, t) TRAJECTORY —
   a Bayesian multi-view STATE-SPACE: each feed an observation model with its own uncertainty +
   coverage window, tied by a shared clock (sync) + identity bridge (legible-number moments). This is
   the causal-physics philosophy applied to the data layer (latent state, measurements-with-noise) and
   the natural home of every per-feed effort. Out of it fall the control/lateral cause, margins, and
   velocity as readouts. PARKED future (out of scope now, #84): jockey-action classification → a direct
   control-effort observable. The per-feed prototypes build the observation models; fusion (#80)
   integrates once ≥2 validate. STATUS: #82 AERIAL render-inversion VALIDATED (0075, faithful
   between-runner lateral) + #83 PATROL fixed-homography VALIDATED (0076, stable metric home-straight
   lateral) ⇒ ≥2-model gate MET, FUSION (#80) UNBLOCKED. Pending feeds: #81 drone stabilization, #60 RF
   margins. Fusion design (#80): a shared-clock + identity-bridge state-space — patrol's t≥105 finish-cam
   numbers / RF saddlecloths are the identity anchors; the aerial odometer + patrol/RF race clocks the
   time sync; each feed enters as an observation model over its coverage window with its own y_lat noise.
   **#80 STEP 1 BUILT + FIRST CROSS-FEED TEST (journal 0077, studies/cv_fusion_spine.py).** The SYNC
   spine works: one shared race_t axis (official sections [27.65,23.30,23.75,24.15], sum 98.85=winner
   1:38.85) — AERIAL via the odometer arclen→time curve (per-row, covers race_t 9-98s), PATROL anchored
   by the winner's finish-cross (vid 117.5s→race_t 98.85, offset 18.65±0.4s), cross-checked by the RF
   on-screen RACE CLOCK (race_t=RF_vid−1.34s, three manual reads). IDENTITY: aerial 11/12 (legend);
   patrol finish-cam OCR reads 11/12 NUMBERS but binds them to head-on tracks UNRELIABLY (the finish
   cam is SIDE-ON so img_y/perp-to-rail conflate lateral with along-track depth; the cut breaks pixel
   continuity). FIRST cross-feed lateral test = a NULL on the overlap (Spearman ρ −0.03, rail/wide
   class-agree 56%) — but the null LOCALISES, it is NOT a coherence refutation: (a) the overlap window
   race_t 65-78s is the BEND (arclen ~1100-1290m), the WORST aerial geometry — aerial-vs-stewards ρ is
   +0.93 on the clean body/home-straight but ~0 on this bend (the 0075 run-in foreshortening, now
   window-quantified); (b) the patrol identity bridge is the bottleneck. COHERENCE POSITIVE: the two
   feeds' lateral SPREADS agree (aerial 7.8m / patrol 9.9m fan, ~half the 21.3m width, identity-free) —
   the latent lateral is real + similarly scaled, fuseable. **STRUCTURAL FINDING (sharper than 0077's
   own NEXT line): on THIS race the two validated lateral feeds have COMPLEMENTARY, not OVERLAPPING,
   clean zones — aerial is faithful on the straights/body but degrades on the bend+run-in
   (foreshortening), while patrol's clean head-on window IS the bend/straight-entry (arclen ~1100-1290m)
   and is OCCLUDED/cut after ~1290m. There is NO arclen band where BOTH are clean — so a strong
   REDUNDANT per-horse cross-feed agreement test cannot be earned by re-windowing alone.** NEXT (step 2)
   — three routes, in leverage order: (i) **AERIAL MINIMAP probe (highest leverage):** the AVV's
   top-down inset is a (near-)orthographic 2nd view, foreshortening-FREE and already number-identified
   (same frame as the legend) — if its resolution yields 12 per-horse lateral positions it is a CLEAN
   whole-race lateral that MOOTS the render-foreshortening problem and could be the fusion backbone
   (flagged unexploited since 0073, never tested); (ii) bridge PATROL identity by RF time-sync
   continuity (NOT the side-on finish-cut img_y) so the bend overlap becomes usable; (iii) COMPLEMENTARY stitching on the shared clock + the identity-free spread match + seam continuity.
   **SUCCESS BAR (user steer, 2026-06-06): FULL-RACE 4-feed coverage remains the HOLY GRAIL / north star
   — keep pursuing it unless DEFINITIVELY proven impossible (a per-arclen coverage census showing some
   band is unreadable by ALL four feeds — aerial render, minimap, patrol, drone, RF — not just the two
   tried so far). PARTIAL high-res reconstruction of individual SEGMENTS is a valued MILESTONE/FLOOR, not
   a substitute for and not a reason to stop short of full coverage.** So: ship confident segments as we
   earn them (each feed owns where it is clean — patrol → bend/straight-entry head-on; aerial render →
   body/straights; RF → margins; drone → back straight; validated per-segment vs stewards, stitched on
   the shared clock) AND keep closing the gaps toward whole-race. Gap-closers, in order: probe (i) the
   minimap (a foreshortening-free feed that could cover the bend/run-in the render loses, AND cross-
   validate the body); then the DRONE back-straight segment (real positions, 0070 number-bridge) and the
   RF margins backbone — the 3rd/4th feeds 0077 has not yet folded in. A band is only conceded after the
   full coverage census fails for it.
   **#80 STEP 2 BUILT — THE FUSION SMOOTHER + FIRST RECONSTRUCTED RACE (journals 0083/0084,
   studies/cv_fusion_smoother.py, data_cache/cv_fusion_trajectories.parquet, fig cv_fusion_recon.png).**
   Per-horse numpy Kalman forward-filter + RTS backward-smoother on the shared race_t grid (5 Hz, latent
   z=[s,ṡ,l,l̇]); the FIRST-PRINCIPLES motion prior is applied JOINTLY in the smoother (bounded long.
   accel σ0.8 m/s² + bounded LATERAL accel σ0.35 m/s² = the no-teleport prior), NOT as a per-feed
   pre-filter — so each feed keeps its measured noise. 3 feeds as observation models over their windows:
   aerial odometer arclen (sd3m)+y_lat (sd6m, →14m on >1300m foreshortening); drone arclen (sd8m)+y_lat
   (sd7m→11m run-in); PATROL = the metric ANCHOR (y_lat sd~1.2m) pinning absolute scale on 1058-1279m.
   RESULTS: **12/12 horses reconstructed full-race** (arclen ~150-1647m); 6 see all 3 feeds. **The fused
   rail-vs-wide GAP SURVIVES + EXTENDS: +3.84m (rail #4/8/11/12 5.4m vs wide #1/2/6/9 9.2m), now CLEAN on
   8/8 stewards-commented horses (vs patrol-alone 4/8), no mis-class — the S8-ready control CLASSIFICATION
   on <1300m.** Per-band y_lat uncertainty (the segment-quality census): tightest ~0.97m at the patrol
   anchor (1200-1300m), inflating monotonically to ~1.72m at the line — the honest run-in gap, visible as
   fanning 2σ bands. **TWO GAPS the fusion itself exposes:** (1) LONGITUDINAL per-horse order is NOT yet
   meaningful (order-vs-RP ρ ~0 at poles) — the only long. observation is the aerial FIELD-SHARED odometer
   (per-horse stagger ~1.2m unobserved); needs the **RF MARGINS feed (the un-built 4th feed, #60)** which
   gives per-horse longitudinal gaps side-on across the whole race incl. the finish. (2) the run-in
   LATERAL (>1300m) — patrol occludes, aerial foreshortens, drone metric nulls. NEXT THRUST: fold in **RF
   (#60)** — it closes the longitudinal gap, is the S3 margins surface, AND covers the run-in side-on;
   then the run-in lateral via a drone run-in homography / the patrol finish-cam. The control LATERAL
   classification deliverable (S8) is now IN HAND on <1300m; S9 (absolute-metre energetics) still awaits
   a whole-race metric.
   **VENUE CORRECTION (2026-06-06): the 2026-06-03 meeting is HAPPY VALLEY, not "ST" — all "ST
   2026-06-03" labels in 0067-0087 are wrong (HV = tight 1450m circuit; explains the sharp render bends).
   CV methodology unaffected (reads the rail directly); the physics track-geometry (ground-loss/draw) must
   use HV geometry for these races.**
   **#80 STEP 3 — 4-FEED FUSION CAPSTONE COMPLETE (journal 0087, studies/cv_fusion_4feed.py, fig
   cv_fusion_recon_4feed.png). HOLY-GRAIL v1 DONE for HV 2026-06-03 R1.** RF folded in as observation-
   model 4: finish margins → metric longitudinal obs at the line (arclen=1650−margin·2.4m, sd~1m), legend
   order → per-horse arclen offset reproducing the RF rank (sd1.8m, imposes ORDER not false metres);
   needed two fixes — back-marker obs (else they float to field-mean, capping ρ~0.6) + loosening the
   aerial FIELD-MEAN odometer sd 3→7m (its identical per-row obs were clamping the field together). NET:
   **per-pole order vs official RP ρ +0.96/+0.80/+0.90/+0.71** (decisive jump from 0084's ~0), **fused
   finish order ρ+0.68, winner #3 RECOVERED (top-2 exact)**, lead-group finish-margin MAE 0.72 len. The
   LATERAL HELD (RF didn't corrupt it): rail-vs-wide gap **+4.26m**, 8/8 covered 7/8 correct; per-band
   lateral sd 1.00m (patrol anchor) → 1.42m (run-in) → 2.24m (start). HONEST-UNCERTAIN still: absolute
   per-horse MID-RACE metres (order observed, not metres — §4b class; deep tail compressed not fabricated)
   + the run-in LATERAL metric (the last gap). **So v1 delivers, for one race: a per-horse (longitudinal
   ORDER-faithful + finish-margin-metric) × (lateral control-CLASSIFICATION + patrol-anchored metric) × t
   reconstruction, all 4 auth-free feeds fused on one clock, 0 outcomes used as input.** PHASE FORK (next):
   (a) SCALE the validated pipeline to a race CORPUS (#75) — the gate to the real S8 payoff; (b) S8
   CONTROL re-emergence (#66) — does the reconstructed lateral trip predict jockey skill (needs ≥a few
   races); (c) close the mid-race METRIC gaps (run-in lateral homography / absolute stagger). Recommend
   (a)→(b): the single-race METHOD is validated end-to-end; the science needs many races.
   **USER STEER (2026-06-06): scale SLOW — acquired a 4-race distance-spread corpus (HV 2026-06-03,
   "C" course, all 4 feeds): R01 1650m (v1), R03+R06 1200m sprints, R08 1800m route. Goal: close the
   metric gaps + refine the algo race-agnostic, NOT the big automated pipeline yet. Official validation
   cached via Chrome MCP → race_official/20260603R{3,6,8}_results.json.**
   **#91 GENERALIZATION TEST — the METHOD GENERALIZES to a 2nd race/distance (HV R03 1200m sprint,
   journal 0088, studies/cv_r03_*.py, fig cv_fusion_recon_R03.png).** Reused R1 study code (geometry
   imported unchanged, only R1-baked constants re-pointed). CLEAN generalizers: AERIAL render
   faithfulness (autocorr +0.93, rail-vs-wide gap +4.0m correct sign), PATROL head-on geometry + WINDOW
   AUTO-DETECT (sprint straight arrives ~20s earlier → found at t=60-66s by the rail-slope-stability
   scan; ratio 0.055/0.10/0.18), RF LEGEND-ORDER backbone (ρ+1.00/+1.00 early), DRONE time-anchor, and
   the FUSION: **winner #7 RECOVERED, per-pole order vs RP +0.99/+0.98 (body) — beats R1**. DEGRADED (the
   refinement signal): finish-margin MAE 1.70 len (R03 finish-cam blurs at the line → re-windowed to the
   legible approach), and the LATERAL metric — patrol numbered gap came out WRONG SIGN (−5.7m) and fused
   lateral gap +2.69m/6-of-9 (vs R1 +4.26m/7-of-8). **REFINEMENT BACKLOG (the real deliverable): the
   dominant cross-cutting weakness is IDENTITY + OCR — aerial OCR cadence does NOT scale (2.4hr→stalled)
   and drops identity to 7/12, patrol silk→number permutes (the sign flip), drone identity had no
   race-agnostic bridge (5/12). Plus: official format parser (done), sections 4→3 (done), clock anchors
   auto-scanned (done), RF margin-window + winner-ref were hard-coded (load-bearing), stewards' lateral
   truth weaker on R03 (interference not trip calls — a validation-surface limit).** NEXT REFINEMENT
   (highest leverage, distance-independent): a UNIFIED race-agnostic IDENTITY module — read numbers ONCE
   from the clearest source (RF crisp side-on numbers), build a per-runner silk-colour signature,
   PROPAGATE to all feeds by silk + shared clock — replacing the slow per-feed OCR and fixing the patrol
   sign-flip. Then validate the hardened pipeline on R08 (route — also surfaces the foreshortening metric
   gap). The run-in lateral metric + absolute mid-race stagger remain the open metric gaps.
   **#92 UNIFIED IDENTITY MODULE — BUILT; CORRECTS R03, REGRESSES R01 (journal 0089, studies/cv_identity.py
   + cv_identity_rerun.py). PARTIAL WIN, NOT promotion-ready.** Solve-once: RF multi-frame-vote saddlecloth
   OCR + patrol finish-cam augment -> silk hue-hist LUT 11/12 on R03 (vs per-feed 7/4/8); RF legend
   name->number -> rank(race_t) backbone 12/12 ORDER (rho +1.00/+1.00 body). PROPAGATE by Hungarian on
   circular hue-EMD CROSS-CONSTRAINED by the RF clock-order. R03 (the case it was built for): the order
   constraint is demonstrably the fix — patrol gap -7.2m (PURE SILK, degenerate) -> +4.4m CORRECT
   (silk+order); fused lateral +1.95m/8-of-10 (vs 0088 +2.69m/6-of-9, classification up, magnitude diluted
   by the single ~1000m patrol band + noisy draw-truth); winner #7 + per-pole +0.99/+0.99 PRESERVED; the
   2.4hr aerial-OCR bottleneck GONE (whole anchor 7-8min, cached npz, av sequential decode replaces cap.set,
   no per-feed re-OCR). HONEST RED FLAG: on the known-good R01 the module gives the WRONG sign (banked +4.1m
   -> lam0 -2.6m / lam0.6 -9.5m) — the RF-anchored LUT is lead-group-biased (lower per-track fidelity than
   R01's own dedicated 102.5-123s finish-cam LUT) AND the weak head-on DEPTH order-proxy makes the
   constraint race-dependent (helps R03's late reliable-rank window, hurts R01's bunched one). So the SAME
   mechanism that fixes R03 breaks R01. BEFORE hkjc.vision promotion: (a) fuse each feed's OWN finish-cam
   LUT (keep R01's richness) with confidence weighting, (b) a stronger order proxy than head-on depth (RF
   rank by time-sync continuity / apply only where the RF rank is unbunched), re-test NO R01 regression,
   THEN R08.
   **DRAW → y_lat(t=0) ANCHOR (user steer 2026-06-06, task #93, to fold into the next fusion pass).** The
   barrier draw is a known PRE-RACE INITIAL CONDITION (NOT an outcome — philosophy-clean, same class as
   distance/weight; the pricer already uses DRAW_* couplings), already in the cached official "Dr." column.
   At the gate horses sit in evenly-spaced stalls across the track → each horse's y_lat(t=0) is known to
   ~±½ stall width (map: draw d → rail_offset(stall1, rail-config) + (d−1)·stall_width; HV "C", width
   21.3m). ADD it to the fusion smoother as a LOW-SD lateral observation at the gate-jump (t=0). Two wins:
   (1) it anchors the lateral state exactly where feeds are weakest (early race, before drone coverage);
   (2) the draw ORDER is a 3rd independent IDENTITY vote (complements silk+clock; breaks the silk-degeneracy
   that flipped R03 patrol's sign). Fold into the post-#92 fusion re-run (do NOT dispatch now — #92 is
   re-running the fusion; avoid collision).
   **#92→#94 IDENTITY HARDENED (journals 0089→0090). #92 proved the clock-order constraint fixes R03's
   patrol sign (−5.7→+4.4m) + killed the 2.4hr aerial-OCR bottleneck (→<1s cached), but REGRESSED R01
   (replaced each feed's OWN finish-cam LUT with RF-only + weak head-on-depth order proxy). #94 fixed it
   with a SINGLE FIXED consensus (no per-race tuning): cost = own-silk-LUT(1.0, distinctness-wtd) +
   RF-anchor(0.45) + reliability-GATED clock-order(0.6) + DRAW-order(0.7), Hungarian one-to-one.** RESULT,
   same scheme both races: **R01 fused rail-vs-wide gap +4.29m (banked +4.26m HELD, NO regression, 8/8,
   winner #3) AND R03 +5.81m (restored from #92's diluted +1.95m, 8/10, winner #7).** The **DRAW→y_lat(0)
   anchor (#93, user idea)** is in the smoother (low-sd 1.5m lateral obs at race_t=0; map y_lat(d)=
   0.5·w+(d−1)·w, w=21.3/14=1.52m, cross-checked vs patrol span) — it is the MAGNITUDE fix (2nd metric
   anchor: start=draw + mid=patrol) AND a 3rd race-independent identity vote; the bounded-lat-accel prior
   bridges in/out shifters (R01 #8 drawn gate-12 raced inside → fused 5.2m rail CORRECT though patrol-alone
   mis-read 14.5m). Honest residual: patrol-ALONE R01 mag +2.34m<+4.1m (draw is a cleaner start-ANCHOR
   than mid-race id-vote; fusion absorbs it, not patched=no overfit). Module `studies/cv_identity.py`
   (consensus_assign, draw_to_metres) — CANDIDATE for src/hkjc/vision once R08-validated.
   **#95 R08 HELD-OUT VALIDATION — PASS, no tuning (journals 0091/0092). The hardened pipeline GENERALIZES
   across the full spread sprint→mile→route.** Winner recovered **3/3** (R01 #3, R03 #7, R08 #6); per-pole
   BODY order vs RP generalizes cleanest — R08 (route) is the corpus BEST at +0.96/+0.95/+0.98/+0.94;
   lateral rail-vs-wide SIGN holds **3/3** (R08 fused +3.97m, patrol-consensus +12.74m). The identity
   consensus + draw anchor are validated on 3 races, one fixed scheme. **THE ONE REMAINING LIMIT, now
   precisely quantified = the AERIAL RENDER absolute lateral METRIC degrades with BEND content (substrate
   foreshortening, NOT the identity/fusion scheme):** aerial within/between ratio 1.59(sprint)→1.6-1.8(mile)
   →**2.55(route)**; per-bend aerial y_lat lag-1 autocorr **+0.93/+0.90 → −0.16** (becomes per-frame noise
   on continuous bends; aerial-alone gap flips to −3.2m WRONG). Fusion holds the SIGN via patrol+draw+drone
   but the mid-race lateral MAGNITUDE is essentially unconstrained on routes (fused y_lat sd balloons
   12→28→39→52m vs R1 1.0-1.4m) except where draw (start) + patrol (mid) anchor it. **NET STATE: the
   reconstruction reliably delivers, across distances, longitudinal ORDER + finish MARGINS + lateral
   CLASSIFICATION/SIGN (who saved ground / went wide) — which is the S8 CONTROL-signal level — but NOT
   absolute mid-race lateral METRES on bend-heavy races (the S9/drag level), which awaits a per-frame bend
   homography (rail-ARC 2nd-fiducial, the deferred hard problem). The CV-REFINEMENT phase is at its natural
   end; the S8 control-classification deliverable is IN HAND across the corpus.**
   **CONSOLIDATED WRITE-UP: `docs/CV_RECON.md`** (the validated 4-feed method: feeds, shared clock,
   identity consensus, draw anchor, fusion smoother, corpus results, the bend-metric limit §7, and the
   study→library promotion candidates). Remaining consolidation step (needs an API decision, defer to
   discussion): promote `studies/cv_identity.py` + `cv_fusion_4feed.py` + the per-feed observation models
   from study-scratch into `src/hkjc/vision`. NEXT-PHASE choice (user deferred to discussion): S8 control
   re-emergence on the corpus / scale the corpus / close the bend metric gap (S9, hard).
   **SMOOTHER REFINEMENTS (user-spotted). #96 LATERAL physical constraints — DONE (journal 0093,
   studies/cv_fusion_constrained.py, _constrained parquets, fig cv_fusion_constrained_qc.png). Track-geom
   spec acquired from the authoritative HKJC racing_course page: HV turf WIDTHS per course A 30.5 / B 26.5
   / C 22.5 / C+3 19.5 m (studies had baked 21.3 m, ~5% low — now W=22.5 for the box); home straight C=334m,
   circuit ~1450m → bend radius ~130m (recorded for the future rail-arc homography). VERDICT (honest both
   ways): the HARD BOX is a real no-regression win — a truncated-Gaussian moment-match caps the unphysical
   tail at ~W/√12; R08 route bend lateral sd 52→3 m (16×) from one physical fact; naive CLIP leaves the 52m
   sd untouched, so the principled truncation is doing the work. Winner 3/3, finish-ρ, every per-pole order
   NON-REGRESSED all 3 races; R08 field y_lat span pulled inside [0,22.5]. BUT 3m sd is "which seventh of the
   track," NOT the 1-2m absolute metres the patrol anchor gives on straights — the box BOUNDS the nonsense,
   does not RECONSTRUCT bend metres; R08 rail-vs-wide classification unchanged (3/6), the box caps sd but
   leaves the posterior MEAN (which drives classification) where it was. The hoped-for gap-closer — JOINT
   cross-horse ORDERING — is sound but NEAR-INERT on routes, for a diagnosed reason: the field is
   longitudinally strung out (gaps ~30m), so only ~2-3 horses are genuinely co-located with lateral reads at
   any instant; "11 ordered bodies across 22.5m" is the premise but the bend reality is "2-3 bodies, order
   read from a foreshortened render." (Caught+rejected an un-gated version that falsely collapsed every band
   to a 0.55m floor; gating constrained-sd by per-node order-reliability, ordering adds only BOX 3.48→ON 3.38
   on the R08 mid bend; where order IS reliable — R01/R03 straights — ON<BOX, e.g. R01 run-in 0.96→0.78.)
   BAKE the BOX + geometry bend-downweight (free, correct, no-regression); the bend ABSOLUTE-METRE gap
   (CV_RECON §7, S9) is NOT closed and still needs the per-frame rail-arc bend homography.
   #97 PHYSICS LONGITUDINAL KERNEL — DONE (journal 0094, studies/cv_fusion_physkernel.py, _physkernel
   parquets, fig cv_fusion_physkernel_qc.png). Built on #96's constrained smoother (box+ordering+geometry+
   lateral-bounds all kept, lateral channel UNTOUCHED). EOM velocity SHAPE v_EOM(t)=v_c·ramp(t)·(1+d_fin·
   kick(arclen)): saturating gate-ramp 1−exp(−t/2.5) (matches solve_eom ~6-7s opening) → cruise plateau →
   one signed finish-deviation d.o.f. over last 12%; levels (v_c, d_fin) fit per-horse from that horse's own
   feeds (no per-horse-θ, no forward-sim — guardrail satisfied). Encoded as a phase-dependent OU velocity
   mean-reversion replacing the free white-accel random walk.
   THE STRUCTURAL LESSON (the §4b gem): a GLOBALLY strong template pull collapsed the squiggle but BROKE THE
   ORDER (R03 winner lost, late-pole ρ +0.10→−0.06, R01 finish-ρ +0.73→+0.38). Root cause, diagnosed not
   tuned: the per-horse VELOCITY DIFFERENCES ARE the running-order signal the RF rank feed carries — a horse
   climbing the field (winner #3 7→1) IS a velocity excess; a shared-cruise template smooths those away =
   ERASES OVERTAKING. The squiggle and the order are the SAME degree of freedom; you cannot smooth one
   without destroying the other with a shared-shape prior. No global-pull setting keeps winner 3/3 AND
   smooths. SHIPPED FIX: phase-dependent pull — strong EOM-shape pull ONLY in the gate-ramp window (~7.5s,
   where there's no running-position signal yet), weak in cruise/finish (residual σ_a 0.55 < white 0.8).
   RESULT: real but MODEST no-regression win. Smoothness (WHITE→PHYS, post-ramp): R01 cruise vel-TV
   17.5→12.5, 20s swing 2.7→1.8, gap-roughness 6.6→4.3; R03 16.0→14.3 / 8.3→7.5 / 55.9→38.2; R08 21.9→19.1 /
   11.1→8.9 / 46.5→32.8. NO-REGRESSION all PASS: winner 3/3 held; finish-ρ R01 +0.73→+0.71, R03 +0.36 flat,
   R08 +0.26→+0.45 (improved); per-pole flat; lateral sign +6.66(8/8)/+6.57(8/10)/+3.50(3/6) all held;
   margin-MAE 2.59→2.84/2.68→2.63/2.25→2.00. ±30% cruise-pull sweep keeps winner 3/3 + min-pole ρ>0 (not a
   knife-edge). HONEST FAILURE MODES: (1) the win is cosmetic-plus — removes wiggle + pins the gate but adds
   NO new longitudinal information (by construction, else it erases order); (2) the over-smoothing-hides-a-
   real-move risk is REAL — exactly why the global version was rejected (it hid #3's 7→1 on R03); (3)
   per-horse fitted v_c≈16.8 nearly identical because the aerial FIELD odometer dominates arclen → little
   cruise differentiation to carry. SMOOTHER v2 = constrained (#96 box+geom+lateral-bounds) + phase-gated
   physics kernel (#97); the _physkernel parquets are the current best per-horse trajectories.
   **BOTH user-spotted refinements now landed (#96 lateral constraints + #97 physics kernel). Per user: PAUSE
   for discussion. Open levers banked for that discussion: #98 horse-known-dimension prior (ruler/packing/
   bend-homography scale constraint, coupled to S9/#67 — feasibility-checked: render horses are perspective-
   faithful but small+clustered+camera-self-scaling, so best as the AUX constraint making the rail-arc bend
   homography solvable, NOT a standalone bend ruler); #99 DIAGNOSTIC WALKTHROUGH VIDEO (synced 4-feed frames
   + top-down recon + overlays, R01 primary / R08 bend-difficulty exhibit — RUNNING, renders while we
   discuss). NEXT-PHASE choice still open: S8 control re-emergence on the corpus / scale corpus / S9 bend
   metre gap (rail-arc homography + horse-size aux) / library promotion scope.**
   **LIBRARY PROMOTION — CORE DONE (#100, user chose D; 193 tests green, ruff clean, full pipeline imports
   with no heavy CV deps; ~2976 LOC promoted, 614-LOC old skeleton deleted). Design: hkjc.vision = PURE
   perception (feeds/ observation models + silk/identity + geometry_cv/ground_contact/ocr primitives + schema;
   never imports data; cv2/av/rapidocr lazy). hkjc.reconstruction = orchestration on top of vision+data+core:
   geometry.RaceGeom (sources width/bend/draw/start-lateral from core.Race — track facts never re-encoded),
   clock (ClockAnchors), fusion (ONE smoother v2 'smooth_field'+'Obs' = #97 phase-gated EOM long-kernel +
   bounds-only lateral + #96 truncated-box/PAVA-ordering/geom-downweight constraints), reconstruct
   (Reconstructor(race,recipe).run()→RaceTrajectory; RaceRecipe = per-race sync/calib inputs). Deps one-way:
   core◄data◄reconstruction, core◄vision◄reconstruction. Physical constants (THOROUGHBRED_LENGTH_M 2.4,
   HORSE_LANE_WIDTH_M 1.10, BARRIER_STALLS) in core.track. Docs: CV_RECON §8 (library layout) + CLAUDE.md
   (now 8 packages). De-dup proven (width=22.5 from core; silk/rail/ocr single-homed). OPEN SEAM (honest):
   Reconstructor.extract_feeds() (raw 4×mp4 → numbered per-feed rows + identity solve + window auto-scan) NOT
   yet wired — the tested pure tail from_feeds() runs clock→fusion→assemble from supplied rows; until
   extract_feeds lands, end-to-end runs still go through studies/cv_r0* adapters (source-of-truth, NOT yet
   archived). REMAINING: (a) wire extract_feeds (prereq to retiring studies); (b) THEN archive studies +
   fold mode_b's AVV machinery up into vision proper & retire the 'mode_b' name (user directive — only after
   its dependents retire); (c) FIX core.geometry.is_bend start-chute approximation for 1650/1800 trips IN CORE
   (separate model-geometry task, not bundled — fusion currently consumes RaceGeom.bend_weight as-is). Tracked
   in task #100.**
   **CV BACKBONE REVAMP — the walkthrough exposed two fatal raw-measurement bugs; #100 promotion PAUSED, the
   per-feed→fuse pipeline is being REPLACED (#101). DIAGNOSIS (all-frames-quantified + video-confirmed): (1)
   PER-HORSE LONGITUDINAL IS NEVER MEASURED — all 3 metric feeds do `arclen = race_t_to_arclen(race_t)` (the
   shared section-clock value, identical for every horse), so the field is shoulder-to-shoulder BY CONSTRUCTION;
   the along-rail s_px IS computed (rail.project) then discarded. Per-horse MEAN-speed spread in the fused
   output was only 0.20 m/s (R01) — pure smoother fabrication, not measurement. (2) RAIL FLIPS — every rail fit
   is a FREE polynomial/PCA on noisy pixels (no track-shape prior): drone angle range 124°, 31-57 frame-to-frame
   jumps>20°/race, worst single jump 110°; aerial discrete breaks. The two share ONE root: NO STABLE METRIC
   BACKBONE (arclen lives along the rail, lateral from it). Raw-feed reality: aerial=per-horse lateral + shared
   odometer scalar (long 11% finite, NaN at finish); DRONE=dense full-race per-horse geometry but 82% ANONYMOUS
   (only #8/9/11 IDed) + worst rail; patrol=finish-straight only; RF=rank+4-5 margins. CRITICAL CORRECTION
   (user, video-verified): the aerial VIDEO is rich in front-to-back (AVV-render along-track DEPTH + the
   saddlecloth LEGEND order strip) — 'aerial can't separate front-to-back' was a CODE artifact (reads odometer),
   never the signal.
   ARCHITECTURE DECISION (user-driven, agreed): STOP doing 4 separate per-feed recons → late fusion. Instead a
   TWO-LAYER pipeline: (A) EXTRACTION per feed = raw observations ONLY, no metric commitment (detections+silk,
   rail pixels, distance-marker poles WITH image positions, legend order, odometer, finish margins); (B) ONE
   JOINT RECON over all of it: a SINGLE RIGID rail (its planar shape DERIVED from the fixed track geometry — not
   re-fit per feed) + a per-feed camera pose + the per-horse trajectory (arclen,y_lat,t), solved so every feed's
   raw observations are explained. Metric commitment happens INSIDE the recon where all constraints are present.
   Rigid-rail method (user): track is a fixed geometric shape → derive centreline (X(d),Y(d)); treat rail as a
   RIGID BODY; per frame fit the homography track-plane→image to rail pixels + distance markers (known-arclen
   landmarks — pin pose even on bends, closing the §7 foreshortening gap) + odometer (absolute anchor). FORKS
   CHOSEN: STAGED joint (poses first, then trajectories jointly given poses — not full BA) + AERIAL FIRST
   (richest geometry: markers+odometer+clean render rail+legend+7 IDed), then add drone/patrol/RF as extra
   observation models. DISCIPLINE (user): fusion SMOOTHER stays OFF until the raw per-frame geometry is sound and
   measurements are in the physical ballpark (finish margins/order vs official, lateral∈[0,W]); validate against
   the actual VIDEO (overlay projected rail+markers+positions on frames), not code/numbers. Reuse: all extraction
   code. Retire: the 4 per-feed metric projections + the late smoother (identity-consensus becomes a recon prior).
   BUILT so far (#101): studies/cv_rigid_rail.py = the rigid HV-'C' centreline (local R≈159m on bends, straight
   elsewhere; global closure 155m off under the equal-radius approx — irrelevant for local homography, flags
   true per-bend radii needed for a global embed). marker IMAGE POSITIONS now plumbed through extraction
   (read_distance_markers returns (value,cx,cy); cv_joint_aerial.py = the new EXTRACTION LAYER + landmark
   inventory, caches raw obs to cv_joint_aerial_R08_obs.npz). NEXT: Stage-1 aerial pose fit + overlay-validate.
   Tracked in #101.**
   **SHARED-CLOCK AUTO-SYNC (user-prompted — replaces the hand-tuned per-feed offsets). All 4 R08 feeds carry
   AUDIO; the broadcast feeds share it. Audio cross-correlation (8kHz mono, FFT): full≡drone lag +0.000s peak
   1.000 (IDENTICAL track); full≈aerial lag −0.032s peak 0.74 (same commentary, <1 frame); full vs patrol peak
   0.011 (NO shared audio — independent mix). ON-SCREEN GRAPHICS: full/RF has a clean broadcast overlay — a race
   CLOCK 'M:SS.ss' (e.g. 0:18.68 @ vid_t 20s → gate-open at full vid_t≈1.32s, the absolute race_t=0 anchor) +
   'NNNN M TO FINISH' (a TRUE per-frame field odometer/arclen, no inference) + sectional + legend strip. aerial
   render has its OWN clock (0:38.81). patrol has NO timer/overlay. SO THE SHARED CLOCK: (1) full/RF race_t =
   read the timer directly (kills the section-clock hand-tuning + gives an independent absolute arclen via TO-
   FINISH); (2) drone race_t = full's (audio lag 0); (3) aerial via audio (−0.032s) AND its render clock (cross-
   check); (4) PATROL — no audio/timer → sync on the FINISH-LINE CROSSING (sharp head-on event; race_t known
   from full's timer; patrol's useful coverage IS the finish straight). Net: 3/4 feeds auto-sync sub-frame; only
   patrol needs one visual anchor. This is a joint-recon input (the shared time axis) — fold the timer/TO-FINISH
   reads + audio offsets in as the clock layer.**
   **HARD INVARIANT (user): NOTHING in the pipeline may be hand-tuned per race — hand-tuned constants are the
   edge-case landmines that yield confident-but-wrong recon. Pipeline INPUT = {4 video URLs + official race
   metadata (draw/distance/course)} and NOTHING else. Everything else auto-detected: sync offsets→audio xcorr +
   full/RF timer OCR + finish detection; sections/section_bounds/winner-finish-t→full/RF 'M TO FINISH'+clock;
   per-feed windows→layout/coverage detection; identity→consensus; lateral scale→rigid-rail homography + known
   width. ALLOWED: physical constants (horse length 2.4m, track width, track shape) — physics, not per-race fudge.
   BANNED: any constant tuned to make one race line up (the old RaceRecipe patrol_offset/rf_offset/drone_gate_vid_t/
   section_bounds/name2num/WINNER_FINISH_RACE_T are all to be REPLACED by detection).**
   **STAGE-1 AERIAL FINDINGS (#101, studies cv_joint_aerial.py / cv_joint_aerial_pose.py / cv_rigid_rail_fit.py
   / cv_debug_overlay.py; R08). (a) FRONT-TO-BACK IS REAL in the aerial (user was right): along-rail projection
   gets winner #6 as leader + 257px median field spread (old pipeline forced 0). But scale-free order is only
   +0.19 vs official — works on STRAIGHTS, collapses on BENDS (separation goes into DEPTH/image-y; the near-rail
   projection can't see it). (b) RAIL WAVINESS (user Q 'why isn't the straight rail straight?') = the free CUBIC
   fit_rail_spline chases non-rail whites (painted numerals, silks, far rail); RIGID fix (RANSAC straight on
   straights / known-R arc on bends) makes f2580 come out FLAT (y 456→456) — but it locks onto the wrong white
   BAND (top boundary, not the running rail at the avatars) → a DETECTION problem. (c) DETECTION vs IDENTITY:
   detect_blobs finds 12-13 blobs/frame (NOT the bottleneck) but the pipeline only NUMBERS 5-7/11 — the loss is
   IDENTITY ATTRIBUTION (tracklet-link + legend assign), worst when bunched (finish). markers: read_distance_markers
   returns 0 on R08 aerial though '100M/110M' poles are visibly painted (reader ROI/OCR miss — fixable, free
   finish-straight anchors). DECISION (user deferred): fine-tuning a vision model is DEFERRED — the AVV is a
   SYNTHETIC render (consistent sprite, KNOWN distinct silk per horse, uniform turf), so classical is cheaper +
   higher-leverage: (1) SILK-COLOUR-KEYED detection+assignment (we already have per-horse silk sigs + per-det Lab;
   search each expected silk, assign to nearest known → fixes the 13→5-7 loss AND fills misses; field=11 + legend
   order are hard constraints); (2) template/shape match for bunched cases; (3) fine-tune ONLY if (1-2) plateau
   <~10/11 (synthetic render makes auto-labelling cheap). USER VIZ STANDARD: extraction-debug overlays must draw
   avatar BOUNDING BOXES (+ ground-contact + rail + ID) — cv_debug_overlay.py. NEXT: build silk-keyed detect/assign
   → re-overlay toward 11/11 → then the rigid-rail homography for bend depth (horse-size 2.4m ruler, #98).**
   **AVATAR GEOMETRY (user-measured, corrects my under-estimate): the field avatar is the full
   HORSE+SHADOW elongated sprite ~50px long × 10-20px wide (~5 grid cells), NOT the ~9px silk patch the
   detector currently latches. CRUCIAL: the avatar ROTATES through the race — its LONG AXIS = the running
   direction = the local RIGID-RAIL TANGENT at that arclen (straight on the home straight, rotating with the
   arc on the bend). So rotation is PREDICTED by geometry, not a free unknown. Revised detection (all coupled
   via the rail): detect the full oriented ~50px body aligned to the rail tangent (dark body+shadow carrying
   the silk patch); NMS at the ~50px oriented footprint kills duplicate/overlap detections (the 13>11 over-
   count); orientation from the rail tangent makes oriented boxes/template-match well-posed despite rotation;
   silk patch→identity, body→position, temporal track→gap-fill. Avatar projected length also varies with
   orientation×depth → ties the 2.4m ruler to the homography. Detection↔rail↔identity are mutually
   constraining = the joint-recon structure at the pixel layer.**
   **DETECTION PROGRESS (#101). Silk-patch blobs over-detect ~2x (18-20/frame vs field 11) → 890 fragmented
   tracks. WINNER (user idea — TURF SUPPRESSION, studies/cv_avatar_preproc.py): auto-detect turf hue (modal
   hue of saturated greenish px, no tuning) → non-turf mask (dark body OR vivid off-green silk, exclude bright
   rail) → morph-close so body+silk+shadow = ONE blob → connected components filtered by ~50x16 footprint.
   RESULT: 9-11 clean blobs/frame (one per horse), per-blob ORIENTATION from moments, clean mask even at the
   bunched finish. ADOPT as the detection foundation. NEXT: swap into the tracker (kills fragmentation →~11
   stable tracks); sample silk colour INSIDE each blob → silk-keyed per-track ID; split occasional merges
   (9-10 when overlapping) via field=11 + spacing/watershed + temporal disambiguation. Also: oriented-box
   detection framework (cv_avatar_detect.py: NMS + Hungarian temporal track + gap-fill) is built and works
   when detections are clean. DIAG OUTPUTS now live in data_cache/diag/ (decluttered: walkthrough frame dirs
   removed).**
   **SINGLE-FRAME DETECTION DIALED IN (user-driven, studies/cv_avatar_edgefill.py). Sequence that won a
   variant bake-off (cv_preproc_variants.py): turf+rail SUPPRESS (auto turf-hue, no tuning) → CANNY edges →
   dilate/close → FILL contours → ∩ non-turf colour mask → close(silk-gap)/open(denoise) → shape/size filter
   (drop rail-bar + noise) → K-MEANS-TO-FIELD on the denoised foreground (k=11 on pixel coords: groups the
   silk-split pieces into one cluster AND splits a genuinely-touching pair; uses field=11 directly — user's
   call, validated). RESULT R08: EXACTLY 11 clean one-per-horse blobs at bend AND bunched finish (was 18-20
   silk-patch over-detect / fragmented colour-mask). The silk-SPLIT cause (user): a bright-yellow silk sits at
   hue~28, inside the turf green band, so it's suppressed AS turf → gap mid-horse. A vivid-silk source-fix
   (protect v>175&s>110 from turf) was TRIED and DROPPED — it leaked bright HV turf (sat~126) into the mask;
   k-means-to-11 handles the split cleanly instead. KEY LESSONS: (1) colour-mask alone fragments (mid-tone body cut) OR over-keeps (loose); EDGE+
   FILL gives solid whole-horse silhouettes — the right primitive. (2) avatar orientation = TRACK/RAIL tangent
   (global per-frame heading), NOT per-blob moments (noisy). (3) silk too small to read for ID even by eye →
   ID via DRAW→start-frame position→track-forward, not silk. NEXT: wire edge+fill detector into the FIXED-
   cardinality tracker (clean ~11 dets → contiguous tracks, fixing the earlier 890-fragment / 0-contiguous
   failure) → seed IDs from the start frame (draw order) → per-horse along-rail longitudinal. AWT GENERALIZATION:
   agent dispatched (background) to test turf-suppression on a Sha Tin all-weather (dirt) aerial — hue auto-
   detect should adapt; awaiting its verdict.**
   **AWT GENERALIZATION — agent verdict (studies/cv_awt_preproc_probe.py, data_cache/video/aerial_awt_20260308_03.mp4
   = ST 2026-03-08 R3 1200m AllWeather; diag awt_preproc_f*.png). VERDICT: hue-agnostic skeleton ADAPTS but needs
   AWT changes. AVV DIRECT-MP4 RECIPE (verified, auth-free, token literal): https://streaminghkjc-a.akamaihd.net/
   hdflash/replay-aerial/{YYYY}/{YYYYMMDD}/{NN}/eng/replay-aerial_{YYYYMMDD}_{NN}_eng_2500kbps.mp4 (NN=race no;
   other feeds swap the replay-aerial token). AVV render is a DELAYED 3rd-party render (same-day may 404 while
   other feeds 200). FINDINGS: AWT background brown/tan hue ~12-19 + LOW sat (~50 vs HV green 35-60 sat~126); the
   modal-hue auto-detect CORRECTLY locks onto brown (premise holds). But the OLD colour-mask detector over-detects
   ~1.7x on straights (low dirt contrast → fragments) and ~3.5x on BENDS (39-43: non-white tan rail + lighter
   outer-dirt run-off survive the green-tuned white_rail v>180&s<50 filter + single-modal-hue too narrow for the
   brown RANGE). AWT CHANGES NEEDED (for later — R08 is HV turf, not now): (1) lean on EDGE+FILL (which we already
   adopted) + silk-anchored detection, not pure bg-subtraction, since dirt avatars don't pop; (2) background =
   LOW-SATURATION generally (S<~40) not a single hue band, to kill all dirt shades; (3) explicit non-white
   rail/marker suppression / restrict to the inter-rail LANE before connected-components (esp. bends); the agent
   ran the OLD detector — our new edge+fill should be RE-TESTED on AWT. Net: corpus generalization is de-risked,
   recipe captured; AWT-specific tuning deferred until the HV pipeline is contiguous. **AWT RECONSTRUCTION IS
   BLOCKED (task #102) until the brown-dirt foreground model is built — the dirt background blends with the
   avatars (disastrous low contrast, user-confirmed). HV turf is unaffected.**
   **CENTROID VARIANTS (user request, cv_avatar_edgefill.py returns BOTH): A = pixel centroid of the k-means
   cluster (pixel-weighted, leans to the bigger fragment); B = unweighted mean of the per-connected-component
   piece centroids (= midpoint of the two blobs for a silk-split). On clean frames they differ only on the
   silk-split horses by ~5-12px; A is the ROBUST default — B gets yanked ~57px when a stray rail/noise fragment
   lands in a cluster (f770 rail-leak), while A's pixel-weighting absorbs it. Use A. (Also surfaced: f770-type
   EARLY/distant frames leak the diagonal rail into the mask — the aspect filter misses segmented rail; rail
   suppression needs strengthening for distant frames, tracked under #101.)**
   **DISTANT-FRAME RAIL LEAK — FIXED + the geometric-bound plan. f770 (early/distant, rail crosses the small
   field) leaked the DIAGONAL rail into the mask; bbox-aspect missed it (a diagonal line has a square-ish bbox).
   FIX: reject components by PCA principal-axis LENGTH (>130px) + ELONGATION (major/minor >6) — orientation-
   independent, catches diagonal rails, keeps horses (~50px) and 2-horse merges (~100px). Plus a FIELD-ROI
   pre-step: morphological-open the non-turf colour mask (removes the thin rail, keeps ~16px horses) → robust
   (1-99 pct) bbox = the field, crop the mask to it. RESULT: f770 A-vs-B 10→1 horse, all frames clean 11.
   These are HEURISTIC interims. DESIGN DECISION (user): the RIGHT detection bound is GEOMETRIC — once the rigid
   rail is reconstructed (#101), restrict detection to the LANE BAND (rail → +track-width perpendicular), which
   excludes rail+infield+stands+distant-bg in one principled, cheap step (rail & detection co-constrain = the
   joint-recon way). Adopt the geometric lane-bound INSIDE the rigid-rail integration; the PCA/opening heuristic
   is the interim until the rail is reliably localized (it still has the wrong-band issue). Centroid choice =
   A (pixel centroid); both returned but A is the robust default.**
   **RAIL RECON DIAL-IN STARTED (#101, studies/cv_rail_recon.py; diag cleared + re-sampled to 16 frames spanning
   the race for coverage). FINDING: naive RANSAC on all white-rail candidates LOCKS THE WRONG BAND on straights/
   finish (picks the dominant far-boundary line ABOVE the horses, not the running rail). FIX (validated f2610):
   HORSE-GUIDED selection — the running rail is the candidate line ADJACENT to the field, so select candidates
   within a track-width perpendicular band of the field axis (PCA of horse dots), then RANSAC → fit now tracks
   the RUNNING SURFACE through the field across the whole race (not the top band). This is the joint coupling:
   detection ↔ rail constrain each other. REMAINING RAIL REFINEMENTS: (1) snap to the inner-rail EDGE (infield
   side of the field) not the field centre; (2) apply the RIGID shape constraint — straight vs arc(R) chosen by
   odometer→track-geometry (cv_rigid_rail.local_radius), so the rail can't wiggle; (3) TEMPORAL smoothness (rail
   evolves slowly frame-to-frame; reject jumps). Then the rail → LANE BAND bounds detection (kills the f770
   infield-hedge false positives) + anchors the longitudinal recon. Diag images: data_cache/diag/cv_rail_recon_*.**
   **RAIL RECON — 16-FRAME CAREFUL SCRUTINY (user-directed) OVERTURNS the horse-guided fix. Verdict per frame:
   the CYAN white-rail candidates ACCURATELY trace the running rail in EVERY active frame — STRAIGHT on the
   straights (back 770-920: f1420/f1590; home 0-300: f2440/f2785 horizontal), an ARC on the bends (f570/740/910/
   1080/1250/1760/1930/2100/2270). The rail is OFFSET from the field and FLIPS SIDE (above vs below) as the
   camera orientation rotates around the lap — the horse↔rail gap IS the y_lat signal. The GREEN fit is OFF
   almost everywhere: (a) a STRAIGHT LINE can't follow the arc, (b) the HORSE-GUIDED PULL (my error) drags it
   onto the field not the rail (on the home straight green goes diagonal chasing the strung field while the rail
   is horizontal). f0 = start GATE (real cam, no render) → no rail. CORRECTED FIX: DROP the horse-guided pull;
   fit a rigid CURVE DIRECTLY to the cyan — straight vs arc(R) by odometer→track-geometry — robust to dashed-
   marker/boundary outliers; flag rail ABSENT when no coherent curve (gate frames). The cyan is already good;
   it needs the right MODEL (curve not line). Do NOT re-introduce horse-guidance to position the rail.**
   **RAIL FIT CORRECTED + VALIDATED (cv_rail_recon.fit_rail_curve). Robust QUADRATIC fit DIRECTLY to the cyan
   rail candidates, no horse guidance: build cyan mask → connected-components → keep components with EXTENT≥40px
   (the continuous rail line; drops short dashed markers/noise) → polyfit deg-2 along the major axis (auto
   horizontal vs vertical) with 3 rounds of 2σ outlier rejection. RE-SCRUTINIZED: green now tracks the rail in
   ALL phases — arc on bends (f570 rail-above, f910/f1930 arc-below), horizontal on straights (f1420 back, f2785
   finish) — vs the old straight+horse-pull that was off everywhere. REMAINING: (1) enforce the RIGID straight-
   vs-arc(R) by odometer→geometry (currently free quad, tracks well empirically) + regularize a≈0 on straights;
   (2) extend the curve full-width via geometry where rail pixels are sparse; (3) TEMPORAL smoothness; (4) verify
   no-rail handling on the gate frames; (5) confirm it never locks the far boundary (held on the checked frames).
   THEN rail → lane-band bounds detection + anchors longitudinal.**
   **RAIL FIT v2 (user caught f1760 FAILED). TWO root causes fixed: (1) BAND TOO NARROW — track_band is centred
   on the HORSES (f1760: y453-620) but the rail sat BELOW it (y620-810), cut off → fit got only a sliver. Fix:
   detect rail candidates in a WIDER band (0.28-0.82·h: below top text/widget, above legend) so the rail is
   caught whether above OR below the field. (2) MULTIPLE LINES — the wider band exposes several horizontal lines
   (inner running rail, outer rail, foreground/legend boundary, dashed markers); robust fit locked the bottom
   boundary (most pixels). Fix: use the field to SELECT WHICH line (connected-component nearest the field
   centroid = the inner running rail the horses run along), then fit the curve to THAT line's pixels + collinear
   inliers (<12px) and refit. CRITICAL DISTINCTION (honours 'no horse-pull'): the field only PICKS which cyan
   line; the curve POSITION comes entirely from the rail pixels. VALIDATED: green on the inner running rail
   across bend-above (f570), bend-below (f910), two-rail bend (f1760), back straight (f1420), finish (f2785).**
   **RAIL v3 — TEMPORAL INNER-RAIL TRACKER (user caught the fundamental flaw; cv_rail_temporal.py). The
   per-frame 'nearest coherent line' LATCHES THE OUTER RAIL when the inner rail is OCCLUDED by the field that
   runs on it (f772/f2200 both wrong) — and the crude median-RESEED (cv_rail_recon_full) corrupted CORRECT fits
   in rapid-motion regions (f752/f760: raw right, reseed wrong). REVERTED the reseed. FIX: PASS1 extract ALL
   candidate rail lines per frame (each large cyan component → its own curve + its image-y at field-x); PASS2
   GLOBAL ROBUST SMOOTHER — init each frame to the candidate nearest the FIELD (inner rail = band nearest the
   horses), then iterate {moving-median over frames → re-select nearest candidate} to lock the dominant inner
   band and REJECT outlier picks (occluded frames). RESULT: smooth coherent inner-rail trajectory across the
   race (3 candidate bands: top boundary ~300 / inner ~450-620 / outer ~770 — tracker holds the inner);
   f772/f2200 outer-latch FIXED. USER KEY INSIGHT: the inner-rail image position is PREDICTABLE from the EXACT
   track geometry + the ODOMETER (index by arclen, curvature from is_bend) — fold in as the prior (stronger than
   frame-smoothness). REMAINING: (a) at OCCLUDED frames emit the PREDICTED curve (geometry-projected / neighbour-
   shape shifted to tracked ry), not a candidate fragment; (b) explicit odometer-arclen indexing + geometry
   curvature prior; (c) keep iterating against the outlier-frame overlays until all clean (user discipline).
   DISCIPLINE: VISUALLY SCRUTINIZE raw+diagnostic images (docs/CV_RECON §0). reseed approach RETIRED.**
   **RAIL REFINEMENTS (SUPERSEDED by v3 temporal tracker) (cv_rail_recon_full.py; user-endorsed, f0/gate DEFERRED to race-start trimming).
   (1) RIGID straight/arc: deg=1 on straights, deg=2 on bends, chosen by arclen(odometer-interp)→TrackGeometry.
   (2) TEMPORAL reseed: per-frame rail summarised by image-y at the field-x (drifts smoothly as the camera pans);
   median-filter + reseed frames whose raw jumps >22px off the local median ('use nearby frames as a prior',
   user). RESULT over 641 frames: 0 fit-failures, 43 (7%) outlier-jumps reseeded, smooth rail trajectory across
   the race (fig cv_rail_recon_temporal.png). (3) full-width extension = minor/partial (curve spans the rail
   pixels). RAIL RECON IS NOW SOLID — detection + rail are the dialed-in foundation. CV DISCIPLINE persisted to
   docs/CV_RECON §0: VISUALLY SCRUTINIZE raw+diagnostic images every time (numbers hide mis-placement).**
   **RF/RACE-FILM FOLDED IN AS THE 4th FEED (#89, journal 0086, studies/cv_rf_longitudinal.py,
   data_cache/cv_rf_longitudinal.parquet, fig cv_rf_qc.png).** RF is real side-on broadcast; on the shared
   clock race_t=RF_vid−1.34 (0077). It DELIVERS the two things 0084's fusion lacked. (1) FINISH MARGINS
   (the S3 surface #60): photo-finish within-frame pixel-gap/body-length(2.4m), open-gap 75th-pct, n≥2 →
   vs the official chart LBW **MAE 0.41 lengths, margin+order Spearman ρ +0.90, 9/10 pairwise, fit
   0.70·official+0.23** (#12 NECK 0.28 vs official 0.3 nailed; 5 horses co-framed with the winner near the
   line). HONEST residual slope 0.70: the leader-tracking camera loses back-markers from frame before their
   gap fully opens → deep margins under-read (#8 worst). (2) per-horse LONGITUDINAL ORDER backbone: the
   bottom LEGEND STRIP is a rendered running-order overlay (~official chart); its saddlecloth-number box is
   too small to OCR but the english NAMES OCR cleanly → name→number LUT bridge, 8-deep order at 42
   timepoints race_t 8.7-90.7s, **VALIDATED vs official running positions ρ +0.98/+1.00/+1.00 at the
   1/4,1/2,3/4 calls** (winner #3's come-from-behind "7-7-6-1" captured). IDENTITY: 8/12 OCR metric at the
   line; whole-race ORDER effectively 12/12 via the legend. **NET: RF resolves the per-horse longitudinal
   ORDER everywhere (the field-odometer ρ~0 gap is CLOSED) + the metric finishing gap at the line; the
   per-horse absolute-metre stagger mid-race stays unresolved by ALL 4 feeds (panning side-on horse pixels
   give order, not absolute arclen — same §4b-metric class as the lateral nulls), but ORDER + finish-margins
   is exactly what the generative finishing-ORDER model + the S3 Wald spread (W'_0 sharpening: tight close
   finish ⇒ small run-in diffusion D, upper tail; long back tail ⇒ lower tail) consume.** All 4 feeds now
   have a validated per-feed observation model. NEXT: enter RF into the 0084 smoother as observation-model 4
   (legend order = per-horse rank/ordering constraint over the whole race; finish margins = metric
   longitudinal gaps at the line) → give the fused trajectory the per-horse longitudinal order it lacks;
   then the run-in lateral (drone run-in homography / patrol finish-cam) remains the last gap.
2. **FIELD / DEPLOYMENT cause ← equilibrium pace / mean-field game** (`run_policy_dataset`) + wiring
   the field/pace coupling into the generative dynamics. (The former S7 + S2 — NOT superseded; a real
   causal dimension, philosophy-pure.)
3. **PREP cause (fitness on the day) ← training-load / trackwork** (Banister), modelled CAUSALLY as a
   modulation of the physiological constants (CP/W' on the day), NOT the deleted gallop-fusion feature.
4. **CONDITION cause (soundness) ← gait CV** (gated on parade footage).
5. **PHYSIOLOGY sharpening** (margins→W', P_max/τ — the former S3/S6) — valid physiology-dimension
   work, not a win-skill dead end.

The 0059 "SUPERSEDED" list is RESCINDED. What is closed: the expectation that one dimension matches
form, and the form-as-measurement route.

## What is preserved / refuted

- **Preserved substrate** (physics + honest references, NOT regression): `energetics`,
  `energetics_dynamic`, `calibration`/`calibration_bayes` (EB population prior), `plackett_luce`/
  `harville_model`/`conditional_logit`/`baselines` (the Harville/market/uniform scoring references).
- **The acausal model is deleted, not refuted** — it fit the market well; it was purged for being
  acausal. Its empirical findings (which meta-signals carried market skill, the draw/collateral/
  peakgap structure) are recorded in the journal as the RE-EMERGENCE TARGETS the causal model must
  regenerate — they are validation surfaces now, never inputs.

## Methodology guardrails

- **Leak-free is non-negotiable:** strictly-prior θ, train-only population per fold, `assert_no_lookahead`
  (drift must be 0), paired fixed-seed `PYTHONHASHSEED=0`. `quick_eval` over-states ~4× — full-scale only.
- **Identification, not one metric doing all the work:** pin each constant from the surface where it
  dominates; cross-surface disagreement is a falsification pointing at missing physics, not a knob to fit.
- **Win% is parked** — it is the LAST validation, never the objective. Optimize path-fit, re-emergence,
  and cross-surface consistency; let the win gate be the honest final judge.
