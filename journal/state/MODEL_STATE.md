# Model State — carry-forward (survives the refactor)

> Self-contained snapshot of where the ABM calibration stands, the methodology, and the
> open residual queue. This is the modelling agent's working memory. After the refactor it
> should become the seed entries of the modelling agent's journal channel.
> Last updated: 2026-06-05.

## Objective (do not lose this framing)

**Fit the market first. Bet-skill (% of market) is a GOODNESS-OF-FIT metric, not an edge hunt.**
% of market = (uniform_LL − model_LL) / (uniform_LL − market_LL), measured leak-free walk-forward.
Do NOT invoke the "private info" excuse to stop — that logic would have quit at 10% skill, yet we
are now ~48%. The edge/trading decision is the user's to make, not a reason to stop calibrating.

**First-principle regularization (physics + game theory).** Every factor introduced must couple as a
real, plausible *mechanism* in the plant (EOM) or policy — never an arbitrary regressor (overfits).
The ABM = plant (power-balance EOM: finite W′ reserve, traffic, drafting, bend ground-loss, mass) +
policy (action = π(state): kick timing/intensity, pace tilt).

## The calibration loop (this is the method)

**diagnose → recalibrate → bake → re-diagnose.**
Market-residual diagnosis: `resid = market_prob − model_prob` (both de-vigged, race-normalized),
regress on each observable mechanism with race-clustered SEs. Sign + |t| say which mechanism the
model mis-weights and by how much. Biggest |t| = the next thing to ground in the physics. Fix it,
re-run, find the next. (Script: `experiments/market_residual_diagnosis.py`.)

## Current calibration constants (baked)

| Constant | Value | Where | Note |
|---|---|---|---|
| **`FORM_HALF_LIFE`** | **2.0 races** | `hkjc.model.constants` | **THE production form is RECENCY-weighted (exp), not the career mean. +7.8pp — the biggest single lever.** Clean half-life optimum at ~2 (career=∞ worst). Career form stale over long HK careers; recency tracks current ability + matches market weighting. `features.recency_form_table`. |
| **`DRAW_GAIN_BY_VENUE`** | **ST 0.003 / HV 0.008** | `hkjc.model.constants` | **Bend ground-loss in the wald/pl drift: `L_eff = dist·(1+DRAW_GAIN·(draw−1))`. Re-added to wald (+1.9pp ST) then made PER-VENUE: it's geometric, so HV (bend-dominated, strongest HK draw bias) wants ~2.7× ST. HV was massively UNDER-penalised by the global 0.003. fit.py ZEROES it on the ST-1000m straight chute (no bends). SIGN LESSON: a subagent read the HV residual as "over-weighted→set 0"; the sweep proved the opposite (0 craters HV to 41%) — always sweep.** |
| **`FORM_GAIN_BY_VENUE`** | **ST 5.0 / HV 4.0** | `hkjc.model.constants` | Form coupling, now PER-VENUE. ST recalibrated 4→5 at the recency baseline (+0.9pp). HV wants LOWER (4.0) — HV form was over-weighted (residual t=−13.7 vs ST −8.4). Resolved per-race; an explicit `form_gain` arg overrides (for sweeps). |
| `SHRINK_MULT` | **0.5** | `hkjc.model.constants` | Multiplier on the sectional v_c EB residual — it was OVER-trusted at 1.0 (peak 0.5 in both career+recency scans; dropping it, 0.0, costs only ~0.8pp — the sectional energetics barely earns its keep for win). |
| `W_PRIME_GAIN` | **1.0** | `hkjc.model.constants` | Stamina coupling; REVERTED 2→1 at the recency baseline. Recency form ABSORBED the reserve signal — re-diagnosis flipped W′ to over-weighted (t=−8.95). Win-prob impact flat over 0–2 (W′ is a margin effect); parsimony + residual sign both say revert to the un-amplified EB estimate. |
| `JOCKEY_GAIN` | **6.0** | `hkjc.model.constants` | Jockey trip/ability; re-swept 5→6 after the recent bakes (shared optimum both venues, +0.4/0.3pp). |
| **`POS_FADE_GAIN`** | **−1.0** | `hkjc.model.constants` | Positional-fade: `v_c += gain·(early_pos−field_mean)`. Non-interacting physics over-rates front-runners / under-rates closers — the mean-field shadow of the pace equilibrium (see THEORY §3b). Clean shared optimum both venues. `features.early_position_table`. |
| **`ALLOMETRIC_B_BY_VENUE`** | **ST 1.1 / HV 1.0** | `hkjc.model.constants` | Allometric mass law `v_c ∝ m^(b−1)`, now applied on the wald path. The persistent #1 ST residual; converts only at ST (+0.5pp; HV 1.0 = off). |
| **`DRAW_GAIN_AWT`** | **0.008** | `hkjc.model.constants` | Per-SURFACE draw: AWT dirt has a stronger rail bias than turf (overrides the venue value). AWT 43→49%. |
| **`DRAW_FS_GAIN_BY_VENUE`** | **HV 0.25 / ST 0.0** | `hkjc.model.constants` | **Crowding-amplified draw — the ONE convertible INTERACTION term (the ABM motivation as a mean-field shadow): `dg = draw_g·max(0, 1+gain·(N−12))`, a wide draw costs more in a crowded field (boxed-in), clamped ≥0 (never reverses). HV-only: PAIRED full-scale test converts only at HV (53.8→54.1%, +0.3pp; collision physics dominates the tightest track), FLAT at ST. `fit.py` wald branch.** |
| **`TRIAL_GAIN_BY_VENUE`** | **ST 1.5 / HV 1.0** | `hkjc.model.constants` | **FIRST NEW-DATA lever. Barrier-trial speed figure (trial v_c vs going×dist par) — the only on-track form a DEBUTANT has; market otherwise efficient on debuts. Full 4-yr panel (94.8% debut coverage): ST 60.0→61.9% (+1.9pp, biggest lever since recency form), HV +0.6pp. Needs `data_cache/barrier_trials.parquet`. `features.barrier_trial_table` (as-of on short stable brand).** |
| **`COLLAT_GAIN_BY_VENUE`** | **ST 1.0 / HV 2.5** | `hkjc.model.constants` | **Competition axis CRACKED. Relational opponent-adjusted (Massey) velocity rating — credits a margin vs a STRONG field more; propagated online (leak-free). A scalar Elo FAILED (collinear w/ rtg); this RELATIONAL form converts. HV +2.3pp (+ECE improves) — closes ⅓ of the ST-HV gap. `features.collateral_form_table`.** |
| **`PEAKGAP_GAIN_BY_VENUE`** | **ST −0.015 / HV −0.025** | `hkjc.model.constants` | **PROCESS feature (not point): current rtg − own prior PEAK rtg (≤0). Carries an ABSOLUTE career landmark the point can't express (a declining horse runs below its lagging rating). Marginal +0.4pp ST / +2.4pp HV (does NOT overlap collateral). Rating velocity/accel did NOT convert (restate form). `features.rating_trajectory_table`.** |
| **`FADE_GAIN`** | **0.1** | `hkjc.model.constants` | Trajectory-shape: final-vs-peak section velocity (W′ emptying late); the drift under-credits horses that hold speed. Shared-sign mean-field const, +0.2/0.1pp. (sustained-speed DON'T-BAKE — priced.) `features.trajectory_trait_table`. |
| `SPEED_NOISE` | 0.12 | `hkjc.model.constants` | Velocity-noise (= log-normal time). Diffusion `diff_coef≈170` ties it (`sim="wald"`). |
| `awt_diff_mult` | 2.0 | `fit.py` (sim="wald") | AWT is 2× as diffusive as turf (surface-dependent D) — fixes AWT over-confidence (12.7%→23.7%). |

Current fit: **ST ≈ 65.0% / HV ≈ 59.7% of market** (wald, full-scale paired). This push (60.0/54.1 → 65.0/59.7:
ST +5.0pp, HV +5.6pp) was driven ENTIRELY by NEW DATA + new STRUCTURAL signal — barrier trials (debut/lightly-
raced form), collateral-form (opponent strength), rating-peakgap (career decline), trajectory fade, trainer
hot-streak (intent momentum), **draw→D (the trip-variance third knob)**, the **coherent trial FUSION**
(precision-weighted v_c posterior), the **W′-recoupling of fade/pos_fade** (reserve knob, distance-aware), and
the **deepened collateral rating** (Jacobi multi-sweep + recency + trip-band) — none a reweighting of priced
features. ST broke 65% on the deepened collateral alone (+1.3pp over fusion, super-additive).
Newly baked since the table above (all paired full-scale, PYTHONHASHSEED=0):
- `TRN_HOT_GAIN_BY_VENUE` (ST 2.0/HV 1.0) — trainer hot-streak / intent.
- `DRAW_VAR_GAIN_BY_VENUE` (ST 0.5/HV 0.0) — draw→D trip variance (the third knob; ST-only, the EVT field-size
  argument predicts variance pays in big fields → ST not HV — see THEORY §3e).
- `FADE_W_GAIN=8.0`, `POSFADE_W_GAIN=−80.0` (FADE_GAIN/POS_FADE_GAIN retired to 0.0) — fade + positional-fade
  moved off the v_c proxy onto the **W′ knob** (reserve-metres, distance-aware via W′/L; THEORY §3d T1).
  ST parity, HV +0.3pp (the gain lands in HV sprints — the W′/L signature).
- `TRIAL_PI_RATIO_BY_VENUE` (ST 2.0/HV 0.0) — coherent barrier-trial FUSION: trial folded into the v_c
  posterior at its OWN precision above the shrink layer, weight fading with race count. ST +0.4pp (debut +6pp);
  HV stays additive. The substrate for folding training gallops next.
- `COLLAT_CONSTRUCTION_BY_VENUE` (ST {n_sweeps:3, half_life:4, band_par}; HV {}) — deepened collateral rating.
  ST +1.3pp on the live base; HV keeps the baked one-sweep (band-par fragments its tight trip range).

**THE PATH IS NEW DATA + NEW STRUCTURE, not more mean-field (this push proved it).** The earlier "saturation"
was a saturation of *mean-field terms on the existing race panel* — every per-horse residual was collinear with
the priced ability channel. The break came from (a) NEW DATA the market prices but the model never saw (barrier
trials) and (b) STRUCTURAL signals (relational opponent-strength; the process/career-landmark of peakgap). The
conversion test (paired fixed-seed full-scale — quick_eval over-states ~4×), NOT the residual t-stat, remains the
gate. Refuted-as-mean-field-term but NOT as an axis (need new data/structure): see the deferred + refuted lists.
mean-field terms.

## ARCHITECTURE (audit findings — the model is a prototype, not gospel)

**The win model is a drift-diffusion / random-utility choice model.** Each horse is a particle
`dx_i = v_i·dt + √(2D)·dW_i` (drift = ability v_target, diffusion = D). The winner is the first to reach
the line — an order statistic of finishing times. Two practical consequences:

- **Fast surrogate (no interactions):** independent finishing times → win probs by argmin. `fit.py`
  exposes `sim` modes (all pure numpy, ~35× faster than the JAX sim, full test set in ~30s):
  `probit` (velocity-multiplicative = log-normal time, the in-place model, 48.2%); `wald` (the proper
  diffusion/first-passage Inverse-Gaussian, 47.7% — ties it); `pl` (closed-form Plackett-Luce / Gumbel,
  46.4%, gives the full order distribution); `time` (additive Gaussian, worse). **Use the probit/wald
  surrogate for the WIN-calibration loop** — interactions are confirmed low-leverage for win (draw, mass,
  speed-figure, traffic all tested inert), so the surrogate matches the full sim (48.1% vs 47.7–49%).
- **The agent-based sim is NOT replaced.** A purely statistical model (probit/PL/Wald) CANNOT
  mechanistically represent interactions — blocking, real-time tactics, jockey policy. That is the sim's
  reason to exist and the physics+game-theory core. The sim is the Monte-Carlo of the SAME many-body
  drift-diffusion *with* interactions; reserve it for the policy layer and place/exotic pricing.

**Noise model:** the in-place velocity-multiplicative noise = log-normal finishing time (correct: time>0);
the diffusion law (Wald) ties it and is the more principled foundation (one physical diffusion coefficient,
and the natural substrate for the interaction program). Distance-dependent noise does NOT help (flat best);
the calibration-by-condition distance gradient was small-sample/AWT noise. Baked gains (FG4, WPG2, JG5,
sn0.12) all sit at interior optima under the surrogate too.

**Future direction (not now):** stat-mech / condensed-matter machinery for interactions — exclusion
processes (TASEP) for blocking, active matter for drafting, mean-field games for jockey policy; an
"interacting Plackett-Luce" with PL as the free theory. See journal 0015.

## NEXT LEVERS (post-60% — NEW DATA + NEW STRUCTURE is the path; proven this push)

The 2026-06 fan-out (barrier trials, competition, trajectory, physiology, rating-trajectory, fusion, W′-recoupling,
deepened collateral, theory) lifted ST 60.0→65.0 / HV 54.1→59.7 entirely via new data + structural signal. The
forward levers, ranked:

**A. The INTERACTION layer — the first-attempt ABM code was deprecated, but the axis is DEFERRED, not dead.**
The kinematic sim is gone as production: the old "kinematic BEATS wald at HV +2-3pp" claim was STALE (it
predated the per-venue wald bakes); at the current baseline it LOSES to wald at HV by **−4.1pp** (48.8 vs
52.9, paired), so the per-venue DRAW/FORM/pos_fade/mass bakes superseded its uncalibrated emergent draw.
The sims are deprecated from the codebase (physics preserved in THEORY.md Appendix). What converts so far
are **mean-field shadow terms on the fast wald path**:
- **DRAW_FS_GAIN (crowding×draw)** — HV +0.3pp (baked, HV-only) — but an INTERIM stopgap, not the answer.
- **Relational pace-pressure** (front-runner W′-burn ∝ rival front-runner count, THEORY §3b #1, driven by
  the converting `early_pos`): HV +0.12pp (noise), ST flat — a 4th flat result on the simplistic pace axis.

**WHY the simplistic interaction treatments don't convert (the key open theory, per the user):** draw and
the pace/competition structure are **deeply coupled to the CONTROL dimension** — the jockey's policy (how the
gate is used to take position, when to commit, how the trip is navigated). A regression / mean-field term
(`DRAW_GAIN`, `DRAW_FS_GAIN`, a scalar contestedness) can only capture the *first-order geometric shadow* of
a fundamentally **geometry × control** phenomenon. The ABM did reproduce the qualitative draw/trip
characteristics — the mechanism is REAL — but it is not modelled deeply enough. The proper treatment is a
joint **differential / mean-field game of jockeys navigating the geometry** (the §3b competition game), with
draw as a boundary condition. **DEFER, don't reject:** keep the draw + interaction axis open and treat the
baked draw terms as interim until geometry + control are modelled together. This is the most promising piece
of deep theory-building left — but it is a theory build, not a regression probe.

**B. BANKED this push (all committed; the proof that new data/structure is the path):**
- **Barrier trials** (`TRIAL_GAIN`, `features.barrier_trial_table`) — full 4-yr panel `data_cache/barrier_trials.parquet`
  (27,909 trials, 94.8% debut coverage). ST +1.9pp. Debutants/lightly-raced need NEW form, not reweighting.
- **Collateral-form** (`COLLAT_GAIN`, `features.collateral_form_table`) — relational opponent-strength (Massey
  velocity rating, NOT scalar Elo). HV +2.3pp, ECE improves. The competition axis, cracked.
- **Rating-peakgap** (`PEAKGAP_GAIN`, `features.rating_trajectory_table`) — PROCESS feature, own-peak deficit.
  HV +2.4pp. Pays because it adds an absolute career landmark the point rating can't express.
- **Trajectory fade** (`FADE_W_GAIN`, `features.trajectory_trait_table`) — late-velocity hold, +0.2/0.1pp; now on the
  **W′ knob** (with positional-fade, `POSFADE_W_GAIN`) — reserve phenomena, distance-aware (THEORY §3d T1). HV +0.3pp.
- **Coherent trial FUSION** (`TRIAL_PI_RATIO`, ST 2.0) — trial folded into the v_c posterior at its own precision,
  weight fading with race count. ST 63.3→63.7% (+0.4pp; debut +6pp). The successor to additive `TRIAL_GAIN`.
- **Deepened collateral** (`COLLAT_CONSTRUCTION_BY_VENUE`, ST Jacobi-3-sweep + recency-4 + trip-band) — ST 63.7→65.0%
  (+1.3pp, super-additive with fusion; ECE improves). HV keeps the one-sweep table (band-par fragments its trip range).
- **Training-load DATA banked** (`src/hkjc/data/trackwork.py`, `data_cache/trackwork.parquet`, 30k+ Banister gallops)
  — scraped & leak-free, NOT yet wired (route into the fusion per Plan #1, not an additive bolt-on).

**C. THE FORWARD PLAN (ranked; user-steered):**
1. **Coherent Bayesian fusion — BANKED for trials (`TRIAL_PI_RATIO`, ST ρ=2.0); EXTEND to training gallops next.**
   The trial now folds into the v_c posterior at its OWN precision above the shrink layer, weight fading with race
   count (ST +0.4pp, debut +6pp). The architecture is proven and was explicitly built as **the substrate for the
   gallop layer**. NEXT: fold the (already-scraped) training gallops into the same posterior — `data_cache/trackwork.parquet`
   is landed (30k+ Banister gallop sessions; `src/hkjc/data/trackwork.py`, probe in `studies/trackwork_fitness_probe.py`).
   The training agent's additive probe converted +0.2/+0.4pp (fresh-work subslice 2×), but the RIGHT home is the
   fusion (a gallop is the densest, noisiest velocity obs — give it its own low precision, like the trial). Needs a
   full ~1.5k-horse backfill (current research subsample ~13% coverage) + `trial_vc_table`-style `gallop_vc_table`,
   then a 3rd precision channel `π_gallop = ρ_g·π_race` in the fit fusion. HV's sparse trial axis may convert once
   the denser gallop prior joins it.
2. **CV / vision (IN PROGRESS).** Drone (aerial) replay path-extraction agent running — the (x,y,t) data feed is
   non-interceptable (confirmed), so CV on the near-top-down drone video: frames → detect → camera-motion homography
   → rail-relative (arc-length x, rail-offset y) coords, prototyped on one race (home straight first). Unblocks the
   geometry × control axis (§3c) with real lateral paths. Gait analysis (biomechanical condition) is the follow-on.
3. **Geometry × control / draw axis (THEORY §3c/§3e).** **draw→`D` (trip variance) BANKED** (`DRAW_VAR_GAIN`, ST 0.5,
   +0.4pp) — the first use of the third knob, and the EVT field-size argument (§3e) retro-explains its ST-only split
   (variance pays in big fields). O1 (scalar-MEAN `g(Course)` modulation) did NOT convert (rejects that structure, not
   the mechanism). Still open: **O3** control-quality path-efficiency jockeyship trait from `trajectory_dataset_full`
   (the CONTROL share, no CV); **draw × distance via real `bend_fraction`**; and the CV lateral paths (#2) for the
   genuine 2-D game. Baked `DRAW_GAIN`/`DRAW_FS_GAIN` stay interim until geometry+control are modelled jointly.
4. **NEW §3e phase-structure levers (theory-derived this session; convertible & leak-free):**
   - **Gate-break trait** — first-call rank *adjusted for draw* (= `early_pos` decomposed into draw-explained vs
     gate-skill), EB-shrunk over prior runs → the start-phase drift; its dispersion optionally → `D`. The "start not
     explicit" scorecard gap, given a structure. Unmodelled today.
   - **Finishing-sectional v_c channel** — last-400 m velocity (pace+going adjusted) as its OWN high-precision channel
     in the fusion (the closing phase is the least tactically-contaminated ability readout — give each PHASE its own precision).
   - **Distance-law of variance** — validate the parameter-free `CV(T) ∝ 1/√L` prediction against per-distance ECE;
     if sprints are under-dispersed, add a distance-dependent `D(L)` floor (start+traffic inject distance-independent variance).
   - **Field-size weighting of the D knob** — prioritise every variance lever (figure-variance→`D_i`, gate-temperament,
     `D(L)`) on ST big fields (the order-statistic left-tail argument; MC already captures it — this is a prioritisation rule).
5. **Other new data:** weather (continuous soil-moisture going + wind; poll-and-log, no history); formline (the
   collateral graph could go deeper / directed). See HKJC_DATA.md.

**Refuted this session as a SIMPLE TERM (do NOT re-litigate as-is — but several are alive with new data/structure):**
carried-weight actwt, rest-days (days-since-race — blind to training; alive via #2), weight-trajectory bw_dev,
**physiology-from-race-history** (layoff/Banister/own-weight all collapse to ~0 partial corr — a race outcome is
itself a coarse training proxy; alive only via real training-load #2), **sustained-speed** trajectory trait (priced),
**rating velocity/acceleration** (restate recent form — only the peakgap LANDMARK converts), the **debut
RTG-suppression** (rating slope tiny, market already calibrated on debuts — the trial PRIOR is the right fix, not
reweighting). NOTE: scalar Elo / opponent-strength was "refuted" earlier but the RELATIONAL collateral-form CONVERTS
(+2.3pp HV) — "a dead first attempt ≠ a dead axis". The CONVERSION test (paired full-scale), not the t-stat, is the gate.

**DEFERRED (NOT refuted — real mechanism, wrong tool):** the draw / pace / competition INTERACTION axis. The
simple regression/mean-field treatments (contestedness×pos_fade, relational pace-pressure) don't convert
because the effect is geometry × CONTROL coupled (see §A above) — revisit as a joint theory build, not a
scalar probe. The baked draw terms are interim stopgaps.

### Parked — tested, do NOT re-litigate (real signal but low-leverage / redundant on WIN)
speed-figure form (rank-form's pace/class invariance wins); **trainer** strike-rate (+0.2pp, redundant
with jockey+rating); distance-dependent noise (flat best); the full power-balance EOM (~1.4 nats behind
kinematic; `sim="eom"` inert infra). **draw mean-field DONE** (no longer parked — DRAW_GAIN=0.003).

### Note on the residual→leverage rule
A high |t| residual ≠ a useful lever; what matters is impact on the WINNER's probability (which dominates
%market). BUT the draw lesson (t=−26.5, parked as "flat ridge" at 48%, then +1.9pp once re-added to the
*production* path) shows a parked verdict can go stale when the baseline/sim changes — the old draw test
was on the kinematic sim, but production switched to wald which had dropped draw entirely. Re-check parked
items against the CURRENT production path before trusting "low-leverage". `bodywt` is the next such re-check.

## Methodology caveats

- **PAIRED fixed-seed full-scale is the real gate (NEW, 2026-06-05).** The 150-race subsampled `quick_eval`
  WILDLY over-states a knob: `draw_fs_gain` read +1.0–1.6pp on quick_eval but only **+0.3pp HV / 0.0pp ST**
  at full scale — a ~4× overstatement. AND the full backtest's per-race MC seed is `hash(race_id)`, which is
  PYTHONHASHSEED-salted → ~±0.3-0.5pp run-to-run noise, so a single-run +0.3pp is indistinguishable from
  noise. FIX: run base and candidate with **`PYTHONHASHSEED=0` fixed** so they share identical per-race MC
  seeds — a paired comparison where the delta is purely the mechanism, not seed noise. Use quick_eval only to
  scout direction/sign; confirm every bake with the paired full-scale test (both venues + ECE).
- Going/class categorical residual test is **uninformative** under race-normalization (race-level
  constants vanish). To check condition calibration, use a calibration-by-condition test (ECE within
  each going/class bucket), not mean-resid-by-level.
- Form residual flipped to t=−9.8 (mildly OVER) at FORM_GAIN=4 — confirms we're at/just past optimum.
  True peak ~3.5; full-scale % of market still strongly favours 4.0 over 2.0, so 4.0 stays. A 3.5
  fine-tune is marginal — not worth a study before the draw/W′/mass fixes.

## Exhausted paths (don't re-litigate)

- Flat additive covariates are exhausted: form+jockey ≈ 41%, policy traits / weight add ≈ 0pp over
  form+jockey. Confirms the mechanistic-only path. (Tactics/pace are ~1–2pp — low ceiling.)
- Mechanism couplings are the only lever left that pays.
