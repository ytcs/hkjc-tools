# Model State — carry-forward (survives the refactor)

> Self-contained snapshot of where the ABM calibration stands, the methodology, and the
> open residual queue. This is the modelling agent's working memory. After the refactor it
> should become the seed entries of the modelling agent's journal channel.
> Last updated: 2026-06-04.

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
| `SPEED_NOISE` | 0.12 | `hkjc.model.constants` | Velocity-noise (= log-normal time). Diffusion `diff_coef≈170` ties it (`sim="wald"`). |
| `awt_diff_mult` | 2.0 | `fit.py` (sim="wald") | AWT is 2× as diffusive as turf (surface-dependent D) — fixes AWT over-confidence (12.7%→23.7%). |
| `bend_curvature` (ST/HV) | 0.0037 / 0.0023 | `hkjc.model.calibration` | Geometric draw bias. Confirmed near-optimal; draw is low-leverage (parked). |

Current fit: **ST ≈ 60.0% / HV ≈ 54% of market** (wald, full-scale). MILESTONE: ST crossed 60% (project
start ~48%). Levers since the recency-form era, in order: **per-venue DRAW/FORM** (HV 47.6→52.5%, +4.9pp
— the model had been ST-calibrated and applied globally; DRAW_GAIN ST 0.003/HV 0.008, FORM ST 5/HV 4);
**positional-fade** POS_FADE_GAIN=−1.0 (+0.5/0.6pp, the pace-equilibrium shadow); **allometric mass** on
wald (ST b=1.1, +0.5pp; the persistent #1 ST residual; HV b=1.0 — doesn't convert there); **AWT
per-surface draw** 0.008 (AWT 43→49%, the AWT market is ~33% softer); **JOCKEY_GAIN 5→6** (+0.4/0.3pp).
HV still trails ST ~6pp (diffuse gap, not one mechanism). The ST-1000m straight chute zeroes the draw term.

**SATURATION NOTE (important for the next agent):** the mean-field-surrogate levers are now saturating.
A growing list of real market residuals do NOT convert to %market when coupled (they're collinear with
priced signals or too noisy per-horse): opponent-strength/Elo, carried-weight (actwt), rest-days
(days-since-RACE), weight-trajectory (bw_dev, won-corr~0), the contestedness×pos_fade interaction.
**The conversion test (quick_eval), not the residual t-stat, is the real gate.** Beyond ~60% the gains
need either the INTERACTION layer (the ABM — theory predictions below) or NEW DATA (below), not more
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

## NEXT LEVERS (post-60% — the mean-field surrogate is saturating)

The cheap per-horse mean-field terms are largely banked or refuted. The remaining headroom is in two
buckets, both bigger investments — this is the strategic fork for the push toward 70%:

**A. The INTERACTION layer (the ABM / theory predictions — no new data).** THEORY §3b develops the race as
a differential / mean-field game and lists ranked, leak-free testable predictions. The mean-field shadow
(POS_FADE) pays; its *interaction* structure is untouched. NOTE: the simplest test — contestedness×pos_fade
(prediction #1) via an `early_pos>0.85` count — did NOT convert (the mean-field term absorbs it / the proxy
is crude). The richer interactions likely need the kinematic ABM, which BEATS wald at HV (+2-3pp) but loses
on ST — so the ABM is the natural HV win model and the substrate for the pace/draw×style couplings. Highest
ceiling; reserve for place/exotics too.

**B. NEW DATA (scraping builds — the data-mining round found these, all in HKJC_DATA.md).** Each opens a
data-starved dimension: **barrier-trial times** (`btresult`) → the only pre-race fitness signal for DEBUTANTS
(the debut market is genuinely soft — winner-rank 0.44 vs 0.68 — and our model is currently WORSE than the
market there, partly because RTG is *anti-predictive* for fill-null debut ratings); **training-load**
(`localtrackwork` timed gallops) → the right substrate for the rest/fitness-fatigue axis (days-since-RACE
is blind to daily training, which is why rest-days didn't convert); **weather** (NEW GraphQL op) → wind +
a *continuous* soil-moisture going proxy (the environment dimension); **formline** → the relational
opponent-linkage for the competition interaction (THEORY §3b pred #5).

**Quick wins still on the table (existing data, no big build):** a DEBUT-specific fix — suppress the
anti-predictive RTG for fill-null debut ratings + lean on jockey (the strongest debut signal, t=5.3);
needs a `prior_starts` feature. Bounded (~0.3-0.5pp global) but a real correctness fix.

**Refuted/parked this session (do NOT re-litigate — residual signal that does NOT convert):** opponent-
strength/Elo (collinear with rtg), carried-weight actwt, rest-days (days-since-race proxy), weight-trajectory
bw_dev (won-corr~0), contestedness×pos_fade, HV form-by-distance (marginal +0.2-0.4pp). The CONVERSION test
(quick_eval %market), not the residual t-stat, is the gate — many high-|t| residuals are already priced.

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
