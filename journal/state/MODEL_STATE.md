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
| `SPEED_NOISE` | 0.12 | `hkjc.model.constants` | Velocity-noise (= log-normal time). Diffusion `diff_coef≈170` ties it (`sim="wald"`). |
| `awt_diff_mult` | 2.0 | `fit.py` (sim="wald") | AWT is 2× as diffusive as turf (surface-dependent D) — fixes AWT over-confidence (12.7%→23.7%). |
| `bend_curvature` (ST/HV) | 0.0037 / 0.0023 | `hkjc.model.calibration` | Geometric draw bias. Confirmed near-optimal; draw is low-leverage (parked). |

Current fit: **ST ≈ 60.0% / HV ≈ 54.1% of market** (wald, full-scale; HV +0.3pp from the new crowding×draw
interaction term). MILESTONE: ST crossed 60% (project
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

The cheap per-horse mean-field terms are largely banked or refuted. After a full three-front push
(2026-06-05: ABM revamp + data + debut), the picture sharpened decisively:

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

**B. NEW DATA is the genuine remaining lever — and barrier trials are #1 (build IN PROGRESS).** Both the
data-mining and debut work converge: **debutants need NEW signal, not a reweighting of existing features.**
The debut RTG-suppression quick-win was built and tested end-to-end and does NOT convert (see refuted list).
The data agent verified the sources LIVE (prototype `studies/newdata_probe.py`, samples in
`studies/newdata_samples/`): **barrier trials** (`btresult`, SSR HTML `?Date=DD/MM/YYYY`) are the clear #1 —
clean `horse_id` join, history queryable to 2018, inherently pre-race, and the trial field is DOMINATED by
debutants (11/19 in a sample were unraced foals). A trial gives a debutant a real `v_c`/W′ velocity prior
(trial_time + 3 sectionals + lbw) — the same energetics machinery. **A historical backfill (2022-2026) is
running** → `data_cache/barrier_trials.parquet`; next step is a leak-free trial-form feature + conversion
test (esp. on the debut subset). Secondary sources: **weather** (GraphQL `Weather` op — continuous
soil-moisture going proxy + wind; needs poll-and-log, no history) and **formline** (reconstruct the
collateral-form graph from our existing `localresults` for full depth). See HKJC_DATA.md.

**Refuted this session (do NOT re-litigate — residual signal that does NOT convert as a simple term):**
opponent-strength/Elo (collinear with rtg), carried-weight actwt, rest-days (days-since-race proxy),
weight-trajectory bw_dev (won-corr~0), HV form-by-distance (marginal), and the **debut RTG-suppression /
debut-jockey-gain** (premise real — fab. RTG corr −0.04 — but rating slope is tiny (~0.04 m/s over the debut
spread vs FORM's ~2 m/s), the model is already calibrated on debuts (mean p 0.065 vs actual 0.068), and ~35%
of "debutants" carry a REAL handicapper rating, so suppression discards signal; the debut-jockey gain HURT,
HV 33→28%). The CONVERSION test (quick_eval %market), not the residual t-stat, is the gate.

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
