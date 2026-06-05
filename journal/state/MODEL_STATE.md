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
| `FORM_GAIN` | 4.0 | `hkjc.model.constants` | Form coupling; held at 4 under recency (sharp peak; over-gain over-concentrates fast). |
| `SHRINK_MULT` | **0.5** | `hkjc.model.constants` | Multiplier on the sectional v_c EB residual — it was OVER-trusted at 1.0 (peak 0.5 in both career+recency scans; dropping it, 0.0, costs only ~0.8pp — the sectional energetics barely earns its keep for win). |
| `W_PRIME_GAIN` | 2.0 | `hkjc.model.constants` | Stamina coupling; held (1–2 within noise under recency, which absorbed some W′). |
| `JOCKEY_GAIN` | 5.0 | `hkjc.model.constants` | Jockey trip/ability coupling. Held under recency. |
| `SPEED_NOISE` | 0.12 | `hkjc.model.constants` | Velocity-noise (= log-normal time). Diffusion `diff_coef≈170` ties it (`sim="wald"`). |
| `awt_diff_mult` | 2.0 | `fit.py` (sim="wald") | AWT is 2× as diffusive as turf (surface-dependent D) — fixes AWT over-confidence (12.7%→23.7%). |
| `bend_curvature` (ST/HV) | 0.0037 / 0.0023 | `hkjc.model.calibration` | Geometric draw bias. Confirmed near-optimal; draw is low-leverage (parked). |

Current fit: **ST ≈ 56% of market** (probit 56.1%, diffusion/wald 55.6%, full-scale; was ~48% before the recency-form switch). HV not yet re-confirmed — **follow-up: `VENUE=HV studies/backtest_st.py`.** A fresh residual re-diagnosis at the 56% baseline is the next housekeeping step (the old queue is dead).

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

## NEXT LEVERS

0. **Housekeeping:** re-diagnose the market residual at the 56% baseline (old queue dead); re-confirm
   HV (`VENUE=HV`); a finer recency half-life + "recency × career" blend check (recency replaced career
   wholesale — adding career back as a weak second term might add a little).
1. **AWT ability-transfer (data-limited).** The noise/over-confidence half is fixed (D_awt=2×D_turf,
   12.7%→23.7%). The rest is genuine: turf-trained ability barely ranks AWT winners (percentile 0.59 vs
   0.70). Surface-specific form/v_c would help but AWT is ~6% of races (data-starved). Hard, localized.
2. **Going×horse affinity** — real physical mechanism, residual-loop-blind (race-level). Common goings
   ~calibrated; rare soft/wet too sparse. Needs the calibration-by-condition instrument.
3. **Critical-power v_c estimator** — the hyperbola `D=v_c·T+W′`. BUT note the sectional v_c is now
   *over-trusted* (SHRINK_MULT=0.5) and barely earns its keep for win, so sharpening it is lower-value
   than it looked. Foundational for place/exotics more than win.
4. **Unused fundamentals (parsimony-gated):** barrier-trial times (direct fitness), gear changes, sire
   distance/going aptitude. Add only if they move the needle.

### Parked — tested, do NOT re-litigate (real signal but low-leverage / redundant on WIN)
draw (bend, t=−17.5); bodywt/mass (allometric b=1 isometric); speed-figure form (rank-form's pace/class
invariance wins); **trainer** strike-rate (+0.2pp, redundant with jockey+rating); distance-dependent
noise (flat best); the full power-balance EOM (~1.4 nats behind kinematic; `sim="eom"` inert infra).

### Parked — real residual but LOW-LEVERAGE (do not re-litigate)
A high |t| residual ≠ a useful lever; what matters is impact on the WINNER's probability (which dominates
%market). The high-leverage lever this session was W′ (+1.4pp). These are real but inert on win:
- **`draw`** (t≈−17.5): not bend_curvature-fixable (full-scale 47.9% vs 47.7%, flat ridge, ECE worse).
- **`bodywt`/mass** (t=+8.6, confirmed real, survives class/rating): steady-state allometric law
  `v_c·(m/m̄)^(b−1)` — optimum b=1 (isometric, no net effect); coupling it only hurts. Full power-balance
  EOM also underperforms the kinematic sim by ~1.4 nats (kept as inert `SIM="eom"` infra).
- **speed-figure form** (velocity vs par): worse than rank-form, which already normalises pace+class.

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
