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
| **`DRAW_GAIN`** | **0.003** | `hkjc.model.constants` | **Bend ground-loss re-added to the wald/pl drift: `L_eff = dist·(1+DRAW_GAIN·(draw−1))`. Production wald had NO draw term (it lived only in the kinematic sim, dropped at the wald switch) — re-diagnosis flagged draw as the #1 residual (t=−26.5). +1.9pp (56→58%); collapsed the residual to t=−1.5. Calibrated magnitude matches the geometric arc-length scale.** |
| `FORM_GAIN` | **5.0** | `hkjc.model.constants` | Form coupling; RECALIBRATED 4→5 at the recency baseline (recency form is sharper/higher-variance than career, flipped form back to under-weighted t=+11.4). Clean optimum at 5 (+0.9pp); 6–7 fall off fast. |
| `SHRINK_MULT` | **0.5** | `hkjc.model.constants` | Multiplier on the sectional v_c EB residual — it was OVER-trusted at 1.0 (peak 0.5 in both career+recency scans; dropping it, 0.0, costs only ~0.8pp — the sectional energetics barely earns its keep for win). |
| `W_PRIME_GAIN` | **1.0** | `hkjc.model.constants` | Stamina coupling; REVERTED 2→1 at the recency baseline. Recency form ABSORBED the reserve signal — re-diagnosis flipped W′ to over-weighted (t=−8.95). Win-prob impact flat over 0–2 (W′ is a margin effect); parsimony + residual sign both say revert to the un-amplified EB estimate. |
| `JOCKEY_GAIN` | 5.0 | `hkjc.model.constants` | Jockey trip/ability coupling. Held under recency. |
| `SPEED_NOISE` | 0.12 | `hkjc.model.constants` | Velocity-noise (= log-normal time). Diffusion `diff_coef≈170` ties it (`sim="wald"`). |
| `awt_diff_mult` | 2.0 | `fit.py` (sim="wald") | AWT is 2× as diffusive as turf (surface-dependent D) — fixes AWT over-confidence (12.7%→23.7%). |
| `bend_curvature` (ST/HV) | 0.0037 / 0.0023 | `hkjc.model.calibration` | Geometric draw bias. Confirmed near-optimal; draw is low-leverage (parked). |

Current fit: **ST ≈ 58% of market** (wald, full-scale; model ECE 0.0060 — better-calibrated than the
market's 0.0069). Session arc: 55.3% → 58.2% via the re-diagnosis loop (FG 4→5 +0.9pp; WPG 2→1 revert;
**DRAW_GAIN 0→0.003 +1.9pp — the big one**). HV not yet re-confirmed — **follow-up: `VENUE=HV
studies/backtest_st.py`.** Re-diagnosis at the 58% baseline: draw NEUTRALIZED (t=−26.5→−1.5), form/W′/v_c
all balanced; the new #1 residual is **bodywt t=+9.6 (UNDER-weighted)** — see NEXT LEVERS.

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

0. **`bodywt`/mass — the NEW #1 residual (t=+9.6, UNDER-weighted).** Market backs heavier horses more;
   the model has no mass effect (ALLOMETRIC_B=1). Was PARKED as "b=1 isometric optimum, coupling hurts" —
   but that was decided at the 48% baseline via the EOM steady-state law `v_c·(m/m̄)^(b−1)`. At 58% with
   the ability channel rebalanced it is the top residual again. **Revisit: is the right coupling a v_c law
   (mass→sustainable power) or a NOISE law (heavier = more reliable, smaller D)?** A direct mass→drift or
   mass→D term in the wald model (not the EOM path) is untried. Prime candidate.
1. **Draw interaction (ABM territory).** The mean-field bend ground-loss is DONE (DRAW_GAIN=0.003,
   +1.9pp, residual t=−26.5→−1.5). The residual has a field-size GRADIENT (small fields over-penalized,
   crowded under — the collision fingerprint: horses aren't collisionless). A mean-field field-size
   modulation (`draw_fs_gain`, wired but unbaked) recovers only ~0.1pp — diminishing returns. The full
   draw×crowding / boxed-in / draw×pace structure is the agent-based sim's job (place/exotics).
2. **`actwt` (carried weight) t=+2.9 UNDER** — handicap weight as a quality proxy (better horses carry
   more). Small but untested as its own term (distinct from bodywt). Cheap to probe.
3. **AWT ability-transfer (data-limited).** Noise half fixed (D_awt=2×D_turf, 12.7%→23.7%). Rest genuine:
   turf-trained ability barely ranks AWT winners (0.59 vs 0.70). Surface-specific form; AWT ~6% (starved).
4. **Going×horse affinity** — real, residual-loop-blind (race-level). Needs the calibration-by-condition
   instrument. **Recency×career form blend** — recency replaced career wholesale; a weak career 2nd term
   might add a little. **Unused fundamentals** (parsimony-gated): barrier-trial times, gear, sire aptitude.

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
