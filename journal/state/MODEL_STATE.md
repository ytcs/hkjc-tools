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
| `FORM_GAIN` | **4.0** | `hkjc.model.constants` | Was 2.0; residual diag showed form t=+32 under-weighted. Interior optimum ~3.5–4.0. ST 43.3%→48.2%. Amplify the GOOD signal, don't lower SPEED_NOISE. |
| `W_PRIME_GAIN` | **2.0** | `hkjc.model.constants` (applied in `fit.py`) | Was 1.0; residual diag showed W′ under-weighted t=+13.2. Sweep peaked 1.5–2.5 (3.0 over-amplifies, W′ noisy). Full-scale ST 47.7%→49.1%; w_prime residual halved (t +13.2→+6.4), v_c over-weight also fell (−8.9→−7.0). |
| `JOCKEY_GAIN` | 5.0 | `hkjc.model.constants` | Jockey trip/ability coupling (+7pp). |
| `SPEED_NOISE` | 0.12 | `hkjc.model.constants` | Race-day noise; calibrated to the ability spread. |
| `bend_curvature` (ST/HV) | 0.0037 / 0.0023 | `hkjc.model.calibration` | Geometric draw bias. **Confirmed near-optimal — NOT the lever for the draw residual (see below).** |

Current fit: **ST ≈ 49% of market** (49.1% at N=3000 after the W′ bake; was 47.7%). HV not yet re-confirmed at W_PRIME_GAIN=2.0 — **follow-up: run `VENUE=HV studies/backtest_st.py`.** mean |resid| 0.0426.

## ARCHITECTURE (audit finding — the model is a prototype, not gospel)

**The win model is a PROBIT on log effective-velocity.** Analytically the kinematic sim's winner =
`argmax(v_target·(1+ε))`, `ε~N(0, speed_noise)` — a probit choice model, temperature = speed_noise.
A direct probit (`sim="probit"` in `fit.py`, pure numpy) **matches the full traffic sim** for win:
48.1% full-scale vs the sim's 47.7–49%, at **~35× speed** (full-scale 1932 races in 28s, no geometry
recompile). The traffic machinery (blocking, bend ground-loss, draw) is confirmed-inert weight for the
WIN target — consistent with draw/mass/speed-figure all testing low-leverage. **Re-base the calibration
loop on the probit** (use `studies/quick_eval.py sim="probit"`, full test set affordable); reserve the
traffic sim for generative exotic pricing (place/trifecta, where ordering + traffic matter). Baked gains
all sit at interior optima under the probit too (FG4, WPG2, JG5, sn0.12 confirmed).

## NEXT LEVERS (the saturated ability signals aren't it — structure is)

1. **Noise-model structure — #1.** `speed_noise=0.12` is a single global constant but sets the ENTIRE
   probability concentration (hence all calibration/ECE). From first principles it should vary: shrink
   with distance (more sections average out), and depend on field size / going. Now a sub-second probit
   sweep. Pair with a **calibration-by-condition** diagnostic (ECE within field-size/distance/going
   buckets) — the residual regression is blind to race-level effects.
2. **Critical-power v_c estimator.** Backbone signal estimated crudely (`min` of last-two-section
   velocities, single race). The proper hyperbola `D = v_c·T + W′` across a horse's multi-distance runs
   gives a robust joint v_c/W′ + a per-horse distance curve. Foundational.
3. **Going×horse affinity** — real physical mechanism, residual-loop-blind (race-level). Needs the
   calibration-by-condition instrument, not the residual regression.
4. **Unused fundamentals (parsimony-gated):** barrier-trial times (direct fitness readout), gear changes,
   trainer form, sire distance/going aptitude. Add only if they move the needle.

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
