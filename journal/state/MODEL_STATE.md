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

## OPEN RESIDUAL QUEUE (re-measured at FORM_GAIN=4, W_PRIME_GAIN=2.0)

1. **`bodywt` (mass) UNDER-weighted — t = +8.6 — #1 actionable.**
   Market prices body mass. The single-sourced EOM (`hkjc.model.eom`, used by `DynamicsRaceModel`)
   now carries mass (`v_c = p_c/(K_LOCO·m·g)`) — but PRODUCTION (`fit.py` → `traffic_jax.win_probs`,
   kinematic) does NOT. Next mechanism: wire race-day mass into the production win-prob path and
   recalibrate. A real coupling, not a flat additive bodywt term. Verify it's mass, not a class proxy.

2. **`v_c` OVER- / `w_prime` UNDER- (t −7.0 / +6.4) — residual ability rebalance.**
   The W′ bake halved this but didn't zero it. A further small W′ push or a slight FORM/v_c trim could
   help — but diminishing returns and W′ is noisy. Lower priority than mass.

### Falsified / low-leverage (do not re-litigate)
- **`draw` OVER-weighted t≈−17.5 is NOT bend_curvature-fixable.** Subsample sweep nominated 0.0030 but
  full-scale gave 47.9% vs 0.0037's 47.7% (within noise), ECE worse; %market is flat over bend in
  [0.003,0.005]. The draw residual is real but **low-leverage** (it shifts mid-pack mass, not the
  winner's prob, which dominates %market). A high |t| ≠ high leverage — prioritise by *winner* impact.
  Possible real fix later: a draw×distance/field interaction, not the flat bend constant.

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
