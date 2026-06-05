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
| `FORM_GAIN` | **4.0** | `backtest_traffic_st.py` (HV imports it) | Was 2.0; residual diag showed form t=+32 under-weighted. Sweep found interior optimum ~3.5–4.0. ST 43.3%→48.2%, ECE 0.0081→0.0069. Lowering SPEED_NOISE instead HURTS — amplify the GOOD signal, don't globally sharpen. |
| `SPEED_NOISE` | 0.12 | same | Unchanged; race-day noise. |
| `bend_curvature` (ST) | 0.0037 | calibration grid | Geometric draw bias (mass×lateral offset). **Suspected stale** — see queue #1. |

Current fit: **ST ≈ 48% of market, HV ≈ 43%.** mean |resid| = 0.0430 (was 0.0457 pre-FORM_GAIN fix).

## OPEN RESIDUAL QUEUE (priority by |t|, measured at FORM_GAIN=4)

1. **`draw` OVER-weighted — t = −17.4 — #1 miss.**
   Unchanged by the form fix ⇒ a *separate* miscalibration of bend geometry. `bend_curvature=0.0037`
   is too strong (likely calibrated on the old 3-month 161-race subset, same staleness that made
   FORM_GAIN=2 wrong). **Cleanest next fix — a single-constant recalibration, structurally identical
   to FORM_GAIN.** RECOMMENDED NEXT STUDY.

2. **`w_prime` (anaerobic reserve) UNDER-weighted — t = +12.9 — #2.**
   Sharpening v_c (form) left the reserve channel relatively under-used; market values stamina more.
   Candidate: a W′ gain, or relax W′ empirical-Bayes shrinkage. Caution: W′ is the noisier estimate —
   amplify carefully, verify it doesn't just inflate variance.

3. **`bodywt` (mass) UNDER-weighted — t = +9.0 — #3.**
   Market prices body mass; `traffic_jax` has NO mass term (it's kinematic). `dynamics.py` (power-balance
   EOM) DOES: `v_c = p_c/(K_LOCO·m·g)`. The two sims have drifted — this residual IS that drift. Fixing
   needs a real mechanistic coupling (single-source the EOM so JAX sim and python sim don't diverge),
   not a flat additive bodywt term. Tangled with class/condition — verify it's mass, not a class proxy.

Lower |t| (monitor, not actionable): actwt +6.0, burden −5.6, style −5.4, rtg +2.8.

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
