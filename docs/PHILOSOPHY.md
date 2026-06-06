# Philosophy & Method

The single source of truth for *how* we work on this model — the governing principles and the
calibration method. Mechanism detail lives in `docs/THEORY.md`; current numbers and the lever
queue live in `journal/state/MODEL_STATE.md`; findings live in `journal/`. This file is the *why*
and the *rules*. Other docs and the agent prompts should point here, not restate it.

---

## 1. Fit the market first

**`% of market` is a goodness-of-fit metric, not an edge hunt.**

```
% of market = (uniform_LL − model_LL) / (uniform_LL − market_LL)     measured leak-free, walk-forward
```

Closing the gap to the market *is* the job. We are not hunting mispricings. Whether any residual
gap is tradable is the **user's** call — never the model's, and never a reason to stop. The
edge/trading decision belongs to the user; the modelling work is pure goodness-of-fit.

**There is always a next mechanism — never invoke "private info / unknowable" to stop.** That
reasoning would have quit at ~10% skill; the model is now far past that. A residual is a missing
mechanism until proven otherwise. Understanding *what the market knows that the model doesn't* is
how we decide whether a residual is a lever to model or genuinely unmodellable private info — and
the bar for the latter is high.

## 2. Theory leads, statistics gate

**Statistical models cannot replace theoretical understanding.** Theory chooses *which* structure
to try; statistics only gate *whether that particular structure* converts.

- A feature failing to convert under a simple/additive coupling rejects **that structure**, not the
  mechanism. Re-derive the correct structure — the right latent, the right precision, the right
  interaction, the right knob — before declaring anything dead.
- **A dead first attempt ≠ a dead axis.** Scalar Elo failed; the *relational* collateral-form rating
  of the same competition axis converts strongly. Draw failed as a mean scalar; it converts as the
  variance knob. Never declare a mechanism or axis dead from a crude regression.
- A parked verdict can go **stale** when the baseline or sim changes. A "low-leverage" item measured
  against an old production path must be re-checked against the current one before it is trusted.

## 3. First-principle regularization (physics + game theory)

The model is regularized by **physics and game theory, not by feature count.** Every factor MUST
couple as a real, plausible mechanism — a term in the equation of motion (the plant) or in the
jockey/trainer policy. An arbitrary regressor that merely correlates is forbidden: it overfits and
breaks the discipline that carried this work from ~10% to its current skill.

- Before adding a signal, answer: **what mechanism does it act through?** If you can't name one,
  don't add it. Every coupling is a real mechanism, not a multiplier bolted onto the output.
- **It must enter in the right place.** The drift-diffusion exposes three independent knobs — the
  aerobic ceiling `v_c`, the anaerobic reserve `W′`, the diffusion `D` (variance). Using a signal is
  not enough; a feature is correctly structured only when it loads the knob its mechanism acts on
  (and, for fused observations/phases, at its **own precision** — a shared shrinkage is the wrong
  structure). See THEORY §3d/§3e.
- **Parsimony gates everything:** a term must be physics/game-theory-derivable **AND** move the
  needle (§5). A first-principles derivation makes a mechanism *a priori* correct, which lets us lean
  less on statistics — but it still has to earn its place on the conversion test.

## 4. The order statistic is the object

The winner is `argmin_i T_i` over realised finishing times — a first-passage / random-utility choice.
Two consequences that shape every structural decision:

- **Variance is as load-bearing as the mean.** A high-`D` horse upsets more than its mean drift
  implies; a metronomic one less. Modelling per-horse variance is not optional polish.
- **A purely additive statistical model cannot represent the many-body terms** (blocking, drafting,
  real-time tactics, jockey policy) — these are products across dimensions, not main effects. That is
  the agent-based sim's reason to exist: the "interacting Plackett-Luce" with PL as the free,
  non-interacting theory. The production Wald/PL surrogate *is* that free theory, which is exactly why
  bolt-on scalar ratings vanish into it and the live signal must be a structural interaction.

## 5. The conversion test is the gate — not the t-stat

Diagnosis points at a miss; **only the conversion test decides a bake.**

- The gate is **% of market (OOS) AND calibration (ECE)** both improving — confirmed at full scale
  with a **PAIRED fixed-seed backtest** (`PYTHONHASHSEED=0`, so base and candidate share per-race MC
  seeds and the delta is pure mechanism, not noise), on **both venues**.
- A high `|t|` residual is **not** a useful lever. What matters is impact on the *winner's*
  probability, which is what dominates % of market.
- The subsampled `quick_eval` **over-states a knob ~4×** (and run-to-run MC noise is ±0.3–0.5pp).
  Use `quick_eval` only to scout direction/sign; never bake on it.

## 6. Leak-free is non-negotiable

A backtest number is worthless if a feature saw the future.

- Every per-horse feature is **strictly prior**: the mean/rate over a horse's prior runs only
  (expanding mean shifted by one, or a strict as-of join with `allow_exact_matches=False`).
  Contemporaneous *boundary conditions* (draw, geometry, field composition, declared weight) are
  pre-race public and allowed.
- Enforce it mechanically: build features through `AsOfView` and run `assert_no_lookahead` on the
  fit/predict before trusting any number. Never weaken a leak-free path.

## 7. The calibration loop (this is the method)

```
diagnose → recalibrate → bake → re-diagnose
```

1. **Diagnose.** Run the model walk-forward; take the signed residual `resid = market_prob −
   model_prob` (both de-vigged, race-normalised). Regress `resid` on each observable mechanism with
   **race-clustered** standard errors. Sign + `|t|` say which mechanism the model mis-weights and by
   how much — the biggest is the next thing to ground in the physics.
2. **Recalibrate.** Fix it with a single mechanism-grounded constant (or the right structural change),
   not a guessed covariate.
3. **Bake.** The constant goes in `hkjc.model.constants` — the single source of baked calibration —
   only after passing the §5 conversion test.
4. **Re-diagnose.** Confirm OOS % of market *and* ECE both improved (not just in-sample), then find
   the next miss.

## 8. New data + new structure beats more mean-field

The mean-field "saturation" was a saturation of *additive terms on the existing race panel* — every
per-horse residual was collinear with the priced ability channel. The flat fundamentals (draw,
carried weight, official rating, crude form) are exhausted: the market already prices the main
effects. Gains come only from:

- **measuring a latent better** (e.g. recency-weighted form ≫ career mean), or
- **adding a missing mechanism** to the production model (e.g. bend ground-loss in the drift), or
- **bringing in new data the market prices but the model never saw** (barrier trials, training
  gallops), or **a structural/relational signal** the additive panel can't express (opponent-adjusted
  collateral form, a career-landmark process feature).

All are mechanism/measurement upgrades, never re-weightings — which is exactly what the theory predicts.

## 9. Separation of concerns (the three agents)

The work is split across three roles that share one journal and never reach into each other:

- **modelling** — owns the model (calibration, the sim, the constants). Makes the model fit reality.
  **Never places bets or designs strategies.**
- **strategist** — consumes the model's probabilities as a fixed input to design betting strategies.
  **Never edits the model.**
- **trading** — executes the chosen strategy live (paper). **Never invents strategies or retunes the
  model mid-meeting.**

If a downstream agent suspects a miscalibration, it **files a journal entry** addressed to modelling
(`audience=["modelling"]`) — it does not patch the model.

## 10. Communicate findings in the journal, not in code comments

Code comments say **what the code does** (and the non-obvious *why*: units, leak-free guarantees,
gotchas). **Findings** — conversion results, ST/HV deltas, ECE numbers, BAKE/DON'T-BAKE stories,
journal references — live in `journal/` and `journal/state/MODEL_STATE.md`, not inline. A constant
gets a one-line what/units; its story belongs in the journal. This keeps the code a clean reference
and the findings searchable and dated.

---

*Mechanism and the dimensional structure of the problem: `docs/THEORY.md`. Current fit, baked
constants, and the lever queue: `journal/state/MODEL_STATE.md`. Data surface: `docs/HKJC_DATA.md`.*
