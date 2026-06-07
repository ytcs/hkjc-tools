# `hkjc.model` — the causal generative latent state-space win model (design + interface frame)

> **`hkjc.model` is the canonical, single, ONLY model namespace.** The acausal additive-regression
> win model (`fit_predict_traffic`, the `features` meta-signal tables, the fitted `constants` gains)
> has been DELETED; the experimental `hkjc.statespace` package was abandoned and FOLDED into
> `hkjc.model` (the module names that moved in: dynamics, identify, field, generative, pricer, gate,
> pipeline, assembly, coupling, latent, conditions, causes, likelihood, pathfit, pure_mechanism).
> There is no separate "statespace package built alongside production" — this **is** the model. This
> document is its code / interface design (formerly `docs/STATESPACE_DESIGN.md`).
>
> **CAUSAL REBUILD (owner-directed) — `docs/CAUSAL_MODEL.md` is the authoritative design.** The
> package is built on a **causal** structure:
> drive the model from the **mechanistic CAUSES only** (physiology / control / prep), inferred from
> the PATH, with ALL meta-signal features STRIPPED. The meta-signals (form, jockey-SR, collateral,
> peak-gap, speed-fig, trainer-hot) are **downstream CONSEQUENCES** of the causes — they must
> **re-emerge** as predictions, and that re-emergence (not %-of-market) is the near-term gate
> (`docs/CAUSAL_MODEL.md` §4). The primary metric becomes **path-fit + re-emergence**; win% is checked
> LAST and is expected to fall then recover fundamentally. The three-layer generative spine, the
> leak-free seam, and the gate harness below are all **reused**; the inputs and the metric change.
> The FIRST BUILD is the pure-mechanism model (`docs/CAUSAL_MODEL.md` §6): physiology latents inferred
> from the first-principles EOM core (`dynamics.py`, 0047; CP anchored on the Hill critical velocity,
> 0048), ZERO meta-signal features, measured on path-fit + FORM re-emergence + the GoF-fall magnitude.
> The Stage-1 `a`-gate work below (RQ-a1, 0042/0043) is the *coupling* precursor — kept as proven
> infrastructure; superseded as the architecture by the causal first build.

**Status:** THE FIRST-PRINCIPLES EOM CORE IS BUILT AND VALIDATED (journal 0047–0049). `dynamics.py`
solves the power-balance EOM under the bang–CP–bang optimal policy: it removes the path-fit level
bias AND the opening-ramp defect BY MECHANISM (no fudge constant), recovers equine-plausible physical
constants (CP≈54 W/kg, W′_0≈225 kJ, v_c≈17.2–17.4 m/s), and matches the heuristic on RMSE while being
pure physics (0047). `identify.py` pins each constant on its dedicated surface and cross-checks
consistency — CP-from-cruise vs CP-from-Hill DISAGREE +1.4 m/s, a structured falsification that
diagnosed the missing field coupling (0048). `field.py` then built the mean-field pace/draft coupling
on the observed field pace: it closes 41–47% of that gap by aerodynamics with NO fitted constant,
flattens the closing fade-hump, and the de-confounded ability level RE-EMERGES as held-out collateral
(corr −0.02→+0.33 ST) — the first validated re-emergence (0049). All leak-free
(`assert_eom_no_lookahead` drift=0), DON'T-BAKE diagnostic foundation.
(`tests/test_dynamics_eom.py` + `test_identify.py` + `test_field.py` green; lint clean.)

**Payoff test RUN (journal 0050).** The pure-mechanism first build was wired through `build_fit_predict`
+ `make_pure_mechanism_pre_fit` and run on the full walk-forward. The GoF FELL TO THE FLOOR
(ST 65.4%→−7.6%, HV 60.0%→+4.6% of market) and **FORM DOES NOT YET RE-EMERGE at the win gate** — the
physics-only win prob recovers only 7%/4% of the empirical form→result gradient (the path Hill v_c
beats the rating floor on held-out form, +0.066 vs +0.015 ST, but sits far below the +0.382 trailing-
form ceiling). The cause is identifiable and leak-free but not yet predictive. Diagnosis: the ability
READ is too coarse — it used the noisy across-distance Hill *slope* instead of the de-confounded
settled-cruise *level* that 0049 showed re-emerges form at +0.33/+0.40. **Named next step: swap the
ability anchor slope→level**, then restore the between-horse ability spread the generative law lost.

**Precursor (0042/0043, kept but SUPERSEDED as the architecture):** the gate-coupling Stage-1 `a`
work — the generative core reproduced the (now-deleted) acausal wald law byte-for-byte at a=OFF
(Acceptance Gate A: max|Δp|=0), and `a` was gate-tested on win + the pool book. VERDICT: `a` does NOT promote on
either (the win AND pool-book gates reject a path-identified latent as a coupling — the evidence the
metric had to invert). The leak-free seam and the gate harness are proven and kept.

Earlier (scaffold) status — typed interfaces + stubs + a runnable end-to-end skeleton — preserved
below for context.

This **is** the win model — the canonical `hkjc.model` package, not a separate experiment beside a
production model (the acausal model it replaced is deleted; the abandoned `hkjc.statespace` scaffold
was folded in here). It expresses the win model as the canonical three-layer generative latent
state-space so the implementation grows staged, mechanism by mechanism. It reuses the retained physics
substrate in the same package (`energetics`, `energetics_dynamic`, `calibration`, `calibration_bayes`,
`plackett_luce`, `harville_model`, `conditional_logit`, `baselines`) and `hkjc.backtesting` (the gate
harness).

## The canonical framework (the spine this honors — `docs/PHILOSOPHY.md`)

Causal, not correlational; first-principles physics, not regression (no epicycles); leak-free
non-negotiable; validated re-emergence-first, market-last. Three layers:

1. **LATENT CAUSES** θ_i = (CP↔v_c aerobic, W′_0 reserve, P_max, a acceleration/gate, m mass, …) —
   PHYSICAL CONSTANTS with units and bounds (PHILOSOPHY §2); hierarchical (population prior + per-horse
   strictly-prior posterior).
2. **GENERATIVE RACE MODEL** θ_i + conditions → the **first-principles EOM under bang–CP–bang control**
   (0047) + the **mean-field pace/draft coupling** (0049), under the diffusion first-passage envelope
   (Wald/IG, CV∝1/√L — 0029/0032) → full predicted PATH per runner → JOINT finishing-ORDER distribution
   → every pool price. No free coupling gain — deployment is the optimal-control solution.
3. **IDENTIFY + VALIDATE.** IDENTIFY by path-reconstruction on the DEDICATED surface with CROSS-SURFACE
   CONSISTENCY (0040/0048; PHILOSOPHY §3, §4); a disagreement is a falsification naming missing physics.
   VALIDATE by RE-EMERGENCE of the stripped meta-signals (PRIMARY; collateral DONE, 0049); the
   pool-book market %market+ECE is the **LAST** yardstick, measured to bound the accepted GoF fall,
   never optimized (PHILOSOPHY §6).

Discipline: this is the ONLY model — the deleted acausal model is a journal-recorded benchmark, not a
live fallback (PHILOSOPHY §9); leak-free mechanical; staged — **Stage 1** physiology causes from the
longitudinal path + FORM re-emergence; **Stage 2** trainer-prep + margins + the field-pace
equilibrium; **Stage 3** CV lateral → jockey-control.

## Module layout

```
src/hkjc/model/
  __init__.py        package surface (re-exports the core abstractions)
  dynamics.py        THE FIRST-PRINCIPLES EOM CORE (journal 0047) — power-balance EOM under bang–CP–bang  [CORE]
                     optimal control; CP↔v_c force-balance root; W′ energy law; numba RK4 integrator
  identify.py        DEDICATED-SURFACE identification battery (journal 0048) — per-constant estimators    [IDENTIFY]
                     (Hill CV, cruise root, P_max ceiling/transient, W′ spent/intercept) + cross-checks
  field.py           MEAN-FIELD pace/draft coupling (journal 0049) — draft (1−draft) aero + pace-matching [§III.4a]
                     on the OBSERVED field pace; reduces to isolated EOM when v_field≤v_c
  causes.py          LatentCauses — the CAUSAL successor to LatentState: crossed horse/jockey/   [CAUSAL §1]
                     trainer effect slots + env; the crossed-effects inference is STUBBED (Stage 2/3)
  pathfit.py         FOUNDATION path-fit eval (journal 0044/0045) — is the generator a good path model?
  latent.py          LatentState θ value object + PopulationPrior (hierarchical prior)   [Layer 1]
  conditions.py      RaceConditions — exogenous pre-race boundary conditions
  likelihood.py      PathLikelihood + SurfaceTerm (profile term reused; margin/lateral stubs) [IDENTIFY]
  generative.py      GenerativeRaceModel (first-passage law → sampled paths) + OrderDistribution [Layer 2]
  pricer.py          PoolPricer — order dist → win/place/forecast/quinella/trifecta
  coupling.py        CouplingCalibrator + CouplingGains — the (SUPERSEDED) gate-coupling seam, 0042/0043 [precursor]
  gate.py            the pool-book market-skill% gate (reuses hkjc.backtesting) — the LAST yardstick [GATE]
  pipeline.py        fit→predict spine + the leak-free contract + the runnable skeleton/smoke
tests/  test_model_skeleton.py · test_dynamics_eom.py · test_identify.py · test_field.py
docs/MODEL_DESIGN.md                 this document
```

> **NB the metric is INVERTED (PHILOSOPHY §6).** `gate.py` is the **LAST** yardstick (measure the GoF
> fall; confirm the causal model eventually surpasses the deleted model's recorded benchmark), **not**
> the coupling-calibrator. The `coupling.py` gate-calibration seam (Stage-1 `a`-gate, 0042/0043) is kept
> as proven infrastructure but **superseded as the architecture** — the EOM core (`dynamics.py`) has
> no free coupling gain; the deployment is the optimal-control solution.

Dependency direction (one-way, mirrors the repo's): the causal modules build on the retained physics
substrate *within* `hkjc.model` (`energetics`, `energetics_dynamic`, `calibration`, `calibration_bayes`)
and the provisional acausal couplings in `hkjc.model.wald_constants` (`DRAW_*`/`FIGVAR_*`/`GATE_BREAK_*`,
slated for mechanistic replacement), plus `hkjc.backtesting` (the gate harness). **Nothing the gate/judge
runs is reachable from the model's inputs** — the objective stays un-game-able (the same discipline as
`hkjc.backtesting`).

## Data flow

```
                 ┌─────────────────────────── IDENTIFY (honest, from the path) ──────────────────────────┐
 measurement     │  PathLikelihood.identify  ──▶  per-run θ̂  ──▶  PopulationPrior (train-only EB)         │
 surface  ──────▶│  (energetics_dynamic:                                │  pooling                        │
 (K-sec profile, │   profile_loglik / estimate_run_profile)             ▼                                 │
  +margins,      │                                       strictly-prior θ_i  (a horse's EARLIER runs only) │
  +lateral)      └──────────────────────────────────────────────────────┬───────────────────────────────┘
                                                                         │
 RaceConditions  ───────────────────────────────────────────────────────┤
 (trip, venue,                                                           ▼
  surface, draws) ───────────────▶  CouplingCalibrator.couple  ──▶  (μ_i, λ_i)     ← gate-CALIBRATED gains (#40)
                                    (θ → first-passage params)          │
                                                                        ▼
                                    GenerativeRaceModel.order_distribution
                                    (Wald first-passage sampler, race-id seeded)
                                                                        │
                                                                        ▼
                                              OrderDistribution  (n_samples × n joint orders)
                                                                        │
                                                ┌───────────────────────┴───────────────────────┐
                                                ▼                                                 ▼
                                       PoolPricer (win/place/                          ┌──────────────────────┐
                                       forecast/quinella/trifecta)                     │  GATE                 │
                                                │                                      │  market skill% + ECE  │
                                                └──────────────────────────────────────▶  over the pool book   │
                                                                                       │  (reuses backtesting):│
                                                                                       │  VALIDATE + CALIBRATE │
                                                                                       │  coupling (loops back) │
                                                                                       └──────────────────────┘
```

The **two objectives are different objects** by design: `PathLikelihood` (IDENTIFY) and `gate`
(VALIDATE+CALIBRATE coupling). θ is identified by the surface; the *coupling strength* of each latent
into the predictive law is set by the gate (#40). The loop: identify θ → couple with current gains →
generate → price → gate → re-calibrate gains → re-couple.

## Key interfaces (signatures)

```python
# latent.py — Layer 1
THETA_AXES = ("v_c", "w_prime", "rho", "a")            # extension point

@dataclass(frozen=True)
class LatentState:
    v_c: float; w_prime: float; rho: float = 0.0; a: float = 1.0
    extra: Mapping[str, float] = {}
    def as_array(self) -> np.ndarray: ...
    @classmethod
    def from_array(cls, vec, extra_keys=()) -> "LatentState": ...
    def with_(self, **changes) -> "LatentState": ...

@dataclass(frozen=True)
class PopulationPrior:                                  # thin adapter over calibration.EnergeticsPopulation
    energetics: EnergeticsPopulation; a_pop: float = 1.0; rho_pop: float = 0.0
    def prior_mean(self, band, rating=None) -> LatentState: ...                      # debut fallback
    def posterior(self, band, rating, *, n_prior, resid_v_c, resid_w_prime,
                  a_prior=None) -> LatentState: ...      # STRICTLY-PRIOR partial pooling (EB λ)

# conditions.py
@dataclass(frozen=True)
class RaceConditions:
    race_id: str; distance: int; venue: str; surface: str
    draws: np.ndarray; field_size: int; band: str = "mile"
    geom: object | None = None; section_dists: np.ndarray | None = None

# likelihood.py — IDENTIFY objective (path-reconstruction loss)
@dataclass(frozen=True)
class SurfaceTerm:  name: str; stage: int; fn: SurfaceTermFn; weight: float = 1.0
def profile_term(*, sigma=0.06, unit="dist", rho=0.0) -> SurfaceTerm   # REUSES profile_loglik (#41)
def margin_term()  -> SurfaceTerm    # Stage-2 STUB
def lateral_term() -> SurfaceTerm    # Stage-3 STUB

@dataclass
class PathLikelihood:
    terms: list[SurfaceTerm]
    @classmethod
    def stage1(cls, **kw) -> "PathLikelihood": ...      # profile-only
    def log_prob(self, theta, conditions, observation, *, max_stage=1) -> float: ...
    def identify(self, conditions, observation, *, recover=None, max_stage=1) -> LatentState: ...

# generative.py — Layer 2
@dataclass(frozen=True)
class OrderDistribution:
    orders: np.ndarray; n: int                          # (n_samples, n) joint finishing orders
    def win_probs(self) -> np.ndarray: ...
    def topk_probs(self, k) -> np.ndarray: ...

class GenerativeRaceModel:
    def __init__(self, *, n_samples=20000, diff_coef=170.0, awt_diff_mult=2.0): ...
    def order_distribution(self, thetas, conditions, *, mu=None, lam=None) -> OrderDistribution: ...
    def sample_paths(self, thetas, conditions): ...      # Stage-2/3 STUB (full x_long,y_lat,v,t)

# pricer.py
class PoolPricer:
    def __init__(self, order_dist: OrderDistribution): ...
    def win(self) -> np.ndarray; def place(self, k=3) -> np.ndarray
    def forecast(self) -> np.ndarray; def quinella(self) -> np.ndarray   # (n,n)
    def trifecta(self): ...                              # STUB (n,n,n + HKJC rules)

# coupling.py — the #40 seam (identified θ → predictive law, gate-calibrated strength)
@dataclass(frozen=True)
class CouplingGains:  a_start_gain: float = 0.0          # Stage-1 set; gate-tuned; 0 ⇒ a OFF baseline
class CouplingCalibrator:
    def __init__(self, gains: CouplingGains | None = None): ...
    def couple(self, thetas, conditions, model) -> tuple[mu, lam]: ...

# gate.py — VALIDATE + CALIBRATE coupling (reuses hkjc.backtesting; un-game-able judge)
def gate_score(preds, *, prob_col="p", market_col="market_win_prob", outcome_col="won") -> dict
def calibrate_coupling(data, folds, build_fit_predict, candidate_gains, ...) -> (best_gains, board)  # STUB

# pipeline.py — the spine + leak-free contract
def predict_race(thetas, conditions, *, calibrator=None, model=None) -> (OrderDistribution, win_probs)
def build_fit_predict(*, n_samples=4000, gains=None, theta_builder=None) -> FitPredict  # for the gate harness
def smoke_race(seed=0, n=8) -> dict                      # runnable end-to-end skeleton
```

## The leak-free contract (made hard by construction)

* **θ is strictly-prior.** A per-runner `LatentState` for prediction can only be produced by
  `PopulationPrior.posterior(..., n_prior, resid_*)`, where `n_prior`/`resid_*` are computed over a
  horse's **earlier runs only** — mirroring `calibration.horse_energetics_trailing`. A race's own
  surface never enters the θ that predicts it. Identifying a run's *own* θ (`PathLikelihood.identify`)
  is a separate, pooling-only step.
* **Population fit is train-only, per fold.** `PopulationPrior` is built from train rows and passed in;
  it is never refit on test. The pipeline owns the fold boundary.
* **The sampler is a pure function of (θ, conditions) + a race-id seed.** No outcome, no odds reach it.
* **The fit→predict adapter is gate-shaped.** `build_fit_predict` returns a `(train, test) -> preds`
  callable, so `assert_no_lookahead` and `walk_forward_evaluate` apply UNCHANGED. The skeleton already
  passes `assert_no_lookahead` (the default θ-builder reads pre-computed strictly-prior columns and
  never touches an outcome).

## Runnable-skeleton status

`tests/test_model_skeleton.py` — **6 tests pass**, lint clean:

* `smoke_race` constructs θ → conditions → `CouplingCalibrator` → `GenerativeRaceModel` →
  `OrderDistribution` → `PoolPricer` and returns a **valid win distribution** (sums to 1, all in
  (0,1)), top-3 place marginals (sum ≈ 3), and an (n,n) forecast (zero diagonal, sums to 1).
* `predict_race` win probs match `PoolPricer.win()`; the faster horse wins most.
* `a_start_gain=0` reproduces the bare uncoupled wald drift (the honest baseline); a non-zero
  gain moves it (coupling is live).
* The Stage-1 `profile_term` reuses `energetics_dynamic.profile_loglik` and returns a finite logL.
* `build_fit_predict` produces proper per-race win distributions and **passes
  `assert_no_lookahead`** (drift < 1e-12).

What is STUBBED (raises `NotImplementedError`, with a pointer here): `margin_term`, `lateral_term`,
`GenerativeRaceModel.sample_paths`, `PoolPricer.trifecta`, `gate.calibrate_coupling`.

## CAUSAL REBUILD — the FIRST BUILD (supersedes the gate-coupling Stage 1 as the architecture)

> Authoritative spec: `docs/CAUSAL_MODEL.md` §6. The pure-mechanism model — physiology latents
> inferred from the EOM core (`dynamics.py`, 0047; CP anchored on the Hill critical velocity, 0048),
> ZERO meta-signal features — measured on **path-fit + FORM re-emergence + the GoF-fall magnitude**
> (NOT win-first). Build it; do not gold-plate.

0. **Interface stubs (design surface, no inference).** Add `hkjc/model/causes.py` with `LatentCauses`
   (the crossed `horse` / `jockey` / `trainer` effect slots + `env`, the causal successor to
   `LatentState`); the crossed inference raises `NotImplementedError` (Stage 2/3). Add a
   `likelihood.crossed_term` stub. The horse-only path is the first build.
1. **`PureMechanismThetaBuilder` (the strip).** A θ-builder that produces per-runner physiology
   `(CP↔v_c, W'_0, P_max, a, m)` inferred from the horse's EARLIER PATHS via the EOM core
   (`dynamics.py` inversion, 0047; CP anchored on the Hill critical velocity, `identify.py`, 0048) +
   the `calibration` EB prior (horse RE only), with EVERY meta-signal feature STRIPPED from the input
   (contrast the now-deleted production θ-stamp, which carried the whole meta-signal stack via
   `v_c_eb`). `rating` enters ONLY as the debut/thin-horse prior mean. Wire through
   `build_fit_predict`; `assert_no_lookahead` = 0.
2. **The re-emergence suite (`studies/causal_reemergence.py`).** Implement the `docs/CAUSAL_MODEL.md`
   §4 tests: predict each held-out stripped meta-signal from its causal latent (FORM/figure/collateral
   from `v_c`; peak-gap from the ability trajectory), and the decisive §4.3 form→result-without-form
   curve. Metrics per §4.2. This is the build's PRIMARY pass/fail.
3. **The GoF-fall measurement.** Run the paired fixed-seed win gate on the pure-mechanism model;
   record %-of-market vs production and decompose which stripped gain carried how much win%. A
   MEASUREMENT, not a gate — the build is rejected only if path-fit is inadequate (vs the descriptive
   baseline on the EOM core) or FORM fails to re-emerge.

**Hard prerequisite: MET.** The unbiased generator is the first-principles EOM core (`dynamics.py`,
0047 — level bias removed by mechanism), with CP anchored on the Hill critical velocity (0048) and
the field/pace coupling closing the dominant residual (0049). Inference runs through the EOM core.

---

## (PRECURSOR) Stage 1 — acceleration → start-phase, gate-calibrated

> Kept as proven infrastructure (the leak-free seam, the byte-exact production port, the gate harness).
> Superseded as the architecture by the causal first build above; the `a`-coupling result (no-promote,
> 0042/0043) stands.

Prioritized. Stage 1's thesis: promote the #41 latent `a` (opening-split ramp) into the win model via a
gate-calibrated **start-phase** coupling, validated on the pool book. Everything is leak-free by the
contract above.

> **Items 1–3 are DONE (journal 0042, RQ-a1).** The production-reproduction bridge that realized
> them (`assembly.py`: `attach_trailing_a`, `make_production_pre_fit`, `production_theta_builder` —
> the byte-exact Acceptance Gate A against the old production model) was REMOVED in the lean purge
> (journal 0051) once production was deleted; the result stands. The byte-exact wald port survives
> in `generative.py` (`_drift_times`/`_effective_length`/`_diffusion_shape`),
> the `a_start_gain` + `a_d_gain` knobs in `coupling.py`, the exact %market+ECE `gate_score` +
> `calibrate_coupling` in `gate.py`, and the gate harness (the then-`statespace`-era `a`-gate study). a=OFF
> reproduces production byte-for-byte (Acceptance Gate A met). `a` did NOT pass the market gate (no
> promote) — see the Status block and journal 0042 for numbers + Stage-1.5 inheritance. Item 4 (fold
> `a`/ρ into the EB hierarchy proper) and item 5 (pool-book gate breadth) remain open.

1. **Real θ assembly (the data seam).** Implement a `theta_builder(train, race_group)` that produces
   per-runner **strictly-prior** θ over real panels: fit `PopulationPrior` train-only (wrap
   `calibration.fit_energetics_population`), compute each horse's earlier-run residuals + trailing `a`
   (the `energetics_dynamic.estimate_run_profile` recovery + the #41 trailing-mean recipe), and call
   `PopulationPrior.posterior(...)`. Mirror `studies/backtest_st.py::load()` for the frame and
   `calibration.horse_energetics_trailing` for the strictly-prior pooling. Wire it into
   `build_fit_predict`. *Acceptance:* `assert_no_lookahead` passes on real folds; at a=OFF the bare
   wald law reproduced the (now-deleted) acausal `fit_predict_traffic` to Monte-Carlo tolerance — the
   byte-exact equivalence baseline recorded in journal 0042, before that model was removed.

2. **`a` → start-phase coupling.** In `CouplingCalibrator.couple`, replace the
   skeleton's local `_START_LREF`/diffusion defaults with `hkjc.model.wald_constants`
   (`GATE_BREAK_LREF`, `DIFF_COEF`, `AWT_DIFF_MULT`, the draw/bend terms — the PROVISIONAL acausal
   couplings slated for mechanistic replacement) so the bare law is byte-for-byte the wald
   drift/diffusion. Keep `a_start_gain` as the single new knob: `μ_i *= (1 − a_start_gain ·
   start_frac · ā_centred_i)`, start_frac = min(1, LREF/L) (sprint-biased per THEORY §3e Phase 1).
   Consider a parallel `a`→**D** variance channel (a troubled break injects variance) as a second gain.

3. **Gate calibration of `a_start_gain` (the #40 step).** Implement `gate.calibrate_coupling`: a small
   grid/optimiser over `CouplingGains` that, for each candidate, builds the leak-free `fit_predict`,
   `walk_forward_evaluate`s it, and scores `gate_score` (market skill% + ECE) across the pool book.
   Replace the skeleton's crude `skill_vs_market` ratio with the exact %-of-market definition the
   production ST/HV backtest uses. *Acceptance:* the search runs leak-free over folds and returns a
   scoreboard; PROMOTE `a` iff the best non-zero gain beats `a_start_gain=0` on market-skill% (the
   honest Stage-0 baseline), broken out by distance band (expect the lift in sprints).

4. **`PopulationPrior` for `a`/`ρ`.** Fold `a` into the EB random-effects hierarchy proper (band×rating
   random effect) instead of the trailing-mean fallback, so a debutant's `a` is the rating-conditioned
   population mean and an established horse trusts its own. Decide ρ: keep population-fixed for Stage 1
   (the #38 engine), flag per-horse ρ as a Stage-3 CV target.

5. **Pool-book gate breadth (bridge to Stage 2).** DONE (journal 0043). `gate.pool_book_score` scores
   WIN/PLA/QIN/FCT/TCE/TRI from `PoolPricer.prob_*` (realized-combination probabilities, Laplace-
   smoothed) against the de-vigged dividends ((1−takeout)/decimal, takeout from `hkjc_facts`), exact
   per-pool %market on the realized combination. The pool-book gate study (the then-`statespace`-era
   harness) re-gated `a` on it (ST, 170 dividend races, paired fixed-seed). VERDICT: `a`→start-phase drift does NOT convert on
   the ordering pools either — it HURTS QIN/FCT/PLA/TRI monotonically, worst in the sprint band (sprint
   QIN −1.01%, FCT −0.86% at g=.075); only WIN improves. The reformulation's "ordering pools reward what
   win discards" wager is FALSIFIED for this latent/coupling — the ordering pools penalise the same
   minor-placing error win diluted. The gate INFRASTRUCTURE is kept (leak-free, the right Stage-2 judge).
   CAVEAT: the dividend prices only the WINNING combination, so %market is on the realized combination —
   a valid but PARTIAL benchmark. (Trifecta/tierce stay down-weighted: soft, higher-vig, noisier de-vig.)

### Later stages (named so the frame holds; not Stage-1 work)
* **Stage 2 — margins + pool book.** Implement `margin_term` (margin likelihood from the sampled Wald
  times), `GenerativeRaceModel.sample_paths` (per-section velocity/position so margins are scorable),
  `PoolPricer.trifecta` (+ HKJC dead-heat/refund rules from `hkjc.core.hkjc_facts`), and the full
  pool-book gate.
* **Stage 3 — CV lateral → control.** Add control/ground-saved latents to `LatentState.extra`, implement
  `lateral_term` over CV paths (THEORY §3c), and the geometry×control coupling.
```
