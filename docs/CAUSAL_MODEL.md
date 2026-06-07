# The Causal Rebuild — the HKJC win model as a causal generative state-space

> **Status: DESIGN (authoritative). Decided by the project owner.** This is a ground-up
> *causal* re-architecture of the win model — and `hkjc.model` now IS this causal model: the
> acausal additive-regression model (`fit_predict_traffic`, the `features` meta-signal tables, the
> fitted `constants` gains, ST 65% / HV 60% of market) has been **DELETED**, not frozen. There is no
> deployable fallback running alongside; the deleted model's 65%/60% survives only as a **benchmark
> recorded in the journal** for the causal model to surpass. The goodness-of-fit (GoF) fall this
> design ACCEPTS is therefore the fall of the only model — paid deliberately for causal confidence.
>
> Companion docs: `docs/THEORY.md` (the generative physics — the validated core, reframed
> cause-not-effect), `docs/MODEL_DESIGN.md` (the package + build-wave items),
> `docs/RESEARCH_PROGRAM.md` (the staged RQ roadmap), `docs/PHILOSOPHY.md` (governing
> principles), `journal/state/MODEL_STATE.md` (production constants + lever queue).
>
> **Evidence key (cite journal entries).** `0037` factor audit: form/figure/collateral/sectional
> = ONE over-measured ability latent (PC1≈25%, 5/5 retrodiction) [VALIDATED]. `0039` ability-fusion
> collapse craters −35/−32pp ⇒ the additive gains are win-calibrated couplings, not honest
> measurements [VALIDATED]. `0040` profile-likelihood pilot: the dense path surfaces acceleration
> `a` (t=+9.98 ST sprint) that win cannot identify [VALIDATED]. `0044`/`0045` path-fit foundation +
> level-bias debiased [VALIDATED]. **`0047` the FIRST-PRINCIPLES EOM core replaces the heuristic
> generator: bang–CP–bang generates the phases, inferred CP≈54 W/kg / W′_0≈225 kJ are equine-plausible,
> the level bias and opening ramp are fixed BY MECHANISM (no fudge) [VALIDATED].** **`0048` the
> dedicated-surface identification battery + cross-surface consistency: CP-from-cruise vs CP-from-Hill
> DISAGREE +1.4 m/s → a structured falsification diagnosing the missing field coupling [VALIDATED].**
> **`0049` the mean-field pace/draft coupling closes 41–47% of that gap by aerodynamics with no fitted
> constant, and COLLATERAL re-emerges (corr −0.02→+0.33 ST) — the first validated re-emergence
> [VALIDATED].**

---

## 0. The principle — cause, not effect

The current model **confounds cause and effect**. Its strongest channels — FORM, JOCKEY,
TRAINER, COLLATERAL, PEAK-GAP, SPEED-FIGURE — are not mechanisms. They are **downstream
consequences** of the true causes:

- **FORM** is high *because* the horse has high intrinsic ability. Form is a noisy *past
  measurement* of ability, not a cause of winning.
- The **JOCKEY** signal exists *because* the jockey exerts better **control** over the path
  (a consequence of control quality, read through outcomes).
- **COLLATERAL** = ability *inferred via opponents*; **PEAK-GAP** = ability's *career
  trajectory*; **SPEED-FIGURE** = ability *read off a past kinematic*.

Feeding these as features makes the model **circular** — "winners win again." It fits well
because it is **tautological**: it confounds cause and effect. The empirical proof is `0037`
(all the ability-proxies are ONE latent over-measured by many instruments) plus `0039` (the
additive gains are not honest measurements — they are win-calibrated couplings amplifying one
shared factor ~7× past its honest slope).

**The rebuild.** With the dense PATH likelihood now available (`0040`/`0044`), **decouple ALL
meta-signals. Build the model on the TRUE MECHANISTIC CAUSES only. Infer the causes from the
paths. The meta-signals must RE-EMERGE NATURALLY as consequences of the correct structure** —
and that re-emergence is the validation. A short-term GoF fall is **accepted**; we trade it for
causal confidence and rebuild GoF back fundamentally.

This document specifies: the causal DAG (§1), the strip list and re-derivation (§2), the
crossed-effects inference with an honest identifiability map (§3), the re-emergence validation
suite (§4), the rebuild metric and staging (§5), and the precise first pure-mechanism build (§6).

---

## 1. The causal generative DAG

### 1.1 The latent CAUSES (exogenous + structural)

The causes are the *true generators*. Every one is a physical/control quantity, never a
regressor:

```
HORSE PHYSIOLOGY (intrinsic, slowly-varying — the per-horse random effect)
  v_c       aerobic critical velocity        — sustainable ceiling
  W'_0      anaerobic reserve                — finite tank above v_c
  a         acceleration / gate-ramp         — start-transient capacity (0040)
  rho       recovery rate                    — reserve replenishment kinetics
  D_i       per-horse consistency            — own finishing-time variance
  m         allometric mass                  — v_c ∝ m^(b-1)

JOCKEY CONTROL quality (the rider random effect)
  c_lat     lateral path-efficiency          — ground saved / position taken (LATERAL)
  c_pol     pace/effort policy               — when to commit reserve (intertemporal)

TRAINER CONDITIONING / day-fitness (the trainer × time random effect)
  f         prep / day-fitness               — how race-ready the horse is brought
  Δ         conditioning trajectory          — improving vs holding a yard's runners

ENVIRONMENT (exogenous, observed)
  G         geometry                         — venue × surface × course rail × bend map
  going     surface state                    — friction / energy return / kickback
  P         field pace                        — the realised aggregate pace (endogenous → MFG)
  Dr        draw                              — the lateral START boundary condition
```

### 1.2 The generative flow (two arrows out of every cause)

The defining feature of the causal model: **the same causes generate BOTH the path/outcome AND
the meta-signals.** The meta-signals are leaves, not parents.

```
                          ┌─────────────────────────────────────────────────────────┐
                          │                  LATENT CAUSES                            │
                          │  physiology(v_c,W',a,rho,D,m)  control(c)  prep(f)  env(G,going,Dr) │
                          └───────────────┬─────────────────────────────┬───────────┘
                                          │ (mechanism)                 │ (consequence)
                                          ▼                             ▼
                         ┌────────────────────────────┐    ┌────────────────────────────────┐
                         │  THE PATH  (this race)      │    │  PAST PATHS / PAST RACES        │
                         │  longitudinal v(t)  [NOW]   │    │  (the same causes, earlier)     │
                         │  + lateral y(t)     [CV]    │    └───────────────┬────────────────┘
                         │  ← pace game P (MFG)        │                    │ (measured, noisily)
                         └──────────────┬──────────────┘                    ▼
                                        │                        ┌──────────────────────────┐
                                        ▼                        │  THE META-SIGNALS         │
                         ┌────────────────────────────┐          │  form, speed-fig,         │
                         │  FINISHING ORDER            │          │  collateral, peak-gap,    │
                         │  = argmin_i T_i             │          │  jockey-SR, trainer-hot   │
                         │  (order statistic)          │          │  ← effects, NOT causes    │
                         └────────────────────────────┘          └──────────────────────────┘
```

Read the two arrows explicitly per family:

- **ability `v_c` → past paths → FORM/figure/collateral/peak-gap.** A high-`v_c` horse ran fast
  in the past ⇒ good form, high speed-figure, beat strong fields (collateral), sits near its peak
  (small peak-gap). These meta-signals are **noisy past-measurements of the SAME `v_c`** that
  drives this race's path. The current model feeds the measurement back in as if it were a cause.
- **control `c` → path positioning → JOCKEY strike-rate.** A skilled jockey saves ground / takes
  the right slot ⇒ the horse finishes ahead of its raw ability ⇒ the jockey accrues a high strike
  rate. Jockey-SR is the **effect of control quality on outcomes**, aggregated over the rider's book.
- **prep `f` → readiness → TRAINER-hot.** A yard bringing horses fit wins a streak ⇒ trainer-hot.

### 1.3 The generative mechanism (the path → order arrow — VALIDATED first-principles physics)

The path → order arrow is the **first-principles EOM** generative core (THEORY §III.2; `0047`),
under the diffusion first-passage envelope (`0029`/`0032`):

```
m·dv_i/dt = η·P_i(t)/v_i − ½ρ·C_dA·v_i² − C_rr·m·g − F_geom      # the power-balance EOM (the plant)
P_i(t) ∈ {P_max ramp, CP cruise, P_max kick}                    # the bang–CP–bang optimal-control policy
dx_i = v_i(t)·dt + √(2 D_i)·dW_i ,  T_i = inf{t : x_i ≥ L_eff,i} ,  winner = argmin_i T_i
+ mean-field pace/draft coupling on the OBSERVED field pace (0049) — perturbs DEPLOYMENT, not constants
```

Every parameter is a **physical constant with units and bounds** (`CP↔v_c, W′_0, P_max, m, C_dA`),
inferred by physically-constrained inversion of the laws — NOT a fitted gain (PHILOSOPHY §2,
no-epicycles). **What changes from the old model is the input**: the old drift was `v_c_eff =
v_c_base + Σ_k g_k·s_k`, a stack of *meta-signal* couplings; the causal rebuild sets `v_i` from the
**inferred causal latents only**, with the meta-signals removed entirely. Each constant is identified
on its **dedicated surface** and **cross-checked for consistency** (THEORY §IV; PHILOSOPHY §4) — the
CP cross-check failing in `0048` is what diagnosed the field coupling `0049` then confirmed.

### 1.4 Identifiability annotation on the DAG (NOW vs needs-CV)

Mark each arrow by whether it is identifiable from the LONGITUDINAL path available now
(`trajectory_dataset_full`: K-section velocity profile) vs the LATERAL (CV) surface (#34):

| Arrow | Identifiable from longitudinal NOW? | Needs lateral (CV)? |
|---|---|---|
| `v_c` → path (cruise/closing velocity level) | **YES** (`0040`, the settled level) | no |
| `W'_0` → path (late fade shape) | **YES** (partial; margins sharpen, Stage 2) | no |
| `a` → path (opening ramp) | **YES** (`0040`, t=+9.98) | no |
| `rho` → path (mid-dip recover) | **WEAK** (ρ≈0 on 94% of runs; assembly.py) | margins help |
| `D_i` → path (own velocity spread) | **YES** (figure-spread) | no |
| `m` → `v_c` (allometric) | **YES** (analytic, ST-only) | no |
| `c_pol` → path (pace/effort policy) | **PARTIAL** (confounded with `v_c`/`W'` longitudinally) | strongly needs CV |
| `c_lat` → path (ground saved, positioning) | **NO** (lateral is invisible to v(t)) | **YES, CV-only** |
| `f` → path (day-fitness level shift) | **PARTIAL** (crossed with horse; thin crossings) | no, but thin |
| pace `P` ← {c_pol} (MFG coupling) | **NO** (single-horse generator has no field) | partial |

**The honest headline: longitudinal paths identify the PHYSIOLOGY causes cleanly; control is
fundamentally LATERAL and is only weakly separable until CV (#34/Stage 3); the pace interaction is
not identifiable from a single-horse generator at all.** This is the spine of the staging (§5) and
the identifiability map (§3.3).

---

## 2. The STRIP LIST and the re-derivation

Every current meta-signal feature, classified by which cause it is an effect of, with the
re-emergence path. (Tables in `src/hkjc/model/features.py`; gains in `constants.py`.)

### 2.1 Ability-effects — STRIP, re-emerge from the ability latent (`v_c`, `W'_0`)

These are noisy *past-measurements* of one ability latent (`0037` PC1). They are the textbook
cause↔effect confound: high ability → these are high. **STRIP all from the model input.** They
re-emerge as the model's PREDICTION of the held-out signal from `v_c`/`W'_0` (§4).

The left column names the meta-signals as they existed in the now-DELETED acausal model (its
`features` tables and fitted `constants` gains — `FORM_GAIN`, `COLLAT_GAIN`, … — all removed); the
causal model neither contains nor re-creates them. They are listed here only to enumerate exactly
what is stripped and which latent each must RE-EMERGE from.

| Stripped meta-signal (deleted acausal artifact) | Effect of | Re-emerges from |
|---|---|---|
| `recency_form_table` / `form_table` (FORM_GAIN) | `v_c` (past path level) | ability latent → predicted form |
| `speed_figure_table` (SHRINK on `u_vc`) | `v_c` (past kinematic) | ability latent → predicted figure |
| `collateral_form_table[_v2]` (COLLAT_GAIN) | `v_c` (ability via opponents) | ability latent + field → predicted collateral |
| `rating_trajectory_table` peak-gap (PEAKGAP_GAIN) | `v_c` trajectory (career landmark) | ability **trajectory** (per-horse RE drift) → peak-gap |
| `finishing_sectional_vc_table` (FINSEC) | `v_c` (closing phase) | ability latent (closing) → predicted finsec |
| `trajectory_trait_table` fade (FADE_W_GAIN) | `W'_0` (reserve emptying late) | reserve latent → predicted fade |
| `barrier_trial_table` / `trial_vc_table` | `v_c` (a different instrument) | ability latent → predicted trial figure |
| `gallop_vc_table` (GALLOP_PI_RATIO) | `v_c` (a third instrument) | ability latent → predicted gallop figure |

**Re-derivation principle.** These instruments do not *measure different attributes*; they measure
the SAME `v_c`/`W'_0` the path identifies, with different precision/noise. So the causal model does
not feed them — it **infers `v_c`/`W'_0` from the path**, and form/figure/collateral are then
*predicted consequences* of that inferred ability. The `0037` audit is exactly this picture
(one over-measured latent); the rebuild stops over-measuring and infers the latent directly.

> **The `0039` tension, stated honestly (see §3.4).** `0039` proved that *honestly* fusing these
> instruments craters win%: the win-calibrated gains do independent **order-statistic tail work** the
> honest posterior discards. The causal model accepts this GoF fall by construction — it is the
> point. The bet is that the **path** (a denser likelihood than win) recovers the tail-relevant
> ability shape honestly, where the win-bit fusion could not. If a stripped instrument *fails* to
> re-emerge from the path-inferred ability, that is a precise pointer (§4) that the path is not yet
> identifying the full ability — not a license to feed the meta-signal back in.

### 2.2 Control-effects — RE-DERIVE causally (coarsely now, fully with lateral CV)

| Feature | Effect of | Re-derivation |
|---|---|---|
| `jockey_table` (JOCKEY_GAIN) | control `c` (path efficiency) → outcomes | a JOCKEY random effect on the path (crossed with horse, §3); fully causal only with CV lateral `c_lat` (Stage 3) |
| `early_position_table` / `style_table` (POS_FADE) | control policy `c_pol` (where the rider puts the horse) + ability | split: the *pace-game shadow* stays on `W'` (mean-field), the *positioning* becomes a control latent (Stage 3) |
| `gate_break_table` | NOT a separate cause — it is `early_pos` (`0037`: R²=0.998) | DROP entirely; the *kinematic* `a` (a capacity cause) replaces it |

**Coarse-now / full-later.** Now: a crossed JOCKEY random effect absorbs the part of the path a
rider systematically shifts (§3) — a coarse causal read of control. Full: with CV lateral paths
(#34), `c_lat` (ground-saved) and `c_pol` (commit timing) become *directly observed* causes, and
the jockey RE re-emerges as their consequence (§4).

### 2.3 Prep-effects — RE-DERIVE causally (coarsely now)

| Feature | Effect of | Re-derivation |
|---|---|---|
| `trainer_table` / `trainer_hot_table` (TRN_HOT_GAIN) | prep `f` / conditioning `Δ` → readiness → outcomes | a TRAINER × time random effect on the path (crossed with horse, §3); coarse now, no new data needed |

### 2.4 What is NOT stripped (genuine exogenous causes / environment)

Keep — these are causes, not effects: `rating` (the handicapper's Stackelberg assignment — an
exogenous condition compressing ability, kept as a population-prior anchor for debut/thin horses,
NOT as an ability proxy fed alongside the path), draw `Dr`, going, geometry `G`, allometric mass
`m`, field size `N`. The allometric law `v_c ∝ m^(b-1)` is a cause→cause arrow and stays.

> **Open tension T1 (for the owner).** `rating` is dual-natured: an *exogenous* handicap weight
> (cause) AND a *measurement* of past ability (effect). The design keeps it ONLY as the population
> prior mean for horses the path cannot yet speak to (debut / lightly-raced), never as a per-runner
> ability channel beside the path. Confirm this split is acceptable, or rule rating fully exogenous.

---

## 3. The inference — separating ability, control and prep from the path

### 3.1 The problem

`v_c`, jockey-control `c`, and trainer-prep `f` **all shape the same path**. A horse that finishes
fast may be able (`v_c`), well-ridden (`c`), or well-prepared (`f`). The single-horse profile
recovery (`estimate_run_profile`) attributes ALL of it to the horse. To decouple cause from effect
we must split the path's systematic part across the three agents that crossed to produce it.

### 3.2 The structure — a CROSS-CLASSIFIED (crossed random-effects) hierarchy over the path likelihood

A given run is produced by a `(horse, jockey, trainer)` triple on `(geometry, going)`. Across the
panel these crossings are dense enough to separate the three effects, exactly as crossed
random-effects models separate teacher/school/student effects. The generative model for the path's
systematic level (per phase, on the log-velocity scale the profile likelihood already uses):

```
log v_obs(run, section) = θ_horse[h]·B_phys(section)            # physiology: v_c, W', a, rho, D
                        + u_jockey[j]·B_ctrl(section)            # control: a phase-indexed rider shift
                        + w_trainer[h's trainer, time]·B_prep    # prep: a slow day-fitness level shift
                        + env(G, going) + ε                      # environment + log-velocity noise

θ_horse  ~ N(μ_pop(band,rating), Σ_horse)     # the EB population prior (generalize calibration.py)
u_jockey ~ N(0, σ²_j)                          # crossed rider random effect
w_trainer ~ N(0, σ²_t)  (random-walk in time)  # crossed trainer × time random effect
```

This **generalizes the existing EB hierarchy** (`calibration.fit_energetics_population`,
`RandomEffects`: band×rating fixed + horse random effect) from a single-grouping model to a
**crossed** model adding jockey and trainer groupings. The horse effect is the physiology cause;
the jockey effect is the control cause (coarse, longitudinal); the trainer-time effect is the prep
cause. **The meta-signals do not appear** — form/jockey-SR/trainer-hot are not inputs; the
*latent effects* are inferred from the path and the crossings.

### 3.3 The identifiability map — what is identifiable NOW vs deferred (HONEST)

This is the load-bearing honesty of the design. The three effects are **NOT equally identifiable
from the longitudinal path.**

**Identifiable NOW (longitudinal profile + crossings):**
- **Horse physiology `θ_horse`** — strong. The settled velocity level (`v_c`), opening ramp (`a`),
  late fade (`W'`), own spread (`D`) are all in the K-section profile (`0040`/`0044`). This is the
  bulk of the systematic path and the Stage-1 deliverable.
- **Trainer-prep `w_trainer`** — PARTIAL. A slow `trainer×time` level shift is separable from the
  fast horse effect **where the crossings are dense** (a yard's horses share `f`); thin where a
  trainer has few horses in a window. Coarse but buildable now, no new data.

**Weakly identifiable NOW, fully needs CV (#34):**
- **Jockey-control `u_jockey`** — **fundamentally LATERAL.** Control is *positioning* and
  *ground-saved* — a lateral (`y(t)`) phenomenon largely INVISIBLE to the longitudinal velocity
  profile. Longitudinally, a rider's systematic shift is **confounded with horse `v_c`** (good
  riders get good horses — the crossing helps but does not fully break it) and with the pace policy.
  So the jockey RE identified now is a **coarse upper bound on control** carrying ability leakage;
  the clean `c_lat`/`c_pol` split is **deferred to Stage 3 (CV lateral, #34)**. State this in every
  Stage-1/2 result: "jockey-control is only partially identified until CV."

**Not identifiable from a single-horse generator (deferred to the interaction layer):**
- **Pace `P` / the MFG coupling** — `0044` measured the fingerprint: a pace-dependent closing miss
  (+0.10 m/s in fast-pace ST races) the single-horse generator structurally cannot fit. The field
  coupling (§III.4) is deferred; until then the pace effect leaks into `W'`/`rho`.

**Thin-crossing caveat (the core identifiability risk).** Where `horse×jockey×trainer` crossings
are thin (a horse always ridden by one jockey for one trainer), the three effects are **aliased** —
the split is weak and the model must shrink hard to the population (or report the combined effect).
Quantify crossing density (bipartite connectivity of the horse–jockey and horse–trainer graphs) as
an entry diagnostic; report effective separability per cell.

### 3.4 Why the path, not win, makes this honest (the `0039` resolution)

`0039` showed the win bit cannot identify the ability coupling honestly (honest fusion craters).
The path is a **denser likelihood** (K sections + margins vs one win bit), so the crossed effects
are estimated against many degrees of freedom per run. The wager (to be tested, §4) is that the
path identifies the *tail-relevant* ability shape that win could not — so the re-emerged form
predicts the form→result relationship **without the win-calibrated gain**. If the path-inferred
ability under-predicts held-out winning (the `0039` under-discrimination re-appearing on the causal
model), that is the **measured GoF fall** we accept and then close by enriching the causal structure
(adding margins, the interaction layer), NOT by re-introducing the gain.

### 3.5 Estimation (build-wave note, not built here)

The crossed hierarchy is fit by EB / penalized likelihood over the path likelihood
(`hkjc.model.likelihood.PathLikelihood`), train-only per walk-forward fold, every per-horse /
per-jockey / per-trainer effect **strictly-prior** (trailing, `assert_no_lookahead` drift = 0).
A `LatentCauses` interface sketch (crossed `horse`/`jockey`/`trainer` effects) is added to
`hkjc.model` as a stub (§6); the inference itself is build-wave work. **All inference runs on the
unbiased generator** — now the **first-principles EOM core** (`0047`, `hkjc.model.dynamics`),
which removed the `0044` level bias by mechanism (a biased generator would bias every inferred
latent). The EOM core also supersedes the old per-horse single-NLS inference where it under-
identifies coupled constants: identify each constant on its **dedicated surface** with **cross-
surface consistency** (THEORY §IV; `0048`).

---

## 4. The RE-EMERGENCE validation suite — the falsifiable test

The validation is **not** win% (that comes last, §5). It is: **does the causal model, which never
saw a meta-signal, PREDICT that held-out meta-signal from its causal latent?** Re-emergence ⇒ the
mechanism is captured. Failure ⇒ a precise pointer to a missing mechanism.

> **First re-emergence VALIDATED (`0049`).** The field-coupled, de-confounded ability level predicts
> a horse's held-out ability — corr **−0.02 → +0.33 ST / +0.20 → +0.40 HV** once the field's pace/draft
> contribution is stripped — *without collateral-form as a feature*. The **COLLATERAL** mechanism
> (row below) is the first stripped meta-signal shown to re-emerge from physics. The rest of the suite
> (FORM, figure, peak-gap, jockey-SR, trainer-hot) is HYPOTHESIZED, to be run on the pure-mechanism
> build (§6). The headline payoff — that the full causal model BEATS production on the market — is
> NOT yet proven.

### 4.1 The general protocol

For each stripped meta-signal `s`:
1. Infer the causal latent from the PATH only (strictly-prior, leak-free), never from `s`.
2. PREDICT `s` from the latent via its consequence arrow (§1.2).
3. Score the prediction on held-out runs the latent never trained on.
4. Compare to (a) the trivial population-mean baseline and (b) the meta-signal's own
   autocorrelation (its trailing mean) as the *ceiling* a direct measurement achieves.

A signal **re-emerges** if (2) predicts held-out `s` materially above (a) and approaches (b).

### 4.2 Per-signal tests and metrics

| Stripped signal | Causal latent | Re-emergence test | Metric |
|---|---|---|---|
| **FORM** (recency rel-finish) | ability `v_c` (+ field) | predict a horse's held-out next relative-finish from path-inferred `v_c` and the field's `v_c` | rank corr ρ(predicted, observed form); R² vs trailing-form ceiling |
| **SPEED-FIGURE** | `v_c` | predict held-out speed-figure from `v_c` (+ going/dist par) | RMSE / R² of figure |
| **COLLATERAL** | `v_c` + opponents' `v_c` | reproduce the Massey opponent-adjusted rating from inferred field abilities | corr with the held-out collateral rating |
| **PEAK-GAP** | `v_c` **trajectory** (per-horse RE time-drift) | predict current-minus-peak from the inferred ability trajectory | corr; sign (declining horses < peak) |
| **JOCKEY-SR** | control `c` (jockey RE) | predict a jockey's held-out strike-rate from the inferred jockey effect | corr(jockey RE, held-out SR); rank agreement on top/bottom riders |
| **TRAINER-HOT** | prep `f` (trainer×time RE) | predict a yard's held-out hot-streak from the inferred prep trajectory | corr; lead-lag (does inferred `f` lead the streak?) |

### 4.3 The decisive joint test — reproduce form→result WITHOUT seeing form

The headline falsifiable test of the whole rebuild: **does the causal model reproduce the
empirical form→result relationship without form as an input?** Concretely — bin runners by their
*observed* recency-form; within each bin, does the causal model's win probability (built from path
ability, form stripped) track the empirical win rate? If the model never saw form yet its
predictions sort by form the way results do, the ability mechanism has **causally re-derived** what
form was a proxy for. Metric: the form-bin calibration curve of the causal model vs the empirical
win-rate-by-form curve; the slope and the monotonicity. (Run the same for jockey-SR-bins and
trainer-hot-bins.)

### 4.4 Failure = a pointer, not a defeat

A signal that fails to re-emerge names a **missing mechanism**:
- FORM fails ⇒ the path is not yet identifying the full ability (needs margins / closing precision).
- JOCKEY-SR fails ⇒ control is lateral and unobserved ⇒ confirms the CV (#34) dependency, not a model error.
- TRAINER-HOT fails ⇒ prep is not a path-level shift we can see ⇒ may need explicit fitness data.
- COLLATERAL re-emerges but FORM does not ⇒ the *relational* read carries ability the *self* read
  misses ⇒ the field-coupling (interaction layer) is the missing piece.

Each failure routes to a specific staging entry (§5). This is the design's engine of progress: the
re-emergence map tells us *which* mechanism to build next.

---

## 5. The rebuild metric and staging

### 5.1 The PRIMARY metric during rebuild (NOT win%)

```
PRIMARY = path-fit quality  +  re-emergence score
   path-fit: held-out per-section velocity RMSE vs the noise floor + the descriptive baseline (0044);
             calibration (PIT); residual structure by phase/pace.
   re-emergence: the §4 suite — does each stripped meta-signal re-emerge from its causal latent.
SECONDARY (diagnostic): the crossing-separability diagnostics (§3.3).
LAST: win% / %-of-market. Expected to FALL sharply on first strip; checked only to MEASURE the fall
      and then to confirm fundamental recovery as causal structure is added. The fall is the fall of
      the ONLY model (the acausal model is deleted) — paid deliberately for causal confidence; the
      deleted model's 65%/60% survives only as a journal-recorded benchmark to surpass.
```

This inverts the old acausal gate (which was %-of-market first). During the causal rebuild, a model
that fits paths well and whose meta-signals re-emerge is **winning even if its win% is below the
deleted model's benchmark** — because it has earned causal confidence that benchmark lacked.

### 5.2 Staging (entry / exit criteria)

**Stage 1 — Strip ability-proxies; build the physiological causal latents from the longitudinal
path; test FORM re-emergence.**
- *Entry:* MET — the unbiased generator is the EOM core (`0047`, level bias removed by mechanism;
  PIT re-centred); CP anchored on the Hill critical velocity (`0048`); the field coupling closes the
  fade-hump (`0049`). The strictly-prior physiology recovery (`CP↔v_c`, `W'`, `a`) is leak-free
  (`assert_no_lookahead` = 0).
- *Build:* the pure-mechanism model of §6 — physiology latents only, ZERO meta-signal features.
- *Exit (ALL):* (1) path-fit ≥ the descriptive baseline (the EOM core meets this with the field
  coupling, `0049`); (2) FORM re-emerges (§4.2 + §4.3 — path-inferred `v_c` predicts held-out form
  above the population baseline and the form→result curve is reproduced); (3) the win% fall is
  MEASURED and bounded (record it; it is allowed to be large); (4) at least figure + collateral also
  re-emerge (collateral DONE, `0049`; they are the same `v_c`).

**Stage 2 — Trainer-prep + the field-pace equilibrium.**
- *Entry:* Stage 1 exits; the crossed `trainer×time` effect is fit train-only with reported
  separability; the margins surface is joinable (`results_rich` finish times, leak-free).
- *Build:* add the trainer RE to the crossed hierarchy; add the margins term to the path likelihood
  (`likelihood.margin_term`); promote the `0049` *observed* field-pace coupling to the **equilibrium**
  pace (the MFG fixed point) for a leak-free predictive use (`0049` §5).
- *Exit:* (1) TRAINER-HOT re-emerges from the inferred prep; (2) margins identify `W'`/`rho` beyond
  the velocity profile; (3) the equilibrium pace coupling reproduces the validated fade-hump fix
  (`0049`) without the contemporaneous field pace; (4) win% recovers materially toward production
  (the first fundamental rebuild of GoF).

**Stage 3 — Jockey-control with CV lateral.**
- *Entry:* the #34 CV lateral pipeline clears cross-view re-identification (`0041` S1) and yields a
  leak-free trailing `ground_saved` / `c_lat` on a backfilled corpus.
- *Build:* add `c_lat`/`c_pol` to `LatentCauses.extra`; the lateral path likelihood term
  (`likelihood.lateral_term`); the geometry × control coupling (THEORY §III.4b).
- *Exit:* (1) JOCKEY-SR re-emerges from the inferred control latent (the test that fails-by-design
  at Stage 1, §4.4, now passes); (2) the control latent supersedes the interim draw shadows; (3) win%
  recovers to / past production — now on a causal footing.

### 5.3 The GoF-fall contract

The fall is expected and bounded by stage: Stage 1 large (ability gains were the bulk of
production's lift — recency-form alone is +7.8pp), Stage 2 partial recovery, Stage 3 full recovery
or better. Each stage records the fall and the recovery as a first-class metric. The fall is
**never** closed by re-introducing a stripped meta-signal — only by adding causal structure.

---

## 6. The concrete FIRST BUILD (the pure-mechanism model)

The minimal model on the EOM core. **BUILT and RUN** (journal 0050): the pure-mechanism θ assembly
(`hkjc.model.pure_mechanism`) wired through `build_fit_predict` + `make_pure_mechanism_pre_fit`, on
the full walk-forward, both venues, leak-free (`assert_no_lookahead` drift = 0).

> **HONEST RESULT (0050) — do not oversell.** The GoF fell TO THE FLOOR (ST 65.4%→−7.6%,
> HV 60.0%→+4.6% of market) and **FORM does NOT yet re-emerge at the win gate**: the physics-only win
> prob recovers only **7% / 4%** of the empirical form→result gradient. The path Hill `v_c` carries a
> WEAK held-out-form signal (+0.066 ST / +0.033 HV) — above the rating floor (+0.015 / −0.022) but far
> below the trailing-form ceiling (+0.382 / +0.302). The cause is identifiable and leak-free but **not
> yet predictive at the win gate**. Diagnosis: the ability READ is too coarse — it used the noisy
> across-distance Hill *slope* instead of the de-confounded settled-cruise *level* that `0049` showed
> re-emerges form at +0.33 / +0.40. **Named next step: swap the ability anchor slope→level**, then
> restore the between-horse ability spread the generative law lost. The build is honest, not (yet) a
> win.

**What it is.** A win model whose per-runner drift is set by the **path-inferred physiological
latents ONLY**, with EVERY meta-signal feature stripped from the input:

```
INPUT (per runner, strictly-prior, leak-free):
  theta_horse = (CP↔v_c, W'_0, P_max, a, m)  ← inferred by physically-constrained inversion of the
                                               EOM (0047, hkjc.model.dynamics) on the horse's EARLIER
                                               PATHS; CP anchored on the Hill critical velocity (0048,
                                               the robust + cross-consistent CP surface); pooled by the
                                               EB population prior (horse RE only for the first build —
                                               jockey/trainer crossings are Stage 2/3)
  env         = (G, going, Dr, N, observed field pace)  ← exogenous / boundary conditions
  rating                                     ← ONLY as the population-prior mean for thin/debut horses
STRIPPED (must NOT appear in the input):
  form, recency-form, speed-figure, collateral, peak-gap, finishing-sectional,
  barrier-trial, gallop, jockey, trainer, trainer-hot, early-pos, gate-break
GENERATOR:
  the first-principles EOM under bang–CP–bang (THEORY §III.2, 0047), drift from theta_horse only,
  + the mean-field pace/draft coupling on the observed field pace (THEORY §III.4a, 0049).
```

**How it is measured (the PRIMARY metric, §5.1 — NOT win-first):**
1. **Path-fit:** held-out per-section RMSE vs the noise floor and the descriptive baseline (`0044`/
   `0045` harness, `hkjc.model.pathfit`); must be ADEQUATE (≥ the descriptive baseline) on the EOM
   core — met by `0047` once the field coupling (`0049`) flattens the fade-hump.
2. **FORM re-emergence:** path-inferred `v_c` predicts held-out recency-form above the population
   baseline (§4.2) AND the form→result curve is reproduced without form (§4.3). This is the build's
   headline pass/fail.
3. **The GoF-fall magnitude:** run the paired fixed-seed win gate, record %-of-market vs production
   (expected to fall sharply). This is a MEASUREMENT, not a gate — the build is not rejected for a
   large fall; it is rejected only if path-fit is inadequate or FORM does not re-emerge.

**Where it lives.** In `src/hkjc/model/` (the canonical, only model namespace): a
`PureMechanismThetaBuilder` (`pure_mechanism.py` — the §6 input, meta-signals stripped), reusing the
corrected `energetics_dynamic` recovery and `calibration` EB prior; wired through the existing
`build_fit_predict` so `assert_no_lookahead` and the gate harness apply unchanged. (The old
production-reproduction θ-stamp, `assembly.production_theta_builder`, was removed with production in
the lean purge, journal 0051.) The re-emergence suite ran (journal 0050); its scratch study was
removed in the purge and will be rebuilt lean — scored vs market/uniform, no production arm.

**Acceptance for the first build:**
- `assert_no_lookahead` drift = 0 (leak-free).
- Path-fit ≥ descriptive baseline on the EOM core (`0047` met this with the field coupling `0049`).
- FORM re-emerges (else: the path is not identifying full ability — a Stage-1.5 pointer, §4.4).
- The GoF fall is recorded with its decomposition (which stripped gain carried how much win%).

**Interface stubs to add now (design surface only, no inference):**
- `hkjc/model/causes.py` — `LatentCauses` (a crossed-effects sketch: `horse`, `jockey`, `trainer`
  effect slots + `env`), the causal successor framing of `LatentState`. Stubs raise
  `NotImplementedError` for the crossed inference (Stage 2/3); the horse-only path is the first build.
- `likelihood.crossed_term` (stub) — the crossed-effects path likelihood (Stage 2).

---

## 7. Open design tensions (for the owner to resolve)

- **T1 — rating's dual nature (§2.4).** Keep `rating` as the debut/thin-horse population prior only,
  or rule it fully exogenous (a pure handicap weight)? It is both cause and effect.
- **T2 — the `0039` GoF fall magnitude (§3.4).** The win-calibrated ability gains were ~7× the honest
  slope and carried order-statistic tail work. Stage 1's win% fall could be very large (recency-form
  alone is +7.8pp; the whole ability stack is most of production's edge over the ~48% start). Confirm
  the appetite: a Stage-1 model could plausibly sit well below production on win% while passing every
  causal test. Accepted per the directive — flagging the likely magnitude.
- **T3 — thin crossings (§3.3).** Where `horse×jockey×trainer` is thin, ability/control/prep are
  aliased. Do we (a) shrink hard to population and report a combined effect, or (b) restrict the
  crossed inference to the dense sub-graph and treat thin cells as physiology-only? Recommend (b) for
  Stage 2, with the separability diagnostic gating which cells get a jockey/trainer effect.
- **T4 — pace interaction timing (§3.3).** RESOLVED in part: the **observed** field-pace coupling is
  built and validated (`0049`) — it closes the cruise-vs-Hill gap and the fade-hump as a *path-
  reconstruction* boundary condition. What remains is the **equilibrium** pace (the MFG fixed point)
  needed for a *predictive* WIN use without the contemporaneous field pace (`0049` §5) — a Stage-2/3
  structural step. The full *lateral* positional draft is CV-only (Stage 3).
- **T5 — generator dependency (§3.5). RESOLVED.** The first build was gated on an unbiased generator;
  `0047` delivered it as the **first-principles EOM core** (level bias removed by mechanism, not a
  nuisance fit), and `0049` closed the dominant remaining residual (the field coupling). The earlier
  "#50 corrected generator" prerequisite is met by the EOM core; inference now runs through it.
