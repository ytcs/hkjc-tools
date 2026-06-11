# Philosophy & Method

The single governing source for *how* we work on this model — the principles and the method. This
file is the *why* and the *rules*; every other doc and every agent prompt points here, not the
reverse. Mechanism detail lives in `docs/THEORY.md`; the causal DAG + strip + re-emergence in
`docs/CAUSAL_MODEL.md`; the staged roadmap in `docs/RESEARCH_PROGRAM.md`; the code/interface design
in `docs/MODEL_DESIGN.md`; current model numbers in `journal/state/MODEL_STATE.md`. The
**doc map** (read order + what each doc owns) is at the end of this file.

> **THE PIVOT (2026-06; owner-directed).** This project has pivoted from an **additive
> market-regression win model** to a **CAUSAL, FIRST-PRINCIPLES PHYSICS generative model**. The
> old model added ~10 signal channels onto a drift-diffusion law, each with a gain calibrated to
> the win outcome; it reached ST 65% / HV 60% of market. We now model the **physical CAUSES** of a
> race result from first principles — Newton's equation of motion, the critical-power energy
> budget, optimal-control deployment, aerodynamic field coupling — and require the strong "signals"
> (form, jockey, collateral, …) to RE-EMERGE as downstream *consequences* rather than be fed in as
> inputs. This file is the governing philosophy. The old additive model has been **DELETED** (§9) —
> the lean single-namespace causal model in `hkjc.model` is the only model; the journal is its
> rebuild trail.
>
> **Status discipline.** Throughout, claims are **[VALIDATED]** (confirmed on the data — the
> physics, the path-fit, the field coupling, the first re-emergences) or **[HYPOTHESIZED]** (the
> not-yet-proven payoff). The causal payoff itself — that stripping the meta-signals and re-deriving
> them from physics matches or beats the deleted model on the market — is **NOT yet proven**: the
> pure-mechanism build has been RUN (journal 0050) and its GoF fell to the floor (ST 65.4%→−7.6%,
> HV 60.0%→+4.6% of market), with **FORM NOT yet re-emerging at the win gate** (the physics-only win
> prob recovers only 7%/4% of the empirical form→result gradient). The cause is identifiable and
> leak-free but not yet predictive; the named next step is swapping the ability anchor from the noisy
> Hill *slope* to the de-confounded cruise *level* (which `0049` showed re-emerges form at
> +0.33/+0.40). **The philosophy is the commitment; the empirical proof is in progress, and is honest
> about how far it has to go.**

---

## The principles

### 1. Causal, not correlational — model the CAUSES, never the EFFECTS

Model the physical/physiological **mechanisms** that *generate* a race result. The strong "signals"
the old model leaned on — **form, jockey, trainer, collateral, peak-gap, speed-figure** — are NOT
causes. They are downstream **consequences** of the true causes (intrinsic ability `v_c`, jockey
control `c`, trainer prep `f`):

- **Form** is high *because* ability is high — form is a noisy *past measurement* of `v_c`, not a
  cause of winning.
- The **jockey** signal exists *because* the rider exerts better control over the path.
- **Collateral / peak-gap / speed-figure** are all ability read different ways (via opponents, via
  career trajectory, off a past kinematic).

Feeding these effects in as features confounds cause and effect and makes the model **circular** —
"winners win again." It fits well because it is **tautological**. The discipline is to *decouple*
them: build on the causes, infer the causes from the path, and require the effects to **RE-EMERGE**
from the causes as validation. *(Evidence: `0037` — the channels are ONE over-measured ability
EFFECT, PC1≈25%, 5/5 retrodiction [VALIDATED]; `0049` — the de-confounded ability level re-emerges
as held-out collateral, corr −0.02→+0.33 ST, without collateral as a feature [VALIDATED, first
re-emergence].)*

### 2. First-principles physics, not regression — the no-epicycles rule

The generative model is the **LAWS**, not a regression: Newton's equation of motion (the plant),
the critical-power / `W′` energy budget, the bang–CP–bang optimal-control deployment, the
aerodynamic field/draft coupling. Every parameter is a **PHYSICAL CONSTANT** with **units and
bounds** — `CP`, `W′_0`, `P_max`, mass `m`, `C_dA` — inferred by physically-constrained inversion
of the laws, never a fitted coefficient on the outcome.

**The Ptolemaic-epicycle test (the forbidden move).** A fitted affine map, a learned gain, a
spline, or any free parameter *without a physical referent* is **FORBIDDEN**. Adding a degree of
freedom to absorb a residual — with no mechanism behind it — is an epicycle: it improves the fit
locally and corrupts the science globally. If a residual will not yield to a law, that is a *missing
mechanism to find*, not a knob to add. *(Evidence: `0047` — the inferred `CP≈54 W/kg` matches
equine gallop physiology, `W′_0≈225 kJ`, `P_max/CP≈1.85–2.18`; the opening ramp R²≈0→0.77 by the
ODE transient with NO fitted `a`; the level bias falls to ≈0 with NO spend-share fudge — these are
*measurements*, not fits [VALIDATED].)*

### 3. Fit the dense surface, not the coarse outcome

The objective that *identifies* mechanism is **path-reconstruction** of the dense measurement
surface — the sectional velocity profile now; lateral path and finishing margins later — **NOT** the
win bit. **Win is too coarse a likelihood to identify mechanism**: one bit per race cannot resolve
how a mechanism should couple. The dense surface has many degrees of freedom per run, so mechanisms
that are collinear or sub-noise against the win outcome become *identified* against the surface.
*(Evidence: `0040` — the sectional profile fits 22–30% better held-out AND surfaces acceleration `a`
(t=+9.98), a latent win alone could not identify [VALIDATED]; `0042`/`0043` — win AND the pool book
both REJECT that strongly path-identified `a` as a *coupling*, confirming the win/pool gates cannot
do identification's job [VALIDATED].)*

### 4. Identify each constant on its dedicated surface — with cross-surface consistency

No single metric carries identification. Each physical constant is **pinned from the surface where
it dominates** (CP from the route cruise *and* from the distance–time / Hill slope; `P_max` from the
gate transient *and* the sprint ceiling; `W′_0` from the Hill intercept *and* the reserve spent
above CP), and it is **OVER-DETERMINED** — the same constant measured several independent ways must
**AGREE**. This is the Newtonian standard: `G` from the falling apple must equal `G` from the Moon's
orbit. **Cross-surface consistency is the physics validation.** A disagreement is a **FALSIFICATION
that points at missing physics** — *never* a number to fudge toward the middle. *(Evidence: `0048` —
CP-from-cruise sits +1.4 m/s ABOVE CP-from-Hill, both venues, a clean structured disagreement → it
diagnoses that the settled cruise is not aerobic-limited (a missing field coupling), NOT a
mis-set constant [VALIDATED]; `0049` — the field/draft coupling then closes 41–47% of that exact gap
by drafting aerodynamics with no fitted constant, confirming the diagnosis [VALIDATED].)*

### 4b. A null is not a conclusion until it clears a de-confounded ceiling check

A surface that shows no signal — or a mechanism that underperforms — has two very different causes:
the effect is genuinely absent (**true null**), or our model of it is **mis-specified** (wrong
observable, functional form, or coupling) so we miss an effect that *is* there. **Never conflate
them** — that is the cardinal diagnostic error. Before concluding "weak/absent", establish the
**effect ceiling**: the most signal *any* model can pull from the surface — a flexible, out-of-sample
predictor used **purely as a DIAGNOSTIC, never promoted into the model** (the same firewall the win
gate has). Then:

- **De-confound correctly** — measure the effect as `ceiling − a no-effect CONDITIONS ceiling`, NOT
  `ceiling − a naive baseline`. A naive baseline can average away the dominant confounds (distance,
  field size, class) and manufacture a *phantom* mis-specification verdict.
- **Sweep the observable, including DERIVATIVES** (gradient, variance, how a quantity *evolves*) —
  not just levels. The effect may live in the change, not the mean.
- **Read it:** `ceiling ≈ baseline ⇒ TRUE NULL` (a substantive finding — the signal is elsewhere);
  `ceiling ≫ our model ⇒ MIS-SPECIFICATION` (re-specify — the gap localizes the error). The
  `baseline→ceiling` gap is the real effect size; the `our-model→ceiling` gap is our specification
  error.

This is the discipline that separated "the path is ability-poor" from "the path is UNDER-extracted"
(`0054`/S1b), and "ST deployment is weak" from "ST deployment is real but lives in the pace GRADIENT,
and our level-only coupling was blind to it" (`0062`/S7c). It applies to **every** dimension's
null/partial result before we accept it. WIN/path stays the parked yardstick (§6); the ceiling check
runs on the dedicated surface.

### 5. Validate by re-emergence

The model must **regenerate the meta-signals it refuses to consume**. If it is built only on the
causes, then ability → form, ability-via-opponents → collateral, control → jockey strike-rate,
prep → trainer-hot must all fall out as *predicted consequences* of the inferred latents on held-out
data. **Re-emergence ⇒ the causal structure is right.** A failure to re-emerge is not a defeat — it
is a **precise pointer to a missing mechanism** (e.g. FORM fails ⇒ the path is not yet identifying
the full ability; JOCKEY-SR fails ⇒ control is lateral and needs CV — confirming a dependency, not a
bug). The re-emergence map is the engine that tells us which mechanism to build next.
*(Evidence: `0049` — collateral re-emerges (corr +0.33 ST / +0.40 HV) from the field-coupled
ability, first re-emergence validated; the full suite is in progress [VALIDATED in part /
HYPOTHESIZED for the rest]. See `docs/CAUSAL_MODEL.md` §4.)*

### 6. The metric hierarchy is INVERTED from the old model

The primary/last ordering has flipped:

```
PRIMARY   = path-fit quality  +  cross-surface consistency  +  re-emergence
SECONDARY = the crossing-separability / identifiability diagnostics
LAST      = win% / pool-book skill%  — PARKED during the rebuild, the FINAL yardstick, NEVER the target
```

The old loop — *“% of market is the goodness-of-fit gate, found via diagnose → add-feature → gate”* —
is **SUPERSEDED**. % of market is now the **final yardstick, not the identifier**. A model that fits
paths well, whose constants agree across surfaces, and whose meta-signals re-emerge is *winning*
even if its win% is below production — because it has earned causal confidence the production model
lacks. A sharp short-term GoF fall is **ACCEPTED** in exchange for causal confidence and a
fundamentally-rebuilt fit. The fall is **never** closed by re-introducing a stripped meta-signal —
only by adding causal structure. *(The GoF-fall contract is staged in `docs/CAUSAL_MODEL.md` §5.3.)*

### 7. Leak-free is non-negotiable

A backtest number is worthless if a feature saw the future — unchanged from the old discipline, and
inherited by every causal latent.

- Every per-horse latent is **strictly prior**: inferred from a horse's *earlier runs only*
  (expanding mean shifted by one, or a strict as-of join with `allow_exact_matches=False`).
  Contemporaneous *boundary conditions* — draw, geometry, going, field composition, declared weight,
  and the **observed field pace** (a legitimate per-section public observable, `0049`) — are pre-race
  / in-context public and allowed for *path reconstruction*. A *predictive* use of field pace needs
  the equilibrium pace from prior constants (`0049` §5).
- Enforce it mechanically: build through `AsOfView` and run `assert_no_lookahead` (drift = 0) before
  trusting any number. Never weaken a leak-free path.

### 7b. No stage trusts its upstream — robustness at every seam

In any multi-stage pipeline (the model's infer→couple→price chain, the CV perception stack, the
backtest assembly), **no downstream step may assume its upstream input is fully correct**. Upstream
stages fail — partially, plausibly, and silently — so a downstream pass that chains blindly converts
a local defect into a global one. The discipline at every seam:

- **Gate the evidence**: admit an upstream output as *evidence* only when it passes the stage's own
  independent check (e.g. a long fitted curve is not a rail until its own support backs it — a
  fabricated arc through screen text passed every extent gate until an occupancy gate asked the
  pixels, journal `0122`).
- **Contain failures**: design links/joins so one wrong input corrupts only its own segment, never
  the chain after it (tracklets that END at a discontinuity instead of guessing across it).
- **Aggregate, don't chain**: where a decision is global (an identity, a constant, a label), decide
  it ONCE from pooled evidence rather than propagating a per-step decision that can lock in an early
  error.

This is the same epistemic stance as §4 (over-determine, cross-check) applied to *engineering seams*
instead of physical constants. *(Owner-directed, banked 2026-06; evidence: the R02A far-rail failure
modes and the f690 fabricated text-arc, journal `0122`.)*

### 8. The market is the ultimate yardstick — and the edge/trading call is the user's

Preserved from the old philosophy, with the timing inverted (§6): the market remains the **final**
measure of goodness-of-fit. **Closing the gap to the market is the job — we fit the market, we do
not hunt mispricings.** Whether any residual gap is *tradable* is the **user's** call, never the
model's, and never a reason to stop modelling. There is always a next mechanism: a residual is a
missing mechanism until proven otherwise, and "private info / unknowable" is a high bar invoked
last, not a place to quit.

The work stays split across **three agents** that share one journal and never reach into each other
(`docs/CAUSAL_MODEL.md`, the agent prompts):

- **modelling** — owns the model (the physics, the causal latents, the calibration). Makes the model
  fit reality. **Never places bets or designs strategies.**
- **strategist** — consumes the model's probabilities as a fixed input to design betting strategies.
  **Never edits the model.**
- **trading** — executes the chosen strategy live (paper). **Never invents strategies or retunes the
  model mid-meeting.**

A downstream agent that suspects a miscalibration **files a journal entry** to modelling
(`audience=["modelling"]`); it does not patch the model. The edge/trading decision belongs to the
user; the modelling work is pure goodness-of-fit.

### 9. The acausal model was DELETED — the causal model is the only model

The additive-regression win model (the old `fit_predict_traffic` assembly, the `features` meta-signal
tables, the fitted `constants` gains — `FORM_GAIN`, `JOCKEY_GAIN`, …) has been **DELETED**, not
frozen. There is no deployable fallback sitting alongside the rebuild; `hkjc.model` is now the **lean,
single-namespace causal physics model and the only model**. Its old ST 65% / HV 60% of market survives
as a **benchmark recorded in the journal** (and `MODEL_STATE.md`), the number the causal model must
eventually surpass — the journal is the rebuild trail, not a live competitor. The model is **NEVER**
re-grown with regression terms or meta-signals (ANTI-REGRESSION below); every forward step is causal
structure. The GoF fall the rebuild accepts (§6) is therefore the fall of the **only** model — paid
deliberately for causal confidence — and is closed only by adding causal structure, never by reviving
a deleted gain.

---

## The method (the causal loop)

The old `diagnose → recalibrate → bake → re-diagnose` market loop is superseded for the rebuild.
The method now is:

```
infer the CAUSES from the dense path  →  cross-check each constant across its surfaces
   →  if a constant disagrees, find the MISSING PHYSICS (not a knob)  →  add the law
   →  validate by RE-EMERGENCE of the stripped meta-signals  →  (LAST, parked) measure win% / pool-book
```

1. **Infer.** Recover the physical constants (`CP/v_c`, `W′_0`, `P_max`, …) per run by
   physically-constrained inversion of the EOM on the dense surface, strictly-prior, leak-free.
2. **Cross-check (§4).** Measure each constant on every surface where it dominates; require
   agreement. A structured disagreement is a falsification — read it as a pointer.
3. **Add the named physics (§2).** Close a diagnosed gap with a *law* (the field/draft coupling for
   the cruise-vs-Hill gap), never an epicycle. Constants stay fixed; new physics perturbs *deployment*.
4. **Validate by re-emergence (§5).** Predict each held-out stripped meta-signal from its causal
   latent. Re-emergence promotes; failure routes to a specific next-mechanism in `RESEARCH_PROGRAM`.
5. **Measure the market LAST (§6).** Run the paired fixed-seed win / pool-book gate only to
   *measure* the GoF fall and, eventually, to confirm the causal model surpasses the deleted model's
   recorded benchmark (ST 65% / HV 60%). `PYTHONHASHSEED=0`, full-scale, both venues; `quick_eval`
   over-states ~4× so is for direction only.

---

## ANTI-REGRESSION — forbidden practices

To prevent drift back to the acausal model, the following are **forbidden**, each with its one-line
reason. Any of these in a build is a bug to revert, not a tradeoff to weigh.

1. **No regression of the outcome on features.** Regressing win (or any outcome) on covariates fits
   the *effect*, not the cause — it is the circular model we left (§1).
2. **No meta-signals as inputs.** Form, jockey-SR, trainer-hot, collateral, peak-gap, speed-figure,
   finishing-sectional, gate-break are EFFECTS; feeding them confounds cause and effect (§1). They
   are targets to re-emerge, never inputs.
3. **No fitted parameter without a physical referent (no epicycles).** Every parameter is a physical
   constant with units and bounds; a free coefficient/affine-map/learned-gain/spline that merely
   absorbs a residual is a Ptolemaic epicycle (§2).
4. **No fudging a cross-surface disagreement.** When two independent measurements of a constant
   disagree, that is a falsification pointing at missing physics — do not average them or tune one to
   match (§4).
5. **No optimizing win% (or pool-book%) directly.** Win is too coarse to identify mechanism and the
   market is the *last* yardstick, not the fitting target; optimizing it re-creates gate-calibrated
   couplings that masquerade as mechanism (§3, §6; `0039`/`0042`/`0043`).
6. **No closing the GoF fall with a stripped meta-signal.** The accepted short-term fall is closed
   only by adding causal structure (§6); re-adding a meta-signal undoes the rebuild.
7. **No reviving the deleted additive model.** The acausal `fit`/`features`/`constants` stack is
   deleted (§9); new mechanism is causal structure in `hkjc.model`, never a re-added regression term
   or meta-signal gain.
8. **No weakening a leak-free path.** Every latent is strictly-prior; `assert_no_lookahead` = 0
   before any number is trusted (§7).

---

## DOC MAP — what each doc is, and the read order

Read in this order; each doc owns one thing and cross-references the others rather than restating
them.

1. **`docs/PHILOSOPHY.md`** (this file) — **the governing principles + the method.** The *why* and
   the *rules*. Start here.
2. **`docs/THEORY.md`** — **the mechanism / physics + the architecture.** The equation of motion,
   the critical-power energetics, the bang–CP–bang optimal control, the first-passage generative
   spine, the phase structure, and the interaction (mean-field pace/draft) couplings. The *what
   generates a race result*, as a causal generative latent state-space.
3. **`docs/CAUSAL_MODEL.md`** — **the causal DAG + the strip list + the re-emergence suite.** The
   authoritative design of the rebuild: cause-vs-effect, what is stripped and how each meta-signal
   re-derives, the crossed-effects identifiability map, the inverted metric and staging.
4. **`docs/RESEARCH_PROGRAM.md`** — **the staged roadmap + the falsifiable research questions.** What
   to build, in what order, with the pass/fail on each RQ and its binding constraint.
5. **`docs/MODEL_DESIGN.md`** — **the code / interface design.** The `hkjc.model` package (the
   canonical, only model namespace): modules, signatures, data flow, the leak-free contract,
   build-wave items.

Reference data (slow-changing, not part of the philosophy chain): **`docs/HKJC_DATA.md`** (the data
surface) and **`docs/HKJC_SPECS.md`** (tracks, competition + betting rules). Current production
constants, the lever queue, and the refuted list: **`journal/state/MODEL_STATE.md`**. The durable
findings record: **`journal/`** (cite journal entries, e.g. `0047`–`0049`, not ephemeral queue
numbers).
