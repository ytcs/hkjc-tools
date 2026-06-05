# A Theory of Horse Racing

The conceptual map of the problem: what actually determines a race result, organised
into dimensions, with the causal structure that ties them together and a note on where
our model currently sits against each. This is a **living outline** — to be expanded
into a full (PhD-thesis-grade) treatment later. For now it is the scaffold that keeps
modelling principled: **every factor we introduce must occupy a named place in this
structure and couple as a real mechanism, never an arbitrary regressor.**

> The governing principle (do not lose it): **fit the market first.** "% of market" is a
> goodness-of-fit metric, not an edge hunt. The model is regularised by physics + game
> theory. There is always a next mechanism — never invoke "private info" to stop.

---

## 1. The generative spine

A race result is an **order statistic of realised finishing times**. For each horse `i`:

```
T_i  =  ∫ ( effective path length ) / v_i(t)  dt   +   ε_i
winner = argmin_i T_i      (the win model is a random-utility / first-passage choice model)
```

Everything in the theory is just **what sets `v_i(t)`** — how much speed a horse can
produce, how that capacity is spent over the trip, how the arena and rivals perturb it,
and what is irreducibly random. The dimensions below are organised by **where in this
equation a factor enters**. Three are *internal* to each horse-in-its-race
(physiology, physics-plant, strategy-control); three are *external* boundary conditions
(environment, competition structure, market); one is the *coupling* between horses; and
one is the *irreducible floor* (stochasticity). Most real effects are **interactions
across dimensions** — the structure is not separable (§ on couplings).

---

## 2. The dimensions

### 2.1 Physiology — the engine ("how fast the horse *can* go today")
The per-horse latent capacity. Sub-parts:
- **Intrinsic ability** — the horse's quality. We proxy it with **form** (recency-
  weighted relative finish) + official rating + sectional `v_c`. **Open weakness: this
  is not opponent-adjusted** — beating a strong field should count more than a weak one
  (see § competition; the next structural lever).
- **Fitness / condition** — readiness on the day: days-since-last-run (freshness vs
  match-fitness), barrier-trial work, trackwork load, the form cycle.
- **Health / soundness** — injury, lameness, bleeding (EIPH); vet flags.
- **Age & development** — maturation curve, peak, decline.
- **Body composition** — body weight and **weight change** (bulked-up vs run-down).
- **Energetics (CP–W′)** — the mechanistic substrate: a sustainable aerobic critical
  velocity `v_c` and a finite anaerobic reserve `W′` depleted above `v_c`. These latents
  are *intrinsic and stable across races* — the "true" features a noisy finish position
  only glimpses.

### 2.2 Physics — the plant ("how effort converts to motion")
The shared equation of motion. `m·dv/dt = P/v − F_resist`, with
`F_resist = K_LOCO·m·g + ½ρ·cda·v²·(1−draft) + bend ground-loss`.
- **Kinematics / power balance** — mass & inertia, drag, the `v_c`/`W′` dynamics.
- **Geometry** — **bend ground-loss**: a wide-drawn horse runs a longer arc round the
  turns (the dominant draw mechanism; now in the production model).
- **Surface mechanics** — going → friction / energy return; AWT kickback; firmness.

### 2.3 Strategy — the control ("how ability is *spent*")
The policy layer: action = π(state).
- **Running style / pace** — front-runner vs closer; *energy deployment of `W′` over the
  trip* (go early vs kick late).
- **Jockey skill & policy** — kick timing/intensity, path-finding, rail-saving, when to
  commit.
- **Trainer strategy** — class placement, race selection, **intent** ("is the horse
  trying today / set up for another day?").
- **Game theory** — the field's tactics are an **equilibrium**: jockeys best-respond to
  each other; collective pace (many front-runners → fast contested pace → favours
  closers) is emergent, not a per-horse property.

### 2.4 Interaction — the coupling ("how horses affect each other")
The many-body terms — **the reason the agent-based sim exists**; a purely additive
statistical model *cannot* represent these.
- **Traffic / blocking / exclusion** — being boxed on the rail, checked, forced to wait
  for a gap (TASEP-like exclusion). "Horses are not collisionless."
- **Drafting / slipstream** — energy saved behind cover (active-matter coupling).
- **Draw × field interaction** — the same wide draw costs more in a crowded field (more
  bodies to be trapped by) than a sparse one — the *interaction* shadow of geometry.

### 2.5 Environment — the arena (exogenous boundary conditions)
Same for everyone, asymmetric in effect; modulates physiology *and* physics.
- **Going / track state** — already partly in physics; broader as a condition.
- **Wind** — a real aerodynamic term (head/tail/cross on the straights; reshapes the
  value of drafting).
- **Track bias** — the "golden rail", watering, where the live ground is on the day.
- **Course configuration** — A/B/C rail position literally changes geometry & bias.
- **Distance & class conditions** — the race's stated shape.

### 2.6 Competition structure — the handicap (the adversarial meta-game)
The dimension that makes this problem *hard*, and the most under-exploited.
- **Ability is relative.** The handicapper actively **compresses the field** — assigns
  weight to engineer a blanket finish. This is *why* win probabilities are close and why
  beating the market is hard.
- **Opponent-strength adjustment.** Raw form is the wrong object; the strength of the
  field a horse beat (or lost to) should reweight it — the win model *is itself a rating
  system at equilibrium* (an "interacting Plackett-Luce"; rating = the consistent
  solution). **This is the leading candidate for the next structural lever.**
- **Endogenous weight.** Carried weight reads as a quality signal (better horse → more
  weight), not merely a physics drag — the two channels must be disentangled.

### 2.7 Market / information — the epistemic overlay (the *target*)
Separate from the physical race. The betting market aggregates information — including
what we cannot directly observe:
- **Private / intent information** — gear changes, jockey bookings, trainer patterns,
  stable confidence, late money.
- **Microstructure** — pool sizes, late drift, CLV.
We **fit** this layer, so it is the yardstick; understanding *what the market knows that
the model doesn't* is how we decide whether a residual is a missing mechanism (keep
modelling) or genuinely unmodellable private info.

### 2.8 Stochasticity — the irreducible floor
What remains even with perfect knowledge.
- **The start** — gate behaviour, a slow break (a distinct phase, not steady-state).
- **In-running incidents** — stumble, interference, clipped heels, no clear run
  ("racing luck"). Often *the* explanation for a result; only partly mechanistic.
- **Random-utility noise** — modelled as the diffusion term (finishing time = first-
  passage of a drift-diffusion → Inverse-Gaussian; strictly positive, `Var ∝ 1/√L`).

---

## 3. The structure is not separable — couplings dominate

Almost every real effect is a **product across dimensions**, not a main effect. This is
why additive statistical models plateau and why the mechanistic sim is necessary:

| coupling | dimensions |
|---|---|
| going × horse aptitude | environment × physiology |
| distance × stamina (`W′`) | physics × physiology |
| pace × running style | strategy × strategy (the equilibrium) |
| draw × field size × pace | geometry × interaction × strategy |
| weight × class | competition × physics |
| surface × ability-transfer (turf↔AWT) | environment × physiology |

A factor that looks dead as a main effect can be alive as an interaction (and vice
versa). The diagnose→recalibrate loop on *main-effect* residuals therefore saturates;
the remaining gains live in the couplings (the sim's domain) and in measuring the
latents better (sectionals/trajectory) and *relatively* (the handicap lens).

---

## 4. What the additive-statistical approach already ruled out

History worth keeping (so it isn't re-litigated). Run through a leak-free walk-forward,
the **flat fundamentals** — draw, carried weight, official rating, and *crude* recent
form, voted/weighted as static covariates — **add little or nothing over the closing win
line**; the win market is efficient w.r.t. all of them. The lesson is not "these don't
matter" but "the market already prices the *main effects*." Gains have come only from:
- **measuring a latent better** (recency-weighted form ≫ career mean: the single biggest
  lever), and
- **adding a missing *mechanism* to the production model** (bend ground-loss re-added to
  the diffusion drift: the second biggest).
Both are mechanism/measurement upgrades, not re-weightings — which is exactly what the
theory predicts.

---

## 5. Where the model sits (scorecard)

| dimension | in the model | gap |
|---|---|---|
| physics — kinematics | `v_c`/`W′` drift-diffusion (Wald) ✓ | — |
| physics — geometry | bend ground-loss in the drift ✓ | — |
| physics — surface | AWT = more diffusive ✓ | turf↔AWT ability transfer |
| **physiology — ability** | recency form + sectional + rating | **not opponent-adjusted** |
| physiology — condition/health | ✗ | fitness cycle, days-since-run, trials, vet |
| physiology — body/mass | tested; ~null on win | real but small lever |
| strategy — pace/policy | infrastructure present, gain off | game-theoretic equilibrium |
| interaction | mean-field draw only; ABM reserved | the many-body coupling |
| **environment** | going partly | **wind / track bias / course bias unused** |
| **competition / handicap** | rating-conditioned mean | **relative / opponent structure** |
| market / information | the fit target | intent signals (gear/vet/trials) unused |
| stochasticity | diffusion noise ✓ | the start + incidents not explicit |

The two highest-leverage, currently-weakest cells: **opponent-adjusted ability**
(§2.6 — physiology measured through the competition lens) and **environment**
(§2.5 — exogenous signal we currently discard).

---

## 6. The research programme (sketch)

1. **Measure the latents better** — opponent-adjusted / relative ability (a coherent
   rating); sharper energetics from sectionals; then trajectory if it pays.
2. **Add missing mechanisms to the production (diffusion) model** — one term at a time,
   each with a physical derivation and an OOS %-market + ECE check.
3. **Let the agent-based sim carry the couplings** — traffic/blocking (exclusion
   processes), drafting (active matter), pace policy (mean-field games) — the
   "interacting Plackett-Luce" with PL as the free theory. Reserved for place/exotics
   and the interaction residual the mean-field surrogate cannot reach.
4. **Bring in exogenous environment** — wind, track bias, course geometry.

Parsimony gates all of it: a term must be **physics/ game-theory-derivable AND move the
needle**. First-principles grounding lets us rely less on statistics — a derived
mechanism is a priori correct — but it still has to earn its place.
