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

## 3b. The competition game, developed — the race as a differential / mean-field game

The competition axis is not a scalar add-on to ability; it is the **structure that
generates the running of the race**. The first attempt to capture it — an Elo /
relative-form rating — failed not because competition is irrelevant but because that
object is the *wrong* object: it is collinear (~0.70) with the official rating the
market already prices, and it discards the only thing competition actually does, which
is to make outcomes **relational**.

**The differential game.** Fix a trip of length `L`. Each horse `i` carries a finite
anaerobic reserve `W′_i` above a sustainable critical velocity `v_{c,i}`. A jockey
chooses an **effort/pace policy** `π_i(t)` (a velocity / reserve-depletion profile over
the trip) to minimise own finishing time `T_i = ∫ ds/v_i(t)`, subject to the **budget**
`∫_{v_i>v_c}(v_i−v_{c,i})dt ≤ W′_i/k` and the position dynamics `dx_i = v_i dt`. State =
`(x_i, W′_i^rem)` + the field's positions (traffic/draft); action = `π_i(t)` (effort and,
discretely, *when to kick*); payoff = `−T_i` (a tournament — only the *order* pays);
coupling = drag/draft and traffic make `v_i` depend on `{x_j(t)}`. The optimal `π_i` is a
**best response to `{π_j}`** → the field's tactics are a **Nash equilibrium**. With many
runners the rivals enter only through the aggregate **pace**, so the large-field limit is
a **mean-field game**: each horse best-responds to the field's pace distribution, which
is itself the aggregate of those responses — a fixed point in pace-policy space.

**Why competition is relational, not scalar.** A horse's effective finishing time
depends on its **style/ability matchup against this field's composition**, not on a
context-free number. A front-runner among closers steals an uncontested lead, dictates a
soft pace, arrives with reserve unspent; the *same* front-runner among other
front-runners is forced into a contested early duel that burns `W′` against the air with
no draft, and fades. Identical horse, different equilibrium. A scalar rating cannot
represent a quantity that is a *function of the opponent set*. The correct generative
object is an **interacting Plackett-Luce**: PL (independent finishing times → argmin) is
the **free / non-interacting theory**; the competition couplings bend PL's per-horse
utilities by *who else is in the race*. Our production Wald/PL surrogate **is** that free
theory — which is exactly why bolt-on scalar ratings vanish into it, and why the live
signal must be a *composition interaction*, not a main effect.

**Two levels: the handicapper (Stackelberg) and the pace game.** Running styles are
**public** (read off past running lines — our `early_pos` feature is the EB-shrunk mean
of normalised early field position). Jockeys observe them and set up the pace
accordingly — best-response on *observable* information, so modellable from form alone.
Above the tactical game sits the **handicapper as a Stackelberg leader**: anticipating
the equilibrium, they assign weight to **compress ability** and engineer a blanket
finish. So competition enters twice — the handicapper compresses the **mean** (already
priced, hence tight win probabilities and an efficient rating channel), but the pace
equilibrium reshuffles the **conditional** outcome by style-composition (*not* fully
priced — our opening).

**The positional fade is the mean-field shadow of this equilibrium.** The production
drift-diffusion is the *non-interacting* theory: context-free drifts, collisionless,
draftless. It therefore over-rates habitual front-runners (never charged for the
contested-pace reserve burn) and under-rates closers (never credited the reserve they
conserve). The confirmed `POS_FADE_GAIN = −1.0` term — front horses win ~1pp *less* than
the physics predicts, closers ~1pp *more*, residual `t ≈ −7` at **both** venues,
surviving form/`v_c`/`W′` controls — is exactly this bias, measured. The shared
sign-and-magnitude across venues is the signature of a **mean-field constant**: the
field-averaged shadow of the pace equilibrium projected onto one per-horse scalar. We
have captured only its **zeroth moment** (the field-mean style offset); the equilibrium's
*interaction* structure — how the fade depends on *who else* is in the race — is
untouched. That is the next layer, and it converts the theory into testable predictions.

**Testable predictions (ranked; all inputs pre-race & public ⇒ leak-free):**
1. **Front-runner penalty scales with the count of rival front-runners** (pace
   contestedness `n_FR`): the fade *grows* with `n_FR`, *vanishes/reverses* at `n_FR=1`.
   Implement as `POS_FADE × contestedness` on the already-paying scalar — lowest-risk,
   highest-information. **Top priority.**
2. **Lone-leader pace-control bonus** (`n_FR=1`): a sign *flip* of the fade. The robust
   form is the *interaction* with `n_FR` (1+2 tested jointly), not the threshold-fragile
   standalone lone-leader cell that was already rejected.
3. **Closer-favoring in contested-pace fields**: closers' bonus scales with `n_FR` — a
   field-level reshuffle the non-interacting model cannot produce.
4. **Field pace-shape modulator**: summarise the `early_pos` distribution into one
   pace-pressure scalar per race modulating the fade gain (generalises 1–3).
5. **Pairwise matchup / collateral-form terms** (interacting-PL): use HKJC `formline`
   (who-met-whom + rating/weight deltas) as *relational* utility bumps — the object the
   scalar Elo groped for, but as an interaction (should survive the collinearity that
   killed Elo). Highest ceiling, heaviest build — after 1–4 confirm the mechanism.
6. **Draw × pace** (inside front-runner controls rail+tempo; wide one burns reserve to
   cross): a genuine many-body term — the agent-based sim's domain, lowest near-term.

Predictions 1–3 are one mechanism (the equilibrium's first moment: fade-scaling-with-
contestedness) from three angles, all a single `contestedness`-modulated extension of
`POS_FADE_GAIN` — cheap, leak-free, on-data. Land those before the heavier `formline`
relational build (5).

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

---

## Appendix: lessons from the deprecated agent-based sims

The project began as a literal **agent-based model** — interacting horses simulated step-by-step
on the real geometry. Two implementations existed (`hkjc.model.traffic` / `traffic_jax`, the
kinematic car-following sim; `hkjc.model.eom` / `dynamics`, the power-balance equation of motion).
**Both were deprecated from the codebase in 2026-06** once the production win model (the `wald`
drift-diffusion first-passage law) plus per-venue mean-field calibration decisively beat them. The
physics they encoded is preserved here so the lessons are not lost with the code.

**What they modelled (still the correct picture of the mechanism):**
- **Power-balance EOM (the "plant").** Each horse is a body of mass `m` driven by metabolic power
  `P(t)` against real resistances, integrated by Newton's law: `m·dv/dt = P/v − F_resist`, with
  `F_resist = K_LOCO·m·g` (locomotion) `+ ½ρ·cda·v²·(1−draft)` (aero drag, cut by slipstreaming a
  leader) `+` bend ground-loss. A finite anaerobic reserve `W′` (joules) depletes above the aerobic
  ceiling `P_c` and forces the late-race fade. This is *why* `v_c = P_c/(K_LOCO·m·g)` falls with
  mass — and it yields the **allometric mass law** `v_c ∝ m^(b−1)` (from `p_c = κ·m^b`) that DID
  survive: it is now applied analytically on the `wald` path (the one converting idea the EOM gave).
- **Kinematic interaction sim.** Car-following (a horse cannot pass through one ahead in its lateral
  band — it is speed-capped/blocked), gap-acceptance lane changes, and bend ground-loss from the
  draw-derived lateral offset. Draw bias and trip-trouble **emerge** from geometry + traffic rather
  than being fitted.

**Why the *code* was deprecated (an engineering verdict, NOT a verdict on the axis):**
- As a first-attempt *implementation*, the sims were low-leverage for WIN. A non-interacting
  first-passage model (`wald`) plus per-venue mean-field terms matches or beats them at ~35× the
  speed; at the calibrated baseline the kinematic sim **lost to `wald` at HV by −4.1pp** (its
  uncalibrated emergent draw is worse than the tuned per-venue `DRAW_GAIN`). So the hand-tuned
  car-following loop is retired — but the *physics it pointed at remains the right picture*, and it
  did reproduce the qualitative draw/trip characteristics that the mean-field terms only approximate.

**Draw and the interaction axis are DEFERRED, not rejected — the key open theory.** Draw is not a
static geometric covariate: its effect is **deeply coupled to the control dimension** (the jockey's
policy — how the gate is used to take up position, when to commit, how the trip is navigated). A
regression / mean-field term (`DRAW_GAIN`, `DRAW_FS_GAIN`) can only capture the *first-order geometric
shadow* of a fundamentally **geometry × control** phenomenon, which is why the simplistic relational
treatments (front-runner fade ∝ rival front-runner count, tested four ways) do not yet convert — not
because the mechanism is absent, but because we have not yet modelled geometry and control jointly.
The proper treatment is a differential / mean-field **game of jockeys navigating the geometry** (the
competition game of §3b), with draw as a boundary condition on that game. Until we have confidently
modelled geometry + control together, treat the baked draw terms as **interim stopgaps** and keep the
axis open. This is the single most promising piece of deep theory-building left.

**Substrate when the axis is taken up** (also for place/exotic pricing, where whole-field ordering
matters more than the knife-edge for first): an **interacting Plackett-Luce** (PL as the free theory)
or condensed-matter machinery — exclusion processes (TASEP) for blocking, active-matter for drafting,
mean-field games for the jockey-control policy — built jointly with the geometry, not a hand-tuned
loop. Rebuild against the conversion test, and remember the methodology lesson: confirm every
candidate with a **paired fixed-seed full-scale** backtest (the subsampled quick_eval over-stated the
one surviving interim term ~4×).
