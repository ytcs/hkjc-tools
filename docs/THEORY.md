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

## 3c. The geometry × control game — draw as a boundary condition on the jockey policy

§3b developed the competition game in the **longitudinal** (along-track, effort/pace)
direction and showed the positional fade is its mean-field shadow. This section develops
the **lateral** (across-track) direction — the part the scalar draw terms (`DRAW_GAIN`,
`DRAW_FS_GAIN`) only shadow. The thesis (journal `0027`): **draw is not a static
geometric covariate. Its effect is a functional of the jockey's control policy** — how the
gate is used to take up position, which lateral path is chosen round the bends (save ground
on the rail vs. go wide for clear air), and when the anaerobic reserve is spent to win or
hold position. The right object is a **two-dimensional differential / mean-field game of
jockeys navigating the real track geometry, with the draw as the initial lateral boundary
condition**. The scalar terms capture only the *first moment* of that game's resolution;
this section writes the game down, identifies its equilibrium structure, and — crucially —
derives the **leak-free observables it predicts** from the data we actually hold.

### 1. State, action, dynamics, payoff (the plant + the control)

**Geometry (the arena).** Encoded in `hkjc.core.track.TrackGeometry`. A horse runs from
`d = distance` (metres-from-finish) down to `d = 0`. The arena is parameterised by
*venue × surface × Course*: the movable running rail (A / A+2 / B / … / C+3, read from
`results_rich.Course` / `st_history.Course`) sets the usable **width** `w(course)` and the
home-straight length. The bend map `bend_segments` partitions the lap into **turns** and
**straights**; on a turn a horse `off` metres off the rail runs extra arc
`≈ off · κ(d) · ds` where `κ(d)` is the local curvature (≈ `1/R`, larger at HV — the
tightest HK circuit, hence its strongest draw bias; ≈ 0 on the ST-1000m straight chute,
which is why `fit.py` zeroes the draw term there). HV is bend-dominated (≈ 470 m + 530 m of
turn in 1450 m); ST is a galloping oval with two ~495 m bends and long straights.

**Agent state.** Each horse-jockey `i` is an agent with state

```
s_i(t) = ( x_i,        # along-track position (metres covered)
           y_i,        # lateral offset from the current inner rail  ∈ [0, w(course)]
           W'_i,       # remaining anaerobic reserve  (≥ 0)
           v_i )       # current speed
```

**Action / control policy.** The jockey chooses `a_i(t) = π_i(s_i, s_{-i}; θ_i)`, a map from
own state and the field state `s_{-i}` to two controls:

```
u_i(t) ≥ 0    # effort above the aerobic ceiling  (depletes W'_i)
ẏ_i(t)        # lateral steering rate  (the path: hug rail vs. peel out for clear air)
```

`θ_i` parameterises the **policy type** — a jockey/horse's habitual style (front-runner vs.
closer, ground-saving rail-hugger vs. wide-and-clear), the object that makes control
*heterogeneous* and therefore *predictable* from past running lines.

**Equations of motion (the plant — same physics as the §2.2 EOM and the deprecated sim,
now 2-D).**

```
ẋ_i = v_i
m_i v̇_i = P_c,i + u_i       (metabolic power)                      [longitudinal]
        − K_LOCO m_i g                                              (locomotion)
        − ½ ρ c_d A v_i² (1 − draft_i(s_{-i}))                      (aero, cut behind cover)
        − χ · 𝟙[bend] · κ(x_i) · v_i² · y_i / R                     (bend ground-loss ∝ lateral offset)
Ẇ'_i = − u_i                  (reserve depletes above ceiling; u_i>0 only when v_i>v_{c,i})
ẏ_i = steering, subject to the EXCLUSION constraint:
      y_i may not coincide with y_j for a horse j with x_j ≈ x_i   (cannot occupy a rival's lane)
```

The **lateral channel is where draw lives**: the start fixes `y_i(0) = y_0(Dr_i)` (a wide
draw starts wide). To save ground a wide-drawn horse must *cross* — spend `ẏ_i < 0` early —
but crossing is contested: the rail is a scarce resource and the exclusion constraint
(a TASEP-like single-file occupancy in each lateral band) means a horse can only take the
rail if a gap opens. So a wide draw imposes a **control dilemma**: (a) commit reserve early
to cross to the rail and save bend-ground, paying a `W'` tax up front and risking being
shuffled back if the gap doesn't come; or (b) stay wide, concede the arc-length penalty but
keep clear air and reserve. The *cost of the draw is whichever horn the equilibrium forces*
— it is **not** a fixed `DRAW_GAIN·(Dr−1)` arc penalty.

**Payoff.** A tournament: only finishing **order** pays. Agent `i` maximises

```
J_i = E[ − T_i ]   or   E[ R(rank_i) ],   T_i = ∫_0^{distance} dx / v_i(t),
```

i.e. a closed-loop best response that trades the bend-ground integral
`∫ χ κ(x) v² y_i dx` (minimised by hugging the rail) against the **clear-air / draft / reserve**
terms (which can favour going wide), under the budget `∫ u_i dt ≤ W'_i` and the lateral
exclusion constraint coupling `i` to the field.

### 2. Why it is a (mean-field) game, and where DRAW enters

Each jockey's optimal `(u_i, ẏ_i)` is a **best response to `{π_j}`**: the value of crossing
to the rail depends on whether rivals leave a gap; the value of leading depends on who else
contests the lead (the §3b pace duel); the draft you get depends on who sits where laterally.
The profile `{π_i}` is therefore a **Nash equilibrium of a differential game** with state
constraints (exclusion) — formally an open-loop/feedback Nash solution of the coupled
Pontryagin/HJB–Isaacs system, one adjoint per agent for `(x, y, W')`. With many runners the
rivals enter only through **aggregate fields** — the *along-track* pace density `ρ(x,t)` and
the *lateral* occupancy density `m(x,y,t)` (how crowded each lane is at each point of the
trip). The large-field limit is a **mean-field game (MFG)**: the HJB for each agent's value
`V_i(s)` is driven by the population densities `(ρ, m)`, which are in turn the Kolmogorov-
forward push-forward of the agents' own optimal controls — a fixed point in
*(pace × lateral-occupancy)* space. The exclusion constraint makes the lateral marginal a
**congestion MFG** (cost of a lane rises with its occupancy `m`), the cleanest formal home
for "a wide draw costs more in a crowded field."

**Draw is the initial/boundary condition, its cost mediated by the equilibrium.** `Dr` sets
`y_i(0)`; the realised draw cost is the equilibrium value-gap
`ΔV_i = V_i(y_0=rail) − V_i(y_0=wide)`, which is **not** a constant — it is a functional of:

- **the rail/curvature** `κ, w(course)` — more arc to lose on a tight track (HV ≫ ST; ≈ 0 on
  the straight chute). This the scalar `DRAW_GAIN_BY_VENUE` already captures (its *zeroth* moment).
- **the field's lateral congestion** `m` — a crowded field (`N` large) means fewer gaps to
  cross, so the wide horse is *trapped* wide → larger cost. This is the **first interaction
  moment**; `DRAW_FS_GAIN` (crowding × draw, HV-only) is its scalar shadow — and an interim
  stopgap, per §Appendix.
- **own and rivals' policy types** `{θ_j}` — a wide horse drawn outside a wall of
  front-runners cannot cross and must go wider still; the *same* wide horse with closers
  inside it can angle down early and save the ground. **Identical draw, opposite cost,
  decided by the style-composition of the neighbours** — exactly the relational quantity a
  scalar cannot encode, and the reason the four scalar pace probes (journal `0024`, `0026`)
  did not convert.

So the equilibrium **mediates** draw: `cost(Dr) = f(Dr; κ, course, N, {θ_j}, θ_i)`. The
production scalar is `cost ≈ DRAW_GAIN·(Dr−1)` — the projection onto the first two arguments
at field-mean policy. The open gain is the **policy-conditioned** correction.

### 3. Testable, leak-free observables it predicts (ranked by tractability)

The binding data constraint (verified on the panels): **we have no lateral coordinate.** No
panel stores `y_i(t)`; the only across-track datum is the *starting* offset `Dr`
(`results_rich.Dr`). What we *do* have is the **longitudinal rank trajectory** — `RunningPosition`
(rank at 3–6 call points, e.g. `"9 9 7 1"` = 9th→9th→7th→1st), per-section `velocity` +
`position` in `trajectory_dataset_full` (400 m sections), and the already-derived policy
panel `run_policy_dataset` (`early_pos_norm`, `late_pos_norm`, `pos_gain`, `fade`,
`kick_frac_obs`, `effort_dtg`, `depleted`, each with leak-free `trail_*` trailing means).
The lateral path and ground-saved are therefore **latent — inferred from the rank trajectory
× draw × geometry, never observed.** Every observable below respects this and is leak-free
(a runner's value uses only its STRICTLY PRIOR runs, the `features.py` discipline; the
contemporaneous geometry/draw/field-composition inputs are all pre-race public).

**O1 — Draw × Course (rail position) interaction. [most tractable]**
The arc-length penalty scales with curvature, and the movable rail (`Course`) changes both
`κ`-exposure and width. Predict: the draw gradient is *steeper on tighter Course settings*
(C/C+3, narrower) than wide (A). Operationalise as a **draw × course** (or draw × `w(course)`,
or draw × bend-fraction) modulation of `DRAW_GAIN`, i.e. `L_eff = dist·(1 + g(course)·(Dr−1))`.
Data: `results_rich.Course` + `track_for(...).width/bend_fraction`. Pure pre-race geometry —
no trajectory, no leak surface, single new constant per course family. **Lowest risk, build first.**

**O2 — Draw × field-pace-makeup (relational congestion). [high value]**
The wide-draw cost rises with lateral congestion, which the *style composition* of the field
proxies: many habitual front-runners + a full field ⇒ a wall the wide horse cannot cross ⇒
larger cost; a field of closers ⇒ the wide horse angles down cheaply. Operationalise:
per-race **pace-pressure / wall scalars** built from the leak-free `early_position_table`
(EB-shrunk prior `early_pos`) — e.g. `n_FR` (count of prior-front-runners), the fraction of
the field drawn *inside* `i` that are front-runners, and `N`. Feature: `DRAW_GAIN × Φ(Dr,
inside-front-runner-count, N)`. This is the relational generalisation `DRAW_FS_GAIN` only
shadows (it uses bare `N`); the prediction is that *conditioning the crowding on who is
inside* converts where bare `N` plateaued. Data: `run_policy`/`early_position_table` + `Dr`.

**O3 — Control-quality / ground-saved signal from the rank trajectory. [medium]**
The equilibrium predicts a measurable **path-efficiency** type: a horse/jockey that
habitually *gains rank without a velocity premium* is saving ground (taking the rail, getting
the gaps) — good control; one that *needs a velocity premium to hold rank* is running wide /
fighting traffic — costly control. Construct a leak-free per-(horse,jockey) trailing index
from `trajectory_dataset_full`: regress section-to-section **rank gain** on the horse's
**relative section velocity**; the residual (rank gained *per unit speed*, i.e. ground-saved
efficiency) averaged over prior runs is a clean control-quality feature. Cross-check against
`run_policy.pos_gain` net of `fade`. Couple it as a per-horse `v_c`/trip bump (it is a real,
stable jockeyship latent, like `early_pos`). Heavier (needs the trajectory join + a
within-race regression) but mechanism-faithful and convertible as a main-effect first.

**O4 — Jockey policy types (the `θ` partition). [medium, enabling]**
Cluster jockeys (and horse×jockey pairs) by their leak-free `trail_*` policy vector
(`trail_early_pos_norm`, `trail_kick_frac_obs`, `trail_fade`, `trail_effort_dtg`, the O3
ground-saved index) into types — e.g. *ground-saver*, *wide-and-clear*, *late-committer*,
*pace-presser*. Not a standalone win feature so much as the **input that makes O2/O3
relational** (the `{θ_j}` the draw cost depends on) and a candidate `DRAW_GAIN` modulator
(a ground-saver suffers a wide draw less). Tractability: the clustering is cheap; its *value*
is realised only through O2/O3, so build after them.

**O5 — Draw × pace × position joint term (the full many-body cell). [least tractable]**
The genuine two-body lateral term — an inside front-runner controls rail+tempo; a wide one
must burn reserve to cross — i.e. `cost(Dr) × pace-pressure × own-style`. This is the
agent-based sim's domain (a 2-D MFG Monte-Carlo with the exclusion constraint), reserved for
place/exotic pricing where whole-field ordering matters more than the knife-edge for first.
Highest ceiling, heaviest build; only after O1–O3 confirm the mechanism converts at all.

**Ranking (tractability × leak-safety, descending): O1 ≻ O2 ≻ O3 ≻ O4 ≻ O5.** O1 is pure
pre-race geometry (one constant); O2 reuses an already-converting leak-free table; O3/O4 add
the trajectory join and a latent index; O5 is the full sim.

### 4. The minimal next experiment (hand-off to a future modelling agent)

**Build O1: a draw × Course (curvature/width) modulation of `DRAW_GAIN`.** It is the single
most tractable convertible step — pure pre-race geometry, one constant family, zero leak
surface, and it directly tests the central claim (the draw cost is *geometry-mediated* before
it is control-mediated) without yet needing the harder control inference.

Concretely: (i) in `fit.py`'s wald drift, replace the per-venue scalar
`L_eff = dist·(1 + DRAW_GAIN·(Dr−1))` with `g(course) = DRAW_GAIN · h(bend_fraction(course),
w(course))` resolved from `track_for(venue, distance, surface)` + `results_rich.Course`
(width and bend-fraction are already on `TrackGeometry`); (ii) scout the shape of
`g(course)` on the residual (`market_residual_diagnosis.py`, draw × course buckets, race-
clustered SEs) to set direction; (iii) **confirm with the PAIRED fixed-seed full-scale
backtest** (`PYTHONHASHSEED=0`, both venues + ECE) — the quick_eval over-states draw-knobs
~4× (the `DRAW_FS_GAIN` lesson), so the t-stat is only a scout; the %market conversion is the
gate. **Decision rule:** if the draw gradient is materially steeper on the tight Course
settings *and* the modulation converts OOS, the geometry-mediation half of the thesis is
confirmed and O2 (relational congestion) is greenlit; if it is flat, the venue scalar already
saturates the geometry moment and the gain is purely in the **control** channel (skip to O2/O3
and the relational/policy terms). Either outcome advances the geometry × control program —
this experiment is designed to *partition* the remaining draw signal into its geometry vs.
control shares.

---

## 3d. How the couplings map onto the generative model — the structure

The drift-diffusion exposes **three independent knobs** a feature can load: the aerobic ceiling
`v_c`, the anaerobic reserve `W′` (the `(1+W′/L)` late boost), and the diffusion `D` (consistency /
racing luck, the Wald shape `λ = L²/2D`). A feature is correctly structured only when it loads the
knob its mechanism actually acts on — using a signal is not enough; it must enter in the right place.

**`v_c` is a precision-weighted posterior over critical velocity.** The ability channels (official
rating, form, sectional `v_c`, collateral-form, barrier trial, speed-figure) are *noisy measurements
of the same latent `v_c`*. The principled combination is the precision-weighted posterior mean
`v_c = Σ_k (π_k/Σπ) v_c^(k)`, `π_k = 1/σ_k²`. The production drift `v_c_eff = v_c_base + Σ_k g_k·s_k`
is the first-order **linearisation** of that posterior, so each gain `g_k` is the channel's relative
precision — which is exactly why gains are calibrated rather than free, why over-coupling a channel
past its precision hurts, and why a channel collinear with a priced one adds nothing. The coherent
empirical-Bayes **fusion** — one shrinkage posterior over *all* velocity observations (official runs,
barrier trials, and training gallops), each down-weighted by its noise, with the correct non-linear
`λ(n_eff)` — is this structure made exact, and the right home for every ability signal. Channels that
load `v_c`: rating (hierarchical-Bayes band×rating mean), form (a rank-censored velocity residual —
its margin-censoring is *why* the velocity-unit figures add over it), sectional `v_c`, **collateral-
form** (a Massey least-squares opponent-adjusted velocity — literally the consistent rating the "win
model is a rating system at equilibrium" framing predicts), trial, speed-figure, and **allometric
mass** (`v_c ∝ m^(b−1)`, with one *venue-invariant* physiological exponent `b ≈ 1.1`; mass is simply
low-leverage where other channels already price it, not a different exponent per track).

**`W′` carries the late-race reserve.** Fade-resistance (final-section vs peak velocity) is the
kinematic signature of the reserve emptying in the closing stages; positional-fade is a front-runner's
*`W′`-burn* spent setting the pace. Both are **reserve** phenomena and belong on the `W′` knob, where
their effect is naturally distance-aware (`W′/L` bites harder over sprints) and the gain reads in
reserve-metres. (Production currently applies them as a `v_c` proxy — a correlated approximation that
raises/lowers the whole velocity curve rather than just the closing kick.)

**`D` carries consistency — the one knob not yet exploited.** Every banked signal moves the *mean*
drift; none moves the per-horse *variance*. But a horse with erratic figures is genuinely less
predictable, so its `D_i` should scale with the spread of its own prior figures. Because the winner is
an **order statistic**, variance is as load-bearing as the mean — a high-variance horse upsets more
often than its mean drift implies, and a metronomic one less.

**Career phase enters as regression to the horse's own mean.** A point rating sits above or below the
horse's career-mean level; the market over-discounts a horse off its recent peak and over-rates one at
its ceiling, so the converting correction is **mean-reversion toward the career mean** (this is what
the rating-trajectory "peak-gap" signal actually captures — a process feature pays only where it adds
an *absolute career landmark* the point rating cannot express, not where it merely restates recent form,
which is already in the form channel).

`JOCKEY` sits slightly apart: part trip-quality (a real `v_c` effect), part an information proxy
(fancied horses get the top riders), so it is partly endogenous to the very market we fit — a
measurement we keep, but not a clean single mechanism.

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
