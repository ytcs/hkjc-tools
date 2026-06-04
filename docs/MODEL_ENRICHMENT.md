# Model Enrichment — mechanistic coupling only (no tacked-on covariates)

**Principle (non-negotiable).** Nothing enters as an additive/multiplicative term on
the model's *output*. Every factor must change the **dynamics** — it modulates the
state evolved by the equations of motion, or it changes how the agent acts within
them — and the effect on win/place/order then *emerges* from integrating those
dynamics. We are modelling the structural physics of a race, not fitting covariances.
A factor we cannot couple mechanistically does not go in.

There are exactly **two entry points**:

- **(A) Physiological state** — the parameters the equations of motion integrate:
  body+carried **mass `m`**, aerobic power ceiling **`P_c`** (≙ critical velocity),
  anaerobic reserve **`W′`**, and their *on-the-day* condition. Fitness, freshness,
  weight, barrier-trial evidence, airway gear, vet state all act here — by setting
  the horse's actual `(m, P_c, W′)` going into the race.
- **(B) Control policy** — how the agent *spends* `W′` over the trip and *navigates*
  traffic/ground-loss: the pace plan `P(t)` and the lateral/positioning decisions.
  Jockey skill, settling gear, running style act here — they are the policy `π`.

## The enabler: upgrade the core to a power-balance equation of motion

Today's core is *kinematic* — it sets a target velocity `v_target = v_c·(1+W′/D)`
and caps it by traffic. That cannot accept mass or drafting mechanistically. The
correct core is a per-horse **Newtonian power balance**, integrated each step (the
traffic sim already steps in time, so this is a core swap, not a new loop):

```
m · dv/dt = F_prop − F_resist
F_prop   = P(t) / v                      # metabolic power → propulsive force
P(t)     = P_c + p_anaer(t),  ∫ p_anaer dt ≤ W′      # aerobic + drawn-down reserve
dW′/dt   = −(P − P_c)               if P > P_c  (spending)
         = +r(P_c − P)              if P ≤ P_c  (recovery)
F_resist = c_loc(m)·g  +  ½ρ·A·v²·(1 − draft_i(t))  +  bend_loss(off, κ)
```

In this single equation **every factor couples through a real term**:

| Factor | Entry | Mechanistic coupling (the actual term it moves) |
|---|---|---|
| **Body + carried weight** | A | `m` in `m·dv/dt` and in locomotion cost `c_loc(m)·g`: heavier ⇒ more force to accelerate, more cost to carry. Not a penalty — it *is* the mass in F=ma. |
| **Freshness / days-since-run** | A | Sets the *initial* `W′` and `P_c` via a recovery-state model (`W′` not fully repleted after a hard recent run; under-raced ⇒ lower `P_c`). Modulates the EOM's inputs, integrated normally. |
| **Barrier trial / recent sectionals** | A | Same CV machinery on the trial's splits → another measurement of `(P_c, W′)` → Bayesian update of the horse's state. Identical physics, more data. |
| **Tongue-tie / airway gear** | A | Raises `P_c` (aerobic ceiling) — a ventilation term, not a dummy. |
| **Drafting / slipstream** | A↔B | `draft_i(t)` reduces aerodynamic drag when sitting behind a leader — a genuine force coupling between agents (replaces today's pure "blocking"). Both a state effect and a reason the policy seeks cover. |
| **Jockey skill** | B | The *quality of the policy* `π`: pace judgment (when to convert `W′`), ground-saving (rail-holding vs gap-acceptance on bends), timing of the finishing effort. A better jockey realises a better trip from the *same* `(m,P_c,W′)`. |
| **Blinkers / settling gear** | B | Changes early-race `P(t)` — less energy wasted fighting for position ⇒ more `W′` left for the finish. A pace-policy modifier. |
| **Running style (front/closer)** | B | The prior on `P(t)` shape (front-loaded vs kicked). Emerges in traffic with everyone else's policy. |
| **Vet flag / Conghua return** | A | State gate: degrades realised `P_c`/`W′` (or censors) — a condition on the physiological inputs. |

Trainer form and "stable in form" are **not** a separate mechanism — they are weak
priors on the horse's condition (the `(P_c, W′)` recovery state), folded into the
freshness/fitness state estimate, never added on the side.

## Why this is also the right call for *edge*
The ST backtests show win/place/exotic are all market-efficient (we recover ~12–17%
of market skill). Bolting on jockey covariates would only re-derive what the market
already prices and would *not* be physics. The mechanistic path is the only one that
can surface something the crowd under-prices: a *correct dynamical model* makes
sharp, falsifiable predictions for **off-equilibrium** situations (unusual pace +
draw + weight + traffic interactions) that rule-of-thumb handicapping integrates
poorly. The value, if any, is in the dynamics — exactly where we should be looking.

## Build order
1. **Swap the kinematic core for the power-balance EOM** (mass, metabolic power,
   drag with drafting). Re-derive `(P_c, W′, m)` from sectionals under the new core.
2. **Recover the policy** `π` per run (the `P(t)` actually executed) from sectional
   pace profiles; this is where jockey/style live.
3. **Couple the state factors** (weight via `m`, freshness/trial via `(P_c,W′)`
   estimation) — already mechanistic once step 1 exists.
4. **Couple drafting and the policy factors**; jockey skill = policy quality, fit
   per jockey from realised-trip efficiency.
Each step must reduce to the prior model when its new term is neutral (mass-invariant,
no draft, neutral policy) — the test that it is coupling, not bolting.
