# Policy Layer Map — inputs × knobs (experiments + data)

The race is a **plant** (the power-balance EOM in `model/dynamics.py`: finite W′ reserve,
traffic, drafting, bends). The **policy** is what the jockey actuates each instant given
what they observe: `action = π(state)`. This file maps every important **policy INPUT**
(observable state the policy conditions on) and **policy KNOB** (control action), each
pinned by a **sim experiment** and, where possible, a **real-data** check. The discipline:
a knob only enters if it's a real control lever on the EOM — nothing tacked on.

Status: ✅ done · 🔬 running/partial · ⬜ TODO

---

## A. Policy INPUTS — the state π may condition on

| Input | How it enters the EOM | Experiment | Data check | Finding |
|---|---|---|---|---|
| **Distance-to-go / progress** | gates W′ deployment timing | `best_response_kick` 🔬 | sectional split profile ⬜ | kick timing is the core feedback variable |
| **Own W′ remaining** | the reserve being spent; empty ⇒ fade | (kick is distance-gated; true W′-feedback ⬜) | — | exhaustion gives an interior optimum (over-aggression punished) ✅ |
| **Gap to leader / rank** | car-following cap; how far back | ⬜ (gap-triggered kick) | sectional running positions ⬜ | not yet a policy input — open-loop only |
| **Field pace config (#front-runners)** | collective early speed ⇒ shared W′ burn | `pace_scenario*`, `best_response` ✅ | running-style mix vs result ⬜ | cannibalisation emerges; lone-speed premium real |
| **Race distance (sprint↔route)** | time for late W′ to convert | `pace_geometry_litmus`, `best_response` dist-sweep ✅ | style×distance win rate 🔬 | front-runner edge ↓ with distance (corr −0.97 sim) |
| **Venue geometry (bends, run-in)** | bend ground-loss + closing room | `pace_geometry_litmus` ✅ | HV vs ST style win rate ⬜ | HV rewards position > ST; emergent from shape |
| **Draw / lateral** | start offset → rail access on bends | backtest draw-slope ✅ | draw bias by venue/dist ✅ (priced) | inside favoured, strongest at HV; market-efficient |
| **Going / condition** | rolling resistance `K_LOCO`, grip | ⬜ | going × style ⬜ | not yet modelled |

## B. Policy KNOBS — the control actions π actuates

| Knob | EOM parameter | Experiment | Data check | Finding |
|---|---|---|---|---|
| **W′-deployment TIMING** | `kick_frac` (hold at v_c → dump reserve inside kick_frac·D) | `best_response_kick`, `policy_intensity_confirm` ✅ | running positions ⬜ | feedback **fixes the dawdling pathology** — a connected closer goes from ~0% (open-loop) to ≈ fair share. BUT with proper SE (N=3000) **no kick timing dominates** on a route (all within 1–2 SE of fair). The earlier "kf 0.40→14.4%" was NOISE (retracted). Tactics are 2nd-order once connected. |
| **Early aggression (open-loop)** | `pace_bias` × `pace_gain` (tilt) | `pace_scenario*`, `best_response` ✅ | `rel_early` distribution ✅ | front-running robustly good in sprints; **closing never pays open-loop** (dawdles, falls back) |
| **W′-deployment INTENSITY** | `kick_overspeed` | ⬜ sweep | — | how hard the kick is; tradeoff vs fade |
| **Lateral path** | `lane_change_prob`, `lane_step`, drift-to-rail | ⬜ sweep | path width from drone CV ⬜ | rail economy vs clean-run tradeoff |
| **Drafting choice** | `draft` relief sitting behind | ⬜ sweep | — | energy saving vs being blocked |
| **Response to blocking** | go-wide vs wait | ⬜ | interference events ⬜ | tactical traffic management |

---

## Findings ledger (chronological, experiment-pinned)

1. **Kinematic sim → no pace game.** Without W′ exhaustion, front-running is trivially good
   and there is zero cannibalisation (lone-speed premium ≈ −0.5pp). `pace_scenario`.
2. **W′-depletion → cannibalisation emerges.** Real reserve burn makes a crowd of
   front-runners self-defeating (lone-speed premium −7pp; interior optimum, b=+2 worse than
   b=+1). `pace_scenario_dynamics`, `best_response`.
3. **Open-loop tilt can't close.** A negative-bias horse dawdles below v_c and falls out of
   it — closing never pays at any distance. Localises the need for **state feedback**. `best_response` dist-sweep.
4. **Distance/venue tactics are emergent, not coded.** Front-runner edge falls with distance
   (corr −0.97) and is larger at HV than ST, from geometry alone. `pace_geometry_litmus`.
5. **Feedback fixes closing's VIABILITY, not its dominance.** The kick policy (hold at v_c,
   pounce inside kick_frac·D) lifts a connected closer from ~0% (open-loop dawdler) to ≈ fair
   share. BUT under proper sampling (N=3000, SE≈2pp) **no kick timing/intensity dominates** on
   a route behind a hot pace — all configs sit within 1–2 SE of fair. The striking "kf 0.40 →
   14.4%" from the first scan (N=1500) was Monte-Carlo NOISE and did not replicate (→4.0%).
   `best_response_kick` then `policy_intensity_confirm`. **Lesson: once horses are connected,
   deployment tactics are 2nd-order (~1–2pp), not decisive.** Methodological note: always carry
   SEs on win-prob sweeps — the EOM is expensive so early runs were under-sampled and misleading.
6. **Net policy-layer verdict.** The only LARGE, replicating effect is over-aggression→fade
   (finding #2, sim + data). Distance/venue gradients (#4) are real but modest. Fine tactical
   optima are noise-dominated at feasible compute ⇒ consistent with the project thesis that
   ability/form dominate and tactics are a marginal overlay.

## Real-data anchors (do real horses behave as the sim predicts?)
- Running style = leak-free trailing `rel_early` (front-runner ↑, closer ↓); trait stability r≈0.45.
- Style as a flat win-covariate HURTS (it's emergent/positional, not additive) — must live in the sim policy.
- **Exhaustion CORROBORATED (`policy_data_style_distance`):** the fastest-early tercile wins
  *below* fair (sprint 0.31×, route 0.64×), the most-conserved tercile *above* — the real-data
  "the speed cut its own throat", matching sim finding #2 (over-aggression self-defeating).
- **Limitation found:** realised `rel_early` is a CONFOUNDED style proxy (pace-setter-fade +
  ability), so it does NOT cleanly validate the distance gradient (front−closer gap is −1.56
  sprint → −0.52 route — muddied). A clean anchor needs sectional **running positions** / drone
  trajectories (the policy *output*), not a speed summary. → ties the policy map to the CV work.
