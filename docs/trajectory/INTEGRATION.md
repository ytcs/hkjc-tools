# Dense Trajectories → Mechanistic Model Integration (spec)

**Status: design only — no code changes implied here.** This document specifies
(1) the canonical per-horse-per-race trajectory schema, (2) exactly how each *port*
of the physics model is sharpened by it, and (3) the migration path from the current
sectional-based calibration. It is the build contract for when the dense positioning
feed (HKJC's RF/GPS-derived "Aerial Virtual View" data; see
`docs/TRAJECTORY_RESEARCH_PLAN.md` Tier-2/V1) lands.

It obeys the enrichment principle (`docs/MODEL_ENRICHMENT.md`): **every data field
maps to a physical term in the equation of motion or to the control policy — nothing
is a tacked-on covariate.** The two entry points stay the same: **(A) physiological
state** `(m, P_c, W′)` and **(B) control policy** `π` (pace `P(t)` + lateral
navigation). The trajectory does not add a third entry point; it *measures* the terms
that today are inferred from 2–4 splits or a coarse `rel_early`.

The structural objects are unchanged. The trajectory feeds:

- the EOM in `src/hkjc/model/dynamics.py` — `m·dv/dt = P/v − F_resist`, with
  `F_resist = K_LOCO·m·g + ½ρ·cda·v²·(1−draft) + bend ground-loss`;
- the interacting simulation in `src/hkjc/model/traffic.py` /
  `src/hkjc/model/traffic_jax.py` — car-following, gap-acceptance lane changes, bend
  ground-loss, and the new pace policy (`pace_bias` per horse × scalar `pace_gain`);
- the calibration layers in `src/hkjc/calibration.py` — EB `(v_c, W′)` and the
  per-venue `bend_curvature` posterior;
- the geometry in `src/hkjc/track.py` — `bend_segments`, `circumference`, per-course
  `width`, `home_straight`.

---

## 0. Coordinate & sign conventions (shared with the existing stack)

To slot beside the current model these conventions are fixed and identical to
`track.py` / `dynamics.py`:

- **`d` = metres-from-finish.** A horse runs from `d = distance` down to `d = 0`.
  The track's `bend_segments` and `is_bend(d)` are defined on `d % circumference`.
- **`s` = finish-progress arc length along the rail**, `s = distance − d`, the same
  `s` the simulators integrate (`ds = v·dt / ground_factor`). `s` is the
  *rail-referenced* progress, **not** the horse's own travelled path length (that is
  a derived field, `path_len`, below). This distinction is the whole ground-loss
  signal.
- **`off` = lateral offset outward from the current inner rail, metres**, ≥ 0. The
  movable rail (`course`) is the zero. Draw → starting `off` today via
  `off0 = clip((draw−1)·lane_sep, 0, width−1)`; the trajectory *measures* `off(t)`
  directly and retires that proxy.
- **`ground_factor = 1 + off·κ` on a bend** (κ = `bend_curvature`); the extra arc per
  rail-metre when `off` wide on a turn. The trajectory lets us measure the LHS
  (`d(path_len)/ds`) and so identify κ directly.
- Right-handed (clockwise) HK tracks: inner rail favours low draws. `handedness` is
  carried so a future left-handed track flips the offset sign convention in one place.

Units everywhere: **metres, seconds, m/s, m/s², watts, joules, kilograms.** Angles in
radians. This matches `dynamics.py` (`G`, `RHO`, `K_LOCO`, `P_c = K_LOCO·m·g·v_c`,
`W′(J) = K_LOCO·m·g·w_prime`).

---

## 1. Canonical trajectory schema

Two parquet datasets, keyed to slot beside the existing `data_cache/*.parquet`
(`sectional_dataset`, `energetics_dataset`, …). The **same canonical keys** used
throughout the library are `(date, race_no, runner)` with `horse_id`; position `i`
in runner arrays is runner number `i+1`. Both datasets join cleanly onto
`energetics_dataset` and `sectional_dataset` on `(date, race_no, runner)`.

### 1a. `trajectory_dataset.parquet` — the dense sample grid (long / tidy)

One row per **(horse, race, time-sample)**. At ~25 Hz a 100 s race is ~2,500 rows ×
~14 runners ≈ 35k rows/race; store row-group-partitioned by `(date, race_no)` and
compress (zstd). Dictionary-encode the string keys. This is the raw substrate; all
physics derivations read from it.

| field | unit | dtype | meaning / physical term it feeds |
|---|---|---|---|
| `date` | — | `str` (DD/MM/YYYY, matches sectional_dataset) | canonical key |
| `race_no` | — | `int32` | canonical key |
| `runner` | — | `int64` | canonical key (HKJC runner no, 1-based) |
| `horse_id` | — | `str` | brand id, joins to energetics/sectional |
| `venue` | — | `str` ("HV"/"ST") | selects `track_for` geometry |
| `t` | s | `float32` | time since flag-fall (gate release), monotone per horse |
| `dt_nominal` | s | `float32` (race-level const, repeated) | nominal sample period (≈1/feed_hz); guards irregular feed |
| `s` | m | `float64` | **rail-referenced finish-progress arc** `= distance − d`. The simulators' `s`. |
| `d` | m | `float64` | metres-from-finish `= distance − s` (redundant, kept for `is_bend(d)` joins) |
| `off` | m | `float32` | **lateral offset from current rail** — feeds `off` in EOM, retires draw→off0 proxy |
| `x` | m | `float64` | world-frame x (rail-origin, track plane) — for path length & curvature |
| `y` | m | `float64` | world-frame y |
| `v` | m/s | `float32` | instantaneous speed along the horse's own path = `d(path_len)/dt` |
| `a` | m/s² | `float32` | instantaneous tangential acceleration `dv/dt` → the LHS `m·dv/dt` |
| `on_bend` | — | `bool` | `track.is_bend(d)` cached at sample (turn vs straight) |
| `rank` | — | `int8` | live finishing-order rank at this `t` (1 = leading on `s`) |
| `quality` | — | `float32` ∈ [0,1] | feed confidence (RF/GPS noise; render is *lossy*) — weight for fits |

Notes:

- `v` and `a` are computed from the **own-path** kinematics (`x,y` differentiated),
  not from `s`; `ds/dt` is `v / ground_factor`. Smooth with a causal Savitzky–Golay
  (e.g. window ~0.5 s) before differentiating to suppress RF jitter; store the
  smoothed `v,a` and keep raw `x,y`.
- `off` sign: positive = wider than the rail. On the ST straight 1000 m chute
  (`SHA_TIN_STRAIGHT`, no bends) `off` is recorded but never multiplies κ.
- `quality` lets every downstream estimator be a *weighted* fit, since the virtual
  view is explicitly RF-noise-limited (good for lateral/ground-loss, not sub-metre
  truth).

### 1b. `trajectory_features.parquet` — per-horse-per-race derived physics (wide)

One row per **(horse, race)** — the model-facing layer, mirroring how
`energetics_dataset` is one row per run. Everything here is an integral/fit over 1a,
so it can be recomputed and is leakage-safe by construction (no other race's data).

| field | unit | dtype | derivation | physical term it sharpens |
|---|---|---|---|---|
| `date,race_no,runner,horse_id,venue,distance` | — | keys + `int32` distance | join keys | — |
| `total_time` | s | `float64` | `t` at `s = distance` | cross-check vs results `FinishTime` |
| `path_len` | m | `float64` | `∮ √(dx²+dy²)` over the trip | **actual ground travelled** (vs `distance`) |
| `ground_lost` | m | `float64` | `path_len − distance` | total extra arc run |
| `bend_ground_lost` | m | `float64` | `∫ off·κ̂ · 1{on_bend} ds` (or geometric `∮ off·dθ` on turns) | **direct κ / `bend_curvature` observable** |
| `mean_off`, `max_off` | m | `float32` | offset moments | lateral path summary |
| `off_profile` | m | `list[float32]` (resampled to K rail-bins, e.g. K=24) | `off(s)` binned by `s` | lane trajectory for traffic calib |
| `v_profile` | m/s | `list[float32]` (K bins on `s`) | binned own-path speed | the **speed-vs-distance curve** for CP/W′ fit |
| `v_section[]` | m/s | `list[float32]` (HKJC sections) | re-aggregate to published section windows | cross-check vs sectional_dataset |
| `section_time[]` | s | `list[float32]` | section times reconstructed from trajectory | **validation vs HKJC published sectionals** |
| `v_peak` | m/s | `float32` | `max v` | anaerobic ceiling sanity |
| `v_c_traj` | m/s | `float32` | CP/W′ fit (port 2.1) | **measured critical velocity** |
| `w_prime_traj` | m | `float32` | CP/W′ fit (port 2.1) | **measured anaerobic distance reserve** |
| `fit_rmse` | m/s | `float32` | residual of the EOM fit to `v_profile` | identifiability / quality weight |
| `pace_bias_meas` | — | `float32` (centred ~0) | measured early/late deployment (port 2.2) | **measured `pace_bias`** |
| `pace_energy_resid` | — | `float32` | departure from energy-neutral (port 2.2) | validates the zero-mean pace shape assumption |
| `power_profile` | W | `list[float32]` (K bins on `s`) | EOM-inverted `P(t)` (port 1c) | the executed metabolic policy `P(t)` |
| `w_bal_profile` | J | `list[float32]` (K bins on `s`) | integrated reserve balance | `W′`-balance over the trip |
| `draft_frac_profile` | — | `list[float32]` ∈[0,1] (K bins) | per-sample draft exposure (port 1d) | the `draft` term in `½ρ·cda·v²·(1−draft)` |
| `draft_exposure` | s·frac | `float32` | `∫ draft_frac dt` | total slipstream benefit (cover) |
| `lead_gap` | m | `float32` mean | mean `s`-gap to horse directly ahead in lane | calibrates `follow_gap` |
| `n_blocked`, `t_blocked` | count, s | `int16`,`float32` | spells capped behind a same-lane horse | calibrates car-following |
| `n_lane_changes` | count | `int16` | outward lateral excursions > threshold | calibrates `lane_change_prob`, `lane_step` |
| `quality_mean` | — | `float32` | mean per-sample `quality` | weight for all fits |

#### Energy-expenditure profile (`power_profile`) — explicit construction

This is the inverse of `dynamics.finish_times`. At each sample, with measured
`v, a, off, on_bend` and (joined) mass `m`, drag area `cda`, draft fraction:

```
P(t) = v · F_prop,   F_prop = m·a + F_resist
F_resist = K_LOCO·m·g  +  ½ρ·cda·v²·(1 − draft_frac)  +  (bend term, in the off-path frame)
dW_bal  = −(P − P_c)·dt
```

Storing `P(t)` (and `W_bal(t)`) per run makes the *executed policy* a first-class
measured object — exactly the `P(t)` that `docs/MODEL_ENRICHMENT.md` build-order
step 2 ("recover the policy per run") asks for, now measured instead of inferred from
section pace. Because `F_resist` uses the *same* `K_LOCO, ρ, cda, draft` constants as
the EOM, `power_profile` reduces to `P ≡ P_c` for a horse that runs an even `v_c`
with no draft on no bend — the reduces-to test.

---

## 2. Port-by-port upgrade

Each port states: the current mechanism, what the trajectory replaces, and the
estimator. Every upgrade reduces to the current model when its new measured term is
neutral.

### 2.1 Engine `v_c` & `W′` — fit the whole speed curve, not 2–4 splits

**Current (`model/energetics.py` `estimate_run_energetics`, `data/sectional.py`):**
`v_c` ≈ the slower of the final two ~400 m section velocities; `W′ = Σ (v − v_c)⁺·t`
over the same few sections. With 2–4 points this is severely under-identified — an
easy winner that eases down reads a low `v_c`; a single fast section swings `W′`. The
EB layer in `calibration.py` exists *precisely* to denoise these.

**Upgrade:** fit the **full `v(s)` profile** (`trajectory_features.v_profile`, K≈24
bins, ~25 Hz underneath) to the critical-power / W′-balance EOM. Per run we now have
~10²–10³ velocity observations instead of 2–4, so `(v_c, W′)` (and optionally `cda`,
`τ`) become **jointly identifiable from a single run** — the partial pooling in
`calibration.py` becomes a *prior*, not a crutch.

**Estimator (per run, weighted by `quality`):** invert the EOM. Two equivalent forms:

1. *Forward (recommended).* Treat the measured early-pace shape as the control input:
   run the `dynamics.py` EOM forward with the horse's *measured* `v_target`/pace
   policy and `off(s)`/`on_bend` from the trajectory, and minimise
   `Σ_k q_k · (v_model(s_k) − v_meas(s_k))²` over `(P_c, W′)` (hence `v_c = P_c/(K_LOCO·m·g)`).
   This respects drag, bends and the finite-reserve fade exactly as the race model
   integrates them, so the recovered `(v_c, W′)` are *self-consistent* with the
   simulator that will use them. Mass `m` from the racecard body-weight + carried
   weight; `cda` fixed at the population value (or jointly fit when `fit_rmse` allows).

2. *Closed-form W′-balance (fast init / cross-check).* In the CP–W′ model the reserve
   balance is `W_bal(t) = W′ − ∫₀ᵗ (P(u) − P_c)⁺ du`. With `P(t)` from the inversion
   above, `P_c` is the level at which `W_bal` is monotone-nonincreasing and hits ~0
   exactly at the observed late-race fade onset; `W′` is the total area spent above it.
   Gives a closed-form initialiser for the optimiser and a residual diagnostic.

**Output:** `v_c_traj, w_prime_traj, fit_rmse` per run. These **replace** the
`v_c, w_prime` columns currently written to `energetics_dataset` (port 2.1 feeds the
same column names, so `fit_energetics_population` / `horse_energetics` consume them
unchanged). The EB machinery in `calibration.py` is unchanged; it just receives
far-lower-variance per-run inputs, so `σ²_within` (the within-horse noise component)
drops and each horse trusts its own runs sooner. **Keep** the `loo`/trailing EB
wrappers for leakage safety.

### 2.2 Pace policy — measure `pace_bias` directly, validate energy-neutrality

**Current (`traffic_jax.finish_times`, `data/historical._classify_running_style`):**
`pace_bias` per horse × scalar `pace_gain` shapes effort as
`vt_eff = vt·(1 + pace_gain·pace_bias·(1 − 2·s/distance))` — a zero-mean-over-distance
tilt (front-loaded for `pace_bias>0`, late for `<0`), asserted energy-neutral. Today
`pace_bias` is set from the coarse `rel_early` / `RunningStyle`
(FrontRunner/Pacer/Closer from `RunningPosition` strings).

**Upgrade:** measure each horse's **actual early/late deployment** from `v_profile`
and fit a *measured* `pace_bias_meas`, and **test** the energy-neutral assumption
rather than assume it.

**Estimator (trajectory → `pace_bias`):**

1. Form the horse's own-trip speed deviation from its *trip-mean* pace,
   `δv(s) = v(s) − v̄`, normalised: `δ̂v(s) = δv(s)/v̄`.
2. Project onto the model's pace basis `φ(s) = (1 − 2·s/distance)` (the exact shape in
   `traffic_jax`): `pace_bias_meas = ⟨δ̂v, φ⟩ / (pace_gain · ⟨φ, φ⟩)`, weighted by
   `quality`. This is a one-line least-squares read of the same tilt the simulator
   applies — so a measured `pace_bias_meas` is **drop-in** for the `pace_bias` arg of
   `finish_times`, replacing the `rel_early`-derived value. (Center per race so the
   field's mean bias is ~0, matching the simulator's convention.)
3. The trailing per-horse mean of `pace_bias_meas` from *prior* runs (same
   walk-forward discipline as `horse_energetics_trailing`) is the forecast-time
   `pace_bias` prior.

**Validate energy-neutral:** with `power_profile`/`w_bal_profile` in hand, compute
the *actual* work timing: `pace_energy_resid = ∫ (P(s) − P̄)·φ(s) ds / ∫ P̄·ds`.
Energy-neutrality predicts the *shape* moves but `∫P dt` over the trip is conserved;
if `pace_energy_resid` is systematically ≠ 0 (e.g. front-runners spend *net more*
total energy, not just earlier), the simulator's zero-mean assumption is wrong and we
either (a) extend the pace basis with a non-zero-mean component coupling to `W′`
spend, or (b) re-center. This is the falsification test the current code cannot run.

### 2.3 Traffic model — calibrate `follow_gap`, `lane_band`, `lane_change_prob`, `lane_step`

**Current:** these four are hand-set constants in `traffic.py` / `traffic_jax.py`
(`follow_gap=6.0`, `lane_band=1.5`, `lane_change_prob=0.3`, `lane_step=1.2`). Blocking
and lane-changes are *emergent* but never validated against observed traffic.

**Upgrade — direct calibration from observed following/blocking/lane-changes:**

- **`follow_gap` (m, min same-lane following distance):** from all sample pairs where
  horse *i* is directly behind *j* in the same lateral band and *i*'s speed is capped
  near *j*'s (`v_i ≈ v_j`, `a_i ≤ 0`), take the distribution of the `s`-gap `s_j − s_i`.
  `follow_gap` ← a low quantile (e.g. 10th–20th pct) of sustained-following gaps — the
  closest horses actually sit. (`lead_gap`, `n_blocked`, `t_blocked` in
  `trajectory_features` are the per-run reductions.)
- **`lane_band` (m, "same lane" half-width):** the lateral `|off_i − off_j|`
  separation below which a trailing horse is actually slowed by the one ahead
  (estimate as the offset gap at which the `v`-capping effect vanishes — a changepoint
  in capped-speed vs lateral separation).
- **`lane_change_prob` (per-step):** observed rate of outward lateral excursions while
  blocked. Count blocked spells (capped behind a same-lane horse) and the fraction
  that initiate an outward move within the next step window; convert to a per-`dt`
  probability. (`n_lane_changes` is the per-run count.)
- **`lane_step` (m per change):** the median magnitude of an outward lateral excursion
  per change event.

Fit these **per venue × course** (rail position changes usable `width`, hence lane
geometry) and per surface. They are nuisance constants today; the trajectory turns
them into measured quantities, removing four hand-tuned knobs. The simulator code is
unchanged — only the default values become data-derived (carry them in a calibrated
dict beside `CALIBRATED_BEND_CURVATURE`).

### 2.4 Geometry — calibrate `bend_curvature` directly from observed ground-loss

**Current (`calibration.py`):** `bend_curvature` (κ) is recovered *indirectly* by the
Bayesian winner-likelihood method — simulate which draw wins over a κ grid, take the
posterior (`bend_loglik` / `bend_posterior` / `hierarchical_bend`), backed by a
moment-match to the empirical mean-finish-vs-draw curve. HV rests on the moment-match
alone (winner-likelihood underpowered: 98 races). Values:
`{"HV": 0.0023, "ST": 0.0037}`. The module docstring *explicitly anticipates this
upgrade*: "the `bend_curvature` likelihood is deliberately factored ... so finer data
(true lateral lane trajectories ...) can later sharpen the *same* posterior."

**Upgrade — direct geometric observation:** κ is *defined* by
`ground_factor = 1 + off·κ` on a bend, i.e. extra path per rail-metre per metre wide.
The trajectory measures both sides:

```
κ̂ = argmin_κ  Σ_samples on bends  q · ( d(path_len) − (1 + off·κ)·ds )²
```

i.e. regress measured incremental **own-path** length (`d(path_len) = √(dx²+dy²)`)
against `ds·(1 + off)` on bend samples; the slope on the `off·ds` term **is** κ. Even
more directly, the geometric `∮ off·dθ` on each turn (`dθ` from the world-frame
heading change) gives `bend_ground_lost` independent of any κ assumption, and
`κ = bend_ground_lost / ∫ off·ds`.

This produces a **per-venue κ with a real standard error from a single race-day** and
**directly resolves the HV under-determination** — HV need no longer rest on the
moment-match. Two ways to fold in:

1. *Replace* the prior in `bend_posterior` with the trajectory-measured κ as a tight,
   informative likelihood term (the docstring's intended path — "sharpen the same
   posterior without changing the machinery"); the winner-likelihood becomes a
   cross-check, not the sole evidence.
2. *Validate* the current `CALIBRATED_BEND_CURVATURE` values against the directly
   measured κ; agreement is strong corroboration, disagreement flags model error
   (e.g. the two-turn ST/AWT bend-boundary *reconstruction* in `track.py`, which is
   noted as approximate — trajectories also let us refit `bend_segments` from where
   `dθ/ds` is actually nonzero).

### 2.5 Drafting / drag — calibrate the `draft` term from observed slipstream positions

**Current (`dynamics.py`):** drafting is a binary `draft_relief ∈ {0, draft}` —
relief `draft=0.4` applied iff *any* horse is within `follow_gap` ahead in the lane
(`drafting = blocking.any(axis=2)`), reducing drag `½ρ·cda·v²·(1−draft)`. Both the
*magnitude* `draft` and the *all-or-nothing* gating are guesses; `cda=0.5` is a guess.

**Upgrade — observed cover → graded draft fraction:** the trajectory gives, per
sample, the exact geometry of cover: longitudinal gap and lateral overlap to the
nearest horse(s) ahead. Define a measured `draft_frac(t) ∈ [0,1]`
(`draft_frac_profile`) from that geometry (decaying with gap, peaking when directly
behind), and calibrate the drag model two ways:

1. *Magnitude.* On the **straights** (no bend term, `on_bend=False`), the EOM tangential
   balance reduces to `m·a = P/v − K_LOCO·m·g − ½ρ·cda·v²·(1−draft_frac)`. Across many
   samples — covered vs in-clear at matched `v` — the deceleration/coast difference
   identifies `cda` and the slipstream relief: regress measured `(P/v − K_LOCO·m·g − m·a)`
   (= drag force) on `½ρ·v²` and on `½ρ·v²·draft_frac`; the first slope is `cda`, the
   second `cda·draft`. This is a real force-balance estimate of both constants.
2. *Shape (gating).* Fit `draft_frac` as a function of (longitudinal gap, lateral
   offset) — recovering how relief decays with distance behind and falls off laterally
   — replacing the binary `blocking.any()` gate with a measured kernel. The EOM then
   takes a continuous `draft_frac(t)` instead of `{0, 0.4}`; with `cda→0` or
   `draft_frac→0` it reduces to the pure energetics model (the existing reduces-to
   test in `dynamics.py`).

`draft_exposure = ∫ draft_frac dt` per run becomes a measured trip-quality feature
(energy saved under cover) — coupling through the drag term exactly as
`MODEL_ENRICHMENT.md` requires (entry A↔B), never as a side covariate.

---

## 3. Migration path (sectional → trajectory, run both in parallel)

The transition is **additive and reversible**: trajectory estimators write the *same
column names* the EB/calibration layers already consume, so the downstream stack
(`Race.traffic_model`, `calibration.fit_energetics_population`,
`CALIBRATED_BEND_CURVATURE`) is untouched and either source can drive it.

**Step 0 — land the data, validate the feed (the gate).**
Build the trajectory `DataSource` (the V1 RF/GPS position feed; CV fallback per the
research plan), writing `trajectory_dataset.parquet`. Before any modelling, run the
**self-consistency check**: reconstruct `section_time[]` from the trajectory and
compare to HKJC's **published sectionals** (`get_sectional_times`, the
`sectional_dataset`). Agreement within feed noise validates the homography/positions;
systematic offset flags a calibration bug. This is the cheapest, highest-leverage
acceptance test and must pass before trusting any derived κ/draft.

**Step 1 — derive `trajectory_features.parquet` in parallel, keep sectionals.**
Compute `v_c_traj, w_prime_traj, pace_bias_meas, bend_ground_lost`, etc. **alongside**
the existing sectional `(v_c, w_prime, rel_early)` — do not delete the sectional
pipeline. Both populate parallel columns. The sectional path remains the fallback for
races with no/low-`quality` trajectory (older races, RF dropouts).

**Step 2 — cross-validate engine parameters.**
Compare `v_c_traj` vs sectional `v_c` per run and after EB pooling. Expect tighter
per-run scatter and lower EB `σ²_within` from trajectories. **Keep** the EB layer
(`fit_random_effects`, `horse_energetics`, trailing variant) unchanged — it now
shrinks less because runs are more informative. Validate that walk-forward
out-of-sample win/place skill is *no worse* (this is the real test, per OQ2 in the
research plan: is the market already efficient w.r.t. denser data?). Promote
trajectory `v_c`/`W′` to the primary `energetics_dataset` only if it wins or ties OOS.

**Step 3 — cross-validate geometry, then promote κ.**
Compare directly-measured κ̂ (port 2.4) against `CALIBRATED_BEND_CURVATURE`
(`{"HV":0.0023,"ST":0.0037}`). On agreement, fold κ̂ in as the informative term in
`bend_posterior` (the path the docstring reserved) — **resolving HV's
under-determination** (winner-likelihood underpowered there). On disagreement,
investigate `bend_segments`/reconstruction before trusting either. Refit
`bend_segments` from measured `dθ/ds` for ST/AWT (boundaries are approximate today).

**Step 4 — promote pace and traffic constants.**
Swap `rel_early`-derived `pace_bias` for `pace_bias_meas` (trailing, leakage-safe),
and run the energy-neutrality test (port 2.2); adjust the pace basis if
`pace_energy_resid` is non-zero. Replace the four hand-set traffic constants with the
per-venue×course calibrated values (port 2.3), carried in a dict beside
`CALIBRATED_BEND_CURVATURE`. Each swap is gated on OOS skill being preserved.

**Step 5 — couple drafting in the EOM.**
With `cda` and the `draft` kernel measured on straights (port 2.5), switch the
`DynamicsRaceModel` from `{0,0.4}` binary relief to the measured continuous
`draft_frac`. Verify the reduces-to test (`cda→0` recovers energetics).

**What to keep throughout.**
- The EB / partial-pooling layer (`calibration.py`) — still essential for thin-history
  horses; trajectory just lowers its variance components.
- The `loo`/trailing leakage discipline — trajectory features are equally leak-prone
  (a run's own κ/pace must not predict that run); reuse the exact walk-forward wrappers.
- The factored `bend_curvature` likelihood — the trajectory plugs into it, as designed.
- The sectional pipeline as a fallback while trajectory coverage and `quality` are
  incomplete; both run in parallel until Step 2–4 OOS checks promote trajectory to
  primary per port.

**What to validate (acceptance gates, in order).**
1. trajectory `section_time[]` ≈ HKJC published sectionals (feed sanity);
2. `total_time` ≈ results `FinishTime`;
3. directly-measured κ̂ ≈ calibrated `bend_curvature` (geometry sanity);
4. `pace_energy_resid` ≈ 0 (or fix the pace basis) — energy-neutrality;
5. reduces-to tests: pace_gain→0, draft→0, even-pace v_c → prior model;
6. **OOS win/place skill not worse** than the sectional pipeline — the only gate that
   authorises retiring a sectional port.

Every promotion is reversible: until gate 6 passes for a port, the sectional value
remains the default and the trajectory value is shadow-computed.
