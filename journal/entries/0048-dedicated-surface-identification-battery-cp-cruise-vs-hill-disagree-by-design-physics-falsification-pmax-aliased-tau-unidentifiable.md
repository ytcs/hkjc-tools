---
title: "Foundation 0048: the DEDICATED-SURFACE IDENTIFICATION BATTERY — identify each EOM constant where it DOMINATES and cross-check it independently. HEADLINE: CP from the route cruise and CP from the distance–time (Hill) slope DISAGREE by a systematic +1.3–1.5 m/s (cruise ABOVE Hill), correlated only +0.25–0.31 — a clean falsification pointing at the SAME missing pace/draft coupling 0047 named (the settled cruise is NOT aerobic-limited; horses cruise above CP on reserve, with the field). Hill CV is the ROBUSTLY-identified CP surface (R²=0.994, slope SE 0.36 m/s). P_max ALIASES across surfaces (opening transient 2× the sprint ceiling). τ is UNIDENTIFIABLE in-race (→0; confirms #52). W′_0 disagrees 2× (Hill 220 kJ vs spent 130 kJ). The single robust+consistent constant for the causal build is the Hill critical velocity."
id: 0048-dedicated-surface-identification-battery-cp-cruise-vs-hill-disagree-by-design-physics-falsification-pmax-aliased-tau-unidentifiable
author: modelling
audience: ["modelling"]
channel: research
kind: result
status: open
thread: 0047-eom-first-principles-generative-core-bang-cp-bang-physics-replaces-heuristic-level-bias-zero-opening-ramp-captured
tags: ["theory", "energetics", "eom", "identification", "identifiability", "dedicated-surface", "cross-check", "consistency", "critical-power", "critical-velocity", "hill", "distance-time", "cruise", "force-balance", "w-prime", "p-max", "tau", "recovery", "aliasing", "falsification", "pace-coupling", "draft", "re-emergence", "first-principles", "physics", "no-regression", "foundation", "draft"]
refs: {"module": "src/hkjc/statespace/identify.py", "study": "studies/identification_battery.py", "test": "tests/test_identify.py", "eom_core": "src/hkjc/statespace/dynamics.py", "eom_journal": "journal/entries/0047-*.md", "panel": "data_cache/trajectory_dataset_full.parquet", "mass_going": "data_cache/results_rich.parquet (DeclarHorseWt + ActWt + Going)", "plot": "studies/out/identification_cp_cruise_vs_hill.png"}
created: 2026-06-06T00:00:00+00:00
updated: 2026-06-06T00:00:00+00:00
---
DRAFT — DEDICATED-SURFACE IDENTIFICATION + CROSS-SURFACE CONSISTENCY of the EOM physical constants.
No market odds, no win gate, no pool book. NO regression of outcome on features — every estimator
INVERTS the SAME EOM physics (`hkjc.statespace.dynamics`, UNTOUCHED) on the surface where one
constant dominates. NEW additive module `hkjc.statespace.identify` + `studies/identification_battery.py`
+ `tests/test_identify.py`. `dynamics.py` logic, production `hkjc.model`, data_cache UNTOUCHED.

## THE DIRECTIVE
The EOM core (0047) infers (CP, W′_0, P_max, τ) by a SINGLE bounded-NLS fit of the whole section
path per run — which UNDER-identifies coupled constants (many (CP, W′_0, P_max) combinations fit the
aggregate path). Identify each constant on the DEDICATED surface where it DOMINATES, with a
dedicated metric, and CROSS-CHECK it measured multiple independent ways. Cross-surface CONSISTENCY
is the deepest validation that the physics is real (G from a falling apple == G from the Moon's
orbit); a DISAGREEMENT is a falsification that names the missing physics.

## THE BATTERY — per constant, the dedicated surface(s) and the cross-check

| constant | dedicated surface(s) | estimate (ST / HV) | identifiability | cross-surface consistency |
|---|---|---|---|---|
| **CP** | (a) route CRUISE force-balance root; (b) distance–time HILL slope | v_c(cruise) **16.5 / 16.2** m/s; CV(hill) **15.0 / 14.9** m/s | (b) Hill **TIGHT** (R²=0.994, slope SE 0.36 m/s, 73% SE<0.5); (a) cruise CP-pinned (±4% to settled-level choice) | **DISAGREE: bias +1.49 / +1.27 m/s (cruise ABOVE hill), corr +0.25/+0.31 (vel), +0.53/+0.67 (power)** |
| **P_max** | (a) opening gate transient; (b) sprint ceiling | open **66.5 / 63.7** kW (117/114 W/kg); ceiling **33.2 / 31.7** kW (58/56 W/kg) | each surface internally ranks horses (corr +0.75/+0.78) but the LEVELS **ALIAS** | **DISAGREE 2×: opening ≈ 2× the ceiling** (bias +35/+36 kW) |
| **W′_0** | (a) Hill intercept D'→J; (b) reserve spent above CP | Hill **220 / 226** kJ; spent **137 / 113** kJ | weak (corr +0.12/+0.12 per horse) | **DISAGREE ~2×** (bias +95/+98 kJ) |
| **τ** | sit-and-kick closers (settle below v_c, then kick) | τ → **~1 s** (both venues) | **UNIDENTIFIABLE in-race** (collapses to 0) | n/a — confirms #52: in-race recovery negligible |
| **C_dA** | sprint ceiling (given P_max); lead-vs-draft | (sensitivity only; draft needs CV) | aliased with P_max at the ceiling | DEFERRED to CV (lateral) |

Coverage: ST 24 433 runs (2 554 route ≥1800m, 10 375 sprint ≤1200m); HV 14 823 (1 745 / 8 490).
CP_cruise: 661/506 horses with ≥1 route run; Hill: 1 194/316 horses with ≥3 distinct distances;
**527/227 horses have BOTH** (the headline n). Median Hill span 450 m, 3 distinct distances.

## (1) THE HEADLINE — CP_cruise vs CP_hill: a CLEAN, SYSTEMATIC DISAGREEMENT (the decisive test)
The two independent CP measurements do **NOT agree**. In velocity space (apples-to-apples — both
are the critical velocity):

| | v_c from route cruise | CV from Hill slope | bias (cruise−hill) | MAD | corr | 95% LoA |
|---|---|---|---|---|---|---|
| **ST** (n=527) | 16.46 m/s | 15.00 m/s | **+1.49 m/s** | 1.49 | +0.25 | [+0.21,+2.76] |
| **HV** (n=227) | 16.19 m/s | 14.94 m/s | **+1.27 m/s** | 1.27 | +0.31 | [+0.42,+2.13] |

The cruise-derived critical velocity sits **~1.3–1.5 m/s ABOVE** the Hill critical velocity,
consistently, both venues, and the correlation is only weak-positive (+0.25–0.31). This is **not**
a noise problem: the Hill fit is essentially exact (R²=0.994 median, slope SE 0.36 m/s, 95% of
horses SE<1.0 m/s) and the cruise read is CP-pinned (only ±4% sensitivity to taking the min vs the
median interior section). Both surfaces are well-measured; they **measure different numbers**.

**The physics the disagreement implicates (a falsification, as designed).** `CP_cruise > CP_hill`
means: *the settled mid-race cruise is NOT the aerobic (CP-limited) steady state.* Even taking the
SLOWEST interior section (the sustainable floor), HK horses cruise ~1.4 m/s above their across-
distance critical velocity — they are running the settled phase **above CP, spending anaerobic
reserve, carried by the field's pace**, not on the isolated minimum-time singular arc. This is the
*same* missing mechanism 0047 named from the closing fade-hump (the §III.4 mean-field pace / draft
coupling): a single-horse EOM treats cruise as the bare aerobic equilibrium, but real horses cruise
hot with the pack. The Hill slope, which integrates a horse's true distance–time envelope ACROSS
races, escapes the per-race pace inflation and recovers the genuine aerobic critical velocity
(15.0 m/s ST — landing exactly on the equine 16–17 m/s race-cruise target's lower aerobic edge, and
on the 0047 note that "the inversion's v_c is the BARE sustainable ceiling; observed cruise runs
above it on reserve"). **So: CP_hill (the distance–time slope) is the trustworthy aerobic CP; the
cruise read is contaminated by the pace coupling and reads ~1.4 m/s too hot.** The two-measurements-
must-agree test FAILS, and the failure is informative: it quantifies the pace inflation of cruise
(+1.4 m/s) and confirms the missing mechanism is the field coupling, not a constant.

## (2) P_max — ALIASED across surfaces (opening transient ≈ 2× the sprint ceiling)
The opening-transient P_max (gate acceleration, P_max/m vs low-v drag, W′ untouched) reads **66.5
kW (117 W/kg)** ST; the sprint-ceiling P_max (force balance at the peak sustained velocity) reads
**33.2 kW (58 W/kg)** — a **2× disagreement**. Each surface RANKS horses consistently (per-horse
corr +0.75/+0.78) but the absolute LEVELS alias hard. The physics: the opening accel is a true
*peak* instantaneous power (a standing-start lunge demands large dv/dt ⇒ large P), while the
sprint-ceiling is a *sustained* power at a window-mean velocity that is already drag-limited and
below the instantaneous ceiling — they measure peak-instantaneous vs sustained-peak, which are NOT
the same quantity. P_max as a single number is therefore **NOT cross-surface identified**; it needs
to be split into P_peak (transient) and P_sprint (sustained), or pinned with the true instantaneous
ceiling from CV paths. The 117 W/kg opening figure is also inflated by the v_end≈2·v̄−v0 ramp
approximation and gate-timing noise — treat the opening surface as an upper bound.

## (3) W′_0 — DISAGREES ~2× (Hill intercept 220 kJ vs reserve-spent 130 kJ)
The Hill distance-capacity D' converted to Joules (D'·F_resist(CV)/η) reads **220/226 kJ**; the
reserve actually spent above CP_cruise (back-solved P(t) from the path, ∫(P−CP)⁺dt) reads
**137/113 kJ** — about half. Per-horse correlation is weak (+0.12). The disagreement is partly
mechanical (the two define W′ in different spaces — distance-capacity vs spent-energy) and partly
the SAME pace contamination: the spent-above-CP estimate uses CP_cruise, which is ~1.4 m/s too hot,
so the reserve "spent above CP" is under-counted (a hotter CP threshold sees less excess). W′_0 is
**not robustly identified** until CP is fixed. The EOM-joint W′_0 (0047: 225/246 kJ) sits at the
Hill end — consistent with the Hill intercept being the cleaner anchor.

## (4) τ — UNIDENTIFIABLE IN-RACE (collapses to ~1 s; confirms #52)
On the dedicated sit-and-kick surface (route runs that settled below v_c then kicked: 1 096/2 554
ST, 730/1 745 HV qualify), τ from the replenishment balance (reserve gained over the off-pace
window = reserve the kick spends) collapses to **~1 s** — i.e. the kick energy is far larger than
any plausible recovery could supply over the median 24.9 s off-pace window, so the algebra drives
τ→0. **The kick is fuelled from baseline W′_0, NOT from in-race recovery.** This is the dedicated
surface PINNING the answer NEGATIVELY: τ is not identifiable in-race because in-race recovery is
negligible at race intensities — exactly #52's finding, now confirmed on its own dominant surface.
The EOM's τ=120 s fixed assumption is harmless (recovery contributes nothing in-race); τ matters
only across-race (settling between efforts), which this longitudinal surface cannot see.

## (5) RE-EMERGENCE — the physically-measured CP weakly predicts held-out form
Lightweight, leak-free, descriptive (correlation only, no outcome/market):
- **Early-half CP vs held-out late-half settled cruise:** ST corr **+0.01** (null), HV **+0.22**.
  The per-race cruise level barely persists at ST — consistent with cruise being pace-driven (a
  field property), not a stable horse constant; it persists modestly at HV (smaller, more horse-
  determined fields).
- **CV(hill) vs mean settled cruise level:** ST **+0.39**, HV **+0.33** — the Hill critical velocity
  DOES track the horse's typical sustainable level (a real, persistent physiological signal),
  reinforcing that CV is the identifiable CP surface and cruise is the contaminated one.

## (6) THE VERDICT — what is trustworthy for the causal rebuild
**ROBUSTLY + CONSISTENTLY identified (trust it):** the **Hill critical velocity CV** (the
distance–time slope) — R²=0.994, slope SE 0.36 m/s, 95% of horses SE<1 m/s, and it re-emerges
(+0.33–0.39 with the sustainable level). It is the one EOM constant identified tightly on its
dedicated surface AND not contaminated by the pace coupling. **This is the CP anchor for the causal
build.** (n: 1 194 ST / 316 HV horses with ≥3 distances.)

**MEASURES INCONSISTENTLY → missing physics named:**
- **CP_cruise ≠ CP_hill (+1.4 m/s):** the settled cruise is NOT aerobic-limited; horses cruise above
  CP on reserve, WITH the field ⇒ the missing **mean-field pace / draft coupling** (THEORY §III.4),
  the SAME mechanism 0047's fade-hump implicated. The disagreement quantifies it (+1.4 m/s cruise
  inflation). Do NOT use the per-race cruise read as CP in the causal build — use Hill CV.
- **P_max aliased 2×:** split into peak-instantaneous (opening) vs sustained-sprint (ceiling); the
  single P_max is not cross-surface identified ⇒ needs the instantaneous ceiling from **CV paths**.
- **W′_0 disagrees ~2×:** not identified until CP is fixed; the Hill intercept (220 kJ) is the
  cleaner anchor (matches the EOM-joint 225/246 kJ), the spent-above-CP estimate is biased by the
  hot CP_cruise threshold.
- **τ unidentifiable in-race (→0):** confirmed; in-race recovery negligible (#52). Valid negative.
- **C_dA:** deferred to CV (the slipstream / lead-vs-draft delta is lateral).

**The deepest reading:** the EOM's per-horse constants are NOT all independently trustworthy from
the longitudinal path. CP (as Hill CV) is. The rest either alias (P_max, W′_0 — need CV / a fixed
CP) or are genuinely in-race-invisible (τ). And the CP cross-check FAILING in a structured,
quantified way (+1.4 m/s, both venues) is the strongest evidence yet that the dominant missing
physics is the field/pace coupling — not a mis-set constant. The causal rebuild should anchor on the
Hill critical velocity, treat the cruise–Hill gap as the pace-coupling observable, and defer P_max /
C_dA to the lateral (CV) surface.

## DISCIPLINE
NO REGRESSION anywhere: every estimate INVERTS the EOM force balance / energy law on its dominant
surface; the consistency checks are descriptive cross-sectional (Bland–Altman, correlation). Mass +
going KNOWN (never fitted). The re-emergence probe is leak-free (early→held-out late, correlation
only). No market / win / pool. NEW module `src/hkjc/statespace/identify.py` + study + 8-test unit
suite (synthetic EOM round-trips: Hill recovers CV to <0.5 m/s; cruise recovers CP to <20%; spent
>0; Bland–Altman). `dynamics.py`, `energetics_dynamic.py`, production `hkjc.model`, data_cache
UNTOUCHED. Lint clean; statespace+dynamics+identify tests green (28). DON'T-BAKE — diagnostic
identification foundation; no production constant changed. Plot:
studies/out/identification_cp_cruise_vs_hill.png (the CP_cruise-vs-CP_hill scatter, the headline).
