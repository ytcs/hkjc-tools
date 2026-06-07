---
title: "S7 (FIELD/DEPLOYMENT cause, step 1): the MEAN-FIELD EQUILIBRIUM PACE — solve the early field pace as a self-consistent fixed point of the field's deployment interaction (leaders set the pace; closers pulled up toward a hot front by their own forwardness), from STRICTLY-PRIOR per-horse opening-pace + running-style only. Validation vs realised opening field pace, leak-free, both venues: it BEATS the naive field-mean-of-prior-paces baseline at HV (corr +0.814→+0.821, RMSE 0.558→0.542, bias −0.130→+0.002) but does NOT beat it at ST (corr +0.598→+0.577, RMSE 0.797→0.843). Honest split read: the interaction carries real predictive pace structure on the tight tactical HV track; on the big galloping ST track the strong per-horse prior already absorbs it and the chase adds bias. First validation of the dimension is PARTIAL (HV yes, ST no). No regression, no outcome-fitted constant."
id: 0061-s7-field-deployment-cause-step1-mean-field-equilibrium-pace-beats-naive-at-hv-not-st
author: modelling
audience: ["modelling"]
channel: research
kind: result
status: open
thread: 0049-mean-field-pace-draft-coupling-deconfounds-cruise-toward-hill-cp-closes-fade-hump-collateral-reemerges
tags: ["field-coupling", "deployment", "mean-field", "mean-field-game", "equilibrium-pace", "pace", "running-style", "front-runner", "closer", "fixed-point", "self-consistent", "leak-free", "prior-only", "predictive-pace", "naive-baseline", "venue-split", "hv-beats", "st-no", "wave-2", "field-cause", "first-principles", "no-regression", "no-fitted-constant", "0049", "0060"]
refs: {"module": "src/hkjc/model/field.py", "study": "studies/equilibrium_pace.py", "test": "tests/test_field.py", "eom_journal": "journal/entries/0049-mean-field-pace-draft-coupling-deconfounds-cruise-toward-hill-cp-closes-fade-hump-collateral-reemerges.md", "wave2": "journal/entries/0060-reorientation-win-diagnostic-not-veto-wave2-expand-causal-network.md", "theory": "docs/THEORY.md §III.4(a)", "data_style": "data_cache/run_policy_dataset.parquet", "data_pace": "data_cache/trajectory_dataset_full.parquet"}
created: 2026-06-06T00:00:00+00:00
updated: 2026-06-06T00:00:00+00:00
---
OPEN — S7, the FIELD/DEPLOYMENT causal dimension step 1 (0060's Wave-2 field cause). BUILDS + validates
the PRE-RACE pace predictor; does NOT wire it into the win drift (deliberate later step). NO market /
win / pool. NO regression: ``base``/``style`` are strictly-prior per-horse observables, the pace
formation is the "leaders set the pace" mechanism, and no coefficient is fitted to any outcome.

## THE DIRECTIVE
0049 conditioned the EOM on the OBSERVED field pace — a contemporaneous boundary condition, fine for
path RECONSTRUCTION but NOT usable to PREDICT a held-out race pre-race (0049 §5 named this as the next
structural step). S7 step 1: MODEL the early field pace as a self-consistent FIXED POINT of the
field's deployment interaction (THEORY §III.4a, the mean-field game of pace), so it is available from
PRIOR-race constants alone — a genuine causal mean-field object, not a regressed pace feature.

## (1) THE PER-HORSE PRIORS — strictly prior, leak-free
Each runner gets two strictly-prior observables (earlier runs only):
- ``base_i`` = own opening-pace tendency [m/s]: the expanding mean (shift-by-one) of its prior
  section-1 velocity from ``trajectory_dataset_full`` (the velocity it runs to its OWN preferred
  deployment, uncontested). Null on debut → field-neutral fallback.
- ``style_i`` ∈ [0,1] = forwardness: ``trail_early_pos_norm`` from ``run_policy_dataset`` (already the
  strictly-prior expanding mean of its early-position rank; 1 = front-runner, low = closer).

## (2) THE MEAN-FIELD GAME (``field.equilibrium_pace`` / ``field.front_group_pace``)
Large-field limit — rivals enter only through the aggregate FRONT pace:
- **Leaders set the pace.** The contested front velocity is the forwardness-weighted mean of the
  bases:  ``L = Σ s_j·b_j / Σ s_j``.
- **Best response.** A horse holds its field-relative position per its style: where the front is
  faster than its own base it is pulled UP toward ``L`` in proportion to its own forwardness (a
  forward horse contests a hot front; a deep closer is barely dragged along); a horse already faster
  than the front holds its base:  ``v_i = b_i + s_i·max(L − b_i, 0)``.
- **Pace formation.** The realised early field pace is the field-mean of the best responses:
  ``V* = mean_i v_i``.

Relational, not scalar (THEORY §III.4a): a field whose FORWARD horses are also its FAST horses runs
ABOVE the plain field-mean (the hot front pulls the closers up); a field of slow front-runners does
not. Bounded in ``[mean(base), max(base)]`` (chasing lifts toward a fast front, never creates speed
beyond the fastest base) and reduces to the plain field-mean of ``base`` for a style-uniform field —
the required null limit (unit-tested). No outcome-fitted knob.

## (3) WHY THE NAIVE COMPOSITION SIGNAL IS WEAK — the diagnosis that shaped the mechanism
A first attempt (front-weighted "chase the aggregate pace V" fixed point) LOST to the naive baseline
at both venues (pure upward bias). The diagnosis: against ``realised − naive`` (what the field-mean of
prior paces misses), the plain style-composition features carry NOTHING — corr(mean_style)≈−0.01/+0.07,
corr(n_frontrunners)≈−0.06/+0.05 (ST/HV). The ONE feature that carries the residual is the
forwardness-weighted base ``front_base`` (corr +0.19 ST / **+0.44 HV**): the field runs hot exactly
when the FORWARD horses are the FAST horses. That is the "leaders set the pace" mechanism above — and
it must pull via the front-group pace ``L``, not via a uniform style mean. (Distance dominates the raw
opening pace, corr −0.49; the per-horse ``base`` already carries it, which is why the naive baseline is
strong and the interaction must add on TOP of it.)

## (4) VALIDATION — predict the realised opening field pace (section-1 field-mean velocity), leak-free
Target = the race's realised section-1 field-mean velocity (the early/mid field pace). Baseline =
the plain field-mean of the prior individual opening paces ``base``. 3176 races with leak-free
coverage (≥5 runners, ≥½ the field with a prior opening pace).

| venue | n | NAIVE corr | EQUILIBRIUM corr | NAIVE RMSE | EQUILIBRIUM RMSE | NAIVE bias | EQUILIBRIUM bias | verdict |
|---|---|---|---|---|---|---|---|---|
| **ST** | 1899 | +0.598 | +0.577 | 0.7966 | 0.8429 | +0.121 | +0.270 | does NOT beat |
| **HV** | 1277 | +0.814 | **+0.821** | 0.5583 | **0.5421** | −0.130 | **+0.002** | **BEATS** |

## (5) THE READ — honest, venue-split
- **HV: the mean-field equilibrium BEATS the naive baseline** on all three (corr +0.814→+0.821, RMSE
  0.558→0.542, bias −0.130→+0.002 — the chase removes the naive under-prediction almost exactly). HV
  is the tight, turning, tactical track where the field bunches and the front group genuinely imposes
  its pace — the dimension carries real, leak-free, pre-race pace structure here, exactly where the
  ``front_base`` residual signal was strongest (+0.44).
- **ST: it does NOT beat** (corr −0.021, RMSE worse by 0.046, bias inflates to +0.27). On the big
  galloping ST track runners are more independent; the strong per-horse ``base`` prior already absorbs
  the pace, and the front-chase adds an upward bias the realised ST pace does not follow (its
  ``front_base`` residual signal was only +0.19). The mechanism is not wrong — it is correctly weak
  where the physics says the front imposes less.

**First validation of the FIELD/DEPLOYMENT dimension: PARTIAL — HV yes, ST no.** This is the expected
shape of a one-dimension causal probe (0060): it earns its keep where its physics dominates. The
honest caveat is that on a strong per-horse pace prior the composition interaction is a small,
venue-dependent increment, not a universal win.

## (6) WHAT REMAINS
- **ST lift correction.** The chase is bias-positive at ST; a venue-aware pull strength would be a
  fitted knob (FORBIDDEN). The right next move is a deeper mechanism — the realised-vs-front pull is a
  FUNCTION of field geometry/congestion (HV congests, ST does not), pointing at the §III.4b lateral
  congestion game (CV / Stage 3), not a tuned coefficient.
- **Early/mid pace, not just opening.** Validated on the section-1 field-mean; extending to a
  mid-race pace and feeding ``equilibrium_pace`` as the leak-free ``v_field`` into the 0049 coupled
  integrator (replacing the observed pace) is the path to a predictive win wiring — the DELIBERATE
  later step, not done here.
- **DRAFT_FRAC / pace-follow magnitudes** unchanged; physiological constants FIXED — the equilibrium
  perturbs the DEPLOYMENT input, not the constants (0047/0048/0049 discipline).

## DISCIPLINE / ARTEFACTS
NEW ``studies/equilibrium_pace.py``; ADDED ``field.equilibrium_pace`` + ``field.front_group_pace`` to
``src/hkjc/model/field.py`` (its only editor); 4 focused tests in ``tests/test_field.py`` (style-uniform
null limit; hot front lifts above mean; bounded by fastest base; front-group weighting). ``generative.py``,
the pricer, ``gate.py``, ``pure_mechanism.py``, ``dynamics.py``, ``identify.py`` UNTOUCHED. Lint clean
(``ruff`` on field + study + test); ``PYTHONHASHSEED=0 pytest -q`` green (176 passed). DON'T-WIRE — this
step BUILDS + validates the pre-race pace predictor only; the win-drift wiring is the next step.
