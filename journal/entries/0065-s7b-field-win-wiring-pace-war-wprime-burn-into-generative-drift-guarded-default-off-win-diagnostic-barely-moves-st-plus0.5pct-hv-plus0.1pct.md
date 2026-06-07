---
title: "S7b WIN-WIRING of the FIELD/DEPLOYMENT cause (the gated next step after 0063): wire the front-group pace-WAR W′-burn into the GENERATIVE win drift as a per-runner reduction of effective anaerobic reserve, GUARDED behind ``GenerativeRaceModel(field_pace_war=)`` defaulting OFF (OFF reproduces the physiology-only model BYTE-IDENTICALLY — the 181 existing tests stay green). Built ``field.pace_war_burn_per_runner`` (the per-runner companion of ``pace_war_gradient``: each contested front-runner held above its own cruise ``base_i`` at the front pace ``L`` spends reserve ``δv_i·t_early`` [m] over the opening window, gated by the front-count pace-war threshold — pure EOM force-balance physics, no outcome-fitted knob), carried it leak-free on ``RaceConditions.pace_war_burn``, and reduced ``W′_eff = max(W′ − burn, 0)`` where the reserve boost (1+W′/L) enters ``generative._drift_times`` ⇒ a front-runner in a hot, front-heavy field carries LESS reserve to the finish ⇒ its drift slows. CONSOLIDATED the forwardness orientation onto the corrected sign (0063): forwardness = 1 − ``trail_early_pos_norm`` (high = closer), retiring the un-inverted feed in ``equilibrium_pace.py`` / ``field_pace_ceiling.py``. MEASURE (``studies/field_win_wiring.py``, paired fixed-seed, leak-free assert_no_lookahead drift=0 both venues): the WIN-diagnostic %market delta physiology-only (OFF) vs +FIELD (ON). VERDICT: the field dimension BARELY moves the win diagnostic — ST 7.98%→8.46% (Δ+0.48%), HV 9.45%→9.54% (Δ+0.10%), ECE ~flat both; the pace war fires in 44–49% of test races. HONEST: a real-but-small causal dimension that barely moves win is a legitimate finding (WIN is the DIAGNOSTIC of cumulative causal coverage, 0060, not the target). SIDE-FINDING: the 0063 orientation fix, applied to 0061's equilibrium-pace study, flips its HV 'BEATS' to MIXED (RMSE better, corr flat) — 0061 was on the wrong sign; the 0063 mechanism |ρ| (0.026 ST / 0.213 HV) is unchanged (it already used the corrected forwardness). No regression, no win-fitted constant; leak-free / strictly-prior."
id: 0065-s7b-field-win-wiring-pace-war-wprime-burn-into-generative-drift-guarded-default-off-win-diagnostic-barely-moves-st-plus0.5pct-hv-plus0.1pct
author: modelling
audience: ["modelling"]
channel: research
kind: result
status: open
thread: 0061-s7-field-deployment-cause-step1-mean-field-equilibrium-pace-beats-naive-at-hv-not-st
tags: ["s7b", "field-coupling", "deployment", "win-wiring", "win-drift", "generative-drift", "win-diagnostic", "win-as-diagnostic", "causal-coverage", "pace-war", "wprime-burn", "w-prime", "effective-reserve", "reserve-reduction", "critical-power", "over-spend", "front-runner", "front-count", "threshold", "threshold-coupling", "guarded", "default-off", "additive", "byte-identical", "acceptance-gate", "orientation-fix", "forwardness", "trail-early-pos-norm", "consolidate", "first-principles", "no-regression", "no-epicycle", "no-fitted-constant", "leak-free", "strictly-prior", "paired-seed", "percent-of-market", "ece", "venue-split", "small-delta", "honest-null", "proxy-to-win", "wave-2", "field-cause", "0049", "0060", "0061", "0062", "0063"]
refs: {"module": "src/hkjc/model/field.py (pace_war_burn_per_runner, PACE_WAR_EARLY_DIST — added; sole editor), src/hkjc/model/generative.py (_effective_w_prime + field_pace_war flag + _WPRIME_FLOOR_FRAC), src/hkjc/model/conditions.py (RaceConditions.pace_war_burn)", "study": "studies/field_win_wiring.py", "test": "tests/test_field.py (+3 per-runner burn tests), tests/test_model_skeleton.py (+2 guard tests: OFF byte-identical, ON slows burnt runners)", "consolidated": "studies/equilibrium_pace.py + studies/field_pace_ceiling.py + studies/field_gradient_coupling.py (forwardness = 1 − trail_early_pos_norm)", "thread_journal": "journal/entries/0063-s7b-respec-field-deployment-coupling-pace-war-wprime-burn-threshold-gradient-mechanism-recovers-hv-composition-signal-st-noise-floor.md", "diagnostic_journal": "journal/entries/0062-s7b-field-pace-ceiling-mis-spec-vs-null-st-s1-pace-is-a-true-null-the-missed-effect-is-in-the-early-pace-gradient.md", "reorientation": "journal/entries/0060-reorientation-win-diagnostic-not-veto-wave2-expand-causal-network.md", "gate": "src/hkjc/model/gate.py (gate_score), studies/pool_book_gate.py (win path)", "theory": "docs/THEORY.md §III.4(a) (mean-field game of pace), §III.2 (critical-power / W′ energy law, eq.3)", "data_market": "data_cache/wpq_dataset_full.parquet", "data_energetics": "data_cache/energetics_dataset_full.parquet", "data_pace": "data_cache/trajectory_dataset_full.parquet", "data_style": "data_cache/run_policy_dataset.parquet"}
created: 2026-06-06T00:00:00+00:00
updated: 2026-06-06T00:00:00+00:00
---
S7b — the WIN-WIRING step the field/deployment thread (0061→0063) deferred as the separate, parked
arbiter. 0063 built + validated the front-group pace-WAR W′-burn GRADIENT mechanism on the PACE proxy
(HV recovers the de-confounded composition signal |ρ|=0.213; ST a ceiling-checked TRUE NULL) but did
NOT wire it into the generative WIN drift, because pace is a proxy and WIN is the parked diagnostic of
cumulative causal coverage (0060; PHILOSOPHY §6). THIS entry wires it and measures whether adding the
2nd causal dimension (after physiology, 0050–0059) MOVES the win diagnostic. No win-fitted constant;
guarded; leak-free.

## THE WIRING (causal, leak-free, GUARDED / additive)
The pace war makes contesting front-runners spend anaerobic reserve EARLY, so they carry LESS reserve
into the finish. Three pieces:
- **``field.pace_war_burn_per_runner(base, style)``** — the PER-RUNNER companion of 0063's
  ``pace_war_gradient``. Among genuine front-runners (forwardness ``style ≥ FRONT_STYLE_CUTOFF``) the
  lead is contested at the front pace ``L`` (the fastest front-runner's own cruise); each front-runner
  held ABOVE its own sustainable opening cruise ``base_i`` runs above CP and spends the over-pace
  ``δv_i = ΔP_i⁺/(dP/dv|_L)`` [m/s] (``ΔP_i⁺ = [P(L)−P(base_i)]⁺``, the EOM force-balance over-spend at
  L, eq.3 — pure resistance physics). The RESERVE it depletes over the opening window is
  ``burn_i = g(n_front)·δv_i·t_early`` [m], ``t_early = PACE_WAR_EARLY_DIST/L`` — the SAME
  distance-reserve units the model's W′ carries (``Σ(v−v_c)·t``, ``energetics.recover_energetics``),
  so it subtracts directly off W′. Gated on the front-count pace-WAR threshold (``front_war_gate``):
  zero for a lone/uncontested leader, a closer, or a single-speed front (all unit-tested).
- **``RaceConditions.pace_war_burn``** — the per-runner burn carried as a pre-race composition input
  (KNOWN before the race from the field's strictly-prior style/pace ⇒ leak-free), ordered like θ.
- **``generative._effective_w_prime``** — ``W′_eff = max(W′ − burn, 0)`` (floor ``_WPRIME_FLOOR_FRAC``
  = 0, no negative reserve), applied where the reserve boost ``v_target = v_c·(1 + W′_eff/L)`` enters
  ``_drift_times``. A burnt front-runner reaches a slower effective target velocity ⇒ a LARGER μ ⇒ a
  lower win prob.

**GUARDED / byte-identical baseline (Acceptance Gate A).** The term is behind
``GenerativeRaceModel(field_pace_war=...)`` defaulting **OFF**. OFF returns W′ UNCHANGED even when a
burn is present on conditions ⇒ μ, the sampled orders, and every existing number are BYTE-IDENTICAL to
the physiology-only model. The win delta is measured ON vs OFF, paired fixed-seed (the model seeds on
race_id, so OFF and ON share the per-race MC seed and Δ is pure mechanism). Verified by a focused
guard test: OFF == baseline (with and without a burn on conditions); ON slows exactly the burnt
runners, monotone in the burn, leaving un-burnt runners byte-identical.

## ORIENTATION FIX consolidated (0063 → here)
``run_policy_dataset.trail_early_pos_norm`` is the strictly-prior trailing position RANK, POSITIVELY
correlated with the section-1 position (1 = leader) ⇒ HIGH = a CLOSER, LOW = a front-runner — the
INVERSE of 0061's docstring. 0063 verified this and inverted inside its own study. Here the corrected
sign is CONSOLIDATED: forwardness = ``1 − trail_early_pos_norm`` is applied at the LOADER in
``equilibrium_pace.py`` and ``field_pace_ceiling.py`` (retiring the un-inverted feed), and
``field_gradient_coupling.py`` is de-double-inverted (the loader now does it once). ``field.py``'s
``equilibrium_pace``/``front_group_pace``/``pace_war_*`` always take the CORRECTED forwardness; the
docstring is repointed.

## MEASURE — the WIN-diagnostic delta (``studies/field_win_wiring.py``; paired fixed-seed; leak-free)
Metric = the exact win %market the gate uses (``gate.gate_score``: (uniform_LL − model_LL)/(uniform_LL
− market_LL)) + ECE. θ = path-energetics, strictly-prior, train-only EB population (the same honest
generator as 0053/pool_book_gate); the field term is the ONLY change ON vs OFF. Test window = the
latest 30% of dates; ``assert_no_lookahead`` drift = 0 both venues.

| venue | N races | runners | pace war fires | physiology-only %mkt | +FIELD %mkt | **Δ%mkt** | Δece |
|---|---|---|---|---|---|---|---|
| **ST** | 602 | 7745 | 267 (44.4%) | 7.98% | 8.46% | **+0.48%** | +0.0002 |
| **HV** | 403 | 4707 | 196 (48.6%) | 9.45% | 9.54% | **+0.10%** | +0.0005 |

(ECE: ST 0.0073→0.0074, HV 0.0029→0.0034 — calibration unchanged to 3 dp. win_LL: ST
2.5003→2.4977, HV 2.4202→2.4199 — a sliver sharper.)

## THE READ — honest (WIN is the diagnostic, not the target)
**The FIELD dimension BARELY moves the win diagnostic.** Both deltas are positive but tiny (ST +0.48%,
HV +0.10% of market), calibration is unchanged. This is a LEGITIMATE finding under the 0060
re-orientation: WIN diagnoses cumulative causal coverage, and a single 2nd causal dimension wired on
top of physiology is expected to add a small increment, not a jump — the all-dimension aggregate
(form, ~45–60%) integrates many causes. The mechanism is correct, leak-free, and first-principles; it
simply carries little WIN information ON TOP of the physiology θ at this stage.

Two honest wrinkles:
- **The venue shape INVERTS vs the pace proxy.** On the PACE observable (0063) HV carried the signal
  (|ρ|=0.213) and ST was a noise floor (0.026). At the WIN gate ST moves slightly MORE (+0.48% vs HV
  +0.10%). This is the load-bearing 0062/0063 caveat realised: pace is a PROXY, and pace-skill does
  not map cleanly to win-skill. The most likely reading is that the pace-war reserve burn perturbs the
  ORDER more where the physiology θ leaves more win headroom (ST's physiology floor 8.0% is lower than
  HV's 9.5%), and HV's tighter, more-congested win outcome already prices the front-crowd effect that
  the burn re-expresses. Neither delta is large enough to over-interpret.
- **The increment is within the range a re-seed could wobble.** +0.10–0.48% on 400–600 races is a real
  but marginal lift; the win step does NOT promote the field dimension as a strong win driver. The
  pace-validated HV gradient signal (0063) does NOT convert to a HV win lift — the parked arbiter the
  0062/0063 caveats flagged has now answered, and the answer is "small, and not where the pace proxy
  pointed."

## SIDE-FINDING — 0061's HV 'BEATS' was on the WRONG SIGN
Re-running 0061's equilibrium-pace study with the consolidated correct forwardness: HV flips from
"BEATS" (0061: corr +0.814→+0.821, RMSE 0.558→0.542) to MIXED (corr +0.814→+0.814 flat, RMSE
0.558→0.550 better, bias −0.130→−0.090). 0061's headline "HV beats the naive baseline" was partly an
artefact of the un-inverted style feed; the corrected equilibrium still improves HV RMSE/bias but no
longer lifts correlation. The 0063 mechanism |ρ| is UNCHANGED (0.026 ST / 0.213 HV) — it already used
the corrected forwardness (it double-inverted before; now the loader inverts once and it reads
directly), so the de-confounded gradient validation stands exactly. The ceiling-gap targets shifted
slightly (CEIL/PACE features now use corrected forwardness: ST gap 0.030→0.012, HV 0.057→0.046) but
the mechanism still meets/exceeds them.

## DISCIPLINE / ARTEFACTS
ADDED to ``src/hkjc/model/field.py`` (sole editor): ``pace_war_burn_per_runner`` + ``PACE_WAR_EARLY_DIST``.
``generative.py``: ``_effective_w_prime`` + the ``field_pace_war`` flag (default OFF) + ``_WPRIME_FLOOR_FRAC``
(the only drift change; gated). ``conditions.py``: ``RaceConditions.pace_war_burn`` (optional pre-race
input). NEW ``studies/field_win_wiring.py``. +3 per-runner burn tests in ``tests/test_field.py``, +2
guard tests in ``tests/test_model_skeleton.py`` (OFF byte-identical incl. with a burn present; ON
slows the burnt runners monotonically, un-burnt byte-identical). Consolidated the forwardness
orientation in ``equilibrium_pace.py``/``field_pace_ceiling.py``/``field_gradient_coupling.py``.
``dynamics.py``, the pricer, ``gate.py``, ``pure_mechanism.py``, ``identify.py`` UNTOUCHED;
``src/hkjc/vision/`` left to its concurrent editor. ``ruff check src/hkjc/model studies/field_win_wiring.py``
clean; ``PYTHONHASHSEED=0 pytest -q`` GREEN (186 passed = 181 baseline UNCHANGED + 5 new ⇒ the OFF
guard is intact). Leak-free: ``assert_no_lookahead`` drift = 0 both venues; θ/base/style strictly-prior,
the burn is a pre-race composition input, no outcome enters. No regression, no win/pool-fitted constant
— the burn magnitude is the fixed EOM force-balance physics, only ON/OFF + the existing physical
front-count threshold are knobs. Ran SYNCHRONOUSLY in the foreground.

## CAVEATS
- **WIN is the parked diagnostic, not the fitting target (PHILOSOPHY §6).** The +0.1–0.5% is a
  measurement of coverage, not a goodness-of-fit win to chase. Do NOT tune the threshold / early window
  / floor against the win delta (that would be the forbidden win-calibrated coupling) — they are
  physical readings; only ON/OFF is the promotion knob.
- **Proxy→win is the real lesson.** The pace-validated HV gradient (0063) does not convert to a HV win
  lift; the dimension is real on pace but near-marginal on win. The deeper field/deployment win signal
  likely needs the §III.4b LATERAL congestion game (CV / Stage 3), where a crowded front actually costs
  GROUND and TRIP — not just early reserve — which the longitudinal pace-war burn cannot capture.
- **The increment is small.** It is a positive, leak-free, first-principles 2nd-dimension contribution
  — kept ON is defensible, but it is not a step-change. The next causal dimension (control/CV lateral,
  0060's frontier) is where the larger uncaptured win coverage most plausibly sits.

## NEXT
The field/deployment dimension is now wired end-to-end (pace-validated + win-measured) and earns a
small, honest place. The win headroom is in the dimensions still unmodelled (0060): CONTROL (CV
lateral path — the §III.4b congestion game that would give the field cause its GROUND/TRIP teeth, not
just the early-reserve burn), PREP (training-load → on-the-day physiology), CONDITION (gait). The field
win-wiring confirms the 0060 thesis empirically: adding one causal dimension moves the diagnostic a
little, exactly as a one-dimension probe should.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
