---
title: "REFACTOR 0051: the LEAN PURGE — DELETE the acausal additive-regression win model (fit.py / features.py / constants.py, ~111k) and FOLD hkjc.statespace into the canonical hkjc.model namespace. hkjc.model now IS the causal-physics generative model; there is no frozen fallback. Tests 190→171 green (−19 = the 3 deleted acausal test files), ruff clean. Docs/state/agent-prompts reconciled to the causal reality and the honest 0050 status (form does not yet re-emerge at the win gate). The journal is the rebuild trail."
id: 0051-lean-refactor-purge-acausal-model-fold-statespace-into-canonical-hkjc-model-namespace
author: modelling
audience: ["modelling", "strategist", "trading"]
channel: research
kind: refactor
status: final
thread: 0046-causal-rebuild-design-cause-not-effect-strip-metasignals-reemergence-validation
tags: ["refactor", "lean", "purge", "anti-regression", "acausal-delete", "namespace-fold", "statespace", "hkjc-model", "canonical-namespace", "causal", "causal-rebuild", "wald-constants", "constants-relocation", "docs-reconcile", "model-state", "claude-md", "agent-prompts", "model-design", "philosophy", "tests-green", "ruff-clean", "journal-as-trail", "0046", "0049", "0050", "queue-57"]
refs: {"deleted": ["src/hkjc/model/fit.py", "src/hkjc/model/features.py", "src/hkjc/model/constants.py"], "relocated": "src/hkjc/model/wald_constants.py", "folded_from": "hkjc.statespace (deleted package)", "folded_into": "src/hkjc/model/", "state": "journal/state/MODEL_STATE.md", "guide": "CLAUDE.md", "design": "docs/MODEL_DESIGN.md", "philosophy": "docs/PHILOSOPHY.md", "causal_model": "docs/CAUSAL_MODEL.md", "prompts": ["src/hkjc/agents/prompts/modelling_agent.md", "src/hkjc/agents/prompts/betting_strategist.md"], "payoff_journal": "journal/entries/0050-pure-mechanism-payoff-test-gof-falls-to-floor-form-does-not-reemerge-at-win-gate-weak-ability-signal.md"}
created: 2026-06-06T00:00:00+00:00
updated: 2026-06-06T00:00:00+00:00
---
FINAL — the LEAN PURGE that the causal pivot (thread 0046) earned. With the pure-mechanism payoff
test run and recorded (0050: the meta-signal-free model is honest and identifiable but does NOT yet
re-emerge form at the win gate), the user directed: keep the codebase lean, **DELETE** the acausal
regression model (not freeze it), and **abandon the `hkjc.statespace` namespace** — it is redundant
with `hkjc.model`, which is canonical. The journal is the full research trail; deleted scratch can
be rebuilt from it. This entry records what was removed, what was kept, and the new shape.

## (1) DELETED — the acausal additive-regression win model (~111k of source)

- **`src/hkjc/model/fit.py`** (39k) — the production additive-regression assembly
  (`fit_predict_traffic`): the EB ability → win-prob pipeline with the meta-signal couplings layered
  on. THE "regression dressed up as physics" the pivot rejected.
- **`src/hkjc/model/features.py`** (57k) — every meta-signal table: recency-form, jockey, collateral
  (Massey), rating-peakgap, trajectory-fade, barrier-trial, training-gallop, gate-break, speed-figure
  dispersion. These were CONSUMED as inputs; the causal thesis is they must RE-EMERGE as consequences.
- **`src/hkjc/model/constants.py`** (15k) — the baked regression-gain tables (FORM_GAIN, JOCKEY_GAIN,
  COLLAT_GAIN, TRIAL_PI_RATIO, …). All dead once `fit.py` went.
- **Tests** of the deleted code: `test_features.py`, `test_gallop_fusion.py`, `test_trial_fusion.py`.
- **25 obsolete studies** that imported the deleted modules (the acausal probes + the production-
  comparison gate studies: `backtest_st.py`, `statespace_*_gate.py`, `causal_reemergence.py`,
  `ability_collapse_probe.py`, the mass/allometric/peakgap/gallop/trial probes, `quick_eval.py`, …).
  Their RESULTS live in journals 0042–0050; the scripts depended on the now-deleted production baseline.

**There is no frozen fallback.** The prior plan (PHILOSOPHY §9, MODEL_STATE) framed the acausal model
as a deployable fallback kept until the causal model surpassed it. The user overrode that: lean beats
a safety net we can rebuild from the journal. Deleting it also enforces ANTI-REGRESSION structurally —
you cannot backslide into a model that no longer exists.

## (2) FOLDED — `hkjc.statespace` → `hkjc.model` (one canonical namespace)

The causal-physics package built alongside production (journals 0045–0050) was moved wholesale into
`hkjc.model` (no name collisions): `dynamics, identify, field, generative, pricer, gate, pipeline,
coupling, latent, conditions, causes, likelihood, pathfit, pure_mechanism`. `hkjc.statespace`
is gone. (`assembly.py` — the production-reproduction bridge that stamped production `v_c_eb` for the
byte-exact Acceptance Gate A: `production_theta_builder`/`make_production_pre_fit`/`attach_trailing_*`
— was DELETED rather than folded; with production gone it was dead code, zero live callers.) `hkjc.model` IS the causal-physics generative win model; `hkjc.model.__init__` re-exports the
full causal surface plus the retained substrate. All `hkjc.statespace.X` references across src/tests/
studies/docs were rewritten to `hkjc.model.X`; `docs/STATESPACE_DESIGN.md` → `docs/MODEL_DESIGN.md`.

## (3) KEPT — physics + honest references (NOT the regression model)

`energetics`, `energetics_dynamic` (critical-velocity / dynamic W'-balance engines); `calibration`,
`calibration_bayes` (EB band×rating population prior — the debut concession the pure build leans on);
`plackett_luce`, `harville_model`, `conditional_logit`, `baselines` (the Harville-on-odds, results-logit,
and market/uniform references the causal model is SCORED against — `core` builds PL, and PHILOSOPHY §8
keeps the market as the final yardstick). The handful of coupling constants the generative pricer still
reads (`DRAW_*`, `FIGVAR_*`, `DRAW_VAR_*`, `GATE_BREAK_*`) were relocated verbatim into
**`src/hkjc/model/wald_constants.py`**, which severs the last dependence on the deleted `constants.py`
and documents them as PROVISIONAL acausal debt (draw→geometry, figvar→D, gate-break→EOM transient).

## (4) VERIFICATION

- **Tests: 190 → 171, all green.** The −19 is exactly the three deleted acausal test files; nothing
  else regressed. The folded package imports cleanly (36 exports after dropping the dead assembly
  bridge); the leak-free `test_model_skeleton` and `test_pure_mechanism` (no-meta-signal-import
  guard) pass against the new layout.
- **`ruff check src/hkjc tests studies` clean.** `pyproject` per-file-ignores re-pointed to `model/`.
- No dangling references to the deleted modules remain in code (only historical provenance mentions).

## (5) DOCS / STATE reconciled to the causal reality + the honest 0050 status

`CLAUDE.md` (working guide), `journal/state/MODEL_STATE.md` (rewritten from the deleted-model constants
table to the causal architecture + identified physical constants + the 0050 verdict + the ranked
queue), `src/hkjc/__init__.py`, `docs/{PHILOSOPHY, CAUSAL_MODEL, MODEL_DESIGN, RESEARCH_PROGRAM,
THEORY}.md`, and the modelling/strategist agent prompts. Every status line states the truth: the GoF
fell to the floor, **form does not yet re-emerge at the win gate** (7%/4% of the gradient), and the
named next step is the ability-anchor **slope → de-confounded LEVEL** swap (0049's +0.33/+0.40 lever).
No doc claims the causal model matches the old 65% or that form re-emerges.

## (6) WHERE THIS LEAVES THE REBUILD

The repo is now a single lean causal-physics codebase with no acausal escape hatch. The next modelling
move is unchanged from 0050's diagnosis and now unobstructed: re-anchor `pure_mechanism` ability on the
de-confounded cruise LEVEL, rebuild a lean re-emergence eval (vs market/uniform — the production arm is
gone by design), and test whether form re-emerges. The fall is closed by adding causal STRUCTURE, never
by reviving the deleted meta-signals.
