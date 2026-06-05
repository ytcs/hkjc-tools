# CLAUDE.md — working guide for this repo

Physics-grounded HKJC horse-racing modelling + the agent scaffolding around it. Read this
first; it tells you how the repo is laid out, how to run things, and which of the three
agents to *become* for a given request.

## The one rule that governs the modelling work

**Fit the market first. Bet-skill (% of market) is the goodness-of-fit metric, not an edge
hunt.** The model is regularized by physics + game theory: every factor must couple as a
real mechanism (a term in the EOM or the policy), never an arbitrary regressor. Do **not**
use a "private info / unknowable" excuse to stop — there is always a next mechanism. The
edge/trading call belongs to the user, never to the model. Full statement:
`src/hkjc/agents/prompts/modelling_agent.md`.

## Layout (seven packages, one-way deps)

`core` ◄ `data` ◄ `model` ◄ `betting`, with `backtesting` + `agents` on top (nothing imports
them); `vision` is an opt-in CV subsystem.

- `hkjc.core` — contracts (`RaceModel`, `DataSource`), `Race`/`TrackGeometry`, `RaceSimulator`,
  and `hkjc_facts` (THE source for site quirks + betting rules — import it, don't hard-code).
- `hkjc.data` — scrapers + persistence (`ticks`, carryover ledger). Acquires only, no edge.
- `hkjc.model` — the production win model: `fit.fit_predict_traffic` (leak-free EB ability →
  win probs via the `wald` first-passage diffusion law + per-venue mean-field couplings),
  `features` (leak-free form/jockey/style/early-pos), `calibration` (EB population), `constants`
  (baked gains). The interacting agent-based sims (kinematic + power-balance EOM) were deprecated
  2026-06 — low-leverage for win; physics preserved in `docs/THEORY.md` (Appendix).
- `hkjc.betting` — `pools`, `carryover`, `clv`, and the `Strategy` seam.
- `hkjc.backtesting` — walk-forward + scoring (% of market, ECE) + **no-lookahead enforcement**
  (`AsOfView`, `assert_no_lookahead`).
- `hkjc.vision` — CV trajectory pipeline (deps in the `vision` group; lazy-loaded).
- `hkjc.agents` — the three prompts + the cross-session `Journal` + paper-trade `OpsLog`.

## Running things

```bash
uv sync --group bayes                              # JAX is required to run the model
uv run python -m pytest -q                         # tests (jax ones gated on the bayes group)
uv run ruff check src/hkjc tests studies           # lint (must stay clean)
uv run --group bayes python studies/backtest_st.py # ST walk-forward (% of market); VENUE=HV for HV
```

Studies are disposable scratch (`studies/`); the library is the durable part. Data panels live
in `data_cache/` (git-ignored). Never weaken a leak-free path — gate features with `AsOfView`
and run `assert_no_lookahead` before trusting a backtest number.

## The three agents — and which user request triggers which

Each agent is a role you adopt by loading its prompt: `hkjc.agents.prompts.load(<name>)`.
They share one journal (`hkjc.agents.Journal`) and talk across sessions via inboxes.

| If the user says… | Become | Prompt | Owns |
| --- | --- | --- | --- |
| "research/improve the model", "calibrate", "diagnose residuals", "/model" | **modelling** | `prompts.load("modelling")` | the ABM + calibration. Never bets. |
| "research a betting strategy", "size stakes", "jackpot/carryover", "/strategy" | **strategist** | `prompts.load("strategist")` | strategies on top of the model. Consumes the model; never edits it. |
| "go live", "paper-trade today's card", "watch the odds", "/trade" | **trading** | `prompts.load("trading")` | the live (paper) runtime. Executes the chosen strategy; feeds observations back. |

Separation of concerns is hard: the strategist and trading agent **consume** model
probabilities and never modify the model — if they suspect a miscalibration, they file a
journal entry addressed to `modelling` (`audience=["modelling"]`), they don't patch it.

On session start, an agent reads its inbox (`Journal.inbox("<name>")`) and its living state
(`hkjc.agents.state.read("<name>")`). It records findings to the journal
(`author=<name>, channel=...`) and updates its `*_STATE.md`.

## Where the modelling work stands

Authoritative state: `journal/state/MODEL_STATE.md` (constants table + the lever queue + the
refuted list — read it before any modelling work). Current production (`SIM=wald`, per-venue
calibration): fit ≈ **62.6% of market (ST) / 59.3% (HV)** (from ~48% at project start), model ECE
~0.005. The path past the mean-field plateau is **NEW DATA + STRUCTURAL signal**, proven by the
2026-06 fan-out: barrier trials (`TRIAL_GAIN`, debut/lightly-raced form), collateral-form
(`COLLAT_GAIN`, relational opponent-strength — a scalar Elo had failed), rating-peakgap
(`PEAKGAP_GAIN`, a process/career-landmark feature), trajectory fade (`FADE_GAIN`). Next: fold trials
(then training gallops) into the EB ability posterior coherently; geometry×control (THEORY §3c). The
interaction ABM was deprecated (physics in THEORY.md Appendix).

The method is the loop: **diagnose** (market-residual regression, race-clustered SEs) →
**recalibrate** (a mechanism-grounded constant) → **bake** (`hkjc.model.constants`) →
**re-diagnose**. The CONVERSION test (%market, confirmed at full scale with a PAIRED fixed-seed
backtest — `PYTHONHASHSEED=0` so base/candidate share MC seeds), NOT the residual t-stat, is the
gate; verify every change improves OOS % of market *and* ECE, not just in-sample.
