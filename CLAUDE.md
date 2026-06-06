# CLAUDE.md — working guide for this repo

Physics-grounded HKJC horse-racing modelling + the agent scaffolding around it. Read this
first; it tells you how the repo is laid out, how to run things, and which of the three
agents to *become* for a given request.

## Principles & method — read `docs/PHILOSOPHY.md` first

The governing principles and the calibration method live in **`docs/PHILOSOPHY.md`** (single
source). In one line: **fit the market first** — % of market is a goodness-of-fit metric, not an
edge hunt — regularized by physics + game theory (every factor a real mechanism, never an arbitrary
regressor), gated by the leak-free paired-fixed-seed conversion test, found via
diagnose → recalibrate → bake → re-diagnose. The edge/trading call is the user's, never the model's.

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

Separation of concerns (PHILOSOPHY §9): strategist and trading **consume** model probabilities and
never modify the model — a suspected miscalibration is a journal entry to `modelling`, not a patch.

On session start, an agent reads its inbox (`Journal.inbox("<name>")`) and its living state
(`hkjc.agents.state.read("<name>")`). It records findings to the journal
(`author=<name>, channel=...`) and updates its `*_STATE.md`.

## Where the modelling work stands

Authoritative state: `journal/state/MODEL_STATE.md` (constants table + the lever queue + the
refuted list — read it before any modelling work). Current production (`SIM=wald`, per-venue
calibration): fit ≈ **65.0% of market (ST) / 59.7% (HV)** (from ~48% at project start), model ECE
~0.006. The path past the mean-field plateau is **NEW DATA + STRUCTURAL signal** (PHILOSOPHY §8),
proven by the 2026-06 fan-out: barrier trials → the coherent precision **fusion** (`TRIAL_PI_RATIO`),
relational collateral-form (`COLLAT_CONSTRUCTION_BY_VENUE` — a scalar Elo had failed), rating-peakgap
(career-landmark process feature), fade/positional-fade on the **W′ knob** (`FADE_W_GAIN`), draw→**D**
trip variance (`DRAW_VAR_GAIN`). Next: fold training gallops into the same fusion; the §3e phase
levers; geometry×control via CV paths (THEORY §3c/§3e). The interaction ABM was deprecated (physics
in THEORY.md Appendix).

The method is the diagnose → recalibrate → bake → re-diagnose loop, gated by the leak-free
paired-fixed-seed conversion test — see **PHILOSOPHY §5 & §7** for the full statement.
