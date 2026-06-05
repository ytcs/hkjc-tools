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
- `hkjc.model` — the ABM: `eom` (single canonical JAX power-balance EOM, mass-aware),
  `fit.fit_predict_traffic` (leak-free EB ability → win probs), `features` (leak-free
  form/jockey/style), `calibration`, `constants` (the baked gains).
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

Authoritative state: `journal/state/MODEL_STATE.md` (constants + the open residual queue).
Current: FORM_GAIN=4.0, SPEED_NOISE=0.12, JOCKEY_GAIN=5.0, bend_curvature ST=0.0037/HV=0.0023;
fit ≈ 48% (ST) / 43% (HV) of market.

**Open residual queue (the modelling agent's worklist, by |t|):**
1. `draw` OVER-weighted, t=−17.4 → recalibrate `bend_curvature` (cleanest single-constant fix).
2. `w_prime` UNDER-weighted, t=+12.9 → amplify the reserve channel (carefully; W′ is noisy).
3. `bodywt`/mass UNDER-weighted, t=+9.0 → wire the mass term (now in the EOM) into production.

The method is the loop: **diagnose** (market-residual regression, race-clustered SEs) →
**recalibrate** (a mechanism-grounded constant) → **bake** (`hkjc.model.constants`) →
**re-diagnose**. Verify every change improves OOS % of market *and* ECE, not just in-sample.
