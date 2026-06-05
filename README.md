# hkjc-tools

A physics-grounded modelling stack for Hong Kong Jockey Club (HKJC) horse racing, and the
data + agent scaffolding around it. The modelling philosophy is **first-principle
regularization**: the race model is an agent-based simulation (a power-balance equation of
motion + a tactical policy), and every factor must couple as a real mechanism rather than an
arbitrary regressor. The goal is to *fit the market* — bet-skill (% of market) is the
goodness-of-fit metric — before any question of edge.

## Install

```bash
uv sync                  # core
uv sync --group bayes    # + JAX (the EOM / fitting; required to run the model)
uv sync --group vision   # + the heavy CV stack (opt-in; the trajectory pipeline)
```

Requires Python >= 3.11.

## Architecture

Seven packages, one-way dependency graph (each imports only from those below it;
`backtesting` and `agents` sit on top and nothing imports them):

```
   core  ◄──  data  ◄──  model  ◄──  betting
     ▲                      ▲            ▲
     └────────  backtesting ────────────┘
   vision ──► core          agents ──► everything (orchestration)
```

| Package | Purpose |
| --- | --- |
| `hkjc.core` | Shared contracts (`RaceModel`, `DataSource`), the canonical `Race` / `TrackGeometry`, the Monte-Carlo `RaceSimulator`, and `hkjc_facts` — the single source for HKJC site quirks + betting rules. |
| `hkjc.data` | Connectors/scrapers (live + historical), the racecard/sectional parsers, and the tick stores (`OddsTickStore`, `JackpotCaptureStore`) + carryover ledger. Acquires and persists only — computes no edge. |
| `hkjc.model` | The agent-based model: the single canonical power-balance EOM (`eom`/`dynamics`), the leak-free fit→predict pipeline (`fit`), the ability features (`features`), calibration, and the baked constants. |
| `hkjc.betting` | Price pools (`pools`), the carryover structural-edge tools, closing-line value (`clv`), and the `Strategy` seam (model probs → stakes). |
| `hkjc.backtesting` | Replay history, **enforce no-lookahead** (`AsOfView` + `assert_no_lookahead`), and score models/strategies (% of market, ECE, walk-forward, selection control). |
| `hkjc.vision` | The opt-in, heavy CV trajectory pipeline (lazy deps). |
| `hkjc.agents` | Orchestration: the three agent prompts (modelling / strategist / trading) and the cross-session `Journal` they communicate through, plus the paper-trade `OpsLog`. |

Docs (the field guides — kept authoritative):
- [`docs/HKJC_KNOWLEDGE.md`](docs/HKJC_KNOWLEDGE.md) — HKJC data sources, endpoints, gotchas.
- [`docs/HKJC_DATA_CATALOG.md`](docs/HKJC_DATA_CATALOG.md) — the modern `en-us` site catalog.
- [`docs/HKJC_BETTING_RULES.md`](docs/HKJC_BETTING_RULES.md) — pools, per-pool takeout, dividends.
- [`docs/CARRYOVER_RESEARCH_PLAN.md`](docs/CARRYOVER_RESEARCH_PLAN.md) / [`CARRYOVER_STRATEGY.md`](docs/CARRYOVER_STRATEGY.md) — the carryover structural-edge line.
- [`docs/TRAJECTORY_RESEARCH_PLAN.md`](docs/TRAJECTORY_RESEARCH_PLAN.md) — the video/physics line.
- [`REFACTOR_PLAN.md`](REFACTOR_PLAN.md) — the package design + migration record.

The machine-readable model state (calibration constants + the open residual queue) lives in
`journal/state/MODEL_STATE.md`.

## Quick start — simulate a race

```python
from hkjc.data.live import live_odds, _fetch_live_races
from hkjc.model import PlackettLuceModel
from hkjc.core import RaceSimulator

card = _fetch_live_races('', '')                   # '' = let the server pick today's card
odds = live_odds(card['Date'], card['Venue'], 1)   # {'WIN','PLA','QIN','QPL'}

model = PlackettLuceModel.from_odds(odds)          # any RaceModel plugs in
sim = RaceSimulator(model, n_samples=200_000)
sim.prob(sim.wins(2))                              # P(runner 3 wins); indices 0-based
sim.expected_profit(sim.wins(2), odds['WIN'][2])   # EV of a $1 win bet
```

## The model — fit ability, predict, score vs the market

```python
from hkjc.model.fit import fit_predict_traffic     # leak-free EB ability → win probs (JAX sim)
from hkjc.backtesting import walk_forward_folds, harness, assert_no_lookahead
```

A full walk-forward backtest (% of market) is in `studies/backtest_st.py`:

```bash
uv run --group bayes python studies/backtest_st.py            # Sha Tin
VENUE=HV uv run --group bayes python studies/backtest_st.py   # Happy Valley
```

## Capture odds, then measure closing-line value

```python
from hkjc.data.ticks import OddsTickStore
from hkjc.betting import clv

store = OddsTickStore('ticks')
store.capture_meeting('', '')           # poll every race once (run on a schedule)
ticks = store.read(pool='WIN')
clv.odds_drift(ticks)                   # open→close drift (negative = steam)
```

## Agents remember across sessions

```python
from hkjc.agents import Journal, prompts

j = Journal()                                       # journal/ (or $HKJC_JOURNAL)
j.record("draw still over-weighted", "t=-17.4 at FG=4",
         author="modelling", channel="research", tags=["residual", "draw"])
j.inbox("strategist")                               # what the strategist should read now
prompts.load("modelling")                           # the modelling agent's system prompt
```

## Live-data caveats (the hard-won ones; full guide in `docs/HKJC_KNOWLEDGE.md`)

The machine-readable mirror of these is `hkjc.core.hkjc_facts`:

- **Venue codes** are `ST` / `HV`; empty strings make the live endpoint return the current meeting.
- **Sectional pages only render with a `DD/MM/YYYY` date** — any other format returns an empty
  "released shortly" page (`hkjc_facts.SECTIONAL_DATE_FORMAT`).
- **Surface detection**: the "ALL WEATHER TRACK"/"AWT" label is authoritative; going codes are a
  weak fallback (GOOD/GD appear in both scales — must not trigger AWT).
- **Live-odds dedup** on the server's `lastUpdateTime`, not wall-clock.
- **Takeout is not uniform**: Win/Place/QIN/QPL ≈ 17.5%; exotics ≈ 19.5–25% (`hkjc_facts.takeout_for`).
- Historical results pages are server-rendered HTML; a non-race date returns an error page.
```
