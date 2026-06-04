# hkjc-tools

Library for scraping Hong Kong Jockey Club (HKJC) horse-racing data and running
basic betting analysis. Intended as the data + modelling layer for an automated
betting agent.

## Install

```bash
uv sync
```

Requires Python >= 3.11.

Docs:
- [`docs/HKJC_KNOWLEDGE.md`](docs/HKJC_KNOWLEDGE.md) — field guide to HKJC's data
  sources, endpoints, domain conventions, feature priors and gotchas.
- [`docs/HKJC_BETTING_RULES.md`](docs/HKJC_BETTING_RULES.md) — full pool catalogue,
  per-pool takeout, dividend mechanics, and jackpot/carryover special payouts.
- [`docs/AGENT_ARCHITECTURE.md`](docs/AGENT_ARCHITECTURE.md) — design for the
  closed-loop, self-improving research agent built on this bedrock.
- [`docs/CARRYOVER_RESEARCH_PLAN.md`](docs/CARRYOVER_RESEARCH_PLAN.md) — **the main
  research line**: the Jackpot Carryover Harvester (build map + research questions).
- [`docs/CARRYOVER_STRATEGY.md`](docs/CARRYOVER_STRATEGY.md) — strategy rationale,
  the structural-edge math, and why model-edge exotics are look-ahead traps.

## Architecture

The library is a **bedrock** for an automated betting agent, organised by concern
with a single direction of dependency. It provides data, a probabilistic engine,
an evaluation harness and a memory; it deliberately does **not** ship any fixed
betting strategy — the agent composes her own bets on top. See
[`docs/AGENT_ARCHITECTURE.md`](docs/AGENT_ARCHITECTURE.md) for the full design.

```
   data         ->   model   ->   sim          eval        ledger
   live/             harville_     RaceSimulator (judges     (durable,
   historical/       model +       (P/EV of ANY  models &    indexed
   speedpro/         RaceModel     bet you        strategies) findings)
   odds_ticks                      compose)
```

| Module | Layer | Purpose |
| --- | --- | --- |
| `hkjc.data.live` | data | Live race cards and live odds via the HKJC GraphQL endpoint. |
| `hkjc.data.historical` | data | Past race results and per-horse form, scraped from HKJC results pages. |
| `hkjc.data.speedpro` | data | SpeedPro energy/fitness figures and speed maps for the current meeting. |
| `hkjc.data.odds_ticks` | data | `OddsTickStore`: capture & persist the live-odds time series (Parquet). |
| `hkjc.data.processing` | data | Build historical datasets across a date range. |
| `hkjc.harville_model` | model | Fits latent horse strengths to odds via DP (Numba); yields a full place-probability matrix. The default odds→strengths fitter. |
| `hkjc.model` | model | `RaceModel` interface (swappable); `PlackettLuceModel` (Harville-on-odds baseline); `ConditionalLogit` results-trained win model (breaks the odds-fitting circularity) + market/uniform baselines. |
| `hkjc.simulation` | sim | `RaceSimulator`: draw finishing orders from any model and estimate the probability/EV of arbitrary bets. |
| `hkjc.eval` | eval | The harness: `clv` (closing-line value), `calibration` (log-loss/Brier/ECE/RPS), `walkforward` (folds + sealed `Lockbox`), `selection` (BH-FDR, Deflated Sharpe), `harness` (fit-on-train/score-on-test). |
| `hkjc.ledger` | memory | `Ledger`: durable, indexed Markdown journal of findings (`record` / `search` / `recent`). |
| `hkjc.features` | — | Polars feature expressions (rating gap, front-runner share, ...). |
| `hkjc.utils` | — | Parsing helpers and a generic Pareto filter. |

The old flat names (`hkjc.live`, `hkjc.historical`, `hkjc.speedpro`) remain
re-exported for convenience; the canonical paths are under `hkjc.data`.

### Bring your own model

`PlackettLuceModel` (fitted to odds via Harville) is only the default. Any object
implementing `RaceModel.sample_orders(size, rng) -> (size, n)` finishing orders
plugs straight into `RaceSimulator` — so the research agent can swap in a
finish-time model, a learned model, or anything else without touching the rest of
the stack.

## Quick start

```python
from hkjc.data.live import live_odds, _fetch_live_races
from hkjc.model import PlackettLuceModel
from hkjc.simulation import RaceSimulator

# Discover the current meeting (empty strings = let the server pick today's card).
card = _fetch_live_races('', '')
odds = live_odds(card['Date'], card['Venue'], 1)   # {'WIN','PLA','QIN','QPL'}

# Fit the default model to the market and simulate the race.
model = PlackettLuceModel.from_odds(odds)
sim = RaceSimulator(model, n_samples=200_000)

# Compose ANY bet yourself. Indices are 0-based (horse i == runner number i+1).
sim.prob(sim.wins(2))                          # P(runner 3 wins)
sim.prob(sim.wins(2) & sim.places(4, k=3))     # joint event, freely composed
sim.expected_profit(sim.wins(2), odds['WIN'][2])  # EV of a $1 win bet at its odds
sim.prob(sim.exact_order([2, 4, 0]))           # straight trifecta 3-5-1
sim.expected_return(my_payout_array)           # arbitrary parimutuel dividends
```

```python
from hkjc.data.historical import get_race_data, get_horse_data

results = get_race_data('2026/05/31', 'ST', 1)   # past race result table
form    = get_horse_data('L276')                  # one horse's full form line
```

### Capture odds over time, then measure closing-line value

```python
from hkjc.data.odds_ticks import OddsTickStore
from hkjc.eval import clv

store = OddsTickStore('ticks')          # Parquet under ./ticks
store.capture_meeting('', '')           # poll every race once (run on a schedule)

ticks = store.read(pool='WIN')          # tidy time series
clv.odds_drift(ticks)                   # open->close drift (negative = steam)
clv.closing_snapshot(ticks)             # the closing line per runner
```

### Remember what you learn

```python
from hkjc.ledger import record, search, recent

record("Exotic takeout is not 17.5%",
       "Trio/Quartet takeout runs ~19.5–25%; the model's 0.175 default only "
       "holds for WIN/PLA/QIN/QPL. Always net edge against the right pool.",
       kind="gotcha", tags=["takeout", "betting-rules"])

search("takeout")    # ranked list[Finding]
recent(5)            # latest entries
```

## Live-data caveats

See `docs/HKJC_KNOWLEDGE.md` for the full field guide; the essentials:

These are properties of the upstream HKJC site, learned the hard way:

- **`live` / `historical` venue codes** are `ST` (Sha Tin) and `HV` (Happy Valley).
  Passing empty strings to the live endpoints lets the server return the current
  meeting.
- **Historical results pages are server-rendered HTML**, but a non-race date
  returns an error page with no tables (the scraper raises `AttributeError`).
  Always target a real meeting date (format `YYYY/MM/DD`).
- **SpeedPro** is only published shortly before a meeting. The legacy `.xls`
  energy-grid endpoint has been retired; figures now come from per-race
  `speedguide` JSON and are **current-meeting only** (not addressable by date).
  Numeric energy/fitness fields are `null` until HKJC publishes them.
- The Harville fit is `O(N·2^N)`; the first call pays a one-off Numba JIT
  compilation cost (~10s), cached thereafter.
