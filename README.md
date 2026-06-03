# hkjc-tools

Library for scraping Hong Kong Jockey Club (HKJC) horse-racing data and running
basic betting analysis. Intended as the data + modelling layer for an automated
betting agent.

## Install

```bash
uv sync
```

Requires Python >= 3.11.

See [`docs/HKJC_KNOWLEDGE.md`](docs/HKJC_KNOWLEDGE.md) for the field guide to
HKJC's data sources, endpoints, domain conventions, feature priors and gotchas.

## Architecture

The library is a **bedrock** for an automated betting agent, in three layers. It
provides data and a probabilistic engine; it deliberately does **not** ship any
fixed betting strategy — the agent composes her own bets on top.

```
data access  ->  model  ->  simulation
live/         harville_   RaceSimulator
historical/   model +     (estimate P/EV of
speedpro      RaceModel    ANY bet you compose)
```

| Module | Layer | Purpose |
| --- | --- | --- |
| `hkjc.live` | data | Live race cards and live odds (WIN/PLA/QIN/QPL) via the HKJC GraphQL endpoint. |
| `hkjc.historical` | data | Past race results and per-horse form, scraped from HKJC results pages. |
| `hkjc.speedpro` | data | SpeedPro energy/fitness figures and speed maps for the current meeting. |
| `hkjc.harville_model` | model | Fits latent horse strengths to market odds via DP (Numba); yields a full place-probability matrix. The default odds→strengths fitter. |
| `hkjc.model` | model | `RaceModel` interface (swappable) + `PlackettLuceModel` reference impl that samples finishing orders from strengths. |
| `hkjc.simulation` | sim | `RaceSimulator`: draw finishing orders from any model and estimate the probability/EV of arbitrary bets. |
| `hkjc.processing` | data | Build historical datasets across a date range. |
| `hkjc.features` | — | Polars feature expressions (rating gap, front-runner share, ...). |
| `hkjc.utils` | — | Parsing helpers and a generic Pareto filter. |

### Bring your own model

`PlackettLuceModel` (fitted to odds via Harville) is only the default. Any object
implementing `RaceModel.sample_orders(size, rng) -> (size, n)` finishing orders
plugs straight into `RaceSimulator` — so the research agent can swap in a
finish-time model, a learned model, or anything else without touching the rest of
the stack.

## Quick start

```python
from hkjc.live import live_odds, _fetch_live_races
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
from hkjc.historical import get_race_data, get_horse_data

results = get_race_data('2026/05/31', 'ST', 1)   # past race result table
form    = get_horse_data('L276')                  # one horse's full form line
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
