# Self-Improving Betting Research Agent — Architecture

How `hkjc-tools` becomes the bedrock for a **closed-loop, self-improving research
workflow** driven by a frontier reasoning model. The agent expands the system
along two axes — **modelling/statistics** and **data** — by *directing engineering
work*, evaluating the result against a rigorous harness, and keeping only what
survives. This document is the contract for that loop. Build the foundations (the
evaluator) before any "self-improvement," or the loop will optimize itself into
noise.

Companion docs: [`HKJC_KNOWLEDGE.md`](HKJC_KNOWLEDGE.md) (data field guide),
[`HKJC_BETTING_RULES.md`](HKJC_BETTING_RULES.md) (pools, takeout, jackpots).

---

## 0. The three load-bearing ideas

Read these first; everything else is mechanism.

### 0.1 Break the circularity (the modelling reframe)

The current `harville_model` **fits latent strengths to market odds**. A model
trained on the market can, by construction, only *reproduce* the market — it can
never beat it. For a research loop that hunts edge, the modelling layer's job has
to flip:

> Produce a finishing-order distribution that **diverges from** the market's
> implied distribution and is **better calibrated against realized results**.
> **Edge = divergence × superior calibration.**

So **odds stop being the target and become one feature among many.** What we score
against is *realized race outcomes* (calibration) and *closing-line value* (the
leading indicator). The Harville-on-odds model remains a strong **baseline to
beat**, not the destination.

### 0.2 The loop is only as good as its evaluator

A self-improving loop is an optimizer: the agent's reasoning is the search, and
**the evaluation harness is the objective function.** HK racing offers only
~700–850 races/year — a *tiny* signal. If the evaluator is gameable, leaky, or
ignores multiple-testing, the loop will enthusiastically overfit noise and
"improve" itself into ruin. **The evaluator is the most important artifact in the
system and the one that must be hardest to change.** (§3.)

### 0.3 Language parity is a first-class requirement, not a nice-to-have

HK racing is a market where a large fraction of informed opinion, tipping,
stable/track commentary, and forum chatter exists **only in Chinese**. A research
process that reads English sources well and Chinese sources poorly has a
*systematic blind spot exactly where the alpha is*. Therefore:

> Every data-axis capability must treat **English (`en-us`) and Chinese
> (`zh-hk`/`zh-cn`) sources as equal-priority inputs.** No source is deprioritized
> for being Chinese-only. Frontier multilingual reasoning is a *core advantage* of
> this approach and must be used to erase, not encode, language bias.

This is enforceable: the source registry (§5) tracks a `languages` field, and the
evaluation harness reports coverage/lift **broken down by source language** so a
language gap shows up as a measured deficiency.

### Scope note: price impact is out of scope (for now)

Our position size is small relative to the HKJC pools, so we **do not model our
own price impact** on the dividend in the near term. The crowd's distribution is
still the payout denominator (we don't move it). Revisit only if stake sizes grow
to a meaningful fraction of a pool.

---

## 1. System at a glance

```
        ┌──────────────────────── KNOWLEDGE LEDGER ────────────────────────┐
        │  hypotheses · experiments · results · killed ideas · data sources │
        │            (append-only; read before every proposal)              │
        └───────────────────────────────────────────────────────────────────┘
              ▲                                                       │
              │ accumulate                                   read before proposing
              │                                                       ▼
   ┌──────────┴───────┐   spec      ┌───────────────┐   artifact  ┌──────────────┐
   │  REASONING AGENT │ ──────────► │  ENGINEERING  │ ──────────► │  EVALUATION  │
   │ (frontier model) │             │  ORCHESTRATOR │             │   HARNESS    │
   │  • hypothesize   │             │ workers in    │             │ calibration  │
   │  • triage results│ ◄────────── │ git worktrees │ ◄────────── │  · CLV · sim │
   │  • decide next   │   metrics   │ + CI gate     │   scored    │  P&L         │
   └──────────────────┘             └───────────────┘             └──────────────┘
              │ promote (only if passes lockbox + multiple-testing control)
              ▼
   ┌────────────────────────────────────────────────────────────────────────────┐
   │  DEPLOYMENT:  paper-trade  →  capped live  →  monitored, with kill-switches  │
   └────────────────────────────────────────────────────────────────────────────┘
```

The agent **never edits production directly**. It emits *specs*; the orchestrator
turns each into an isolated engineering job that produces a scorable artifact;
results (including dead ends) land in the ledger so the loop compounds instead of
rediscovering. Crossing from research to real money is a separate, gated step.

### Layer map onto the existing library

| Loop component | Built on | Status |
| --- | --- | --- |
| Probabilistic engine | `harville_model`, `model.RaceModel`, `simulation.RaceSimulator` | **exists** (bedrock) |
| Data access | `data.live/historical/racecard/sectional/...` | **exists** |
| Feature assembly | `data.feature_store` (`DataSource` + outcome schema) | **exists** |
| Odds tick store + CLV | — | **to build (step 1)** |
| Evaluation harness | — | **to build (step 2)** |
| `DataSource` protocol + feature store | extends data layer | **to build (step 3)** |
| Results-trained `RaceModel` | new `RaceModel` impl | **to build (step 4)** |
| Orchestrator / ledger / governance | tooling around the above | **to build** |

---

## 2. The closed loop, step by step

1. **Propose.** The agent reads the ledger (what's been tried, what's promoted,
   what's dead) and the current leaderboard, then emits a **hypothesis spec**:
   a model change, a feature, or a new data source — with a stated expected
   effect and how it will be measured. One hypothesis = one unit of work.
2. **Build.** The orchestrator spawns an isolated engineering job (own git
   worktree/branch) that implements the spec: writes the scraper/feature/model,
   plus a contract test. Independent jobs run in parallel.
3. **Evaluate.** The harness scores the artifact on the walk-forward backtest
   (calibration → CLV → simulated P&L), with multiple-testing control. Output is
   structured metrics, not prose.
4. **Verify (adversarial).** Before anything is believed, independent skeptic
   agents try to *refute* the result — look-ahead leakage? embargo violation?
   regime-specific fluke? FDR not applied? Default to "refuted" under doubt.
5. **Triage & record.** The agent interprets metrics, writes the result (win or
   kill) to the ledger, and decides the next move (ablate a winner, combine
   survivors, abandon a line). 
6. **Promote.** A surviving candidate that beats baseline *and* passes the
   one-shot lockbox check (§3) graduates — first to paper trading, then to capped
   live under governance (§7).

The loop runs continuously on a cadence (§6): poll odds on schedule, sweep
research nightly, re-score the leaderboard, surface a digest.

---

## 3. The evaluation harness (the crux)

Three scoring layers, in increasing trust and decreasing frequency. A candidate
must clear each to reach the next.

### 3.1 Layer 1 — Calibration vs. realized results
- Score the model's win/place probabilities against **actual finishing orders**
  with proper scoring rules: **log-loss / Brier / ranked probability score**, plus
  **reliability curves** (do horses given 20% actually win ~20%?).
- Dense-ish signal: *every* historical race yields a score even with zero betting.
  This is the **primary fitness function for the modelling axis** and the cheapest
  way to kill a bad model early.

### 3.2 Layer 2 — Closing-Line Value (CLV)
- Did the model's *pre-close* probability beat the **closing** market? In efficient
  pari-mutuel markets, **CLV is the strongest known leading indicator of true
  edge**, and it's available for **every race whether or not we bet** — partially
  rescuing the sparse-feedback problem.
- Requires the **odds tick store** (step 1): we must record the odds trajectory to
  know the close and to measure our timing. The odds-drift signal (informed money
  arriving late — already a prior in `HKJC_KNOWLEDGE.md` §4.3) is learned here.

### 3.3 Layer 3 — Simulated P&L under pari-mutuel realism
- `RaceSimulator` already computes P/EV of arbitrary bets. The harness wraps it
  into a **bankroll simulation** over a walk-forward period, using the **correct
  per-pool takeout** (`HKJC_BETTING_RULES.md`) and the **live dividend denominator**
  where available. (We do **not** model our own price impact — small position.)
- Reports ROI, turnover, drawdown, hit rate, and a **multiple-testing-deflated**
  performance metric. This is the closest proxy to truth and the last gate before
  paper trading.

### 3.4 Anti-overfitting discipline (non-negotiable)
With <1000 races/year these are what stand between "research" and "fooling
ourselves":
- **Walk-forward only, strict time embargo.** Never random splits — horses,
  jockeys, trainers and going conditions persist across races, so a random split
  leaks the future. Train on the past, test on the strictly-later future, with a
  gap.
- **A locked final hold-out ("lockbox")** the agent never sees during search.
  Promotion reads it **exactly once**; a candidate that needs multiple lockbox
  peeks is disqualified.
- **Multiple-testing control.** The agent will try thousands of hypotheses; naïve
  "best of N" *guarantees* phantom edges in data this small. Apply
  **Benjamini-Hochberg FDR / deflated Sharpe** and record the trial count so
  selection is penalized honestly.
- **Provenance on every score:** data version, feature version, model version,
  harness version. Irreproducible → not a result.
- **Language-coverage breakdown:** every harness report slices lift by source
  language (§0.3) so a Chinese-source blind spot is *measured*, not hidden.

> The harness is **versioned and append-only.** Changing the scorer invalidates
> prior results and is logged as an event. It is the system's constitution.

---

## 4. Axis 1 — modelling & statistical methods

The `RaceModel` interface is already the plug. Self-improvement here = a stream of
candidate models/fitters/features, each auto-scored by Layer 1 then Layers 2–3.

- **New `RaceModel` implementations:** finish-time models (log-normal / Weibull on
  sectional & final times), **learned strength functions** (gradient-boosted or
  small neural maps from features → strengths), Bayesian shrinkage for small-field
  noise, and **ensembles/stacks** of Harville-on-odds + a results-trained model.
- **New fitters:** the headline change — replace "fit θ to odds" with "**fit θ to
  results**, with market odds as a *prior/feature*." This is §0.1 in code and the
  first candidate that can structurally beat the market.
- **New features** (`DataSource`s feeding `data.feature_store`): each is a cheap,
  independently-scorable unit — ideal granularity for a parallel fan-out. NB the
  standard-fundamentals priors (`HKJC_KNOWLEDGE.md` §4) were **tested and do not
  beat the market** (journal 0011/0023); the open hope is interactions and
  information the market lacks (sectionals, trip notes) — add interactions
  energy × distance) the old equal-weight voting scheme couldn't express.
- **Selection:** candidates compete on the walk-forward leaderboard, FDR-corrected;
  promotion needs lockbox confirmation. The agent reads leaderboard + ledger to
  choose the next experiment (ablate winners, kill losers, combine survivors).

The pari-mutuel pools themselves (`HKJC_BETTING_RULES.md`) define the **bet-search**
problem that sits on top of the probability model: given a calibrated distribution
and live odds, which tickets (across Win… Triple Trio, jackpot windows included)
maximize edge net of the right takeout. That's optimization over `RaceSimulator`
events, not modelling — keep the two concerns separate.

---

## 5. Axis 2 — data acquisition

A uniform **`DataSource` protocol** parallel to `RaceModel`: every source — odds
polls, video, Chinese tipsters — implements one contract (timestamped, race-keyed
records → a **feature store**) plus a parser and a contract test. Onboarding any
new source becomes a uniform engineering job the agent can direct, and every
source is scored the same way: *does adding it improve walk-forward calibration /
CLV after FDR correction?* Sources that don't earn their keep are archived in the
ledger, not silently retained.

```python
class DataSource(Protocol):
    name: str
    languages: list[str]          # e.g. ["en-us"], ["zh-hk"], ["zh-hk","en-us"]
    def fetch(self, meeting) -> Iterable[Record]: ...   # timestamped, race-keyed
    # Record -> normalized features in the feature store, keyed by (race, runner, ts)
```

Priority order (effort vs. payoff):

### 5.1 Odds time-series — **build first, highest ROI**
You already poll WIN/PLA/QIN/QPL. Persist **every** poll into a tick store keyed
by `(date, venue, race, pool, runner, ts)`. Unlocks:
- **CLV** (Layer 2 of the harness) — the dense feedback signal.
- **Odds-drift / steam detection** — informed money leaves a signature in the
  trajectory; "predict the close from early ticks" is a genuinely *forward*,
  non-circular target.
- Pool-size/liquidity series from per-pool `investment` (already exposed).

Pure engineering, no new technology, enormous downstream leverage. Also capture
the live `jpEsts` / `estimatedDividend` fields so jackpot carryover (§ jackpots)
is observable.

### 5.2 Video / livestream mining
Track-workout MP4s (endpoint in `HKJC_KNOWLEDGE.md` §1.5), race replays, and
public channels (e.g. the HKJC YouTube channel). A **perception pipeline**:
frame-sample → VLM/CV → **structured trip events** ("boxed in at the 800m", "wide
no cover", "raced green", "blocked run", "slowly away"). These explain *why a
horse's result understated its true ability* — information not yet in the odds.
Staged so the agent can direct it: download → segment → annotate → **entity-link
to runner numbers** → emit features. Heavy; sequence it after 5.1/5.3 prove the
loop.

### 5.3 Chinese-language alternative data — **equal priority (§0.3)**
Forums, tipster columns, stable/track commentary, the `zh-hk` halves of HKJC's own
JSON (SpeedPro already ships `zh-hk`). Pipeline: ingest → (translate as needed) →
**NER linking mentions to horses/jockeys/trainers** → stance/sentiment/claim
extraction → features. Frontier models are *strong* at exactly this messy
multilingual extraction, so this plays to the loop's advantage rather than being a
chore. The harness's per-language coverage report (§3.4) keeps this honest: if
Chinese sources aren't pulling weight, that's a measured finding, not an excuse to
skip them.

---

## 6. Orchestration — directing complicated engineering tasks

This is what lets the agent "expand the system," and it maps onto primitives the
runtime already provides:

- **Fan-out workers in isolated git worktrees** for parallel, independent
  experiments (one feature / model / scraper each) so concurrent jobs never
  collide on disk.
- **Pipeline (no barriers)** for staged work: a video-mining job can still be
  downloading while a feature experiment is already being scored. Barriers only
  where a stage genuinely needs all prior results (e.g. dedup before verification).
- **Adversarial verification** of any claimed edge before promotion (§2.4):
  independent skeptics prompted to refute, majority-refute kills it.
- **A CI gate:** nothing merges unless contract tests + a harness regression pass.
- **Scheduled cadence (cron):** poll odds on schedule; nightly research sweep;
  re-score the leaderboard; emit a human-readable digest.
- **A carryover detector** over the jackpot pools every meeting
  (`HKJC_BETTING_RULES.md` §5): flag meetings where the accumulated Triple Trio /
  Six Up / Double Trio jackpot makes a pool +EV on *mechanics alone*, and frame the
  buy as a Flexi-unit covering/optimization problem. Cheap, mechanical, and one of
  the few *structurally* +EV opportunities independent of modelling skill.

---

## 7. Governance — research vs. real money

Non-negotiable for a loop that can touch capital:

- **A bright line between research (sim/paper only) and live capital.** The agent
  operates freely on the research side. Crossing to live money is gated on:
  lockbox pass + a paper-trading track record + an explicit human approval.
- **Hard caps & kill-switches** on the live side: per-race and per-day stake
  limits, automatic halt on drawdown breach, and on any data-integrity alarm
  (stale odds, scraper drift, schema change upstream).
- **Full provenance** on every live strategy: traces to the exact data/feature/
  model/harness versions and the experiments that justified it. Reproducible or it
  doesn't ship.
- **Responsible-use posture:** this is a personal research system operating within
  HKJC's own pari-mutuel framework; respect the scraping discipline in
  `HKJC_KNOWLEDGE.md` §5 (polite poll intervals, ≤3 concurrency) and the site's
  terms.

---

## 8. Build order

Foundations before self-improvement. Steps 1–2 are the spine; without them the
"self-improving" part is just an efficient way to overfit 800 races.

1. ✅ **Odds tick store + CLV harness.** Persist every poll; compute CLV. Smallest
   effort, unlocks the densest signal. *(Axis 2 + harness Layer 2.)* — shipped:
   `hkjc.data.odds_ticks`, `hkjc.eval.clv`.
2. ✅ **Walk-forward evaluation harness** with calibration, FDR control, and a
   lockbox. The objective function — build before any optimizer runs against it.
   *(Harness Layers 1 & 3.)* — shipped: `hkjc.eval.{calibration,walkforward,
   selection,harness}`. *(Layer-3 bankroll P&L still to wire once stake-sizing is
   decided — see §9.)*
3. ✅ **`DataSource` protocol + feature store**, with the `languages` field and
   per-language coverage reporting. The uniform seam for the data axis. — shipped:
   `hkjc.data.source` (protocol), `hkjc.data.feature_store` (`FeatureStore`,
   per-source + per-language coverage, the **Win/Place/Quinella outcome schema**).
   Concrete feature-source adapters grow alongside step 4.
4. ✅ **Results-trained `RaceModel`.** Break the circularity (§0.1). — shipped:
   `hkjc.model.ConditionalLogit` (McFadden conditional logit, fit to realized
   winners, market prob as one feature). `conditional_logit_fit_predict` plugs into
   the step-2 harness; `strengths()` feeds `PlackettLuceModel` for Place/Quinella.
   Validated: beats uniform ~37% on multiclass log-loss out-of-sample. *(Next: add
   real fundamentals + market feature and run vs the market baseline on live data.)*
5. 🟡 **Memory + orchestrator + carryover detector.** Memory shipped
   (`hkjc.ledger`). Carryover detector shipped (`hkjc.bets.carryover` +
   `hkjc.data.live.live_jackpots`); the **Jackpot Carryover Harvester** is the
   chosen first strategy (see `docs/CARRYOVER_STRATEGY.md`, journal 0012). Still
   need: live-capture logging, the Flexi cover layer, and the orchestrator.
6. ⬜ **Heavy data:** Chinese alt-data pipeline (§5.3) and video mining (§5.2), each
   onboarded through the §5 protocol and judged by the §3 harness.

---

## 9. Open questions (to resolve with the user before/while building)

- **Bankroll & stake sizing:** fixed-fraction / Kelly / capped-Kelly? Risk budget?
- **Which pools to target first:** simple Win/Place/Quinella edge, or go straight
  for the jackpot-carryover windows where mechanics alone can be +EV?
- **Live vs. paper horizon:** how long a paper-trading track record before any real
  capital, and what approval ritual crosses that line?
- **Compute envelope** for the video pipeline (the most expensive axis).
- ~~**Ledger substrate:**~~ *resolved* — flat Markdown entries + a rebuildable
  `index.jsonl`, in-repo under `journal/` (`hkjc.ledger`). Revisit only if it
  outgrows a single-file index.
