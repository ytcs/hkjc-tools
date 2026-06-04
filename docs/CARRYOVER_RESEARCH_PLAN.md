# Jackpot Carryover Harvester — Research Plan (main line)

This is the project's **primary research direction**. The thesis (journal 0012,
`docs/CARRYOVER_STRATEGY.md`): jackpot pools (Double Trio, Triple Trio, Six Up)
carry forward when unwon, so on a carryover meeting the aggregate pool can be
structurally +EV (negative effective takeout) — an edge that doesn't require
beating the crowd's handicapping, only correctly identifying and harvesting the
window. The win market being efficient (journal 0011) does not threaten it.

This document maps **everything we need to build** and **every research question
that decides whether the edge is real and reachable**, with sequencing and kill
gates. Forward live capture is the spine: result pages carry no carryover/pool-size
data, so we accumulate the dataset prospectively.

---

## 0. The EV equation (what every work item is in service of)

For a Flexi cover `S` (a chosen set of combinations) of a jackpot pool:

```
EV(S) =  Σ_{w in S} P(w) · payout(w)              −  cost(S)

payout(w) = your_units(w) · ( f_win·NetPool_settled + Carry ) / ( crowd_units(w) + your_units(w) )
            ( + consolation / Six-Win-Bonus terms )

cost(S)   = |S| · unit_stake
NetPool_settled = (1 − deductions) · G_settled
```

Every term maps to a build item and a research question:

| Term | What it needs | Build | Research |
|---|---|---|---|
| `Carry` (C), `G_settled`, `estimatedDividend` | live capture over the betting window | B1,B2 | A, B |
| `P(w)` per combination | per-leg Trio model × independence | B5 | C |
| `crowd_units(w)` (dilution) | crowd combo-popularity + late money | B6 | **B (crux)** |
| `f_win`, deductions, consolation, bonus | exact settlement rules | B8 | E |
| `S`, `unit_stake`, `your_units` | cover optimizer + bankroll | B7,B12 | D |

**The crux is dilution (`crowd_units(w)`).** The aggregate pool can be +EV while
*your* EV is negative, because a publicized carryover draws the crowd onto the
same favourite-heavy combinations you are most likely to win — you get diluted
exactly where you hit. Most of the research weight is here.

---

## 1. What we need to BUILD

Status: ✅ done · 🟡 partial · ⬜ todo.

**Data acquisition**
- **B1 ✅ Live jackpot capture store** — `data.jackpot_ticks.JackpotCaptureStore`
  polls `live_jackpots()` (C, G(t), estimatedDividend(t)) across the betting window
  and persists the time series; the runner also captures per-leg WIN odds via
  `OddsTickStore`. The trajectory is what lets us measure late-money dilution.
- **B2 ✅ Settled-result joiner** — `data.carryover_ledger.settled_jackpot_result`
  + `build_carryover_rows`: after a meeting, join `get_race_dividends` (won /
  not-won, settled dividend, winning combo) to the last pre-close capture.
- **B3 ✅ Carryover ledger** — `data.carryover_ledger.CarryoverLedger`: durable,
  upserting dataset of jackpot events (C, G_close, est dividend, eff_takeout, edge,
  won?, settled dividend). Runner: `experiments/capture_jackpots.py` (capture /
  settle). Still ⬜: wire into a scheduler (part of B11) + implied winning-unit
  count (needs B6).
- **B13 ⬜ Historical backfill (research-grade)** — scrape HKJC Racing News
  carryover/estimated-dividend announcements (and any jackpot-history pages) to
  estimate event **frequency and magnitude** without waiting a full season. Coarse
  but immediately available; informs Gate 1.

**Modelling & detection**
- **B4 🟡 EV detector v2** — `bets.carryover` has the naive `effective_takeout =
  (1−net) − C/G`. Add the realistic EV: anchor on `estimatedDividend` + a
  **stressed late-pool haircut**, and flag only when both naive and realistic agree
  at a high C/G threshold with margin.
- **B5 ⬜ Per-leg Trio model** — `top_set(3)` probabilities per leg; start with
  `PlackettLuceModel.from_odds(closing WIN)`, calibrate against realized trios,
  keep it pluggable for a better (form/pace) model later.
- **B6 ⬜ Dilution / crowd-combo model** — estimate `crowd_units(w)` and the
  settled dividend from estimatedDividend, pool size, and combo-popularity priors
  (favourites over-bet). **The hardest and most important model.**
- **B8 ⬜ Settlement engine** — encode TT **85/15 winning/consolation split**, the
  consolation-combination definition, Six Win Bonus, dead-heat splits,
  scratching/re-banding of legs, and the minimum-dividend floor. Contract-tested
  against known historical dividends. A wrong rule silently flips +EV to −EV.

**Strategy & risk**
- **B7 ⬜ Cover optimizer** — given leg probs + pool/carry + dilution + bankroll,
  select the Flexi cover maximizing EV (or E[log wealth]); combinatorial → greedy +
  `utils.pareto_filter` over (hit-probability, EV). Output ticket set, cost, p_hit,
  EV, variance, cover miss-rate.
- **B12 ⬜ Risk / bankroll** — Kelly-fraction sizing for a lottery-shaped payoff,
  drawdown / risk-of-ruin controls, per-window cap. Sizing must survive long
  unconfirmed-edge stretches.

**Validation & ops**
- **B9 ⬜ Paper-trading loop** — forward, log what we *would* bet and the realized
  outcome → an honest track record (backtest data is too scarce for confirmation).
- **B10 ⬜ Scoring + power analysis** — `eval.walkforward` (meeting embargo) +
  `eval.selection` (FDR/deflated-Sharpe) + sealed lockbox over the accumulating
  ledger; **report statistical power** (how many events to confirm +EV), not just a
  point estimate.
- **B11 ⬜ Orchestrator** — schedule capture, run the detector each meeting, alert
  on a flagged +EV window, emit a digest. Cron-driven.

---

## 2. RESEARCH QUESTIONS (what decides if the edge is real and reachable)

### Theme A — Opportunity frequency & magnitude  *(is it even worth it?)*
- **RQ1** How often does each pool (DT/TT/6UP) carry over, and what is the **C/G
  distribution**? How many structurally +EV windows per season?
- **RQ2** Current fresh-pool `G` and carry `C` magnitudes (the literature notes a
  long-run decline from ~HK$10M to ~HK$2M fresh pools)?
- *Answer via:* B13 backfill now, B1/B3 forward. *Kill:* if only ~0–2 thin windows
  per season, the strategy is too dark to pursue — shelve.

### Theme B — The dilution crux  *(do YOU capture the aggregate edge?)*
- **RQ3** On carryover meetings, how **concentrated** is crowd money over
  combinations — do bettors pile onto favourite combos (diluting you where you're
  likeliest to win)?
- **RQ4** How much does the pool **grow from bet-time to close** (late money), and
  how reliably does `estimatedDividend` predict the **settled** dividend?
- **RQ5** How many other **sharp harvesters** compete, and does competition on
  *publicized* windows erase the edge?
- *Answer via:* B1 time-series capture + B6 model + B2 settled join. *This is the
  central question; most uncertainty lives here.*

### Theme C — Leg model quality  *(does the cover hit efficiently?)*
- **RQ6** How well-calibrated is win-odds Harville for per-leg **Trio**
  probabilities (journal 0011: the win model is only barely calibrated)? Does a
  better leg model (form/pace/SpeedPro) materially improve cover efficiency?
- **RQ7** Are legs effectively **independent** for cover purposes (race
  independence says yes; watch correlated scratchings / track-condition shifts)?
- *Answer via:* B5 + the calibration harness.

### Theme D — Cover economics  *(is there a profitable cover size?)*
- **RQ8** Under realistic dilution, is there **any** cover size with EV > 0 after
  cost? What is the optimal cover and its variance / drawdown profile?
- **RQ9** **Bankroll sizing**: appropriate Kelly fraction for this payoff shape;
  risk of ruin; how to size while the edge is statistically unconfirmed.
- *Answer via:* B7 + B12 simulation over the ledger.

### Theme E — Settlement mechanics  *(don't silently corrupt EV)*
- **RQ10** Exact **consolation** definition + 85/15 split, **Six Win Bonus**,
  dead-heat splits, scratching/re-banding, minimum-dividend floor. Encode precisely
  and verify against known past dividends.
- *Answer via:* B8 + the rule PDFs (betting-rules §7 open items).

### Theme F — Execution & statistical viability
- **RQ11** Bet limits, Flexi mechanics, ability to place large combinatorial
  covers, **timing** (near-close accuracy vs shut-out risk), rebate eligibility.
- **RQ12** **Power**: given event rarity + lottery variance, how long to confirm
  +EV? Is forward **paper-trading** the only honest validation path before capital,
  and what track record crosses the line to real money?

---

## 3. Sequencing, milestones & decision gates

- **M1 — Start the data flywheel (immediately).** B1 + B2 + B3 running every
  meeting; B13 backfill in parallel. → answers **Theme A**.
  **Gate 1: ✅ PASSED** (journal 0017) — TT/Six-Up roll over frequently; fresh
  pools ~HK$2M vs carries HK$3–30M ⇒ C/G regularly ≥ 0.15 (often ≥ 1). **Refinement
  for Gate 2: target *modest, under-publicized* carryovers** — chase money floods
  headline HK$60M+ jackpots and erodes the edge. Cloud capture (Modal) deployable;
  see `cloud/`. Remaining: deploy + accumulate, and the full B13 scrape (Racing
  News announcements for C; localresults-by-date for won/carried).
- **M2 — Make EV computable.** B8 (settlement) + B4 (EV v2) + B5 (leg model
  calibration). → **Theme E**, partial **C**.
- **M3 — Crack dilution.** B6 from accumulated captures. → **Theme B**.
  **Gate 2:** net +EV after realistic dilution **and** competition? If no → mirage,
  stop.
- **M4 — Build the cover.** B7 + B12; begin B9 paper-trading. → **Theme D**.
- **M5 — Validate & operate.** B10 (power) + B11 (orchestrator); ongoing paper
  track. **Gate 3:** forward paper P&L confirms edge with adequate power before any
  real capital.

## 4. Top risks / honest priors
- **Dilution eats the aggregate edge** (RQ3–5) — the most likely way this fails.
- **Statistically dark** — may be genuinely +EV yet take seasons to confirm (RQ12);
  treat as a research instrument and paper-trade first.
- **Cover leans on a barely-calibrated leg model** (RQ6) — high miss-rate flips the
  sign.
- **Settlement mis-encoding** (RQ10) — silent EV corruption.
- **Competition** — the publicized carryover that flags the window to us flags it
  to every other harvester.
