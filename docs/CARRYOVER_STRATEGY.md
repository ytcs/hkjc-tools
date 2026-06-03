# Jackpot Carryover Harvester — strategy & build plan

Output of the quant-research workflow (`quant-arb-strategy`: 6 designs → adversarial
verify → synthesis). This is the chosen first trading strategy. See journal entry
0012 for the decision record.

## Why this one (and not the model-edge exotics)

The WIN market is efficient (journal 0011), and **every model-edge exotic play**
the team proposed (QIN/QPL coherence, cross-pool place arb, steam-residual, τ
place-bias) shares two fatal problems on our stack:

1. **Not backtestable on data we have.** Result pages publish only the *winning*
   combination's dividend — never the pre-close crowd matrix of the *losing*
   combinations you must price to detect an overlay. There is no historical exotic
   crowd-odds capture and no odds-tick store yet.
2. **Look-ahead.** Selecting bets on the realized settlement dividend (e.g.
   `bets.price_dividends`'s `edge`) peeks at the post-race price the walk-forward
   embargo forbids. (That function is now documented as a diagnostic only.)

The τ place-bias play was additionally **empirically falsified** (calibration
optimum at τ=1.0, bias sign backwards).

The **Jackpot Carryover Harvester** is different on the axis that matters: its edge
is **structural**, not a model-beats-crowd claim, and its inputs are already wired.

## The structural edge

Double Trio / Triple Trio / Six Up carry their net pool forward when not won. On a
carryover meeting, winners share `net_fraction·G + C` (today's net pool plus the
carried-in jackpot `C`), so new money faces:

```
effective_takeout = (1 − net_fraction) − C/G        (net_fraction ≈ 0.85)
aggregate_edge    = net_fraction + C/G − 1
```

The aggregate pool is structurally **+EV when `C/G > 1 − net_fraction` (~0.15)** —
a redistribution of forfeited money, not an edge anyone can refute on calibration.
Implemented in `hkjc.bets.carryover` (`effective_takeout`, `aggregate_edge`,
`flag_carryover`).

## Data path (forward, not historical)

Result pages carry **no** carryover/pool-size data (only dividends when won), so a
historical jackpot ledger is not scrapeable. The sound source is **live capture**:
`jpEsts.jackpot` (= `C`), `poolInvs.investment` (= `G`) and `estimatedDividend`,
now extracted by `hkjc.data.live.live_jackpots()`. These populate only while a
meeting's pools are open, so the detector must be **polled during the live window
and logged**.

## Honest weaknesses (respect these)

- **Endogenous dilution** — you bet pre-close but settle at the final crowd, and a
  publicized carryover invites other harvesters onto your combos. Price EV against
  `estimatedDividend` **plus a stressed late-pool haircut**, not the naive
  `(t·G − C)/(G + C)`.
- **Cover dependence** — a partial Flexi cover leans on a win-odds Harville
  `top_set(3)` that is only barely calibrated (journal 0011); a high cover
  miss-rate silently flips +EV to −EV.
- **Statistically dark** — only a handful of carryover events per season and
  lottery-shaped variance; you can be correctly +EV and draw down for seasons.
  Report power, not just a point estimate. **Detector/research instrument first;
  not a blank-cheque deployment.**

## Build plan

1. **Scope guard** — detector + research instrument. Pre-register the C/G threshold
   and τ/cover-size grid so FDR/deflated-Sharpe (journal 0007) has an honest trial
   count.
2. **Detector** — ✅ `bets.carryover` (effective takeout / aggregate edge / flag).
3. **Live capture** — ✅ `live.live_jackpots()` (C, G, estimatedDividend per leg).
   Next: wire into the orchestrator to log every meeting going forward.
4. **EV two ways** — naive `(t·G − C)/(G + C)` *and* the realistic one anchored on
   `estimatedDividend` + stressed late pool; flag only when both agree at a high
   C/G threshold.
5. **Cover layer** — per leg, `PlackettLuceModel.from_odds(closing WIN)` →
   `RaceSimulator.top_set(horses, 3)`; legs independent → cross-leg product; greedy
   Flexi cover maximizing `Σ p·payout − cost` under a bankroll cap; surface
   non-dominated covers with `utils.pareto_filter`. Report cover miss-rate.
6. **Score** — run simulated cover P&L through `eval.walkforward` (meeting-day
   embargo) + `eval.selection` (FDR/deflated-Sharpe) + a sealed lockbox.
7. **Pin payout mechanics before any stake** — Triple Trio consolation split, Six
   Win Bonus, dead-heat splits, scratching/re-banding of legs (betting-rules §7).
   A wrong consolation/floor silently corrupts the EV switch.
