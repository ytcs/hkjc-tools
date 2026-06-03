# HKJC Betting Rules, Bet Types & Special Payouts

Exhaustive reference for every HKJC pari-mutuel pool, its winning condition, the
takeout that eats into its dividend, and the jackpot/carryover mechanics that make
some pools structurally interesting. This is the **payout side** of the edge
equation — the model produces probabilities, but *profit* is decided here.

> **Provenance & staleness warning.** Compiled from the official HKJC betting
> rules, the AOSBS racing guide, and HKJC Racing News. **Takeout rates and rebate
> thresholds are tuned by HKJC per season and per promotion** — treat every
> percentage below as a *prior to verify*, not a constant. The authoritative
> sources are the per-section rule PDFs under
> `https://www.hkjc.com/english/betting/template_betting_rule_files/pdf/` and the
> beginners' guide at `https://special.hkjc.com/racing/info/en/betting/`. The
> system should ideally **scrape the live `estimatedDividend` / `jpEsts` fields**
> (already exposed by the GraphQL endpoint, see `HKJC_KNOWLEDGE.md` §1.1) rather
> than hard-coding any of this.

---

## 1. Why this matters for the agent

Three structural facts drive everything below:

1. **Takeout is not uniform across pools.** It ranges from ~17.5% (Win/Place/
   Quinella) to ~25% (Quartet/First 4). A model edge of 5% is profitable in a
   17.5%-takeout pool and *underwater* in a 25%-takeout pool. **The model's edge
   must always be measured net of the specific pool's takeout.** The
   `harville_model` default `takeout_rate = 0.175` is only correct for the
   WIN/PLA/QIN/QPL family — see §4.
2. **Jackpot pools carry money forward.** Double Trio, Triple Trio and Six Up
   accumulate an un-won jackpot across meetings. On a carryover meeting the
   *effective takeout can go negative* — the pool pays out more than was bet into
   it that day. These are the rare windows where even a market-efficient pool
   becomes structurally +EV, and they deserve a dedicated detector (§6).
3. **Multi-race pools are products of single-race distributions.** Races are
   independent draws, so the joint probability of a Triple Trio is the product of
   three single-race Trio probabilities — each of which the `RaceSimulator`
   already estimates. The combinatorics explode (legs × combinations), so these
   are a *search/optimization* problem over tickets, not a modelling problem.

---

## 2. Single-race pools

| Pool | Code | Win condition | Field-size rule | Takeout (prior) |
| --- | --- | --- | --- | --- |
| **Win** | `WIN` | Pick the 1st horse | — | ~17.5% |
| **Place** | `PLA` | Pick a horse that finishes in the paying places | **≥7 runners → top 3** pay; **4–6 runners → top 2** pay; **≤3 runners → no Place pool** | ~17.5% |
| **Quinella** | `QIN` | 1st & 2nd, **any order** | needs ≥... (small fields may suspend) | ~17.5% |
| **Quinella Place** | `QPL` | **Any two of the top three**, any order | requires a 3-place race (≥7 runners) | ~17.5% |
| **Forecast** | `FCT` | 1st & 2nd, **exact order** | — | ~19.5% (merged-pool, see note) |
| **Tierce** | `TCE` | 1st-2nd-3rd, **exact order** | — | ~22% (cut from 25%) |
| **Trio** | `TRI` | Top 3, **any order** | — | ~19.5% (merged with Forecast) |
| **First 4** | `FF` | Top 4, **any order** | — | ~25% (merged with Quartet) |
| **Quartet** | `QTT` | 1st-2nd-3rd-4th, **exact order** | — | ~25% |

Notes:
- **Forecast ↔ Trio merge (2020):** Forecast was relaunched into a *common pool*
  with Trio at a unified ~19.5% takeout (Trio's takeout dropped from 25%). Treat
  Forecast and Trio as economically linked.
- **First 4 ↔ Quartet:** similarly share liquidity; both are top-4 pools (any
  order vs exact order).
- **Place place-count is field-size-dependent** — this is the single most
  important rule to encode correctly. `simulation.places(horse, k=...)` must be
  called with `k=3` for 7+ runner fields and `k=2` for 4–6 runner fields. Getting
  `k` wrong silently corrupts every Place/QPL EV.

### 2.1 Composite Win (CWA / CWB / CWC)

Composite Win pools group runners (e.g. by bracket / coupled entries / overseas
commingling) and pay on the *group* containing the winner. The GraphQL endpoint
exposes `CWA/CWB/CWC` odds types (`HKJC_KNOWLEDGE.md` §1.1). Minimum-dividend
rule is gentler here: **no dividend below unit bet + $0.10** (vs + $0.50 for the
main pools).

---

## 3. Multi-race pools

| Pool | Code | Win condition | Jackpot? |
| --- | --- | --- | --- |
| **Double** | `DBL` | Winner of **2** nominated races | No (consolation available) |
| **Treble** | `TBL` | Winner of **3** nominated races | No (consolation available) |
| **Double Trio** | `DT` | The **Trio** (top-3 any order) in **2** nominated races | **Yes** |
| **Triple Trio** | `TT` | The **Trio** in **3** nominated races | **Yes** (+ consolation) |
| **Six Up** | `6UP` / `SixUP` | **1st or 2nd** in each of **6** nominated races | **Yes** (+ Six Win Bonus) |

- **Triple Trio consolation:** backers who correctly select the Trio in all three
  legs but whose *units* still miss the top dividend share a consolation pool;
  exact split is in the rules. Large estimated dividends (HK$10M–$100M+ on
  carryover / national-day meetings) come from this pool.
- **Six Up "Six Win Bonus":** an extra bonus when all six selections actually
  *win* (not just place 1st/2nd).
- **All-Up / Cross-Pool All-Up:** winnings from one leg are automatically
  reinvested into the next across 2–6 races/pools — a compounding bet wrapper, not
  a pool of its own.

---

## 4. Pool economics — takeout, net pool, dividends

For a standard pari-mutuel pool:

```
gross_pool  = sum of all stakes into the pool
net_pool    = gross_pool × (1 − takeout_rate)
dividend_per_unit(combo) = net_pool × winning_share(combo) / stakes_on(combo)
```

- **Takeout (`t`)** is the house commission; it is the *only* structural edge the
  operator keeps, and it differs by pool (§2/§3). Net pool is what's redistributed
  to winners.
- **Dividend is endogenous to the crowd.** Your payout depends on how much money
  is on *your* combination — popular combos pay less. This is why the model must
  predict **probabilities**, while the *odds* (and thus the dividend) come from the
  live pool. Our position is small relative to the pool, so **we ignore our own
  price impact** (see Architecture doc) — but the crowd's distribution is still the
  denominator of every payout.
- **Minimum dividend:** no dividend below **unit bet + $0.50** for Win, Quinella,
  Quinella Place and Double; below **unit + $0.10** for Composite Win/Place. A
  raging favourite can be capped at this floor.
- **Unit bet / Flexi:** standard minimum is **HK$10** per bet; **Flexi Bet** lets
  you stake a **$1 unit** (with a variable minimum ticket total) so you can buy
  partial units of a large multi-combination ticket.
- **Rebates** (loss-rebate promotions) effectively *reduce takeout* for high-stake
  losing tickets — e.g. Quinella / Quinella Place tickets with ≥HK$10,000 of
  losing bets on one line have received a **12% rebate**. A rebate is an edge
  source in its own right and must be modelled as negative takeout on the rebated
  fraction.

> **Implication for `harville_model`:** the `takeout_rate` argument must be set
> **per pool** when fitting or evaluating. Fitting strengths to QIN/QPL at 17.5%
> and then evaluating a Trio ticket at the same 17.5% overstates Trio EV by ~2 pts.

---

## 5. Jackpot & carryover mechanics (the special payouts)

Double Trio, Triple Trio and Six Up are **jackpot pools**. On top of the normal
takeout, a slice is skimmed into a reserve that **carries forward to the next
meeting if nobody wins**:

```
Jackpot Reserve Deduction   ≈ 7.5% of pool
Operator Reserve Deduction  ≈ 0.5% of pool
```

- On a normal day, ~85% of the pool is the net dividend pool for winning
  combinations; the jackpot reserve builds quietly.
- **If there is no winning unit, the entire net pool carries forward** and is
  **aggregated with the next meeting's 85% net pool** for the winning combinations.
- On a **carryover ("Fabulous"/"Snowball"/National-Day) meeting**, the pool paying
  out can *exceed* the money bet that day. The effective takeout on the *new*
  money is **negative** — a structural +EV window independent of any modelling
  edge. HKJC publishes estimated dividends (HK$10M–$100M+) ahead of these.

This is why §6 calls for a **carryover detector** as one of the earliest,
highest-value, purely-mechanical pieces of the system.

---

## 6. What the system must do with this

1. **Encode every pool as a `RaceSimulator` event** (single-race) or a **product
   of events** (multi-race), with the correct field-size `k` for Place/QPL. The
   existing predicates already cover the single-race shapes:
   `wins`→WIN, `places(k)`→PLA, `top_set(2)`→QIN, `top_set(3)`→TRI,
   `exact_order`→FCT/TCE/QTT, `all_place`/`any_place`→QPL family.
2. **Carry a per-pool takeout table** (verified live from `estimatedDividend`
   where possible) and always compute edge *net of the right takeout*.
3. **Scrape the live pool / jackpot fields** (`jpEsts`, `estimatedDividend`,
   per-pool `investment`) so EV uses the *actual* dividend denominator, not a
   model of it.
4. **Run a carryover detector** over the jackpot pools every meeting: flag
   meetings where accumulated jackpot makes a pool +EV on mechanics alone, and
   size the ticket-buying as a covering/optimization problem (Flexi units across a
   combinatorial cover) rather than a single bet.
5. **Treat rebates as negative takeout** when evaluating high-stake Quinella/QPL
   lines.

---

## 7. Open verification tasks

- [ ] Confirm current-season takeout for **every** pool from the live rule PDFs
      (numbers above are priors; HKJC adjusts them).
- [ ] Confirm exact **Place field-size thresholds** and small-field pool
      suspension rules.
- [ ] Pin down **Triple Trio consolation** and **Six Win Bonus** split formulas.
- [ ] Confirm **refund/scratching** treatment per pool (Win/Place refund vs
      exotic re-banding when a selection is scratched).
- [ ] Capture **dead-heat** division rules per pool.
- [ ] Map the live GraphQL odds-type codes to these pools end-to-end and validate
      against published dividends on a few past meetings.
