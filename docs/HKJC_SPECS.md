# HKJC Specs — tracks, competition rules, betting rules

The **reference data** of Hong Kong racing: the physical tracks, the handicap/
competition system, and the pari-mutuel pools with their economics. This is the
authoritative, slow-changing layer. The track geometry is encoded in `hkjc.data.tracks`
(`TrackGeometry`, `TRACKS`, `track_for`; the first and last are also exported from
`hkjc.data`) and the incident codes in `hkjc.data.historical.incidents`; import those
rather than hard-coding them. The competition and betting rules are documented here only.

> Venue codes used everywhere: **`ST` = Sha Tin**, **`HV` = Happy Valley** — the only
> two local courses. Two surfaces: **Turf** and **AWT** (All-Weather Track, dirt).

---

## 1. Tracks & geometry

The turf **running rail is movable**: it is shifted outward meeting-to-meeting to
spread wear, which **changes the usable width and the draw-bias geometry per race** —
geometry is *not* fixed per venue. The `Course` field (`A`, `A+2`, `B`, `B+2`, `B+3`,
`C`, `C+3`) sets the rail position. Official widths (encoded in `hkjc.data.tracks`):

| Course | Sha Tin width | Happy Valley width |
| --- | --- | --- |
| A | 30.5 m | 30.5 m |
| A+2 | 28.5 m | 28.5 m |
| B | 26.0 m | 26.5 m |
| B+2 | 24.0 m | 24.5 m |
| C | 21.3 m | 22.5 m |
| C+3 | 18.3 m | 19.5 m |

`hkjc.data.tracks` holds a `TrackGeometry` per *venue × surface × configuration* (the
`TRACKS` dict) with a resolver `track_for(venue, distance, surface)`. The four encoded
geometries:

- **Happy Valley** — turf, ~1,450 m circuit, right-handed, **bend-dominated** (near-
  continuous turning: home straight ~[0,300]; Sanatorium/Crescent/"The Rock" bends
  ~[300,770]; back straight ~[770,920]; HKFC/Stable bends + top turn ~[920,1450]).
  ⇒ **the strongest draw bias in HK** (low draws hold the short inner path). Distances
  1000 / 1200 / 1650 / 1800 / 2200 m.
- **Sha Tin turf** — ~1,900 m galloping oval, right-handed, **430 m home straight**.
  Long straights ⇒ a **weaker** geometric draw bias than HV. Bend boundaries are a
  two-turn reconstruction from circumference + home straight (the straight/turn split
  and total are firm; the exact bend split is approximate).
- **Sha Tin straight 1000 m** — a separate sprint **chute with no bends** ⇒ **no bend-
  geometry draw bias** (only a cross-track watering/going gradient, deliberately not
  modelled as geometry). `track_for("ST",1000)` returns this, not the oval.
- **Sha Tin AWT** — dirt, 1,560 m, right-handed, 365 m home straight, **fixed single
  rail** (uniform 22.8 m). Surface inferred from the "ALL WEATHER TRACK" label (going
  codes alone are ambiguous — both scales have GOOD). AWT has distinct physics: lower
  achievable speeds, kickback, and **ability that may not transfer from turf**.

Draw bias is most acute at HV and the short ST turf trips, flat on the ST 1000 m chute.
The `draw` info page (`DrawStatistics`) publishes realised W%/P% by course-distance.

## 2. Going scales

String → short code, as used to filter form by surface condition:

- **Turf:** `FIRM→F, GOOD TO FIRM→GF, GOOD→G, GOOD TO YIELDING→GY, YIELDING→Y,
  YIELDING TO SOFT→YS, SOFT→S, HEAVY→H` (firm → soft/wet).
- **AWT:** `WET FAST→WF, FAST→FT, GOOD→GD, SLOW→SL, WET SLOW→WS, RAIN AFFECTED→RA,
  NORMAL WATERING→NW`.

Turf firmness is also measured instrumentally: **penetrometer** and **hammer**
readings are exposed on the live card (see HKJC_DATA). Common goings (G/GF) dominate
the sample; soft/wet/heavy are rare and data-sparse.

## 3. The competition system (the handicap)

HK racing is overwhelmingly **handicap** racing. This is the adversarial meta-game
that shapes every win probability and is essential context for modelling:

- **Official rating (`Rtg` / `currentRating`)** — the handicapper's assessment of a
  horse's ability, in points. `internationalRating` is the global-scale equivalent.
  After each run the rating is **revised** (up for over-performing, down for under-);
  the live card exposes the **rating move (`Rtg +/-`)**.
- **Carried weight (`Wt` / `handicapWeight`, in lbs)** — assigned to *equalise* the
  field: higher-rated horses carry more. So weight is **endogenous** — it reads as a
  quality proxy (better horse → more weight), not merely a physical drag.
- **Weight-for-Age (WFA)** scale adjusts for age/sex in set-weight races.
- **Class / grade** — races are banded by rating range (Class 1 highest … Class 5,
  plus Group/Listed pattern races). A horse "out of handicap" carries below its mark.

The handicapper's explicit goal is a *blanket finish* — to compress ability
differences so every horse has a chance. This is **why win probabilities are close,
why beating the market is hard, and why "% of market" is the right yardstick**. It also means raw form must be read **relative to the field
faced** (opponent-strength-adjusted), not in absolute terms.

## 4. Runner & result field semantics

- **`Dr` / `barrierDrawNumber`** — draw (gate). Low draws favoured at bend-heavy
  course-distances (see §1).
- **`last6run`** — recent finishing-positions string.
- **Runner `status` `Standby`** = reserve (exclude it when building a card from GraphQL); scratched
  runners flagged via `changeHistories` / `scratched`.
- **`Pla`** — finishing position; may carry an **incident code** instead of a number:
  `DISQ, DNF, FE, ML, PU, TNP, TO, UR, VOID, WR, WV, WV-A, WX, WX-A, WXNR` (filtered
  out of the results table — but the *fact* that a horse failed to complete is itself
  informative).
- **Time strings differ by page** (parse each with its own rule):

  | Page | Field | Format | Unit of last part |
  |---|---|---|---|
  | results (`LocalResults.aspx`) | `FinishTime` | `m:ss.ff` | hundredths |
  | horse page (`Horse.aspx`) | `FinishTime` | `m.ss.ff` | hundredths |
  | racecard | `Best Time` | `m.ss.ff` / `ss.ff` | hundredths |
  | barrier trials | batch `Time`, horse `Time` | `m.ss.ff` / `ss.ff` | hundredths |
  | barrier trials | `Sectional Time` | `ss.ff ss.ff ss.ff` | hundredths |
  | trackwork | gallop total `(…)` | `m.ss.f` / `ss.f` | **tenths** |
  | trackwork | gallop splits | `ss.f` per furlong (201.168 m) | tenths |
- **`RunningPosition`** — space-separated positions at each call (up to 4 read: start, 2nd, 3rd,
  finish; finish falls back to the 3rd call if missing). Running **style**: `--` if no start
  position; `FrontRunner` if mean(start, 2nd) ≤ 3 and start ≤ 3; `Closer` if start ≥ 6 and
  start − finish ≥ 1; else `Pacer`. `FavoriteRunningStyle` = mode over the 5 most recent runs.
- **`RaceId`** — synthesised as `{YYYYMMDD}{Venue}{RaceIndex}`.
- **Beaten margins (`LBW`)** — lengths as `8-1/2L`, `8 1/2L`, `3/4L`, `8L`; named short margins
  `NK`/`N` neck ≈ 0.3 L, `HD` head ≈ 0.2 L, `SH` short head ≈ 0.1 L, `DH` dead heat 0; blank or `-`
  for the leader.
- **Jockey allowance** — apprentice claim printed in the jockey cell as `NAME (-2)` (lbs).
- **Horse ids** — brand `[A-Z]\d+` (e.g. `K400`) vs registration `HK_YYYY_<brand>`; which page
  takes which is in HKJC_DATA §2.
- **SpeedPro figures** (published shortly before a meeting; `null` until then):
  *energy required* (race-level pace demand), per-horse *SpeedPro energy* and *energy
  difference* (surplus vs requirement), *fitness rating*.

---

## 5. Betting pools

> **Profit is decided here, not in the model.** The model produces probabilities; the
> *payout* depends on the pool's takeout and the crowd's money. Edge must always be
> measured **net of the specific pool's takeout**.

### 5.1 Single-race pools

| Pool | Code | Win condition | Field-size rule | Takeout |
| --- | --- | --- | --- | --- |
| **Win** | `WIN` | the 1st horse | — | 17.5% |
| **Place** | `PLA` | a paying-place finisher | **≥7 runners → top 3**; **4–6 → top 2**; **≤3 → no pool** | 17.5% |
| **Quinella** | `QIN` | 1st & 2nd, any order | small fields may suspend | 17.5% |
| **Quinella Place** | `QPL` | any two of the top three | needs a 3-place (≥7) race | 17.5% |
| **Forecast** | `FCT` | 1st-2nd, exact order | — | 19.5% |
| **Tierce** | `TCE` | 1st-2nd-3rd, exact order | — | 25% |
| **Trio** | `TRI` | top 3, any order | — | 23% |
| **First 4** | `FF` | top 4, any order | — | 25% (merged with Quartet) |
| **Quartet** | `QTT` | 1st-2nd-3rd-4th, exact order | — | 25% (merged with First 4) |
| **Composite Win** | `CWA/CWB/CWC` | the *group* containing the winner | bracket/coupled/commingled | not listed on the pools page |

Takeout = 100% minus the share of the pool paid out as dividends, from HKJC's local-pools page
(special.hkjc.com, betting-info / racing / beginners-guide / local-pools; read 2026-11-21): 82.5% paid out
for WIN/PLA/QIN/QPL/Double, 80.5% Forecast, 77% Trio, 75% Tierce/First 4/Quartet/Treble/Double Trio/Triple
Trio/Six Up. The merged Quartet & First 4 pool also deducts 0.5% as a Jackpot Reserve Deduction.

- **Place place-count is field-size-dependent** — the single most important rule to
  encode right: 3 places for 7+ runners, 2 for 4–6. A wrong count silently corrupts
  every Place/QPL expected value.
- **First 4↔Quartet** are one merged pool; treat them as economically linked.
- Composite Win minimum dividend is gentler: no dividend below unit + $0.10 (vs +$0.50).


### 5.2 Multi-race pools

| Pool | Code | Win condition | Jackpot? |
| --- | --- | --- | --- |
| **Double** | `DBL` | winner of 2 nominated races | No (consolation) |
| **Treble** | `TBL` | winner of 3 nominated races | No (consolation) |
| **Double Trio** | `DT` | the Trio in 2 nominated races | **Yes** |
| **Triple Trio** | `TT` | the Trio in 3 nominated races | **Yes** (+ consolation) |
| **Six Up** | `SixUP` (GraphQL) / `SIX UP` label on result pages | 1st or 2nd in each of 6 nominated races | **Yes** (+ Six Win Bonus) |

Multi-race pools are **products of single-race distributions** (races are independent
draws), so a Triple Trio probability is the product of three Trio probabilities — each
already estimable by the `RaceSimulator`. The combinatorics explode, so these are a
*search/optimisation over tickets* (Flexi covers), not a modelling problem.
**All-Up / Cross-Pool All-Up** is a compounding wrapper (reinvest each leg's winnings),
not a pool of its own.

## 6. Pool economics

```
gross_pool = Σ stakes ;  net_pool = gross_pool × (1 − takeout)
dividend_per_unit(combo) = net_pool × winning_share(combo) / stakes_on(combo)
```

- **Takeout is the operator's only structural edge** and **differs by pool** (17.5%
  WIN/PLA/QIN/QPL/Double, 19.5% Forecast, 23% Trio, 25% Tierce/First 4/Quartet/Treble and
  the jackpot pools). A 5% model edge is profitable at 17.5%, underwater at 25%: use the
  per-pool rate.
- **Dividend is endogenous to the crowd:** your payout falls the more money is on your
  combination. The model predicts *probabilities*; the *odds/dividend* come from the
  live pool. We assume our stake is small vs the pool (ignore own price impact), but the
  crowd's distribution is the denominator of every payout.
- **Minimum dividend:** no dividend below unit + $0.50 (WIN/QIN/QPL/Double); + $0.10
  (Composite). A raging favourite can be capped at this floor.
- **Unit bet / Flexi:** minimum HK$10 per bet; **Flexi Bet** allows a $1 unit (variable
  minimum ticket total) — buy partial units of a large multi-combination ticket.
- **Declared dividends are quoted per HK$10 unit** on result pages (decimal odds = dividend / 10).
- **Rebates:** loss-rebate promotions effectively reduce takeout (e.g. ≥HK$10,000 of
  losing QIN/QPL on one line has received a 12% rebate) — model as negative takeout on
  the rebated fraction.

## 7. Jackpot & carryover mechanics

Double Trio, Triple Trio and Six Up are **jackpot pools**. A slice is skimmed into a
reserve that **carries forward if nobody wins**:

```
Triple Trio: Jackpot Reserve Deduction   7.5% of pool
Double Trio: Operator Reserve Deduction  0.5% of pool (from the Club's revenue)
Six Up:      Operator Reserve Deduction  2%   of pool (from the Club's revenue)
```

- Normal day: ~85% is the net dividend pool; the reserve builds quietly.
- **No winning unit ⇒ the entire net pool carries forward**, aggregating with next
  meeting's net pool.
- On a **carryover ("Fabulous"/"Snowball"/National-Day) meeting**, the pool paying out
  can *exceed* the money bet that day: the effective takeout on the **new** money is
  **negative** — a structural +EV window independent of any modelling edge. For a cover
  `S`: `effective_takeout = (1−net_fraction) − C/G`, +EV when carried-in `C/G > ~0.15`.
  The GraphQL `raceMeetings` op exposes `jpEsts.jackpot` (=C), `poolInvs.investment` (=G) and
  `estimatedDividend`, joined on (pool, leg races) (HKJC_DATA §1.3); they are null until pools open.
  Result pages show a jackpot dividend only when the pool was won and never the carried-in amount,
  so C must be captured live during the betting window.

## 8. Open verification tasks (rules are priors — HKJC tunes them per season)

- [ ] Confirm the Composite Win takeout (not on the local-pools page) from the live rule PDFs
      (`hkjc.com/.../template_betting_rule_files/pdf/`).
- [ ] Confirm exact **Place field-size thresholds** and small-field suspension rules.
- [ ] Pin down **Triple Trio consolation** and **Six Win Bonus** split formulas.
- [ ] Confirm **refund/scratching** treatment per pool (WIN/PLA refund vs exotic
      re-banding) and **dead-heat** division per pool.
- [ ] Prefer scraping the live `estimatedDividend` / `jpEsts` fields over hard-coding.
