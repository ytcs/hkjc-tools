# HKJC Tribal Knowledge

Hard-won knowledge about HKJC's data sources, endpoints, domain conventions and
gotchas. Distilled from the (now removed) Flask dashboard and from probing the
live site. Treat this as the field guide for extending the data layer or building
new tooling — most of it is **not** obvious from the upstream site.

Venue codes used everywhere: **`ST` = Sha Tin**, **`HV` = Happy Valley**. These
are the only two local courses.

---

## 1. Data sources & endpoints

### 1.1 Live odds + race cards — GraphQL  *(used by `hkjc.live`)*

```
POST https://info.cld.hkjc.com/graphql/base/
Headers: Origin/Referer = https://bet.hkjc.com, Content-Type: application/json
```

Two operations (full query bodies live in `src/hkjc/live.py`):

- **`raceMeetings`** — the whole meeting: every race, runners, draws, ratings,
  weights, jockey/trainer, gear, `last6run`, win odds, and pool metadata.
- **`racing`** — odds nodes for given `oddsTypes` and `raceNo`.

Passing `date=""`/`venueCode=""` (or `null`) makes the server return the
**current meeting**; the code then prefers the first `HV`/`ST` meeting in the
response. Always reconcile the returned `Date`/`Venue` with what you asked for —
the server can hand back a different meeting.

**Pool / odds types** the endpoint accepts:
`WIN, PLA, QIN, QPL, CWA, CWB, CWC, IWN, FCT, TCE, TRI, FF, QTT, DBL, TBL, DT, TT, SixUP`
and forecast pools `JKC` (jockey challenge), `TNC` (trainer challenge).
The library currently only parses `WIN, PLA, QIN, QPL` — the rest are available
for the taking.

**Rich fields exposed but NOT yet extracted** (roadmap for the agent):
- `winOdds`, `finalPosition`, `deadHeat` (live result as it comes in)
- `jockey`, `trainer`, `gearInfo`, `last6run`, `currentRating`,
  `internationalRating`, `allowance`, `trainerPreference`, `trumpCard`
- `changeHistories` — **scratchings / jockey & weight changes** (top 3)
- `penetrometerReadings`, `hammerReadings` — **turf firmness measurements**
- `jpEsts` / jackpots and `estimatedDividend` per pool
- `totalInvestment`, per-pool `investment` (pool sizes → liquidity signal)

### 1.2 Historical results — HTML  *(used by `hkjc.historical`)*

```
https://racing.hkjc.com/racing/information/English/Racing/LocalResults.aspx?RaceDate={YYYY/MM/DD}&Racecourse={ST|HV}&RaceNo={n}
Headers: Origin/Referer = https://racing.hkjc.com
```

- Date format here is **`YYYY/MM/DD`** (slashes), unlike the GraphQL/ISO
  `YYYY-MM-DD` used in `hkjc.live`.
- Server-rendered HTML; parsed with BeautifulSoup `html.parser`.
- Result table is `div.performance table`; the race header is `div.race_tab table`.
- Horse brand/ID is embedded in the `Horse` cell as `NAME (CODE)`, e.g.
  `RISING WORLD (L276)`.

### 1.3 Horse form — HTML  *(used by `hkjc.historical`)*

```
https://racing.hkjc.com/racing/information/English/Horse/Horse.aspx?HorseNo={code}
```

`{code}` is the horse **brand number** (e.g. `K281`, `L276`), available as
`HorseNo`/`HorseID` from both the live card and the results table. Form table is
`table.bigborder`.

### 1.4 SpeedPro — JSON  *(connector removed — deprioritized)*

> The `hkjc.speedpro` module was removed (judged unlikely to produce edge given the
> market-efficiency results). The endpoint is documented here in case it's revisited.

```
https://racing.hkjc.com/racing/speedpro/assets/json/speedguide/race_{n}.json
```

Per-race file for the **current meeting only** (not addressable by date). Decode
with `utf-8-sig` (BOM). Structure: `{ "en-us": {...}, "zh-hk": {...} }`, each with:
- `RaceMap` — base64 `data:image/jpeg` **speed map** image.
- `SpeedPRO` — list of runners with `energyrequired` (race-level),
  `speedproenergy`, `speedproenergydifference`, `fitnessrating`, plus per-runner
  recent-run energy history (`lastrun`, `lastrun_2nd`…`lastrun_5th`,
  `bestlast12months`, `bestatdistance`).
- `RaceInfoEng` — date (**`DD/MM/YYYY`** here!), course, track, class.

> ⚠️ **Retired endpoint:** the old energy-grid spreadsheet
> `…/speedpro/assets/excel/{YYYYMMDD}_energygrid_en.xls` now **404s for every
> date**. The JSON above replaced it.

### 1.5 Track-workout videos  *(NOT in the library — preserved from dashboard)*

Morning track-work ("twstarter") clips, ~2500 kbps MP4:

```
https://streaminghkjc-a.akamaihd.net/hdflash/twstarter/{YYYY}/{YYYYMMDD}/{RR}/novo/twstarter_{YYYYMMDD}_{RR}_novo_2500kbps.mp4
```

where `{RR}` is the **zero-padded** race number (`01`, `02`, …). Example for
race 3 on 2026-06-03:
`…/twstarter/2026/20260603/03/novo/twstarter_20260603_03_novo_2500kbps.mp4`.

### 1.6 Tips index  *(NOT in the library — preserved from dashboard)*

Pundit tips for a race (classic ASP page):

```
https://racing.hkjc.com/racing/English/tipsindex/tips_index.asp?RaceNo={n}
```

### 1.7 Misc assets seen in the dashboard

- Horse silk/icon SVGs under `https://consvc.hkjc.com/.../Racing/horse/horse.svg`.
- Replays/race video also live under the `racing.hkjc.com` "VideoReplay" column
  in the form table (dropped during cleaning in `historical._clean_horse_data`).

---

## 2. Domain conventions

### 2.1 Tracks, courses, going

- **Track types:** `Turf` and `AWT` (All Weather Track).
- **Movable running rail (the `Course`):** the turf rail is shifted out
  meeting-to-meeting to spread wear, which **changes the usable width and the
  draw-bias geometry per race** — geometry is *not* fixed per venue. The `Course`
  field (`A`, `A+2`, `B`, `B+2`, `B+3`, `C`, `C+3`) sets the rail position.
  Official HKJC widths (encoded in `hkjc.track`):

  | Course | Sha Tin width | Happy Valley width |
  | --- | --- | --- |
  | A | 30.5m | 30.5m |
  | A+2 | 28.5m | 28.5m |
  | B | 26.0m | 26.5m |
  | B+2 | 24.0m | 24.5m |
  | C | 21.3m | 22.5m |
  | C+3 | 18.3m | 19.5m |

- **Geometry is reference data, resolved per race.** `hkjc.track` holds a
  `TrackGeometry` per *venue × surface × configuration* and a resolver
  `track_for(venue, distance, surface)`. The canonical `Race` object
  (`hkjc.race`) carries `(venue, distance, surface, course, going, runners)` and
  exposes `.geometry`, `.width`, `.bend_fraction`, and one-call model builders
  (`.traffic_model()`, `.energetics_model()`, `.plackett_luce()`) — the agent-based
  modelling seam. The four encoded geometries:
  - **Happy Valley** (`HAPPY_VALLEY`): turf, 1,450m, right-handed, **bend-dominated**
    — home straight ~[0,300]; Sanatorium/Crescent/"The Rock" bends ~[300,770]; back
    straight ~[770,920]; HKFC/Stable bends + top turn ~[920,1450] (bend map from the
    sectional sampling points). Near-continuous turning ⇒ the **strongest draw bias
    in HK** (low draws hold the short inner path). Distances 1000/1200/1650/1800/2200m.
  - **Sha Tin turf** (`SHA_TIN_TURF`): 1,900m galloping oval, right-handed, 430m home
    straight. Long straights ⇒ a **weaker** geometric draw bias than HV (the model
    reproduces this: `bend_fraction` is lower). Bend boundaries are a two-turn
    reconstruction from circumference + home straight (approximate; the straight/turn
    split and total are firm).
  - **Sha Tin straight 1000m** (`SHA_TIN_STRAIGHT`): a separate sprint **chute with no
    bends** ⇒ **no bend-geometry draw bias** at all (only the cross-track watering/going
    gradient, which the geometry deliberately does not model). `track_for("ST",1000)`
    returns this, not the oval.
  - **Sha Tin all-weather** (`SHA_TIN_AWT`): dirt, 1,560m, right-handed, 365m home
    straight, **fixed single rail** (uniform 22.8m). Surface inferred from the
    "ALL WEATHER TRACK" label (going codes alone are ambiguous — both scales have GOOD).

  Draw bias is most acute at HV and the short ST turf trips; flat on the ST 1000m
  chute. See the official course page and `DrawStatistics` for realised bias.
- **Going codes** (string → short code), as used to filter form by surface:

  Turf: `FIRM→F, GOOD TO FIRM→GF, GOOD→G, GOOD TO YIELDING→GY, YIELDING→Y,
  YIELDING TO SOFT→YS, SOFT→S, HEAVY→H`

  AWT: `WET FAST→WF, FAST→FT, GOOD→GD, SLOW→SL, WET SLOW→WS, RAIN AFFECTED→RA,
  NORMAL WATERING→NW`

### 2.2 Pools / bet types (HKJC)

- **WIN** — 1st. **PLA** (place) — top **3** if ≥7 runners, top **2** if 4–6, none
  if ≤3.
- **QIN** (quinella) — pick the first two in any order.
- **QPL** (quinella place) — pick any two of the top three.
- Exotics: **FCT** forecast (1st-2nd ordered), **TCE** tierce (1st-3rd ordered),
  **TRI** trio (top-3 any order), **QTT** quartet, **FF/First-4**, **DBL/TBL**
  double/treble, **DT/TT** double/triple trio (**jackpot pools**), **6UP**.
- The `simulation.RaceSimulator` predicates map onto these: `wins`,
  `places(k=2|3)`, `exact_order` (FCT/TCE/straight exotics), `top_set`
  (QIN/TRI/QTT box), `all_place`/`any_place`.

> **Takeout is NOT uniform across pools** (~17.5% for WIN/PLA/QIN/QPL up to ~25%
> for QTT/FF), and the jackpot pools carry money forward. The model's
> `takeout_rate=0.175` default is only valid for the WIN/PLA/QIN/QPL family — see
> [`HKJC_BETTING_RULES.md`](HKJC_BETTING_RULES.md) for the full pool catalogue,
> per-pool takeout, dividend mechanics, jackpot/carryover rules and special
> payouts.

### 2.3 Runner attributes

- `Dr` / `barrierDrawNumber` — **draw** (gate); low draws favoured at some
  HV/ST course-distance combos.
- `Rtg` / `currentRating`, `internationalRating` — official handicap rating.
- `Wt` / `currentWeight` and `Handicap`/`handicapWeight` — carried weight in lbs.
- `last6run` — recent finishing positions string.
- Runner `status` of **`Standby`** = reserve (excluded by the live parser);
  scratched runners flagged via `changeHistories` / `scratched`.

### 2.4 Result / form fields

- `Pla` — finishing position; may carry **incident codes** instead of a number:
  `DISQ, DNF, FE, ML, PU, TNP, TO, UR, VOID, WR, WV, WV-A, WX, WX-A, WXNR`
  (filtered out during cleaning).
- `FinishTime` — `m.ss.ff` on horse pages, `m:ss.ff` on race pages; both parsed
  to seconds.
- `RunningPosition` — sectional positions; classified into **running style**:
  `FrontRunner` (avg of first two sectionals ≤3 and led early), `Closer`
  (started ≥6, gained ≥1), else `Pacer`. `FavoriteRunningStyle` = mode of last 5.
- `RaceId` — synthesised as `{YYYYMMDD}{Venue}{RaceIndex}`.

### 2.5 SpeedPro figures

- **Energy required** — race-level pace/energy demand.
- **SpeedPro energy** (per horse) and **energy difference** — the horse's energy
  vs requirement; positive = surplus. Only published **shortly before a meeting**;
  fields are `null` until then.
- **Fitness rating** — published alongside.

---

## 3. The edge concept: raw vs fitted odds

The dashboard's core idea (now generalised into the model/simulation layer): fit
the **Harville model** to the market's exotic pools, convert back to implied
WIN/PLA/QIN/QPL odds, and compare to the **raw** market odds. Cells where the
**fitted odds differ from raw** flag where the market's pools disagree with a
self-consistent joint model — i.e. candidate mispricings to bet into.

- `harville_model.fit_harville_to_odds(odds)` returns both `prob_fit` (the true
  probability model: `theta`, `W`, `Qin`, `Q`, `b`, full place matrix `P`) and
  `odds_fit` (those probabilities re-expressed as market-style odds with takeout
  re-applied) — directly comparable to the input odds.
- Default pool weights `λ = {WIN:1.0, QIN:2.0, QPL:1.5, PLA:0.7}` encode the
  belief that **early win odds are noisier** (informed money comes late) while
  exotic pools are more stable signals. Default `takeout_rate = 0.175` (17.5%).

---

## 4. Feature hypotheses & ranking — *priors, now mostly FALSIFIED*

> **Update — these were tested and do NOT beat the market.** The standard
> fundamentals below (market odds, draw, weight, **official rating**, **recent
> form**) were run through the walk-forward harness and add nothing over the
> closing win line — form even *hurts* (journal 0011 & 0023). The win market is
> efficient w.r.t. all of these. Kept as a record of what was tried and ruled out;
> a model edge must come from information the market lacks (sectionals/trajectory,
> trip comments, alt-data), not from re-weighting these.

### 4.1 The features we watched, and the "good" direction

Each runner was scored against these per-race attributes (dashboard column →
belief):

| Feature | Preferred when | Rationale (prior) |
| --- | --- | --- |
| **Draw** | gate ∈ {3, 4, 5} | Mid-low barrier: clear of the rail trouble, not stuck wide. Course/distance dependent — strongest at HV and short ST trips. |
| **Running style** | `FrontRunner` | Front-runners control pace; HV especially rewards leading. |
| **Last finish time** | fastest in race | Recent raw speed, crude (ignores class/going/distance of that run). |
| **Official rating** | highest in race | Class. See also `RtgDiff` below. |
| **Fitness rating** | ≥ 3 | SpeedPro fitness; a readiness floor rather than a ranking. |
| **SpeedPro energy difference** | highest | Energy surplus vs the race's energy requirement. |
| **Adjusted energy difference** | highest | An adjusted variant of the above (course/going-adjusted). |
| **Win odds** | shortest (>0) | Market favourite — the crowd's aggregate signal. |
| **Place odds** | shortest (>0) | Market place signal. |

Derived features in `hkjc.features` (Polars expressions over a meeting frame):

- **`RtgDiff`** = `max(Rtg) over race − Rtg` — gap to the top-rated runner. We
  believed *relative* class within the race beats absolute rating.
- **`FRPct`** = share of the field whose `FavoriteRunningStyle` is `FrontRunner`
  — a **pace-setup** feature: many front-runners → a fast/contested pace that
  favours closers; lone leader → pace advantage. (A race-level feature, used to
  re-interpret each runner's style.)

### 4.2 How we ranked horses (the aggregation)

A deliberately dumb ensemble — **feature voting**:

1. For each feature, mark the horse(s) that are best on it (max/min, or that pass
   the absolute check like draw ∈ {3,4,5} or fitness ≥ 3).
2. Each horse's score = **count of features it leads** (every feature weighted
   **equally**, one vote each).
3. The horse(s) with the highest count are flagged "most preferred" (a `N★` badge).
4. **Uniformity guard:** any feature on which the whole field shares one value is
   dropped for that race — a non-discriminating feature casts no vote. (Worth
   keeping in any successor: don't let dead features dilute the signal.)

Known weaknesses (improvement targets for the agent): equal weights are
arbitrary; hard thresholds (draw 3–5, fitness ≥3) are uncalibrated; ties are not
broken; no feature **interactions** (e.g. draw × style × pace, energy × distance);
last-finish-time isn't normalised for class/going/distance. This voting scheme was
a visual aid, never validated against results.

### 4.3 Pool-reliability prior (model weights)

Inside the Harville fit, pools are trusted unequally:
`λ = {WIN:1.0, QIN:2.0, QPL:1.5, PLA:0.7}` — **exotic pools (QIN/QPL) weighted
above WIN/PLA** because we believed informed money arrives late in the win pool,
making early win odds noisier than the exotics. Also a prior; tune against data.

### 4.4 How we evaluated *bets* (when strategies still existed)

The removed trade tooling ranked candidate tickets by:
- an **edge threshold** of `ExpValue ≥ 0.05` (require ≥5% modelled edge), then
- a **Pareto frontier** over (`WinProb` ↑, `ExpValue` ↑) **within each cover-set
  size** — i.e. for each number of legs, keep only tickets not dominated on both
  win-probability and EV. `hkjc.utils.pareto_filter` still implements this and is
  a reusable way to surface non-dominated options on any 2+ objectives.

---

## 5. Scraping discipline (don't get rate-limited)

The dashboard polled HKJC politely; reuse these defaults for any new fetcher:

- **Odds poll interval:** ~90 s, with ~10 s random jitter (odds barely move
  faster than this except near the off).
- **Concurrency:** ≤ **3** simultaneous requests.
- **Inter-request delay:** random **80–250 ms**.
- **Timeout:** ~10 s per request.
- Library-side TTL caches: live races 1000 s, live odds 30 s, HTML pages 3600 s,
  SpeedPro 300 s — tune rather than hammer.
- Send a real browser `User-Agent` and the matching `Origin`/`Referer`
  (`bet.hkjc.com` for GraphQL, `racing.hkjc.com` for HTML/JSON).

---

## 6. Gotchas (learned the hard way)

- **Non-race dates** return a styled "No information." error page with **zero
  `<table>` elements** — `historical` raises `AttributeError`. The strings
  `race_tab`/`performance` appear in that page's **CSS only**, so don't test for
  them; check for actual tables. Always target a real meeting date.
- **Two date formats:** ISO `YYYY-MM-DD` for `hkjc.live`; slashed `YYYY/MM/DD`
  for `hkjc.historical`; `DD/MM/YYYY` inside SpeedPro JSON. Easy to mix up.
- **SpeedPro staleness:** the `race_n.json` files can serve a previous meeting's
  cached data, and energy/fitness are blank until published. Don't assume the
  JSON reflects today.
- **QIN/QPL diagonals** are self-pairs with zero probability → `1/p = inf`;
  handled with `nan_to_num`/`np.nan`. Expect NaNs on matrix diagonals.
- **Harville cost is `O(N·2^N)`** with a one-off Numba JIT warm-up (~10 s) on the
  first call per process; cached thereafter. Keep fields ≤ ~20 runners.
- Empty-string `date`/`venue` to the live endpoints is a **feature** (server
  picks the current meeting), not a bug — but verify what came back.
