# HKJC Data — the complete data surface

A comprehensive survey of **every HKJC data source we know of** — what it exposes, how
to fetch it, what's already wired, and (the point of this doc) the **deeply-buried,
high-value data we do not yet mine**. The thesis behind the whole project: an edge must
come from information the efficient win market underweights, so knowing the full data
surface *is* the search space.

> Two API surfaces. **(A) Live odds + cards: GraphQL** at `info.cld.hkjc.com/graphql`
> only. **(B) Everything else: server-side-rendered HTML** at `racing.hkjc.com` —
> `requests.get()` returns full HTML (parse with BeautifulSoup), no browser needed, *if*
> you use the right params. There is no public per-page JSON for racing data.

Implementation status legend: ✅ wired in `hkjc.data` · 🟡 partial · ⬜ not wired · 📜 not
wired, with a complete fetch-and-parse recipe below.
Of the §1–§3 sources, only the results table (`LocalResults.aspx`) and `displaysectionaltime` are
wired today (both via `hkjc.data.race_info`); replay videos (§4), silks and track geometry stay
wired in their own modules.

---

## 1. Live: GraphQL (odds, cards, pools, jackpots)

```
POST https://info.cld.hkjc.com/graphql/base/
Origin: https://bet.hkjc.com
Referer: https://bet.hkjc.com
Content-Type: application/json
Accept: application/json
User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/141.0.0.0 Safari/537.36
body: {"operationName": ..., "variables": {...}, "query": "<exact text, §1.2>"}
```
No auth, cookie or token. A 30 s timeout works; treat any non-200 as an error.
The response is `{"data": {"raceMeetings": [...]}}`.

Two operations (exact query texts in §1.2):
- **`raceMeetings`** 📜: variables `{"date": "YYYY-MM-DD" | null, "venueCode": "HV" | "ST" | null}`.
  Returns the whole meeting: every race, runners, draws, ratings, weights, jockey/trainer, gear,
  `last6run`, win odds, pool metadata.
- **`racing`** 📜: variables `{"date": "YYYY-MM-DD", "venueCode": "HV"|"ST", "raceNo": int,
  "oddsTypes": ["WIN","PLA","QIN","QPL"]}`. Returns odds nodes per pool for one race.

Passing `date=""`/`venueCode=""` (or `null`) returns the **current meeting** (pick the first
`HV`/`ST` entry); **always reconcile the returned `Date`/`Venue`** with what
you asked for — the server can hand back a different meeting.

**Pool / odds types accepted:** `WIN, PLA, QIN, QPL, CWA, CWB, CWC, IWN, FCT, TCE, TRI,
FF, QTT, DBL, TBL, DT, TT, SixUP`, plus `JKC` (jockey challenge), `TNC` (trainer
challenge). The odds recipe (§1.3) covers `WIN, PLA, QIN, QPL`; the rest are unextracted ⬜.

**Rich fields exposed but NOT yet extracted** (a roadmap):
- `winOdds`, `finalPosition`, `deadHeat` — live result as it lands.
- `jockey`, `trainer`, `gearInfo`, `last6run`, `currentRating`, `internationalRating`,
  `allowance`, **`trainerPreference`** (the trainer's own ranking of their runners — intent),
  **`trumpCard`** (bool stable-confidence flag). Both confirmed **populated pre-race** — these
  are a zero-extra-fetch intent signal, just extract them.
- `changeHistories(filters:["top3"])` — **scratchings / jockey & weight changes**; returns a
  *current/latest* feed (ignores the `date` param — same records for any past date queried).
- `penetrometerReadings`, `hammerReadings` — turf firmness, but observed **empty** on upcoming
  AND recent-past meetings (only populated transiently raceday morning). **Unreliable — prefer
  the `weather` op's soil-moisture fields (§1.1) for firmness.**
- `jpEsts` / jackpots and `estimatedDividend` per pool 📜 (join recipe in §1.3).
- `totalInvestment`, per-pool `investment` — **pool sizes → liquidity signal** 📜.

### 1.1 `weather` operation — wind + soil moisture (NEW, the ENVIRONMENT feed) ⬜
Same endpoint, `operationName: Weather`, `variables: null`, `Origin: https://racing.hkjc.com`.
The server **whitelists by exact query body** (any field-set change → `WHITELIST_ERROR`).
Returns **both ST & HV**, `max-age=15`. Per racecourse:
- `sectional[]` — **four anemometers (A/B/C/D)** around the track, each with
  `avgCorrectedWindDirection` (compass, e.g. `N80W`), `avgCorrectedWindSpeed`, gust dir/speed,
  `gpsPosition` (lat:lon:elev). → the **real per-straight wind vector** (head/tail/cross),
  projectable onto each straight's heading via the GPS (wind exposure, draft).
- `weatherStation` (loc E) — the above wind fields **plus** `temperature`, `relativeHumidity`,
  `pressure`, `solarRadiation`, `sunshineHour`, daily max/min, `rainPrecipitationDailyTotal`,
  **`soilVolumeticWaterContent`**, **`soilMoistureLoss{1Min,DailyTotal,24H}}`**. → a **continuous
  going/firmness proxy** far richer than the categorical going string (going-affinity was found
  noise-limited by soft-going *sparsity*; a continuous firmness covariate sidesteps that).

### 1.2 Exact query texts (`raceMeetings`, `racing`)

Send the `query` string byte-for-byte. The `weather` op is whitelisted by exact body (§1.1); whether
these two are is untested, so do not reformat them. Each block below is the string **minus its
leading newline**: the real string starts with `\n`. The `raceMeetings` string ends with `\n`, the
`racing` string ends with `}` (no newline). Check with sha256 of the full string.

`raceMeetings` (`"operationName": "raceMeetings"`; 3678 chars; sha256 `2fbb1cf9ea40c4eec1e74e5308920bb4545d2e74b765ebb2ef20273c0ccb121a`):

```graphql
    fragment raceFragment on Race {
    id
    no
    status
    raceName_en
    raceName_ch
    postTime
    country_en
    country_ch
    distance
    wageringFieldSize
    go_en
    go_ch
    ratingType
    raceTrack {
        description_en
        description_ch
    }
    raceCourse {
        description_en
        description_ch
        displayCode
    }
    claCode
    raceClass_en
    raceClass_ch
    judgeSigns {
        value_en
    }
    }

    fragment racingBlockFragment on RaceMeeting {
    jpEsts: pmPools(
        oddsTypes: [WIN, PLA, TCE, TRI, FF, QTT, DT, TT, SixUP]
        filters: ["jackpot", "estimatedDividend"]
    ) {
        leg {
        number
        races
        }
        oddsType
        jackpot
        estimatedDividend
        mergedPoolId
    }
    poolInvs: pmPools(
        oddsTypes: [WIN, PLA, QIN, QPL, CWA, CWB, CWC, IWN, FCT, TCE, TRI, FF, QTT, DBL, TBL, DT, TT, SixUP]
    ) {
        id
        leg {
        races
        }
    }
    penetrometerReadings(filters: ["first"]) {
        reading
        readingTime
    }
    hammerReadings(filters: ["first"]) {
        reading
        readingTime
    }
    changeHistories(filters: ["top3"]) {
        type
        time
        raceNo
        runnerNo
        horseName_ch
        horseName_en
        jockeyName_ch
        jockeyName_en
        scratchHorseName_ch
        scratchHorseName_en
        handicapWeight
        scrResvIndicator
    }
    }

    query raceMeetings($date: String, $venueCode: String) {
    timeOffset {
        rc
    }
    activeMeetings: raceMeetings {
        id
        venueCode
        date
        status
        races {
        no
        postTime
        status
        wageringFieldSize
        }
    }
    raceMeetings(date: $date, venueCode: $venueCode) {
        id
        status
        venueCode
        date
        totalNumberOfRace
        currentNumberOfRace
        dateOfWeek
        meetingType
        totalInvestment
        country {
        code
        namech
        nameen
        seq
        }
        races {
        ...raceFragment
        runners {
            id
            no
            standbyNo
            status
            name_ch
            name_en
            horse {
            id
            code
            }
            color
            barrierDrawNumber
            handicapWeight
            currentWeight
            currentRating
            internationalRating
            gearInfo
            racingColorFileName
            allowance
            trainerPreference
            last6run
            saddleClothNo
            trumpCard
            priority
            finalPosition
            deadHeat
            winOdds
            jockey {
            code
            name_en
            name_ch
            }
            trainer {
            code
            name_en
            name_ch
            }
        }
        }
        obSt: pmPools(oddsTypes: [WIN, PLA]) {
        leg {
            races
        }
        oddsType
        comingleStatus
        }
        poolInvs: pmPools(
        oddsTypes: [WIN, PLA, QIN, QPL, CWA, CWB, CWC, IWN, FCT, TCE, TRI, FF, QTT, DBL, TBL, DT, TT, SixUP]
        ) {
        id
        leg {
            number
            races
        }
        status
        sellStatus
        oddsType
        investment
        mergedPoolId
        lastUpdateTime
        }
        ...racingBlockFragment
        pmPools(oddsTypes: []) {
        id
        }
        jkcInstNo: foPools(oddsTypes: [JKC], filters: ["top"]) {
        instNo
        }
        tncInstNo: foPools(oddsTypes: [TNC], filters: ["top"]) {
        instNo
        }
    }
    }
```

`racing` (`"operationName": "racing"`; 889 chars; sha256 `6103150c8dcad13852d99ba224c89e0f0935e6ba0908087b06aa0edcff31a27f`):

```graphql
query racing($date: String, $venueCode: String, $oddsTypes: [OddsType], $raceNo: Int) {
    raceMeetings(date: $date, venueCode: $venueCode) {
        pmPools(oddsTypes: $oddsTypes, raceNo: $raceNo) {
            id
            status
            sellStatus
            oddsType
            lastUpdateTime
            guarantee
            minTicketCost
            name_en
            name_ch
            leg {
                number
                races
            }
            cWinSelections {
                composite
                name_ch
                name_en
                starters
            }
            oddsNodes {
                combString
                oddsValue
                hotFavourite
                oddsDropValue
                bankerOdds {
                    combString
                    oddsValue
                }
            }
        }
    }
}
```

**Request details.**
- Endpoint: `POST https://info.cld.hkjc.com/graphql/base/` (the `/graphql` in the summary above is shorthand).
  Headers `{"Origin": "https://bet.hkjc.com", "Referer": "https://bet.hkjc.com", "Content-Type": "application/json",
  "Accept": "application/json", "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like
  Gecko) Chrome/141.0.0.0 Safari/537.36"}`,
  timeout 30 s, body = `{"operationName"?, "variables": {...}, "query": "<text>"}` sent with `requests.post(json=...)`.
- `date` is sent exactly as given (`YYYY-MM-DD`); `venueCode` `"ST"`/`"HV"`. Current meeting: send `null` for both
  (an empty string also selects the current meeting; prefer null).
- `oddsTypes` in the `racing` query is a list in the caller's order; typical values are `["PLA"]` (single-pool
  fetch) and `["WIN", "PLA", "QPL", "QIN"]` (the four main pools).

### 1.3 Extraction recipes

- **Meeting choice.** `data.raceMeetings` is a list: take the first entry with `venueCode` in
  `HV`/`ST`, else the first. Compare its `date`/`venueCode` with the request and continue with the
  server's values (warn).
- **Card** (`raceMeetings`): per race `no`, `raceName_en`, `raceClass_en`, `raceCourse.displayCode`, `distance`, `go_en`, `raceTrack.description_en`; per runner with
  `status != "Standby"`: `no`, `name_en`, `barrierDrawNumber`, `currentRating`, `currentWeight`,
  `handicapWeight`, `horse.code` (brand). Map unparseable ints to null (if they are coerced
  to 0 instead, treat 0 rating/weight as missing).
- **Odds** (`racing`): for each `raceMeetings[].pmPools[]` and each `oddsNodes[]`: `combString`,
  `oddsValue` (float; unparseable -> NaN), pool `oddsType`, pool `lastUpdateTime`. `combString` is
  the runner number for `WIN`/`PLA` and `"a,b"` for `QIN`/`QPL` (fill both `[a,b]` and `[b,a]`;
  diagonal stays NaN). Array size = number of non-Standby runners in the race from `raceMeetings`;
  index = runner number − 1. For a time series, poll uncached and dedupe on `lastUpdateTime` (server
  clock), not on the polling clock (an in-process cache would repeat snapshots).
- **Jackpots / pool sizes** (`raceMeetings`): index `jpEsts[]` by `(oddsType, tuple(leg.races))`,
  then for each `poolInvs[]` row output `pool`=`oddsType`, `leg`=`leg.number`,
  `races`=comma-joined `leg.races`, `investment` (today's gross G), matching `jpEsts.jackpot`
  (carried-in C), `jpEsts.estimatedDividend`, `mergedPoolId`. Jackpot pools: `DT`, `TT`, `SixUP`.
  These values are **null until the meeting's pools open**; poll during the betting window and log.

## 2. Historical / info: server-rendered HTML (`racing.hkjc.com`)

### Param conventions (and the gotchas that bite)
- **Most info routes:** `racedate=YYYY/MM/DD` (slashes, zero-padded) + `Racecourse=ST|HV`
  + `RaceNo=N`. Note the capital `R` in `Racecourse`/`RaceNo`.
- **`displaysectionaltime` ONLY:** `racedate=DD/MM/YYYY` (day-first), lowercase, **no
  `Racecourse`**. Wrong format → empty "released shortly" page. *(Handled
  in `hkjc.data.sectional`.)*
- **`draw`:** `racedate` + `Racecourse`, no `RaceNo` (returns all races).
- **`btresult`:** `Date=DD/MM/YYYY` (param named `Date`, capital D, day-first), no venue/race.
- **`trackworkresult`:** `horseid=HK_YYYY_<brand>` (the long registration id).
- **Legacy `.aspx` pages** (`racing.hkjc.com/racing/information/English/...`): results take
  `RaceDate=YYYY/MM/DD&Racecourse=ST|HV&RaceNo=N`; the horse page takes `HorseNo=<id>` (presumably the
  brand; unverified).
- **Horse ids:** the *brand* is `[A-Z]{1,2}\d+` (e.g. `K400`, `AJ313`), printed in name cells as `NAME (K400)`
  (regex `\(([A-Z]{1,2}\d+)\)`); the *registration* is `HK_YYYY_<brand>` (e.g. `HK_2023_J183`), found in
  links as `horseid=...` (regex `horseid=(HK_\d{4}_([A-Z]{1,2}\d+))`). Racecard "Brand No." is the brand.
- **Other date shapes inside pages:** trackwork gallop entries `DD/MM` (no year, §2.2),
  `trackworkresult` rows `DD/MM/YYYY`, horse-page form `Date` `dd/mm/yy` (or `dd/mm/YYYY`).
- **Requests:** plain `GET`, no cookies. Modern routes need only
  `User-Agent: Mozilla/5.0` and `Referer: https://racing.hkjc.com` (timeout 20 s); for the legacy
  `.aspx` pages send `Origin`/`Referer: https://racing.hkjc.com`, `Content-Type: text/plain`,
  `Accept: */*` and the full Chrome UA of §1 (timeout 10 s). Pass `response.content` (bytes) to
  BeautifulSoup `html.parser`; no explicit charset is needed. No header is a secret.
- Three date formats coexist: ISO `YYYY-MM-DD` (GraphQL), slashed `YYYY/MM/DD` (most
  HTML), `DD/MM/YYYY` (sectional page **and** inside SpeedPro JSON). Easy to mix up.

### The high-value pages (modern `/en-us/local/information/` routes)

| Page | Route | Key data | Status |
|---|---|---|---|
| **Race Card** | `racecard?racedate=Y/M/D&Racecourse=ST&RaceNo=N` | **27 cols**: ratings, **Rtg +/- (rating move)**, gear (`V-/B2/TT`), **Horse Wt + Wt+/- (body-weight delta)**, Best Time, **days-since-run**, WFA, season stakes, priority, owner/sire/dam, import cat | 📜 §2.2 |
| **Comments on Running** | `corunning?date=YYYYMMDD&raceno=N` | per-horse free-text stewards' trip line ("raced wide, stayed on final 200m…"); past races served (earlier seasons redirect to `/archive/...`). The `racedate=...&Racecourse=...` form ignores the date and returns the latest meeting | 📜 §2.2 |
| **Sectional Time** | `displaysectionaltime?racedate=D/M/Y&RaceNo=N` | per-horse section positions + split times (1st–6th Sec.) + race splits | ✅ `data.sectional` |
| **Results** | `localresults?racedate=Y/M/D&Racecourse=ST&RaceNo=N`; wired via legacy `/racing/information/English/Racing/LocalResults.aspx?RaceDate=Y/M/D&Racecourse=ST&RaceNo=N` | result table + **full exotic/jackpot dividends** + **incident table** + running positions | ✅ results table · 📜 dividends §2.2 |
| **Horse page** | `horse?horseid=HK_2023_J183`; legacy `/racing/information/English/Horse/Horse.aspx?HorseNo=<id>` (id presumably the brand, e.g. `J183`; unverified) | record, location (Conghua/ST), import; tabs: Form, Rating/Wt/Placing, **Performance by Distance**, **Trackwork**, **Veterinary**, Movement, Pre-import, **Pedigree** + replay links | 📜 legacy form table §2.2 |
| **Trackwork archive** | `trackworkresult?horseid=HK_2023_J183` | one horse's **full dated work log over years** (SSR), **served only while the horse is in training**: a retired horse's id redirects to its race-record page with no trackwork, so historical trackwork is survivor-selected | 📜 §2.2 |

### Per-runner "angle" pages (server-rendered, all ⬜ — the buried gold)
> **Pre-race window caveat (important):** these pages are SSR-fetchable with plain `requests.get`,
> but populated **only for the imminent upcoming meeting** — for *past* meetings they serve
> "…available shortly", for *future* "No information". The "0 tables" you'd otherwise see is a
> stale-date artifact, **not** JS-rendering. Scrape them in the pre-race window with the upcoming date.
> **Exceptions:** `btresult` (barrier trials) and `trackworkresult` (per-horse trackwork archive)
> serve past dates; so does `corunning` with `date=YYYYMMDD&raceno=N`.
- **`exceptionalfactors`** — per horse: `Out of Handicap, WFA, Gear, Tongue Tie, Bit Change
  From/To, Rtg, Dist, Days since Last Run, New Trainer, Date returned from Conghua`, **plus a
  `Trainer | Multi Entries` table** (which stable saddles 2+ runners — pair with `trainerPreference`).
  Concentrated INTENT signals (§2.3/§2.7).
- **`veterinaryrecord`** (alt `overecord?horseid=…`) — per horse: dated `Date, Details (free
  text: "Lame right front…", "Throat surgery…"), Passed On` (cleared-to-race date). The **HEALTH
  cell** (§2.1) — recency of last issue + injury class via NLP.
- **`localtrackwork?racedate=YYYY/MM/DD&Racecourse=ST|HV&RaceNo=N`** 📜 — race-keyed: for every
  runner, ~3 weeks of dated logs across `Barrier Trial / Gallop / Trotting / Swimming / Equine
  Treadmill / Aqua Walker / Spelling`; **gallops carry split/total TIMES** and **CONGHUA vs SHA TIN
  location** (training-base movement). Imminent meeting only; for history use `trackworkresult`
  (§2.2). Fitness-cycle (§2.1).
- **`btresult?Date=DD/MM/YYYY`** 📜 (note `Date=`, day-first) — **barrier trials with batch
  `Time`+`Sectional Time`** and per-horse `Jockey, Trainer, Draw, Gear, LBW, Running Position, Time,
  Result, Comment` — direct fitness read **before a debut/return**, when form is silent. Past dates
  are served (recipe §2.2).
- **`formline`** — groups today's runners by a **shared past race**; per rival: `Wt, Rtg, Draw,
  Jockey, Last Jockey, Placing, Time, Margin, Win Odds, Weight Carried +/-, Rating +/-`. **The
  pre-assembled OPPONENT-LINKAGE / collateral-form graph** — raw material for the competition axis (§2.6).
- **`draw`** — barrier **W%/P%/F% over last 3 seasons** for the exact course/distance + a
  Favourites strike-rate row — realised draw-bias (validate/residual-correct the track geometry).
- **`racereportext`** / **`racereportfull`** — per-horse past incident history / full
  stewards report (panel, going, per-race incidents).
- **`latestonhorse`** — late declared-starter changes / scratchings feed.
- **`formline`**, **`jockeyfavourite`/`trainerfavourite`**, `jkcstat`,
  `jockeyprofile`/`trainerprofile` (some need extra params / partial JS).

### 2.2 Page recipes (not wired)
**Racecard.** URL `https://racing.hkjc.com/en-us/local/information/racecard?racedate=YYYY/MM/DD&Racecourse=ST|HV&RaceNo=N`
(slashes literal, not percent-encoded; venue upper-case as given). Headers `{"User-Agent": "Mozilla/5.0",
"Referer": "https://racing.hkjc.com"}`, timeout 20 s. Input dates `YYYY-MM-DD`, `YYYY/MM/DD` or `DD/MM/YYYY` (dashes
-> slashes; a first part that is not 4 digits is day-first). `Best Time` `m.ss.ff` / `ss.ff` -> seconds (float, 2 dp;
`-` -> null); integer columns strip `,` and `+`, `-`/`--` -> null; the output adds `date`, `venue`, `race_no` columns;
unknown header cells are dropped (only the mapped columns are kept).


URL templates (`{date}` formats: racecard,
localtrackwork, LocalResults `YYYY/MM/DD`; btresult `DD/MM/YYYY`; `{venue}` `ST`/`HV`):

```
https://racing.hkjc.com/en-us/local/information/racecard?racedate={date}&Racecourse={venue}&RaceNo={race}
https://racing.hkjc.com/en-us/local/information/corunning?date={yyyymmdd}&raceno={race}
https://racing.hkjc.com/en-us/local/information/btresult?Date={date}
https://racing.hkjc.com/en-us/local/information/localtrackwork?racedate={date}&Racecourse={venue}&RaceNo={race}
https://racing.hkjc.com/en-us/local/information/trackworkresult?horseid={horseid}
https://racing.hkjc.com/racing/information/English/Horse/Horse.aspx?HorseNo={horse_no}
https://racing.hkjc.com/racing/information/English/Racing/LocalResults.aspx?RaceDate={date}&Racecourse={venue_code}&RaceNo={race_number}
```

All pages: `GET https://racing.hkjc.com/en-us/local/information/<route>?...`, plain `requests`, then
BeautifulSoup. "cells" = `get_text(" ", strip=True)` over a row's `td`+`th`. An unpublished page
gives no matching table -> return an empty frame. `raise_for_status()` on HTTP errors.

**Race card** (`racecard?racedate=YYYY/MM/DD&Racecourse=ST|HV&RaceNo=N`). Pages nest wrapper
tables: parse every `<table>` whose text contains `Brand No.` and keep the one giving the most runner
rows. Header row = first row whose cells contain both `Brand No.` and `Horse No.`; body rows must
have exactly as many cells as the header and a digit `Horse No.` (skips reserves). Header -> field:
`Horse No.`→runner, `Last 6 Runs`→last_6_runs, `Colour`→colour, `Horse`→horse, `Brand No.`→horse_id
(brand), `Wt.`→carried_wt, `Jockey`→jockey, `Over Wt.`→over_wt, `Draw`→draw, `Trainer`→trainer,
`Int'l Rtg.`→intl_rtg, `Rtg.`→rtg, `Rtg.+/-`→rtg_change, `Horse Wt. (Declaration)`→decl_horse_wt,
`Wt.+/- (vs Declaration)`→wt_change, `Best Time`→best_time, `Age`→age, `WFA`→wfa, `Sex`→sex,
`Season Stakes`→season_stakes, `Priority`→priority, `Days since Last Run`→days_since_last_run,
`Gear`→gear, `Owner`→owner, `Sire`→sire, `Dam`→dam, `Import Cat.`→import_cat. Integer columns
(runner, carried_wt, draw, intl_rtg, rtg, rtg_change, decl_horse_wt, wt_change, age,
days_since_last_run, season_stakes): strip `,` and `+`; `""`, `-`, `--` → null. The jockey cell
carries an apprentice allowance, `NAME (-2)`: regex `\(([-+]?\d+)\)` → `jockey_allowance` (default
0), name = cell without it. `Best Time` is `m.ss.ff` or `ss.ff` (hundredths).

**Comments on Running** (`corunning?date=YYYYMMDD&raceno=N`). First table
whose text contains `Comment` and `Horse`. Data rows: ≥ 6 cells and `cells[1]` all digits.
`cells[1]` runner no., `cells[2]` `NAME (BRAND)`, `cells[4]` gear, `cells[5]` comment (`cells[0]`,
`cells[3]` were not read; labels unrecorded). Trouble flag used (case-insensitive search), meant to
separate interference from tactical lines like "held up":
`no clear run|no clear (?:racing )?room|no clear passage|short of room|no room|denied (?:a )?(?:clear )?(?:run|passage)|blocked|checked|hampered|tightened|squeezed|crowded|bumped|interfered|carried (?:wide|out)|forced (?:wide|out)|trapped|in(?:to)? a pocket|behind a wall|had to steady|steadied(?! away| from the start)|had to switch|shuffled|awkwardly placed|lost ground|wide no cover|no luck`

**Barrier trials** (`btresult?Date=DD/MM/YYYY`). Walk all `<table>`s in document order, carrying the
current batch. A table **without** an `<a href>` containing `horseid` is a batch header; on its text
apply `Batch\s*(\d+)\s*-\s*(.+?)\s*-\s*(\d+)m` (DOTALL; batch, course, distance m),
`Going:\s*([A-Za-z /]+?)\s+Time:`, `Time:\s*([\d.:]+)` (batch winner's time) and
`Sectional Time:\s*([\d.\s]+)` (3 space-separated splits). A table **with** horse links: each row
with a `horseid` link and ≥ 10 cells is a runner: 0 `NAME (BRAND)`, 1 jockey, 2 trainer, 3 draw,
4 gear, 5 LBW, 6 running position, 7 time, 8 result, 9 comment; long id from the link
`horseid=(HK_\d{4}_[A-Z]{1,2}\d+)`. Times `m.ss.ff`/`ss.ff` (hundredths; `m:ss.ff` tolerated). LBW →
lengths: upper-case, NBSP → space; `""`/`-` → 0 (leader); `N` 0.3, `NK` 0.3, `HD` 0.2, `SH` 0.1,
`DH` 0; else `^\s*(\d+)?[\s-]*(?:(\d+)/(\d+))?\s*L?\s*$` = whole + num/den (`8-1/2L`, `8 1/2L`,
`3/4L`, `8L`); else null. Trials run mostly Mon/Tue/Thu/Fri, sometimes Wed/Sat, never Sun; other
days give no tables. **Date list:** the page's first `<select>` lists trial dates as option values
`DD/MM/YYYY` covering roughly the trailing 13 months; older dates must be probed. **Backfill used:**
candidates = dropdown dates ∪ every Mon–Sat in range; 1 request/s, one retry after 2 s on a network
error; dedupe on (date, batch sequence, brand). Batch numbers restart at 1 per venue on days with sessions at two
venues, so identify a batch by its position in the page (date, sequence), not its printed number. Column 8 is a verdict
(`Passed`, `Failed`, ...), not a placing; the placing is the last running-position call. Earlier seasons redirect to
`/archive/...` with the date as `YYYY/MM/DD`.

**Trackwork, race-keyed** (`localtrackwork?racedate=YYYY/MM/DD&Racecourse=ST|HV&RaceNo=N`,
imminent meeting only). Tables containing a `horseid=(HK_\d{4}_([A-Z]{1,2}\d+))` link; header = first
row; the Gallop column is the header cell equal to `Gallop` (case-insensitive). Each runner row:
long id and brand from the link, name = `cells[1]` cut at the first `\s+[A-Z]\s|\s+\d`. The Gallop
cell holds consecutive entries `DD/MM: LOCATION SURFACE s1 s2 [s3…] (total) (rider) [companion]`,
e.g. `04/06: SHA TIN AWT 33.0 30.0 25.9 (1.28.9) (H Y Yuen)`; match all with
`(\d{2}/\d{2}):\s+([A-Z][A-Z ]+?)\s+([A-Za-z]+)\s+([\d.][\d.\s]*?)\s*\(([\d.]+)\)\s*\(([^)]*)\)`
(location = upper-case words SHA TIN / CONGHUA; surface token AWT, TurG, AWTG, RGal, SmT, …;
trailing mixed-case companion horse ignored). Splits are **per furlong** (201.168 m), one decimal;
the bracketed total is `m.ss.f` or `ss.f` (**tenths**, unlike race times). Session year = race year,
except a session month later than the race month belongs to the previous year (December work before
a January race). Worked distance = n_splits × 201.168 m; mean velocity = distance / total (fallback:
sum of splits). **Upcoming dates:** every `<option value>` matching `^\d{4}/\d{2}/\d{2}$` on this page.

**Trackwork archive** (`trackworkresult?horseid=HK_YYYY_<brand>`, full history, one request per
horse). Tables whose first row starts `Date, Type`; columns 0 date `DD/MM/YYYY`, 1 type
(`Gallop`, Trotting, Swimming, Aqua Walker, Spelling, Barrier Trial), 2 racecourse/track
(`Sha Tin AWT`, `Conghua TurG` → location + surface), 3 workouts `<splits> (total) (rider)` with no
`DD/MM:` prefix: `([\d.][\d.\s]*?)\s*\(([\d.]+)\)\s*\(([^)]*)\)`. Only `Gallop` rows are timed.
**Backfill used:** 1 request/s, one retry after 2 s; dedupe on (brand, session_date). Keep only
sessions strictly before the race day when joining to races.

**Horse page, legacy** (`https://racing.hkjc.com/racing/information/English/Horse/Horse.aspx?HorseNo=<id>`;
which id is unverified; presumably the brand, as on the results page).
Name = `<title>` text before `- Horses -`. Form table = `<table class="bigborder">`: headers from
`<thead>` `td`s (else the first row), column names stripped of non-word characters; drop rows with
empty `Date`. Used columns (cleaned names): `Pla` (first token; incident codes
dropped), `RaceIndex`, `Date` (`dd/mm/yy`, or `dd/mm/YYYY` if longer than 8 chars), `RCTrackCourse`
(`Venue / Track / Course`, split on ` / `), `Dist`, `Dr`, `Rtg`, `ActWt`, `DeclarHorseWt`,
`WinOdds`, `FinishTime` (`m.ss.ff`; `--…` = no time), `RunningPosition` (space-separated calls),
`VideoReplay` (ignored). `RaceId = {YYYYMMDD}{Venue}{RaceIndex}`.

**Dividends** (same results page; wired URL
`https://racing.hkjc.com/racing/information/English/Racing/LocalResults.aspx?RaceDate=YYYY/MM/DD&Racecourse=ST|HV&RaceNo=N`).
Table = `div.dividend_tab` → first `<table>`; absent = non-race date or not yet declared. Rows with
≥ 3 cells are `(pool, combination, dividend)`; skip header rows (pool `POOL`/`DIVIDEND`, or
combination starting `winning`). Rows with exactly 2 cells continue the previous pool (row-spanned
cell: PLACE and QUINELLA PLACE pay several combinations). Dividend: strip `,`, parse float, else
null. Dividends are **HK$ per HK$10 unit** (decimal odds = dividend / 10). Page
labels → codes: WIN→WIN, PLACE→PLA, QUINELLA→QIN, QUINELLA PLACE→QPL, FORECAST→FCT, TIERCE→TCE,
TRIO→TRI, FIRST 4→FF, QUARTET→QTT, DOUBLE→DBL, TREBLE→TBL, DOUBLE TRIO→DT, TRIPLE TRIO→TT,
SIX UP→`6UP` (GraphQL calls it `SixUP`: normalise before matching: filtering result-page
codes on `SixUP` silently drops Six Up). **Jackpot history:** a DT/TT/Six Up dividend appears
on the pool's final leg **only when won**; carryover meetings are inferred by absence over cards that
offered the pool; the carried-in amount is not on result pages (capture `jpEsts` live, §1.3). The
scanner walked races 1–11 per (date, venue), treating a failure on race 1 as "no meeting", with a
0.1 s pause between meetings.

## 3. SpeedPro — JSON (current meeting only)

```
https://racing.hkjc.com/racing/speedpro/assets/json/speedguide/race_{n}.json
```
Per-race, **current meeting only** (not addressable by date). Decode with `utf-8-sig`
(BOM). `{ "en-us": {...}, "zh-hk": {...} }`, each with: `RaceMap` (base64 speed-map
image), `SpeedPRO` (per-runner `energyrequired`, `speedproenergy`,
`speedproenergydifference`, `fitnessrating`, recent-run energy history `lastrun…
lastrun_5th`, `bestlast12months`, `bestatdistance`), `RaceInfoEng` (date **`DD/MM/YYYY`**,
course, track, class). ⚠️ Can serve a **previous meeting's cached data**, and
energy/fitness are blank until published shortly before the off. The old
`…/excel/{YYYYMMDD}_energygrid_en.xls` endpoint now **404s** — the JSON replaced it.

## 4. Trajectory / positioning data (the dense-kinematics frontier)

A dense per-horse `(x, y, v)` time series is far richer than one finish time. What a live
network/JS inspection of the HKJC site shows:

- **Provider is Trakus** — an **RF saddlecloth-tag** system sampling **~30 Hz**. So a
  dense per-horse trajectory **does exist upstream at ~30 Hz**.
- **BUT it is NOT a public data feed.** The "**Aerial Virtual View**" is delivered as a
  **pre-rendered flat MP4 (pixels only)**, served from Akamai; the 3D virtualisation is
  rendered server-side. Its own disclaimer ("accuracy affected by signal interference…
  refer to Real Replay") confirms it's an RF-positioning render (lossy — good for
  lateral-lane / ground-loss, not sub-metre truth). **There is no `(x,y)` JSON/websocket
  to tap.**
- **Therefore two routes to dense data:**
  1. **Sectionals (structured, ✅).** `data.sectional` gives ~6 dense points/horse
     (positions + split times) with no CV.
  2. **CV from the replay videos (pixels only).** This is what `hkjc.recon` does (method in
     docs/RECON.md). The virtual view is more tractable than broadcast footage (clean
     silk icons, rail lines, distance markers `420M/920M` for calibration) but needs
     per-frame moving-camera registration, and the render is **lossy**.

### Video / replay URLs (for CV ingest or manual review)
Each past run on the horse page links an iframe player:
```
https://racing.hkjc.com/contentAsset/videoplayer_v4/video-player-iframe_v4.html
    ?type=replay-full | replay-aerial | replay-aerial-virtual | trackwork
    &date=YYYYMMDD&no=NN(zero-padded)&lang=eng&videoParam=PAD
```
Media resolves to Akamai `streaminghkjc-a.akamaihd.net/hdflash/...` with renditions
`_700kbps/_1200kbps/_2500kbps.mp4` and `.m3u8` (HLS). The **exact manifest path is built
client-side in `JCPlayer.js`** — replay that construction or capture the network request
to get a direct URL.

#### Confirmed direct-MP4 replay feeds (auth-free, deterministic — no player/JS needed)
The renditions follow a **fixed, scrapable pattern** (verified by HEAD/range requests, 2026-06):
```
https://streaminghkjc-a.akamaihd.net/hdflash/{TOKEN}/{YYYY}/{YYYYMMDD}/{NN}/eng/{TOKEN}_{YYYYMMDD}_{NN}_eng_2500kbps.mp4
   {NN} = zero-padded race number;  {YYYY} = year of {YYYYMMDD}
```
The player's documented `type=` list is INCOMPLETE — `replay-drone` is unlisted yet live, and the
player even mis-guesses the drone filename. The four feeds we use, each a DIFFERENT instrument
(complementary, not redundant):

| `{TOKEN}` | Feed | Camera | Carries / best for |
|---|---|---|---|
| `replay-aerial` | **Aerial Virtual View** | synthetic 3D render of the tracking feed | render-inversion → clean (long, lat); legend running-order strip; odometer/distance markers; minimap; clock+splits. CONTROL (if faithful) + identity + sections. |
| `replay-drone` | **Drone** | real, high, moving/zooming | real horse positions; lateral on the stable top-down back-straight; legible numbers at the finish-zoom. No global longitudinal (camera cuts/pans). |
| `replay-full` | **Race film (RF)** | real, SIDE-ON tracking | longitudinal GAPS → MARGINS / beaten-lengths (→ the W'/spread surface); a clock-adjacent speed/section readout (velocity cross-check); legend strip; legible finish numbers. Lateral weak (foreshortened). |
| `replay-patrol` | **Patrol** | real, FIXED high mount, head-on early | **FIXED camera ⇒ CONSTANT homography** (no per-frame rail jitter); **head-on ⇒ lateral = horizontal image coordinate directly**; the stewards' INTERFERENCE camera (bump/carried-wide events cross-check the official running commentary). Cleanest LATERAL geometry. Limits: 720p, head-on occlusion, identity head-on (bridge from finish), straight-only coverage. |

Also live: `replay-aerial-virtual`, `trackwork`.
Some races' `replay-aerial` file carries live race film with no render (e.g. 2025-01-15 HV R3: every sampled
frame is TV film; `replay-full` and `replay-patrol` are the usual feeds and `replay-aerial-virtual` is 404). Such a
race has no render to reconstruct from; the frames stage stops with "unknown render style".
The official per-horse **running commentary / stewards' trip notes** (the external lateral
ground-truth: "caught wide", "bumped", "shifted across") are in the **results page HTML**
(`...results...`) and, as one free-text line per runner, on the **`corunning`** page (recipe §2.2).
They are not on the sectional page, which is server-rendered (not JS-only) but only answers with
`racedate=DD/MM/YYYY` (§2). Cache them alongside.
Morning track-work clips: `…/twstarter/{YYYY}/{YYYYMMDD}/{RR}/novo/twstarter_…_2500kbps.mp4`
(`{RR}` zero-padded). Pundit tips: `…/tipsindex/tips_index.asp?RaceNo={n}`.

**Alternative source — YouTube archives via `yt-dlp`.** The HKJC CMS resolver
(`getVideoInfo`) needs the browser session and is awkward to script; the direct MP4s above avoid
it. There are also **full race live-stream archives on YouTube** — the side
broadcast view **plus commentary plus on-screen LIVE ODDS** — downloadable with `yt-dlp`
(playlists like `youtube.com/playlist?list=PLMVIVrWyo-Z_3Bddw6rIoyqhzeP35h1ij`). The broadcast
SIDE view suits off-the-shelf detection, carries saddlecloth numbers + silks for identity, and the
burned-in odds/commentary are extra signals.

## 5. Scraping discipline (don't get rate-limited)

- **Odds poll:** ~90 s interval with ~10 s jitter (odds barely move faster except near
  the off). **Concurrency ≤ 3.** Inter-request delay **80–250 ms** random. **Timeouts used:** 10 s legacy `.aspx`, 20 s
  modern HTML routes, 30 s GraphQL. Send a real browser `User-Agent` + matching `Origin`/`Referer`
  (`bet.hkjc.com` for GraphQL, `racing.hkjc.com` for HTML/JSON).
- TTL caches: the results page is cached 3600 s in process (`historical._soupify`). Suggested
  TTLs: live card 1000 s, live odds 30 s, SpeedPro 300 s — reuse rather than hammer.
- **Backfills** (barrier trials by date, trackwork archive by horse): ~1 request/s, one retry after
  2 s on a network error, then skip and log. `hkjc.data` disk-caches the race facts
  (`data_cache/race_official/`), silks and replay videos; nothing else is disk-cached.
- **Missing values:** HTML pages show `""`, `-`, `--`; GraphQL numbers can be unparseable. Map
  all of them to null.

## 6. Gotchas (learned the hard way)

- **Non-race dates** return a styled "No information." page with **zero `<table>`
  elements** — check for actual tables, *not* for the strings `race_tab`/`performance`
  (those appear in that page's CSS too). Always target a real meeting date.
- **SpeedPro staleness** — `race_n.json` can serve a previous meeting; energy/fitness
  blank until published. Don't assume it reflects today.
- **QIN/QPL diagonals** are self-pairs with zero probability → `1/p = inf`; guard with
  `nan_to_num`. Expect NaNs on matrix diagonals.
- Empty-string `date`/`venue` to live endpoints is a **feature** (server picks the
  current meeting), not a bug — but verify what came back.

---

## 7. Mining priority (highest unrealised value first)

1. **`weather` GraphQL op** (§1.1) — per-section **wind vectors** + **soil-moisture** firmness.
   Same endpoint as §1, plain-HTTP.
   Highest leverage, lowest integration cost.
2. **`racecard`** — the 27-field pre-race card (ratings + **rating moves** + gear +
   **body-weight deltas** + days-since-run + pedigree). Biggest single fetch.
3. **`trainerPreference` + `trumpCard`** (GraphQL, already in our query body) — zero extra fetch;
   extract two runner fields = stable-intent ranking + confidence flag.
4. **`formline`** — the pre-assembled opponent-linkage / collateral-form graph → the competition
   axis (§2.6). This is the raw data for a
   game-theoretic treatment of form.
5. **`exceptionalfactors` + `veterinaryrecord` + `localtrackwork` + `btresult`** —
   gear/first-time/vet/work-load/barrier-trial **condition, health & intent** angles.
6. **`corunning`** — running comments → **trip/trouble NLP** (the stochasticity/racing-luck cell).
7. **`draw` stats** — realised W%/P%/F% by course-distance, to validate the draw geometry.
8. **Full exotic dividends + `investment`/`jpEsts`** — pool sizes & carryover.
9. **Sectionals → energetics**; CV trajectory only for place/exotics, and only if it ever pays.

### New routes found (beyond the above) — mostly Next.js (browser or backing GraphQL needed)
`…/local/info/windtracker` (UI for the `weather` op) · `…/local/page/conghua-movement-records`
(training-base moves) · `…/local/page/jtcombo-debutants-statistics` (jockey/trainer-combo + debutant
strike rates) · `…/local/page/last-run-reminder` (layoff/first-up) · `…/local/page/racing-course-time`
(course standard/record times — distance/surface benchmarks) · `…/local/info/changes` (scratchings).
