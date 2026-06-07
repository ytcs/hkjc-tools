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

Implementation status legend: ✅ wired in `hkjc.data` · 🟡 partial · ⬜ TODO.

---

## 1. Live: GraphQL (odds, cards, pools, jackpots)

```
POST https://info.cld.hkjc.com/graphql/base/
Headers: Origin/Referer = https://bet.hkjc.com, Content-Type: application/json
```

Two operations (full query bodies in `hkjc.data.live`):
- **`raceMeetings`** ✅ — the whole meeting: every race, runners, draws, ratings,
  weights, jockey/trainer, gear, `last6run`, win odds, pool metadata.
- **`racing`** ✅ — odds nodes for given `oddsTypes` and `raceNo`.

Passing `date=""`/`venueCode=""` (or `null`) returns the **current meeting** (the code
prefers the first `HV`/`ST`); **always reconcile the returned `Date`/`Venue`** with what
you asked for — the server can hand back a different meeting.

**Pool / odds types accepted:** `WIN, PLA, QIN, QPL, CWA, CWB, CWC, IWN, FCT, TCE, TRI,
FF, QTT, DBL, TBL, DT, TT, SixUP`, plus `JKC` (jockey challenge), `TNC` (trainer
challenge). The library parses `WIN, PLA, QIN, QPL` — **the rest are unextracted** ⬜.

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
- `jpEsts` / jackpots and `estimatedDividend` per pool ✅ (via `live.live_jackpots`).
- `totalInvestment`, per-pool `investment` — **pool sizes → liquidity signal** ✅.

### 1.1 `weather` operation — wind + soil moisture (NEW, the ENVIRONMENT feed) ⬜
Same endpoint, `operationName: Weather`, `variables: null`, `Origin: https://racing.hkjc.com`.
The server **whitelists by exact query body** (any field-set change → `WHITELIST_ERROR`).
Returns **both ST & HV**, `max-age=15`. Per racecourse:
- `sectional[]` — **four anemometers (A/B/C/D)** around the track, each with
  `avgCorrectedWindDirection` (compass, e.g. `N80W`), `avgCorrectedWindSpeed`, gust dir/speed,
  `gpsPosition` (lat:lon:elev). → the **real per-straight wind vector** (head/tail/cross),
  projectable onto each straight's heading via the GPS — fills THEORY §2.5 (wind), reshapes draft.
- `weatherStation` (loc E) — the above wind fields **plus** `temperature`, `relativeHumidity`,
  `pressure`, `solarRadiation`, `sunshineHour`, daily max/min, `rainPrecipitationDailyTotal`,
  **`soilVolumeticWaterContent`**, **`soilMoistureLoss{1Min,DailyTotal,24H}}`**. → a **continuous
  going/firmness proxy** far richer than the categorical going string (going-affinity was found
  noise-limited by soft-going *sparsity*; a continuous firmness covariate sidesteps that).

## 2. Historical / info: server-rendered HTML (`racing.hkjc.com`)

### Param conventions (and the gotchas that bite)
- **Most info routes:** `racedate=YYYY/MM/DD` (slashes, zero-padded) + `Racecourse=ST|HV`
  + `RaceNo=N`. Note the capital `R` in `Racecourse`/`RaceNo`.
- **`displaysectionaltime` ONLY:** `racedate=DD/MM/YYYY` (day-first), lowercase, **no
  `Racecourse`**. Wrong format → empty "released shortly" page. *(This bit us; handled
  in `data.sectional`.)*
- **`draw`:** `racedate` + `Racecourse`, no `RaceNo` (returns all races).
- Three date formats coexist: ISO `YYYY-MM-DD` (GraphQL), slashed `YYYY/MM/DD` (most
  HTML), `DD/MM/YYYY` (sectional page **and** inside SpeedPro JSON). Easy to mix up.

### The high-value pages (modern `/en-us/local/information/` routes)

| Page | Route | Key data | Status |
|---|---|---|---|
| **Race Card** | `racecard?racedate=Y/M/D&Racecourse=ST&RaceNo=N` | **27 cols**: ratings, **Rtg +/- (rating move)**, gear (`V-/B2/TT`), **Horse Wt + Wt+/- (body-weight delta)**, Best Time, **days-since-run**, WFA, season stakes, priority, owner/sire/dam, import cat | ⬜ **top add** |
| **Comments on Running** | `corunning?racedate=...&RaceNo=...` | per-horse free-text trip line ("raced wide, stayed on final 200m…") | ⬜ **trip/trouble NLP — gold** |
| **Sectional Time** | `displaysectionaltime?racedate=D/M/Y&RaceNo=N` | per-horse section positions + split times (1st–6th Sec.) + race splits | ✅ `data.sectional` |
| **Results** | `localresults?racedate=Y/M/D&Racecourse=ST&RaceNo=N` | result table + **full exotic/jackpot dividends** + **incident table** + running positions | 🟡 (dividends partial) |
| **Horse page** | `horse?horseid=HK_2023_J183` | record, location (Conghua/ST), import; tabs: Form, Rating/Wt/Placing, **Performance by Distance**, **Trackwork**, **Veterinary**, Movement, Pre-import, **Pedigree** + replay links | 🟡 (legacy `Horse.aspx` ✅) |

### Per-runner "angle" pages (server-rendered, all ⬜ — the buried gold)
> **Pre-race window caveat (important):** these pages are SSR-fetchable with plain `requests.get`,
> but populated **only for the imminent upcoming meeting** — for *past* meetings they serve
> "…available shortly", for *future* "No information". The "0 tables" you'd otherwise see is a
> stale-date artifact, **not** JS-rendering. Scrape them in the pre-race window with the upcoming date.
- **`exceptionalfactors`** — per horse: `Out of Handicap, WFA, Gear, Tongue Tie, Bit Change
  From/To, Rtg, Dist, Days since Last Run, New Trainer, Date returned from Conghua`, **plus a
  `Trainer | Multi Entries` table** (which stable saddles 2+ runners — pair with `trainerPreference`).
  Concentrated INTENT signals (§2.3/§2.7).
- **`veterinaryrecord`** (alt `overecord?horseid=…`) — per horse: dated `Date, Details (free
  text: "Lame right front…", "Throat surgery…"), Passed On` (cleared-to-race date). The **HEALTH
  cell** (§2.1) — recency of last issue + injury class via NLP.
- **`localtrackwork`** — per horse, dated logs across `Barrier Trial / Gallop / Trotting /
  Swimming / Equine Treadmill / Aqua Walker / Spelling`; **gallops carry split/total TIMES** and
  **CONGHUA vs SHA TIN location** (training-base movement). Fitness-cycle (§2.1).
- **`btresult?Date=DD/MM/YYYY`** (note `Date=`, day-first) — **barrier trials with batch
  `Time`+`Sectional Time`** and per-horse `Draw, Gear, LBW, Running Position, Time, Result, Comment`
  — direct fitness read **before a debut/return**, when form is silent; feeds `v_c`/`W′` priors.
- **`formline`** — groups today's runners by a **shared past race**; per rival: `Wt, Rtg, Draw,
  Jockey, Last Jockey, Placing, Time, Margin, Win Odds, Weight Carried +/-, Rating +/-`. **The
  pre-assembled OPPONENT-LINKAGE / collateral-form graph** — raw material for the competition axis (§2.6).
- **`draw`** — barrier **W%/P%/F% over last 3 seasons** for the exact course/distance + a
  Favourites strike-rate row — realised draw-bias (validate/residual-correct the modelled geometry).
- **`racereportext`** / **`racereportfull`** — per-horse past incident history / full
  stewards report (panel, going, per-race incidents).
- **`latestonhorse`** — late declared-starter changes / scratchings feed.
- **`formline`**, **`jockeyfavourite`/`trainerfavourite`**, `jkcstat`,
  `jockeyprofile`/`trainerprofile` (some need extra params / partial JS).

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

The model wants a dense per-horse `(x, y, v)` time series, not one finish time. The
verdict from a live network/JS recon (`FEED_RECON`):

- **Provider is Trakus** — an **RF saddlecloth-tag** system sampling **~30 Hz**. So the
  dense per-horse trajectory the model wants **does exist upstream at ~30 Hz**.
- **BUT it is NOT a public data feed.** The "**Aerial Virtual View**" is delivered as a
  **pre-rendered flat MP4 (pixels only)**, served from Akamai; the 3D virtualisation is
  rendered server-side. Its own disclaimer ("accuracy affected by signal interference…
  refer to Real Replay") confirms it's an RF-positioning render (lossy — good for
  lateral-lane / ground-loss, not sub-metre truth). **There is no `(x,y)` JSON/websocket
  to tap.** Always re-confirm the feed hunt before building CV.
- **Therefore two routes to dense data:**
  1. **Sectionals (structured, ✅).** `data.sectional` gives ~6 dense points/horse
     (positions + split times) with no CV — the practical source for `v_c`/`W′`/
     finishing-speed features. **Open question: are ~6 points enough to fit energetics,
     or is the market already efficient w.r.t. sectionals too?**
  2. **CV from the Aerial Virtual View (pixels-only fallback).** The `hkjc.vision`
     package is the scaffold (detect → identify → homography → project → trajectory).
     The virtual view is more tractable than broadcast footage (clean silk-icons, rail
     lines, distance markers `420M/920M` for calibration) but needs per-frame moving-
     camera homography; it's **lossy** and expensive. A single-race PoC ran on HV
     2026-06-03 R1 (recovered `(d, off)` tracks, fragmented). **Gate any CV build behind
     a confirmed sectional signal** — don't build a trajectory pipeline for ~zero edge.

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
player even mis-guesses the drone filename. The four feeds we use, each a DIFFERENT instrument for a
DIFFERENT part of the causal program (journal 0073 — they are complementary, not redundant):

| `{TOKEN}` | Feed | Camera | Carries / best for |
|---|---|---|---|
| `replay-aerial` | **Aerial Virtual View** | synthetic 3D render of the tracking feed | render-inversion → clean (long, lat); legend running-order strip; odometer/distance markers; minimap; clock+splits. CONTROL (if faithful) + identity + sections. |
| `replay-drone` | **Drone** | real, high, moving/zooming | real horse positions; lateral on the stable top-down back-straight; legible numbers at the finish-zoom. No global longitudinal (camera cuts/pans). |
| `replay-full` | **Race film (RF)** | real, SIDE-ON tracking | longitudinal GAPS → MARGINS / beaten-lengths (→ the W'/spread surface); a clock-adjacent speed/section readout (velocity cross-check); legend strip; legible finish numbers. Lateral weak (foreshortened). |
| `replay-patrol` | **Patrol** | real, FIXED high mount, head-on early | **FIXED camera ⇒ CONSTANT homography** (no per-frame rail jitter); **head-on ⇒ lateral = horizontal image coordinate directly**; the stewards' INTERFERENCE camera (bump/carried-wide events cross-check the official running commentary). Cleanest LATERAL geometry. Limits: 720p, head-on occlusion, identity head-on (bridge from finish), straight-only coverage. |

Also live: `replay-aerial-virtual`, `trackwork`. The official per-horse **running commentary / stewards'
trip notes** (the external lateral ground-truth: "caught wide", "bumped", "shifted across") are in the
**results page HTML** (`...results...`), NOT the (JS-only, empty) sectional page — cache them alongside.
Morning track-work clips: `…/twstarter/{YYYY}/{YYYYMMDD}/{RR}/novo/twstarter_…_2500kbps.mp4`
(`{RR}` zero-padded). Pundit tips: `…/tipsindex/tips_index.asp?RaceNo={n}`.

**Easiest video acquisition (the CV unblock) — YouTube archives via `yt-dlp`.** The HKJC CMS
resolver (`getVideoInfo`) needs the browser session and is awkward to script (the CV feasibility
probe's one hard blocker). But there are **full race live-stream archives on YouTube** — the side
broadcast view **plus commentary plus on-screen LIVE ODDS** — downloadable with `yt-dlp`
(playlists like `youtube.com/playlist?list=PLMVIVrWyo-Z_3Bddw6rIoyqhzeP35h1ij`). This is the
practical substrate for any CV build: the broadcast SIDE view is the tractable one for
off-the-shelf detection (the feasibility probe got 7 horses + 11 jockeys with no fine-tune vs 0 on
the overhead), it carries saddlecloth numbers + silks for identity, and the burned-in odds/commentary
are bonus signals. (Not a priority now — recorded for when CV path-extraction / gait is taken up.)

## 5. Scraping discipline (don't get rate-limited)

- **Odds poll:** ~90 s interval with ~10 s jitter (odds barely move faster except near
  the off). **Concurrency ≤ 3.** Inter-request delay **80–250 ms** random. **Timeout
  ~10 s.** Send a real browser `User-Agent` + matching `Origin`/`Referer`
  (`bet.hkjc.com` for GraphQL, `racing.hkjc.com` for HTML/JSON).
- Library TTL caches: live races 1000 s, live odds 30 s, HTML pages 3600 s, SpeedPro
  300 s — tune rather than hammer.

## 6. Gotchas (learned the hard way)

- **Non-race dates** return a styled "No information." page with **zero `<table>`
  elements** — check for actual tables, *not* for the strings `race_tab`/`performance`
  (those appear in that page's CSS too). Always target a real meeting date.
- **SpeedPro staleness** — `race_n.json` can serve a previous meeting; energy/fitness
  blank until published. Don't assume it reflects today.
- **QIN/QPL diagonals** are self-pairs with zero probability → `1/p = inf`; handled with
  `nan_to_num`. Expect NaNs on matrix diagonals.
- **Harville cost is `O(N·2^N)`** with a one-off Numba JIT warm-up (~10 s) per process;
  keep fields ≤ ~20 runners.
- Empty-string `date`/`venue` to live endpoints is a **feature** (server picks the
  current meeting), not a bug — but verify what came back.

---

## 7. Mining priority (highest unrealised value first)

1. **`weather` GraphQL op** (§1.1) — per-section **wind vectors** + **soil-moisture** firmness.
   Fills the model's single weakest cell (ENVIRONMENT), same endpoint already wired, plain-HTTP.
   Highest leverage, lowest integration cost.
2. **`racecard`** — the 27-field pre-race card (ratings + **rating moves** + gear +
   **body-weight deltas** + days-since-run + pedigree). Biggest single fetch; feeds every model.
3. **`trainerPreference` + `trumpCard`** (GraphQL, already in our query body) — zero extra fetch;
   extract two runner fields = stable-intent ranking + confidence flag.
4. **`formline`** — the pre-assembled opponent-linkage / collateral-form graph → the competition
   axis (§2.6). The simplistic Elo/relative-form treatments failed; this is the raw data for a
   deeper game-theoretic treatment.
5. **`exceptionalfactors` + `veterinaryrecord` + `localtrackwork` + `btresult`** —
   gear/first-time/vet/work-load/barrier-trial **condition, health & intent** angles.
6. **`corunning`** — running comments → **trip/trouble NLP** (the stochasticity/racing-luck cell).
7. **`draw` stats** — realised W%/P%/F% by course-distance, to validate the modelled draw geometry.
8. **Full exotic dividends + `investment`/`jpEsts`** — pool sizes & carryover (strategist).
9. **Sectionals → energetics**; CV trajectory only for place/exotics, and only if it ever pays.

### New routes found (beyond the above) — mostly Next.js (browser or backing GraphQL needed)
`…/local/info/windtracker` (UI for the `weather` op) · `…/local/page/conghua-movement-records`
(training-base moves) · `…/local/page/jtcombo-debutants-statistics` (jockey/trainer-combo + debutant
strike rates) · `…/local/page/last-run-reminder` (layoff/first-up) · `…/local/page/racing-course-time`
(course standard/record times — distance/surface benchmarks) · `…/local/info/changes` (scratchings).
