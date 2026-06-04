# HKJC Data Catalog — the modern `en-us` site

The modern `racing.hkjc.com/en-us/local/information/*` routes expose far more than
the library currently scrapes. **All are server-side rendered** — `requests.get()`
returns full HTML (parse with BeautifulSoup), no browser needed — *if* you use the
right params. (Live odds remain GraphQL-only at `info.cld.hkjc.com/graphql`; there
is no public per-page JSON for racing data.)

> Map of what to build next. Implemented so far: `sectional` (✅), `historical`
> result/dividend pages, `live` GraphQL, `speedpro`. The rest are high-value TODOs.

## Param conventions (and the gotchas)
- **Most info routes:** `racedate=YYYY/MM/DD` (slashes, zero-padded) + `Racecourse=ST|HV` + `RaceNo=N`. Note capital `R` in `Racecourse`/`RaceNo`.
- **`displaysectionaltime` ONLY:** `racedate=DD/MM/YYYY` (day first), lowercase `racedate`, **no `Racecourse`**. Wrong format → empty "released shortly" page. *(This bit us; now handled in `data.sectional`.)*
- **`draw`:** `racedate` + `Racecourse`, no `RaceNo` (returns all races).

## Highest-value pages

| # | Page | URL (route under `/en-us/local/information/`) | Key data | Value |
|---|---|---|---|---|
| 1 | **Race Card** | `racecard?racedate=Y/M/D&Racecourse=ST&RaceNo=N` | **27 cols**: ratings, **Rtg +/- (rating move)**, gear (e.g. `V-/B2/TT`), **Horse Wt + Wt+/- (body-weight delta)**, Best Time, Days-since-run, WFA, season stakes, priority, owner/sire/dam, import cat | **Top add** — full pre-race handicapping in one fetch |
| 2 | **Comments on Running** | `corunning?racedate=...&Racecourse=...&RaceNo=...` | per-horse free-text running line ("raced wide, stayed on final 200m…") | trip/trouble NLP features — gold |
| 3 | **Sectional Time** ✅ | `displaysectionaltime?racedate=D/M/Y&RaceNo=N` | per-horse section positions + split times (1st–6th Sec.) + race splits | pace/closing-speed; energetics model |
| 4 | **Results** | `localresults?racedate=Y/M/D&Racecourse=ST&RaceNo=N` | result table + **full exotic/jackpot dividends** + **incident table** + running positions | supersedes `LocalResults.aspx` |
| 5 | **Horse page** | `horse?horseid=HK_2023_J183` | header (record, location Conghua/ST, import), tabs: Form, Rating/Wt/Placing, **Performance by Distance**, **Trackwork**, **Veterinary**, Movement, Pre-import, **Pedigree** | richer than `Horse.aspx`; + replay links |

## Per-runner "angle" pages (server-rendered)
- **`exceptionalfactors`** — Out of Handicap, WFA, **Gear / Tongue-Tie / Bit change**, days-since-run, **new trainer**, date returned from Conghua. First-time-gear angles.
- **`veterinaryrecord`** — vet/soundness flags per horse.
- **`localtrackwork`** — Barrier Trial / Gallop / Swimming / Treadmill / Aqua-Walker / Spelling work load.
- **`racereportext`** — per-horse past incident history; **`racereportfull`** — full stewards report (panel, going, per-race incidents).
- **`btresult`** — barrier-trial results (surface/dist/going/time, comment) — form before debut/return.
- **`draw`** — barrier W%/P% over last 3 seasons for the course/distance — draw-bias features.
- **`latestonhorse`** — late declared-starter changes / scratchings feed.
- **`formline`** — collated recent form lines. **`jockeyfavourite` / `trainerfavourite`**, `jkcstat`, `jockeyprofile`/`trainerprofile` (some need extra params / partial JS).

## Video / aerial replay (the user's "aerial replay")
Each past run on the horse page links an iframe player:
```
https://racing.hkjc.com/contentAsset/videoplayer_v4/video-player-iframe_v4.html
    ?type=replay-full | replay-aerial | replay-aerial-virtual | trackwork
    &date=YYYYMMDD&no=NN(zero-padded)&lang=eng&videoParam=PAD
```
Media resolves to Akamai: `https://streaminghkjc-a.akamaihd.net/hdflash/...` with
renditions `_700kbps.mp4 / _1200kbps.mp4 / _2500kbps.mp4` and `.m3u8` (HLS). The
**exact manifest path is constructed client-side in `JCPlayer.js`** — to get a
direct file URL, replay that path construction or capture the network request in a
browser. `type=replay-aerial` is the overhead view best suited to trajectory CV.

## Recommended additions (priority order)
1. **`racecard`** — the 27-field pre-race card (ratings + rating moves + gear + body-weight deltas + days-since-run + pedigree). Biggest single win; feeds every model incl. the carryover leg model.
2. **`corunning`** — running comments (trip lines) → NLP trouble features.
3. **`localresults`** — full exotic dividends + incidents (already partly via `historical.get_race_dividends`).
4. **`exceptionalfactors` + `veterinaryrecord` + `localtrackwork`** — gear/first-time/vet/work-load angles.
5. **Aerial replay** resolution (browser network capture of the Akamai manifest) → the Tier-2 trajectory CV pipeline.

*(Catalog produced by a recon subagent; URLs all fetched HTTP 200. Date examples used 2026/05–06 meetings.)*
