# HKJC Trajectory Feed Recon — "Hunt the Feed"

**Date:** 2026-06-04
**Method:** WebSearch + WebFetch for provider ID; live browser automation
(chrome-devtools MCP) against `racing.hkjc.com` to capture network traffic on the
results page, the video player (`replay-full` and `aerial`), and the sectional-time
page; plus direct fetch + static analysis of the player's JavaScript.
**Browser automation: AVAILABLE and used.** Nothing below is hypothetical — every
URL was observed live unless explicitly flagged as "manual playbook / not yet tested".

---

## TL;DR (the decision)

- **Provider is Trakus**, not TPD / not Swiss-Timing-Omega / not StriderHK. Trakus is an
  **RF (radio-frequency) saddlecloth-tag** system sampling **~30 Hz** — i.e. the dense
  per-horse `(x,y)` trajectory the model wants **does exist upstream at ~30 Hz**.
- **BUT it is NOT publicly fetchable as a data feed.** The "Aerial Virtual View" is
  delivered to the browser as a **pre-rendered flat MP4 video (pixels only)**, served from
  Akamai. The 3D virtualisation is rendered **server-side** (by Trakus/HKJC); the client
  receives video, never coordinates.
- There is **no JSON / WebSocket / `.dat` / `.bin` positional asset** loaded by the player.
  The only API is `getVideoInfo`, which returns **boolean availability flags only** (no data).
- The only structured public trajectory-ish data remains the **~6-point sectional
  positions + split times** the repo already scrapes (server-rendered HTML).
- **Recommendation:** the "perfect trajectories with no CV" path is **closed** via public
  endpoints. If dense `(x,y)` is required, the realistic options are (a) **CV on the aerial
  MP4** (the easiest CV target HKJC offers — clean render, distinct silks, distance markers),
  or (b) a **commercial/data-licensing request to Trakus/HKJC**. Do **not** assume a hidden
  feed exists — we looked hard and it doesn't surface client-side.

---

## UPDATE 2026-06-04 — the DRONE replay is the live overhead CV target (data-licensing ruled out)

Two corrections to the original recon, found by driving the **Multi-Angle Race Replay** player:

1. **The Trakus "Aerial Virtual View" is effectively dead** — the player constructs
   `…/hdflash/aerial/<…>/aerial_<NN>_<bitrate>kbps.mp4` (videoId `aerial_<NN>`, no date/lang)
   and it **404s on every race tested**, including ones whose `getVideoInfo` claims
   `videoFound:true`. (That per-type `{"videoFound":true}` is a **generic truthy — unreliable**;
   it even returns `true` for `drone` on races the detailed payload marks `videoDroneFound:false`.)
2. **There is a real OVERHEAD video: the `replay-drone` mode.** The Multi-Angle player exposes
   **4 angle buttons — Race Replay / Patrol Replay / Leading Horse / Drone Replay.** Clicking
   *Drone Replay* loads a genuine drone-camera MP4 (observed `206`, downloads fine):

   ```
   https://streaminghkjc-a.akamaihd.net/hdflash/replay-drone/<YYYY>/<YYYYMMDD>/<NN>/<lang>/replay-drone_<YYYYMMDD>_<NN>_<lang>_<bitrate>kbps.mp4
   #                                              ^^^^^^^^^^^^ type = "replay-drone" (NOT "drone"/"aerial")
   # filename stem is the FULL pattern  replay-drone_<date>_<no>_<lang>  (like replay-full), NOT drone_<no>
   # bitrate ∈ {700, 1200, 2500};  lang = eng
   # e.g. .../replay-drone/2026/20260603/01/eng/replay-drone_20260603_01_eng_2500kbps.mp4
   ```
   This was the bug behind every 404: the stem `drone_<NN>` / `aerial_<NN>` is wrong; the served
   asset is `replay-drone_<date>_<NN>_<lang>`.

**Availability:** read `videoDroneFound` from the **`type:"replay-full"`** `getVideoInfo` response
(it discriminates — the per-type call does not). In a scan of recent meetings only the **current
meeting (2026-06-03 HV, races 1–9)** had `videoDroneFound:true`; older dates were all `false` — so
drone coverage looks like a **rolling recent window** (download promptly; don't assume back-catalogue).

**The view (verified visually):** a near-top-down **oblique aerial** that follows the field —
horses clearly separated, **both inner & outer rails crisp and continuous** (strong homography
landmarks → maps to `hkjc.track` geometry). This is a genuine **Mode-B** source, **materially
better than the side-pan `replay-full`** and effectively *replaces* the discontinued virtual aerial.

**Access:** no geo-block observed from this (non-HK) host; `replay-drone` served `206`
range-readable and directly `curl`-downloadable. To capture a URL for a new race, open the player
iframe `…/video-player-iframe_v4.html?type=replay-full&date=<YYYYMMDD>&no=<NN>&lang=eng&…`, click
**Drone Replay**, and read the `media` request — the direct `type=drone`/`type=aerial` iframe
builds the wrong (404) URL.

**Decision update:** **data-licensing is OUT (prohibitively expensive — confirmed by owner).** So the
single trajectory path is **CV on the `replay-drone` MP4** (Mode B), gated behind a one-race PoC
(`experiments/trajectory/poc_visualize.py`). Side-pan `replay-full` remains a Mode-A fallback.

---

## 1. Provider identification (with evidence)

**Vendor: Trakus, Inc.** — confirmed by multiple independent sources:

- HKJC's own SCMP launch coverage (2014): *"Aerial Virtual Replay video is provided by
  external vendor — Trakus … overhead views are a graphical representation of the race
  constructed using the data collected by Trakus … wireless tracking of horse saddlecloths
  by antennae situated around the track."* The article quotes HKJC marketing exec Richard
  Cheung on *"issues with Trakus due to the very high density of radio signals in Hong Kong"*
  — which is the exact source of the in-app disclaimer the repo already noted (mobile-phone
  signal interference). 
  <https://www.scmp.com/sport/racing/article/1588883/aerial-virtual-replays-add-digital-experience-new-season>
- Trakus technology spec: **RF, not GPS/satellite.** Small durable radio tags in the saddle
  towel; 6–10 trackside antenna enclosures; system *"knows the location of each horse
  dynamically (30× per second)"* → **~30 Hz**, with per-segment speed, trip distance, and
  gap-to-leader computed server-side.
  - <https://www.trakus.com/trakus-system/>
  - <https://www.trakus.com/wp-content/uploads/2016/06/trakus_FAQ.pdf>
  - <https://www.drf.com/sites/default/files/Trakus%20FAQ.pdf>
  - <https://www.equibase.com/about/abouttrakus.cfm>

**Corrections to the repo's prior assumptions (`TRACETORY_RESEARCH_PLAN.md` Tier-2):**
- The plan said "rendered from an **RF/GPS** horse-positioning system." → It is **RF only**
  (Trakus uses no GPS). Worth fixing the note for accuracy.
- The plan listed **TPD** as a candidate provider. → For the *aerial virtual view / tracking*,
  it is **Trakus**. (TPD/Swiss-Timing were not found anywhere in HKJC's stack during recon.)
- The implied data rate is confirmed: **~30 Hz** native (not the ~25 Hz guessed), so the
  upstream signal is rich — the constraint is purely *access*, not existence.

> Caveat on currency: the Trakus contract is documented from 2014. HKJC could in principle
> have re-tendered since. Recon of the live 2025/26 player still resolves the feature as
> `type=aerial` and the in-app disclaimer still matches Trakus's RF-interference signature,
> so Trakus (or a like-for-like RF successor) remains the best-supported answer. Treat the
> *vendor name* as high-confidence-historical, the *"RF ~30 Hz, server-rendered, video-only
> delivery"* characterisation as high-confidence-current (directly observed).

---

## 2. Endpoints observed live (exact URLs, methods, shapes)

All captured this session via chrome-devtools MCP on a real Chrome against `racing.hkjc.com`.

### 2.1 Results landing (SPA)
- `GET https://racing.hkjc.com/en-us/local/information/localresults`
- XHR/fetch on load = **analytics only** (Adobe `edge.adobedc.net`, Datadog RUM
  `browser-intake-datadoghq.com`, Demdex). Race result data is **server-rendered** into the
  document; no result/position JSON API fires. Media (result charts) come from
  `consvc.hkjc.com/general/-/media/...`.

### 2.2 Video player — the core target
Opening any replay redirects `racing/video/play.asp` → an **iframe player**:
`https://racing.hkjc.com/contentAsset/videoplayer_v4/video-player-iframe_v4.html?type=<TYPE>&date=<YYYYMMDD>&no=<NN>&lang=<eng|zh-TW>`

The iframe calls **one** data API, then loads a flat MP4:

**`POST https://racing.hkjc.com/contentAsset/api/getVideoInfo`**
- Request body (JSON-as-text/plain):
  `{"type":"replay-full","date":"20250406","raceNo":"09","rc":null,"lang":"en-us"}`
- `replay-full` response (note: **flags only, zero positional data**):
  ```json
  {"data":{"videoFound":true,"videoPtFound":true,"noPTBar":false,"noLeading":false,
   "noPatrol":false,"videoExpired":false,"replayShort":false,"videoDroneFound":false,
   "multiAngleDate":"20120616","highQualityDate":"20110801"}}
  ```
- `aerial` response: `{"data":{"videoFound":true}}` — **still flags only.**

**Video assets (pixels only), served from Akamai `streaminghkjc-a.akamaihd.net/hdflash/`:**
- Full replay: `…/hdflash/replay-full/2025/20250406/09/eng/replay-full_20250406_09_eng_1200kbps.mp4`
  (and `_2500kbps.mp4`) — observed `200 / 206` (real video).
- **Aerial virtual view:** `…/hdflash/aerial/2025/20250406/09/eng/aerial_09_2500kbps.mp4`
  (and `_700kbps.mp4`) — observed `404` for this particular race (no aerial render produced
  for it), but the **URL template is built client-side**, confirming aerial = a plain MP4 in
  the same bucket. There is **no companion data file** requested alongside it.

  URL pattern (from player config `videoBasePath = //streaminghkjc-a.akamaihd.net/hdflash/`):
  ```
  //streaminghkjc-a.akamaihd.net/hdflash/<TYPE>/<YYYY>/<YYYYMMDD>/<NN>/<lang>/<videoId>_<bitrate>kbps.mp4
  TYPE ∈ {replay-full, patrol, leading, aerial, …};  bitrate ∈ {700,1200,2500}
  ```

### 2.3 Player JavaScript — static confirmation there is NO data render
Fetched and scanned the player scripts:
- `…/jcPlayer/JCPlayer.js` builds sources purely as
  `videoBasePath + videoId + '_1200kbps.mp4'` / `'_2500kbps.mp4'`. No coordinate parsing.
- `…/jcPlayer/jcPlayerClient.js`: the **only** `canvas` usage is `captureFrame()` which does
  `ctx.drawImage(video,…)` — i.e. it screenshots the `<video>` element. This is positive
  proof the aerial view **is a `<video>` consumed as pixels**, not a WebGL/Canvas scene
  driven by data.
- Keyword sweep across all player scripts for `trakus / .json / .dat / .bin / websocket /
  wss:// / webgl / THREE / positions / coord / x_pos` → **no positional data feed**. (The
  `.json/.dat/.bin` substrings appear only inside the minified `video.min.js` library and
  the Adobe launch tag — generic, not race data.)

### 2.4 Sectional Time & Position page (the structured data that DOES exist)
- Live URL (SPA shell): `https://racing.hkjc.com/en-us/local/information/displaysectionaltime?RaceDate=DD/MM/YYYY&RaceNo=N`
  (legacy `…/Racing/DisplaySectionalTime.aspx?RaceDate=MM/DD/YYYY&RaceNo=N` 302-redirects here).
- On load: **analytics XHR only** — the sectional table is **server-rendered HTML**. This
  matches the repo's existing `hkjc.data.sectional.get_sectional_times()` scraper (journal
  0020). Granularity = **~6 points/horse** (per-~400 m section: running position + split
  time). **No finer (sub-section / 30 Hz) public endpoint exists.**

---

## 3. Is the trajectory feed-extractable, or pixels-only?

| Layer | What's there | Public access |
|---|---|---|
| Trakus RF raw `(x,y)` ~30 Hz | **Exists** (upstream, server-side) | **NO public feed found** (no JSON/WS/binary) |
| Aerial virtual view | Server-rendered 3D, delivered as MP4 | **Pixels only** (`…/hdflash/aerial/…mp4`) |
| Sectional positions + splits | ~6 pts/horse | **YES** — server-rendered HTML (already scraped) |
| Full/patrol/leading replays | Broadcast footage | **Pixels only** (MP4) |

**Verdict: pixels-only for anything denser than ~6 sectional points.** The 30 Hz data is
generated and consumed entirely on HKJC's side to produce the MP4; the public surface never
exposes coordinates.

---

## 4. Geo / auth constraints

- `getVideoInfo` and the MP4s loaded **without login** in this session; cookies present were
  Adobe/Datadog analytics + an `isEU=false` flag (GDPR routing), not an auth gate.
- HKJC video is widely **geo-restricted to Hong Kong**; betting/live content commonly blocks
  non-HK IPs. This session's fetches succeeded for on-demand replay assets, but **expect
  geo-blocking on live streams and possibly on some replays** — verify from an HK IP / VPN
  before relying on bulk download. The `isEU` cookie shows server-side region logic is active.
- Akamai tokenisation: the observed aerial/replay MP4 URLs were **un-tokenised, directly
  cacheable** (`akamai-grn` headers, `cache-control: max-age=180` on the API; plain `206`
  range reads on the MP4). So **if** an aerial render exists for a race, the MP4 is likely
  **directly downloadable** by URL — good for a CV pipeline. (Confirm per-race; some content
  may add tokens.)

---

## 5. Prioritised recommendation

1. **Do NOT invest in finding a hidden positional feed — there isn't one client-side.** This
   recon was thorough (network capture on results/player/sectional pages + static JS audit).
   Re-checking won't change it short of reverse-engineering HKJC's internal systems.
2. **Tier-1 (sectional, ~6 pts) stays the cheap baseline** — already built; run the
   energetics kill/confirm there first (as the existing plan says). Decide on density *after*
   seeing whether even a faint `v_c` signal survives.
3. **If dense `(x,y)` is judged worth it, the aerial MP4 is the CV target — not broadcast
   footage.** It is the most tractable video HKJC offers: clean synthetic render, distinct
   silk colours, schematic rail + distance markers (e.g. 420M/920M) for homography, fixed
   virtual-camera behaviour. Gate behind a single-race PoC. Caveats: aerial is **not produced
   for every race** (`videoFound:false` / 404 cases — saw one live), it is **lossy** (RF
   noise, the disclaimer), and availability is "2–3 days after raceday".
4. **Parallel cheap shot: a licensing email to HKJC / Trakus** for the raw tracking data.
   Since the data demonstrably exists at 30 Hz, a B2B data feed may be purchasable — far
   cheaper/cleaner than CV if they'll sell it. Low effort, high potential payoff.
5. **Fix the plan's provenance note:** Trakus = **RF only (no GPS)**, **~30 Hz**, **delivery
   is server-rendered video (not a client data render)**. Drop TPD as the assumed provider.

---

## 6. MANUAL recon playbook (to reproduce / extend from an HK IP)

Browser automation worked here, but to re-verify (especially geo-gated cases) from a real HK
session:

1. Open a raceday that **has** an aerial render (Results → a recent ST/HV meeting →
   look for the "Aerial Virtual View" / overhead icon on the Results or Horse page; available
   ~2–3 days after the raceday).
2. DevTools → **Network** tab → check **Preserve log** and **Disable cache**.
3. Filter the request list to **`Fetch/XHR`** first, then **`Media`**, then **`WS`**.
4. Click the **Aerial Virtual View** play button. Watch for:
   - `POST …/contentAsset/api/getVideoInfo` → inspect JSON (flags only expected).
   - The MP4: `…/streaminghkjc-a.akamaihd.net/hdflash/aerial/<YYYY>/<YYYYMMDD>/<NN>/<lang>/aerial_<NN>_<bitrate>kbps.mp4`.
     If this `200`s, **right-click → Copy URL → download it** for the CV pipeline.
   - **Red flag to hunt:** any `.json`, `.dat`, `.bin`, `.pbf`, or **WS/`wss://`** request
     firing *with* the player. If one appears, open it — that would be the coordinate feed
     (this session found none; a different/newer player build is the only way one shows up).
5. In the **WS** filter, confirm zero websockets during aerial playback (none seen here).
6. For sectional data: open
   `racing.hkjc.com/en-us/local/information/displaysectionaltime?RaceDate=DD/MM/YYYY&RaceNo=N`
   — data is in the HTML (View Source), already handled by the repo scraper.

**What "we win" would look like:** a request whose response is an array of per-horse,
per-tick `{id, x, y, t}` (or a binary blob). We did not see it. Absence is well-evidenced,
not assumed.

---

## 7. Sources
- SCMP — Aerial virtual replays / Trakus at HKJC: <https://www.scmp.com/sport/racing/article/1588883/aerial-virtual-replays-add-digital-experience-new-season>
- Trakus system (RF, 30 Hz): <https://www.trakus.com/trakus-system/> · <https://www.trakus.com/wp-content/uploads/2016/06/trakus_FAQ.pdf> · <https://www.drf.com/sites/default/files/Trakus%20FAQ.pdf> · <https://www.equibase.com/about/abouttrakus.cfm>
- HKJC results: <https://racing.hkjc.com/en-us/local/information/localresults>
- HKJC video player iframe / `getVideoInfo` API / Akamai `hdflash` MP4s — observed live this session (see §2).
- HKJC sectional page: <https://racing.hkjc.com/en-us/local/information/displaysectionaltime>
