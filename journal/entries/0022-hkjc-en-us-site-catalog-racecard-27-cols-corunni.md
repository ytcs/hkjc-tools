---
id: 0022-hkjc-en-us-site-catalog-racecard-27-cols-corunni
title: HKJC en-us site catalog: racecard (27 cols), corunning, vet/trackwork/stewards, aerial replay
kind: finding
status: open
tags: ["data-catalog", "racecard", "corunning", "aerial-replay", "exploration"]
refs: {"doc": "docs/HKJC_DATA_CATALOG.md"}
related: []
created: 2026-06-04T01:09:43+00:00
updated: 2026-06-04T01:09:43+00:00
---
Recon catalogued the modern racing.hkjc.com/en-us/local/information/* routes — ALL server-rendered (requests, no JS) except live odds (GraphQL). Param convention racedate=YYYY/MM/DD&Racecourse=ST&RaceNo=N (sectional is the DD/MM/YYYY no-Racecourse exception). HIGH-VALUE additions vs current library: racecard (27 cols incl. rating moves, gear, body-weight deltas, days-since-run, pedigree); corunning (per-horse running comments = trip/trouble NLP); localresults (full exotic dividends + incidents); exceptionalfactors/veterinaryrecord/localtrackwork (gear/first-time/vet/work angles); racereportfull (stewards); btresult (barrier trials); draw (barrier bias stats). Aerial replay video: iframe videoplayer_v4/...?type=replay-aerial&date=YYYYMMDD&no=NN&lang=eng; media on Akamai HLS/MP4, manifest path built client-side in JCPlayer.js (capture via browser for direct URL). Full catalog: docs/HKJC_DATA_CATALOG.md. Priority add: racecard (feeds every model incl. carryover leg model).
