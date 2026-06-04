---
id: 0021-b13-jackpot-won-events-scraper-built-results-sid
title: B13 jackpot won-events scraper built (results side)
kind: result
status: open
tags: ["jackpot-carryover", "B13", "scraper", "frequency"]
refs: {"module": "hkjc.data.jackpot_history"}
related: []
created: 2026-06-04T01:09:43+00:00
updated: 2026-06-04T01:09:43+00:00
---
hkjc.data.jackpot_history: scrape_meeting_jackpots(date,venue) + scan_jackpot_history(start,end) reuse the tested get_race_dividends to harvest DT/TT/6UP dividend rows = the WON-event record (pool, winning combo, dividend) across a date range. Verified live: 2026-05-20..31 found 2 TT wins at HV (div HK$176,882 and HK$117,996). This is the reliable outcome side of the carryover frequency dataset; carried-in C still needs the Racing News announcement scrape (fuzzier; TODO informed by the data catalog).
