---
id: 0020-sectional-data-unblocked-url-format-was-the-bug
title: Sectional data UNBLOCKED: URL format was the bug; now server-rendered & scraped
kind: result
status: open
tags: ["trajectory", "sectional", "unblocked", "scraper"]
refs: {"module": "hkjc.data.sectional"}
related: []
created: 2026-06-04T01:09:43+00:00
updated: 2026-06-04T01:09:43+00:00
---
The 'released shortly' failure was a URL-FORMAT bug, not missing data. Working format (user-provided): displaysectionaltime?racedate=DD/MM/YYYY&RaceNo=N (lowercase racedate, day-first date, NO Racecourse). With that, the page is SERVER-RENDERED (requests.get returns full HTML, no browser/JS). Built hkjc.data.sectional.get_sectional_times(date, race_no) -> tidy long frame (date,race_no,runner,horse_id,finish_order,total_time,section,position,section_time). Verified live (2026-05-06 R4: winner J146 splits 13.74/22.76/23.78/24.01/25.14). Per-section position+time parsed; margins/sub-splits left for refinement. This turns finish-time -> dense per-section velocity profile = the energetics-model foundation. NEXT: wrap as a SectionalSource + the v_c/W' fit + harness kill/confirm.
