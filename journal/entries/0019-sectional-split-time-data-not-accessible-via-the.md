---
id: 0019-sectional-split-time-data-not-accessible-via-the
title: Sectional split-time data not accessible via the SPA (this env); RunningPosition is the partial
kind: gotcha
status: open
tags: ["trajectory", "sectional", "data-blocked", "spa"]
refs: {}
related: []
created: 2026-06-04T00:53:57+00:00
updated: 2026-06-04T00:53:57+00:00
---
Trajectory Tier-1 data spike (browser). The 'Sectional Time & Position' page is a JS SPA; the legacy .aspx 302s to it and serves no server-side table. Across multiple races/dates the SPA renders 'Information will be released shortly' and fires NO data XHR (only analytics) -> split-time data is not served in this environment. Available partial signal: result pages' RunningPosition (sectional POSITIONS at each call) + total FinishTime, which we already scrape. Implication for the trajectory line: the clean sectional split-TIME source is blocked here; fall back to RunningPosition as a semi-dense signal and prioritise the video/CV path for true density, or revisit the sectional API later (it may be a deferred GraphQL call or unprovisioned in this env).
