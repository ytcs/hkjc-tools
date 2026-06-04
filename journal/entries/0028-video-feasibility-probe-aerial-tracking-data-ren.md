---
id: 0028-video-feasibility-probe-aerial-tracking-data-ren
title: Video feasibility probe: aerial = tracking-data RENDER; CV-extractable but low-EV
kind: finding
status: open
tags: ["video", "aerial", "tracking-data", "feasibility", "cv"]
refs: {"url_pattern": "streaminghkjc-a.akamaihd.net/hdflash/replay-aerial/..."}
related: []
created: 2026-06-04T05:35:26+00:00
updated: 2026-06-04T05:35:26+00:00
---
Probed the aerial replay. FINDINGS: (1) directly downloadable MP4, deterministic URL streaminghkjc-a.akamaihd.net/hdflash/replay-aerial/{YYYY}/{YYYYMMDD}/{NN}/eng/replay-aerial_{YYYYMMDD}_{NN}_eng_{1200|2500}kbps.mp4 (1080p/25fps). (2) The 'aerial' is a SPLIT screen: real broadcast (left) + 'Aerial Virtual View' (right) = a SYNTHETIC top-down reconstruction rendered from HKJC's horse POSITION TRACKING (GPS/RFID; disclaimer warns mobile-phone interference may reduce accuracy). (3) So the 2D positions WE WANT (lateral, ground-lost) EXIST and are visible as per-horse silk sprites -- but only as a RENDERED VIDEO; getVideoInfo API exposes no raw position feed (videoFound/videoDroneFound metadata only; a real 'drone' video may also exist). (4) CV extraction feasible but NOT trivial: naive saturation-blob on the synthetic panel gave 97 false blobs (textured track); needs silk-template matching + temporal tracking + homography -- a real days/weeks build. KEY: the stewards' running comments we already TESTED (gate, journal 0026) are the EXPERT interpretation of THIS SAME tracking data + all angles, and they don't beat the market -> CV re-deriving the positions (less expertly, from a disclaimed-accuracy render) is very unlikely to. The one genuinely-new signal is QUANTITATIVE ground-lost (meters) for the energetics model, which stewards don't encode numerically -- but cost+accuracy+4x-efficiency-prior make EV low. VERDICT: feasible, not worth the full build; revisit only if a raw position-data feed is found or the energetics model shows promise and needs ground-lost.
