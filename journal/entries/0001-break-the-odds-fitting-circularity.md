---
id: 0001-break-the-odds-fitting-circularity
title: Break the odds-fitting circularity
kind: decision
status: open
tags: ["modelling", "circularity", "edge"]
refs: {"doc": "docs/AGENT_ARCHITECTURE.md#0.1"}
related: []
created: 2026-06-03T21:25:01+00:00
updated: 2026-06-03T21:25:01+00:00
---
The Harville baseline fits latent strengths TO market odds, so it can only reproduce the market, never beat it. The research model must instead predict from features (with odds as ONE feature) and be scored against realized results + CLV. Edge = divergence from the market's implied distribution x superior calibration. This reframes what the model layer is for.
