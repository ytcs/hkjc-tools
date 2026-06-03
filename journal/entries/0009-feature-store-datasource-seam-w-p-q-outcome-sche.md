---
id: 0009-feature-store-datasource-seam-w-p-q-outcome-sche
title: Feature store: DataSource seam + W/P/Q outcome schema + language coverage
kind: decision
status: open
tags: ["feature-store", "datasource", "outcome-schema", "language-parity"]
refs: {"modules": ["hkjc.data.source", "hkjc.data.feature_store"]}
related: []
created: 2026-06-03T21:44:44+00:00
updated: 2026-06-03T21:44:44+00:00
---
Step 3 shipped hkjc.data.source (DataSource protocol: name/languages/role/provides/fetch) and hkjc.data.feature_store (FeatureStore assembles sources into one runner-keyed frame on canonical keys date/venue/race_no/runner; union-universe join so partial-coverage sources don't drop rows). Outcome schema: paying_places() encodes the field-size place rule (3 if >=7, 2 if 4-6, else 0), add_win_place_outcomes() derives won/placed/finish_pos. coverage_report()/language_coverage() make a Chinese-vs-English gap a measured number. Pure adapters results_to_outcomes() and market_features_from_win() are unit-tested; the data layer imports neither model nor eval (clean direction).
