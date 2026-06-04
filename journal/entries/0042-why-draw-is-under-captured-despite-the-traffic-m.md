---
id: 0042-why-draw-is-under-captured-despite-the-traffic-m
title: Why draw is under-captured despite the traffic model targeting it: rail position sets draw bias (course B edge +0.15 vs C+3 +0.01), modelled as fixed course-A; plus draw is pace-contingent
kind: finding
status: open
tags: ["draw", "traffic-model", "rail-position", "course", "track-bias", "diagnostics", "policy"]
refs: {}
related: []
created: 2026-06-04T10:54:23+00:00
updated: 2026-06-04T10:54:23+00:00
---
Diagnostic on why the factor triage showed draw still holds residual info (1.04m dLL) over the traffic model that is explicitly built to encode draw. Three layers, data-grounded on the 139-race ST OOS panel:
1. SHAPE: the model's win-by-draw curve is too FLAT - empirical inner(1-4) 0.121 vs model 0.100 (outer ~0.054 both). Model under-produces the inner-draw edge.
2. RAIL POSITION (the big one): the backtest hardcoded course='A' for ALL races, but the real split is A 287 / B 283 / C+3 212 / C 141 runner-rows (~70% NOT course A). And the empirical draw edge VARIES HUGELY by rail position: course A inner-outer edge +0.080, B +0.154, C +0.019, C+3 +0.007. So draw bias is a first-order function of where the running rail is set - NOT a fixed geometric constant. The model imposes one A-level bias everywhere => over-imposes draw bias on C/C+3 (rail out, little bias) and under-imposes on B. Draw residual is larger on the mis-modelled courses (inner resid +0.035 vs +0.024 on A) confirming it. CHEAP FIX: pass the real per-race course; better, calibrate a per-course bias factor (we have the data).
   Mechanistic insight: rail movement is a TRACK-BIAS / going phenomenon (moving the rail out rests the inner and relocates the fast/fresh ground), not just a width change - our track.py models course only as a width cap on lateral offset, so the bias magnitude barely changes by course. Need a going/rail track-bias term.
3. PACE-CONTINGENCY (the deep gap): even with correct course+calibration, the model's draw effect is a STATIC, horse-agnostic geometric average (everyone runs the same even-pace policy, same lane behaviour). Real draw acts THROUGH tactics: early speed to cross/gain position, trapped-on-rail vs clear, jockey ground-saving. That is the draw x pace x style x jockey interaction the even-pace model cannot produce - exactly what step 2 (the pace/positioning POLICY) adds. So the under-captured draw is itself evidence for prioritising the policy layer.
CAVEAT: per-course win rates are on ~11-22 races each - noisy; the spread (B high, C+3 low) is directionally meaningful (well-known HK rail/track bias) but magnitudes are uncertain. Magnitude of the draw residual is small (~0.001 nats) - the value here is the INSIGHT, not the edge.
