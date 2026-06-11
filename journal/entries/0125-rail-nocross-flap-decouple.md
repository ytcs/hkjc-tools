---
title: No-cross gate flapped pair<->band on touching close-pair fits — midline split now adjudicates (owner zones f642-696, f1868-1916)
id: 0125-rail-nocross-flap-decouple
author: modelling
audience: ["modelling", "all"]
channel: handoff
kind: finding
status: open
thread:
tags: ["cv", "aerial", "rail", "identity", "config-a", "r02a"]
refs: {"refs": "0124", "task": 104}
created: 2026-06-11T00:00:00+00:00
updated: 2026-06-11T00:00:00+00:00
---
## Owner reports (manual review): bend f642-696 "a lot of trouble"; f1868-1916 "racing rail swapped back and forth"

Diagnosis (replicated the tracker's inner steps at f1866/f1874): the partner rail WAS recovered
every frame (upper line, median gap 10-12 px from the running fit) but the two fits CONVERGE to
gmin ~1 px at one end, and the no-cross gate (`gmin > NOCROSS` ~6 px) rejected the honest
partner outright. Next frame's fresh acquisition happened to pass, then failed again ->
**pair<->band alternation every ~2 frames**. Two visible symptoms, one cause:

- f1868-1916: in band frames the single racing-labeled curve hugs the LOWER (fixed-hugging)
  member; split frames put green on the UPPER member -> on screen the green jumps between the
  two white lines ~6x (the "swap back and forth"). Worse, each 1-frame pair became its OWN
  graph component named on ~zero vote evidence (w=0.006).
- f642-696 bend: same flap + fragmented mini-tracklets; f684/f686 fell below the confidence
  span and were sliver-labeled from a STALE pre-band tracklet -> lone-blue flicker mid-band.

NOTE: naming itself never inverted (audited: green = field-adjacent member wherever a pair
exists); the oscillation was the band fit's POSITION alternating with the split assignment.

## Fix — let the ONE pair mechanism adjudicate touching fits

`track_sequence`: when a recovered partner fails ONLY the no-cross test (gmin <= NOCROSS but
median gap plausible), run `_decouple_pair` (midline-side split of the pooled support) BEFORE
rejecting: a real close pair re-splits into two genuinely separated curves (accepted, repaired);
a coincident/spurious duplicate fails the split and still dies. This is the same `_split_band`
mechanism that already owns every "two rails fit as one structure" case — no new geometry, no
new thresholds.

## Validation

- **R08 tripwire: PASSES untouched** (config-C rails never touch; the branch never fires).
- f1858-1922: ONE stable pair component f1864-1920 (green upper y~610-622, blue lower y~636-647
  throughout; single 1-frame G dropout at f1912). Verified on the render (f1874: green on the
  field-adjacent upper rail).
- f642-712: continuous pair tracklets f666-f710+ replace the band/sliver alternation; the
  f684/f686 stale-sliver blue flicker is GONE (now proper pairs, field-adjacency verified on
  f672/f684 crops). f652-664 remain honest band singles (merged pair, racing by content).
- Suite 193 passed; ruff clean.

## Remaining in these zones (known classes, 0124 inventory)

Pair-corridor float (fits ride between the two whites by ~10-25 px on some rotated frames) and
occasional 1-frame dropouts -- Phase-2 (odometer-anchored smooth-in-s) geometry, not identity.
