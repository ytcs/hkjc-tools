---
title: Legend-templated silk detection — shadow-invariant runner anchor (R&D finding)
id: 0112-legend-templated-silk-detection-rnd
author: modelling
audience: ["modelling", "all"]
channel: handoff
kind: finding
status: open
tags: ["cv", "aerial", "detection", "silk", "legend", "shadows", "handoff"]
refs: {"refs": "0111"}
created: 2026-06-08T12:30:00+00:00
updated: 2026-06-08T12:30:00+00:00
---
## TL;DR

The silk-patch anchor **works and is the right primitive**: each AVV avatar carries a small bright
saturated non-green jockey patch that the vivid-silk detector lights up on while **completely
ignoring cast shadows** — the failure mode that makes the dark-body / `detect_avatars` route
unusable. On the R02A shadow stress frame `detect_avatars` returns **24** for a 12-horse field
(horses + shadows), while the silk detector returns 16 blobs all on horse bodies, none on shadows.

The legend-templating idea also works as an **adaptive, race-specific colour prior** and gives
**identity for free**, BUT with a hard structural limit I did not expect: **the AVV legend only ever
shows ~8 entries** (the current running order / top-8), not the full field. So legend-matching can
anchor at most ~8 runners and **cannot by itself hit the full field count**. It is a high-precision
*partial* anchor, not a complete detector.

**Recommendation:** adopt the silk patch as the detection primitive (replace dark-body); use the
legend as an adaptive colour/identity prior over the runners it lists; do **not** make the legend the
count authority. Pair silk anchors with a body/centroid refine + the field-size count constraint that
already lives in `detect_avatars`.

## What was built

`studies/cv_legend_detect.py` (study only — no production code touched; imports `detect_avatars`,
`track_band`, `render_active` for comparison). Pipeline, all derived adaptively from the frame:

1. **Locate the legend strip** by structure — the lowest sustained row-band of saturated non-turf
   boxes in the lower panel (no hardcoded coords).
2. **Count + segment swatches** by the *regular pitch*: non-green saturation column profile →
   autocorrelation pitch (constrained 95–135 px) → phase-aligned comb → median-relative per-slot
   presence test. Robust to dark silks dropping out, touching neighbours, and the R02A dark wall.
3. **Per-swatch colour signature** = top-2 LAB chroma clusters (k-means), not the mean (mean of a
   red+white halved silk = useless pink).
4. **Silk detection in the track band** — turf suppressed adaptively (modal hue ± tol), keep small
   (4–90 px) vivid (S>90, V>70) non-turf blobs; chromatic-LAB per blob.
5. **Tolerant soft match** silk→legend (min LAB ΔE over both top-2 sigs, greedy, unique; ΔE ≤ 34).
6. **Graceful fallback** — if the legend can't be parsed, return the generic vivid-silk blobs (no
   crash; verified by forcing a parse failure).

Overlays for every test frame in `data_cache/diag/legend_*.png`.

## Measured results (test frames; field in parens)

| frame | field | legend parsed | raw silks | matched | `detect_avatars` |
|-------|-------|---------------|-----------|---------|------------------|
| R08 336 | 11 | 7 (true 8) | 7 | 4 | 11 |
| R08 406 | 11 | 6 | 18 | 4 | 13 |
| R08 600 | 11 | 7 | 26 | 7 | 11 |
| R08 1080 | 11 | 8 | 19 | 4 | 11 |
| R03 700 | 12 | 9 (true 8) | 29 | 5 | 12 |
| **R02A 900** | **12** | 9 (true 8) | **16** | **5** | **24 (shadow blowup)** |

Match quality is tight: matched-silk LAB ΔE median **20–32**, max **≤ 34** across all frames — these
are genuine colour agreements, not tolerance-stretched noise. Legend parses to within ±1–2 of the
true 8 entries on every race including the one where the legend sits over a dark rail wall (R02A).

## Does it work?

- **Silk anchor — yes, strongly.** Shadow-invariant and rail-invariant as hypothesised. This is the
  load-bearing finding: it kills the shadow-explosion that no dark-body tweak can fix.
- **Legend colour/identity prior — yes, as a partial anchor.** Adaptive (no hand-tuned hues), tolerant
  match, identity as a byproduct. Robust to background (turf vs wall) and to ±style drift in
  swatch size/spacing/count because everything is read from the data.
- **Legend as the count authority — no.** The legend caps at ~8 entries (running order), so it
  structurally cannot localise an 11–12 field on its own.

## Recall / precision

- **Matched-silk precision: high** (all matches ΔE ≤ 34, landing on real horses; zero shadow hits).
- **Recall vs field: ~33–64%** (matched 4–7 of 11; 5 of 12). Bounded *above* by the ~8-entry legend
  and *below* by silks that are dark/desaturated (a black or near-turf silk has no vivid patch) or
  occluded in a bunch.
- **Raw silk layer precision is low** (18–29 blobs for an 11–12 field): turf-sparkle and silk-pixel
  fragmentation over-fire. The legend match is what cleans this up; without a legend the generic
  fallback needs an NMS/area-consensus pass before it is usable.

## Failure modes

1. **Legend ≠ field.** Top-8-only listing is the dominant limitation. (Worth confirming whether it is
   always 8, or "leaders shown", across more races.)
2. **Dark / low-chroma silks** (black, navy, near-turf green) produce no vivid patch → missed, and
   no legend signature to match. These need the body/shadow-base channel as a backstop.
3. **Bunched starts / overlap** — touching silks merge or one occludes another (R08 336: only 7 raw
   silks visible in the tight pack).
4. **Raw silk over-detection** from turf texture — must be suppressed by match or consensus.

## Robustness to style drift

Good by construction: strip located by structure, count read from data, pitch by autocorrelation,
colours in LAB with tolerance, soft matching, generic fallback. The one brittle constant is the
autocorrelation pitch window (95–135 px) — if HKJC rescales the legend this needs widening, but it is
a single, obvious, data-checkable knob, not a colour/coord table.

## Recommendation (for the detector rework)

1. **Replace the dark-body primitive with the silk-patch primitive** in the AVV detector. This alone
   removes the shadow false-positives.
2. **Use the legend as an adaptive colour + identity prior** over the runners it lists — match silks
   to it for free high-confidence IDs; keep the tolerant ΔE match.
3. **Keep the field-size count constraint** (as `detect_avatars` already does) as the count authority,
   seeded by silk anchors; refine each anchor's centroid/orientation with the dark body + shadow-base.
4. **Backstop dark-silk runners** with the body channel so recall is not capped at the vivid silks.

Net: legend-templated silk detection is **viable as the anchor + identity layer**, not as a
standalone full-field detector. Promote the silk primitive; treat the legend as a partial prior.

Artifacts: `studies/cv_legend_detect.py`;
overlays `data_cache/diag/legend_R02A_f00900.png` (shadow stress: silks on bodies, none on shadows,
vs `detect_avatars`=24), `data_cache/diag/legend_R08_f00600.png` (spread field, 7 legend matches).
