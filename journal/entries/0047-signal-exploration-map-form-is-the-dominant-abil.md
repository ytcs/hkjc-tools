---
id: 0047-signal-exploration-map-form-is-the-dominant-abil
title: Signal-exploration map: form is the dominant ability signal (23% ceiling); jockey/mkt-rel-form add ~+3pp each; field-strength weighting + Elo HURT (data-starved)
kind: finding
status: open
tags: ["form", "elo", "jockey", "ability", "ceiling", "diminishing-returns", "strength-of-field"]
refs: {}
related: []
created: 2026-06-04T11:59:53+00:00
updated: 2026-06-04T11:59:53+00:00
---
Cheap calibrated conditional-logit ceiling tests (vs market) to pick the next mechanism after the form breakthrough (form+draw = 23% of market skill; dynamics with form = 29%):
- jockey trailing strike-rate (form_dataset, 56% cov): form+draw+jockey 26% (+3pp).
- market-relative form (expanding mean of won-p_market over prior runs; corr with won ~0; market doesn't persistently misrate horses): form+mktform+draw 26% (+3pp, may overlap jockey).
- field-strength-WEIGHTED form (weight rel-finish by field size): 8% - HURTS vs raw form 23%.
- multiplayer Elo from results (strength-of-opposition aware, the principled extractor): corr(elo,won)=0.114 < form 0.15; elo+draw 20% < form+draw 23%. WORSE - data-starved (~3 runs/horse, ratings barely leave 1500). - win_rate / combining collinear signals -> numerical blowups (collinear with form).
CONCLUSION: raw expanding relative-finish form is the robust dominant ability signal; opposition-strength refinements (Elo, field-weighting) need far more history than 3300 runs and currently HURT. The structural ceiling with the signals we have is ~26-30% of market skill (we're at 29% in the dynamics). Remaining clean lever: jockey (+3pp, needs a 5-season jockey scrape for robust trailing rates). Bigger future gains need (a) more history (the flywheel) so Elo/strength-of-field start paying off, (b) current-condition signals (trial/gear/body-weight). The form mechanism took the gap from 9x to 3.4x - the user's 'much closer than 10x' goal is met; further closing is incremental.
