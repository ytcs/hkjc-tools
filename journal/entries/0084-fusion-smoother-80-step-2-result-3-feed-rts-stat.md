---
title: FUSION smoother #80 step 2 RESULT — 3-feed RTS state-space reconstructs ALL 12 horses' (arclen,lateral,t) trajectory full-race; the +4.1m rail-wide gap SURVIVES fusion (+3.84m) and EXTENDS to 8/8 commented horses; run-in y_lat uncertainty inflates honestly 0.98m(patrol anchor)->1.72m(line); longitudinal is field-shared (per-horse order awaits RF margins)
id: 0084-fusion-smoother-80-step-2-result-3-feed-rts-stat
author: modelling
audience: ["all"]
channel: research
kind: result
status: open
thread: 0060-reorientation-win-diagnostic-not-veto-wave2-expand-causal-network
tags: []
refs: {}
created: 2026-06-07T04:58:32+00:00
updated: 2026-06-07T04:58:32+00:00
---
RESULT — #80 STEP 2, the FUSION STATE-SPACE SMOOTHER + first reconstructed-race trajectories. studies/cv_fusion_smoother.py (study-only; src/hkjc untouched; the 3 input parquets read-only/not mutated; official sections/RP/stewards VALIDATION-only). ruff clean (src/hkjc + studies). ST 2026-06-03 R1, Class5 1650m turf C (21.3m).

THE MODEL (faithful to the #80 design). Per-horse independent numpy Kalman forward-filter + RTS backward-smoother on ONE shared race_t grid (5Hz, 448 nodes, race_t 9.2-98.6s). Latent z=[s,s_dot,l,l_dot]. PHYSICS motion prior applied JOINTLY in the smoother (NOT a per-feed pre-filter): bounded longitudinal accel (white-accel sigma 0.8 m/s^2, near-constant cruise) + BOUNDED LATERAL accel (sigma 0.35 m/s^2 -> slow sideways drift) — this smoothness prior is the principled home of the no-teleport/jitter concern. Each feed enters ONLY over its coverage window with its own noise: aerial odometer arclen (field-shared, sd 3m) + y_lat (sd 6m, inflated 14m on >1300m foreshortening, 0075); drone time-anchored arclen (sd 8m) + y_lat (sd 7m, inflated 11m on the run-in where its metric §4b-nulls, 0080); PATROL = the METRIC ANCHOR y_lat (sd 1.2m / confidence, 0082/0076) pinning absolute scale on its 1058-1279m window. State covariance propagated -> uncertainty bands.

SHARED CLOCK (0077 reused): aerial t->field-odometer arclen->race_t (official-section curve); drone race_t=(vid-3.0)*0.8931 (0080) -> arclen; patrol already on race_t/arclen/y_lat_m. IDENTITY union: aerial 11/12 (legend) + drone 10/12 (silk->number) + patrol 8/12 (silk LUT) => UNION 12/12.

RESULTS (the first real reconstruct-the-race output):
- 12/12 HORSES reconstructed over the WHOLE race (race_t 9.2-98.6s == arclen ~150-1647m). Per-horse feed mix: 6 horses see all 3 feeds (#1,2,5,6,8,9,12), #3 aerial+patrol, #4 aerial-only, #10/#11 aerial+drone, #7 drone-only (a 771-1646m partial). Output cv_fusion_trajectories.parquet (5193 rows; runner,race_t,arclen,y_lat_m,sd_arclen,sd_ylat).
- FUSED TIGHT-vs-WIDE GAP SURVIVES FUSION and EXTENDS: rail #4/8/11/12 mean 5.39m vs wide #1/2/6/9 mean 9.23m => GAP +3.84m (stewards/patrol ref +4.1m), now covering 8/8 stewards-commented horses (patrol-alone covered 4/8 — fusion roughly doubles the validated coverage). All 4 rail horses read <=6.7m, all 4 wide horses >=9.5m: a clean rail/wide split with NO mis-class. The +4.1m gap survives AND extends to more horses — the deliverable's headline question answered YES.
- PER-BAND y_lat UNCERTAINTY (the segment-quality census — what each segment is worth): start/accel(150-450m) 1.44m; backstr/bend(450-850m) 2.03m (thin: drone window starts ~770m, so 450-770m is aerial-only); bend->straight(850-1300m) 1.00m (the BEST — co-located with the patrol anchor + the cross-validated <1300m aerial/drone ordering zone); RUN-IN(>1300m) 1.41m mean. THE HONEST GAP IS VISIBLE IN THE FINE PROFILE: sd is tightest ~0.96-0.98m at 1200-1300m (the patrol metric anchor) and inflates MONOTONICALLY through the run-in: 1.18(1300-1350) -> 1.50(1450) -> 1.72m at the line (1600-1650m). i.e. as soon as we leave the patrol metric anchor and only foreshortened ORDERING feeds remain, the absolute-metric band widens ~75% — exactly the honest run-in gap the recon figure must show.

THE RECON FIGURE (cv_fusion_recon.png, 2 panels): TOP = arclen vs race_t, all 12 horses + 2-sigma bands on the shared clock; BOTTOM = y_lat vs arclen, the lateral/control trip, 12 horses + 2-sigma bands, with the gold PATROL metric-anchor window (visibly TIGHTEST bands there) and the crimson >1300m run-in divider where the bands visibly FAN OUT. Feed-coverage strip annotated.

HONEST LIMITS (no overclaim — this is a FIRST fusion):
1. The LATERAL is the real fused product and it is validated at the rail-vs-wide CLASSIFICATION / ORDERING level (+3.84m gap, clean split), NOT a clean whole-race ABSOLUTE metric — the metric is anchored only WHERE PATROL SEES IT (1058-1279m); elsewhere the scale rides on the prior + the §4b-null ordering feeds, hence the honestly-widening run-in bands. This matches the per-feed verdicts (all 3 feeds §4b-null on the absolute metric; patrol the only within/between<1).
2. The LONGITUDINAL recon is the FIELD ODOMETER (frame-shared distance-run common to all horses) + coarse drone arclen — per-horse STAGGER is essentially UNOBSERVED (field arclen std ~1.2m at a pole). So the fused lead-ORDER vs official RP is NOT meaningful (rho -0.57/-0.56/+0.31/+0.03 at the 450/850/1250/1640m poles — dominated by sub-metre smoother artifacts, reported with the caveat). A real per-horse longitudinal order needs the RF MARGINS feed — the un-built 4th feed. fused y_lat(<1300m) vs official mean-RP Spearman is only +0.10 because mean-RP conflates lead/back with rail/wide; the rail-vs-wide gap is the cleaner validated axis (and 0075's +0.45 was that conflated corr on aerial-alone).
3. Single race; per-horse independent (field interactions = future work); 4 of the patrol-anchored horses are in the ambiguous pink-silk family (down-weighted by confidence).

NET: the 4-feed-design fusion (3 feeds folded; RF margins pending) now produces a single per-horse latent (arclen, lateral, t) trajectory for ALL 12 horses on the shared clock under a first-principles physics motion prior, with propagated uncertainty that correctly TIGHTENS at the patrol metric anchor and WIDENS on the run-in. The validated deliverable is the rail-vs-wide CONTROL classification (S8-ready: who saved ground / went wide), with absolute metres pinned only on the patrol window — solid where solid, honestly-wide where thin. Builds directly on 0083 (partial).

ARTIFACTS: studies/cv_fusion_smoother.py; data_cache/cv_fusion_trajectories.parquet (5193 rows, 12 runners); data_cache/figs/cv_fusion_recon.png. Inputs reused read-only: cv_s0d_aerial_render.parquet (#82/0075), cv_drone_numbered.parquet (#87/0080), cv_patrol_numbered.parquet (#86/0082).
