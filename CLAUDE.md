# CLAUDE.md — working guide for this repo

First-principles-physics HKJC horse-racing modelling + the agent scaffolding around it. Read
this first; it tells you how the repo is laid out, how to run things, and which of the three
agents to *become* for a given request.

## Principles & method — read `docs/PHILOSOPHY.md` first

The governing principles live in **`docs/PHILOSOPHY.md`** (single source). In one line: a
**causal generative model** — latent physical CAUSES (horse physiology, jockey control, prep,
environment) generate the race PATH, the path produces the finishing ORDER, the order prices
every pool — built from **first-principles physics, not regression** (every parameter a physical
constant with units, never a fitted coefficient; "no epicycles"). Meta-signals (form, jockey,
collateral) are downstream CONSEQUENCES that must **RE-EMERGE** from the structure — never inputs.
Identification comes from fitting **dedicated data surfaces with cross-surface consistency**;
path-fit + re-emergence + consistency are the PRIMARY metrics. **% of market / pool-book is the
LAST validation — parked, never the fitting target.** The edge/trading call is the user's.

> History: an acausal additive-regression win model (meta-signal feature tables + fitted gains)
> reached ~65% of market but was **purged 2026-06** as "regression dressed up as physics." The
> rebuild trades short-term goodness-of-fit for causal confidence; the journal is the full trail.

## Layout (eight packages, one-way deps)

`core` ◄ `data` ◄ `model` ◄ `betting`, with `backtesting` + `agents` on top (nothing imports
them); `vision` is an opt-in CV subsystem and `reconstruction` sits on top of `vision` + `data`/`core`.

- `hkjc.core` — contracts (`RaceModel`, `DataSource`), `Race`/`TrackGeometry`, `RaceSimulator`,
  and `hkjc_facts` (THE source for site quirks + betting rules — import it, don't hard-code).
- `hkjc.data` — scrapers + persistence (`ticks`, carryover ledger). Acquires only, no edge.
- `hkjc.model` — **the causal-physics generative win model** (the former `hkjc.statespace` was
  folded in here 2026-06; it IS the model now). Three layers: `latent`/`identify`/`field`/
  `dynamics` (latent θ inferred from the path via the EOM + Hill critical-velocity);
  `generative`+`pricer` (θ+conditions → first-passage law → `OrderDistribution` → pool prices);
  `coupling`+`gate` (calibrate coupling, validate on the pool book). `pure_mechanism` is the
  meta-signal-free build. Retained substrate: `energetics`/`energetics_dynamic` (physics),
  `calibration`/`calibration_bayes` (EB population prior), `plackett_luce`/`harville_model`/
  `conditional_logit`/`baselines` (the Harville/market/uniform references the model is scored
  against). `wald_constants` holds the provisional draw/figvar/gate coupling gains (slated for
  mechanistic replacement).
- `hkjc.betting` — `pools`, `carryover`, `clv`, and the `Strategy` seam.
- `hkjc.backtesting` — walk-forward + scoring (% of market, ECE) + **no-lookahead enforcement**
  (`AsOfView`, `assert_no_lookahead`).
- `hkjc.vision` — pure CV perception: per-feed observation models (`feeds/`, incl. the aerial-render
  `aerial_detect` blob detector + `aerial_rail` running-rail backbone), the AVV-render `aerial_layout`
  + `aerial_ocr` (odometer/legend), silk `identity`, shared primitives (`geometry_cv`/`ground_contact`/
  `ocr`/`silk`), output `schema`. Never imports `data`; heavy CV deps (the `vision` group) lazy-load.
  (The old `mode_b` package was retired 2026-06 — its layout/OCR moved to `aerial_layout`/`aerial_ocr`,
  its blob detector into `aerial_detect`, and its legacy lateral-path pipeline dropped.)
- `hkjc.reconstruction` — fuses the 4 replay feeds into per-horse `(arclen, y_lat, t)` trajectories:
  `geometry` (sources width/bend/draw from `core.Race`), `clock`, `fusion` (the constrained
  physics-kernel smoother), `Reconstructor`. Methods + state: `docs/CV_RECON.md`.
- `hkjc.agents` — the three prompts + the cross-session `Journal` + paper-trade `OpsLog`.

## Running things

```bash
uv sync --group bayes                                 # JAX is required to run the model
PYTHONHASHSEED=0 uv run pytest -q                      # tests (jax ones gated on the bayes group)
uv run ruff check src/hkjc tests studies               # lint (must stay clean)
uv run --group bayes python studies/pathfit_eom.py     # path-fit quality of the EOM generative core
uv run --group bayes python studies/identification_battery.py  # cross-surface constant consistency
uv run --group bayes python studies/field_coupling_test.py     # mean-field pace/draft de-confounding
```

Studies are disposable scratch (`studies/`); the library is the durable part and the **journal is
the durable research trail** (we can rebuild deleted scratch from it). Data panels live in
`data_cache/` (git-ignored). Never weaken a leak-free path — gate features with `AsOfView` and run
`assert_no_lookahead` before trusting any number. Use paired fixed-seed (`PYTHONHASHSEED=0`).

## The three agents — and which user request triggers which

Each agent is a role you adopt by loading its prompt: `hkjc.agents.prompts.load(<name>)`.
They share one journal (`hkjc.agents.Journal`) and talk across sessions via inboxes.

| If the user says… | Become | Prompt | Owns |
| --- | --- | --- | --- |
| "research/improve the model", "identify the physics", "test re-emergence", "/model" | **modelling** | `prompts.load("modelling")` | the causal-physics generative model. Never bets. |
| "research a betting strategy", "size stakes", "jackpot/carryover", "/strategy" | **strategist** | `prompts.load("strategist")` | strategies on top of the model. Consumes the model; never edits it. |
| "go live", "paper-trade today's card", "watch the odds", "/trade" | **trading** | `prompts.load("trading")` | the live (paper) runtime. Executes the chosen strategy; feeds observations back. |

Separation of concerns (PHILOSOPHY §8): strategist and trading **consume** model probabilities and
never modify the model — a suspected miscalibration is a journal entry to `modelling`, not a patch.

On session start, an agent reads its inbox (`Journal.inbox("<name>")`) and its living state
(`hkjc.agents.state.read("<name>")`). It records findings to the journal
(`author=<name>, channel=...`) and updates its `*_STATE.md`.

## Where the modelling work stands

Authoritative state: `journal/state/MODEL_STATE.md` (read it before any modelling work). The model
is mid-rebuild on causal foundations. The EOM generative core is in place and its constants match
equine physiology (CP ≈ 53–54 W/kg, v_c ≈ 17 m/s, W'_0 ≈ 225–246 kJ); the opening-ramp is captured
by mechanism (R²→0.77); the mean-field pace/draft coupling de-confounds cruise toward the Hill CP
and **collateral-form re-emerges** from it (journal 0049).

The decisive open result (journal **0050**): the meta-signal-free `pure_mechanism` build's % of
market **fell to the floor** when meta-signals were stripped (ST 65→−8%, HV 60→+5%) and **FORM does
NOT yet re-emerge** at the win gate (the physics-only model recovers ~7%/4% of the empirical
form→result gradient). The rebuild is honest and identifiable but **not yet predictive at the win
gate**. Diagnosis: the ability READ is too coarse — it used the noisy Hill *slope*, not the
de-confounded cruise *level* (which 0049 showed re-emerges form at +0.33/+0.40). **Next step:** swap
the ability anchor slope→level, then wire the field/pace coupling into the WIN drift (not just the
path de-confound). The fall is closed by adding causal STRUCTURE, never by re-feeding meta-signals.

The method: identify each physical constant from the surface where it dominates → demand
cross-surface consistency → de-confound → test re-emergence; win% is the parked final yardstick.
See **PHILOSOPHY** for the full statement and `docs/MODEL_DESIGN.md` for the architecture.
