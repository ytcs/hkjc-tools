# HKJC-tools full refactor plan

Status: proposed, awaiting approval. No backward-compat required. Tests ported, not dropped.

## 0. Goals & invariants

1. **Seven packages, clean separation of concerns**: `core`, `data`, `model`, `betting`,
   `backtesting`, `vision`, `agents`.
2. **Library = reusable mechanism, never a specific study.** Study scripts are disposable and
   will be rebuilt against the new library. (User decision.)
3. **One-way dependency graph** (a package may import only from packages below it):

   ```
   core  ◄──  data  ◄──  model  ◄──  betting
     ▲                      ▲            ▲
     └─────────  backtesting ───────────┘     (backtesting imports model+betting; neither imports it)
   vision ──► core            agents ──► everything (orchestration layer, nothing imports agents)
   ```
   Rationale: `backtesting`/`eval` must never be importable *by* `model`, so the objective
   function can't be gamed from inside the model.
4. **No-lookahead is a first-class, enforced primitive** — one implementation both model-fitting
   and backtesting consume (§5). Not re-implemented per layer, not left to convention.
5. **The EOM is single-sourced** — one set of equations, shared by the numpy and JAX backends,
   *with the mass term in both* (§4). This also fixes residual-queue #3 (bodywt, t=+9.0).

## 1. Target package tree

```
src/hkjc/
├─ core/
│  ├─ race.py            # Race + Runner dataclasses (from race.py)
│  ├─ track.py           # TrackGeometry, track_for() (from track.py)
│  ├─ contracts.py       # RaceModel ABC (from model/base.py), DataSource protocol (from data/source.py)
│  ├─ hkjc_facts.py      # NEW: single source for HKJC domain constants + quirks (§8)
│  ├─ simulation.py      # RaceSimulator monte-carlo composer (from simulation.py)
│  └─ utils.py           # html table parse, type coercers, date helpers, pareto (from utils.py)
│
├─ data/                 # connectors + persistence only; computes NO edge
│  ├─ live.py            # GraphQL live odds/pools (unchanged surface)
│  ├─ historical.py      # results + form-line HTML scraper
│  ├─ racecard.py        # pre-race racecard (27 cols)
│  ├─ sectional.py       # per-section split times/positions  (DD/MM/YYYY quirk -> core.hkjc_facts)
│  ├─ ticks.py           # NEW: one partitioned append-only tick store (merges odds_ticks +
│  │                     #      jackpot_ticks + carryover_ledger pattern)
│  ├─ feature_store.py   # runner-keyed wide-frame assembler (outcome rules -> core.hkjc_facts)
│  └─ corunning.py
│
├─ model/                # the ABM + how it is fit. THIS is the modelling agent's library.
│  ├─ eom.py             # NEW: single-sourced equations of motion (backend-agnostic step fn) §4
│  ├─ dynamics.py        # numpy DynamicsRaceModel  -> calls eom.step under numpy
│  ├─ traffic_jax.py     # JAX kernel             -> calls eom.step under jnp (NOW mass-aware) §4
│  ├─ energetics.py      # critical-velocity + W' estimation from sectionals
│  ├─ calibration.py     # EB population fit, distance bands, bend-curvature grid
│  ├─ calibration_bayes.py
│  ├─ features.py        # NEW: EXTRACTED leak-free ability features (form, jockey, style) §3
│  ├─ fit.py             # NEW: EXTRACTED EB per-horse fit + predict pipeline §3
│  ├─ constants.py       # NEW: baked calibration constants (FORM_GAIN=4.0, SPEED_NOISE=0.12,
│  │                     #      JOCKEY_GAIN=5.0, PACE_GAIN=0.0, bend_curvature) — single source
│  ├─ baselines.py       # market + uniform predictors
│  ├─ plackett_luce.py   # Harville/PL reference baseline
│  └─ harville_model.py  # numba DP solver
│
├─ betting/              # prices pools, sizes stakes; embeds NO model, NO live loop
│  ├─ pools.py           # pool pricers + dividend math (takeout-by-pool -> core.hkjc_facts)
│  ├─ carryover.py       # carryover detect + effective takeout
│  ├─ carryover_cover.py # cover optimiser
│  ├─ clv.py             # closing-line value (moved from eval/)
│  └─ strategy.py        # NEW: Strategy protocol — (probs, pool_state) -> stakes
│
├─ backtesting/          # replays history, enforces no-lookahead, scores model+odds strategies
│  ├─ walkforward.py     # folds + Lockbox (from eval/walkforward.py)
│  ├─ temporal.py        # NEW: AsOfView — the enforced no-lookahead accessor §5
│  ├─ scoring.py         # log-loss/brier/ece/rps/reliability (from eval/calibration.py)
│  ├─ selection.py       # BH-FDR, deflated/probabilistic Sharpe
│  ├─ harness.py         # walk_forward_evaluate + aggregate
│  └─ leakage_test.py    # NEW: future-perturbation assertion harness §5
│
├─ vision/               # promoted from experiments/trajectory/; CV deps optional (lazy)
│  ├─ schema.py pipeline.py detect.py track.py identify.py homography.py project.py export.py
│
└─ agents/               # orchestration: prompts + the cross-session journal §6,§7
   ├─ journal/           # NEW from-scratch multi-agent comms store §6
   │  ├─ store.py        # entries (markdown) + index.jsonl + inbox/state views
   │  ├─ schema.py       # Entry dataclass (author, audience, channel, thread, kind, status, refs)
   │  └─ ops_log.py      # high-volume paper-trade log (parquet, not markdown)
   ├─ state/             # living per-agent state docs (mutable, not append-only)
   │  ├─ MODEL_STATE.md          # seeded from /MODEL_STATE.md (residual queue, constants)
   │  ├─ STRATEGY_STATE.md
   │  └─ TRADING_STATE.md
   └─ prompts/
      ├─ modelling_agent.md      # §7.1 — carries the physics+GoF philosophy verbatim
      ├─ betting_strategist.md   # §7.2
      └─ trading_agent.md        # §7.3

studies/                 # NOT a package. Disposable research scripts, rebuilt post-refactor.
tests/                   # ported 1:1 to the new module paths; encode contracts + tribal knowledge
docs/                    # HKJC_KNOWLEDGE / DATA_CATALOG / BETTING_RULES kept (field guide)
cloud/                   # modal scheduler (unchanged)
```

## 2. Mapping summary (old → new)

| Old | New | Note |
|---|---|---|
| `race.py`, `track.py`, `simulation.py`, `utils.py` | `core/` | straight move |
| `model/base.py`, `data/source.py` | `core/contracts.py` | the two protocols co-locate |
| `data/*` | `data/` | 3 tick stores → `data/ticks.py` |
| `model/*` | `model/` | + EOM single-source, + extracted features/fit |
| `eval/walkforward,selection,harness,calibration` | `backtesting/` | scoring renamed from calibration.py |
| `eval/clv.py` | `betting/clv.py` | CLV is an odds metric |
| `bets/*` | `betting/` | + Strategy protocol |
| `experiments/trajectory/` | `vision/` | promoted, CV deps optional |
| `ledger/*` | replaced by `agents/journal/` | rebuilt from scratch (§6); knowledge entries extracted first |
| `experiments/*.py` (studies) | deleted | **after** §3 extraction; rebuilt later |

## 3. CRITICAL extraction before deleting studies

`experiments/backtest_traffic_st.py` is *de-facto model code*, not a study. Extract its guts into
the library, THEN delete the script:

- `current_form_table()` → `model/features.py: form_feature()`. Leak-free expanding mean of prior
  relative-finish. **The dominant ability signal (18% of market).** Must survive.
- `jockey_table()` → `model/features.py: jockey_feature()`. Leak-free trailing strike rate (+7pp).
- `running_style_table()` → `model/features.py: style_feature()`. Trailing rel_early (pace policy input).
- `_horse_residuals()` + `fit_predict_traffic()` → `model/fit.py`. The EB per-horse ability estimate
  (shrinkage λ=τ²/(τ²+σ²/n)) + the predict pipeline that maps (sectional EB + FORM_GAIN·form +
  JOCKEY_GAIN·jockey) → effective v_c and calls the sim. **This is the model.**
- Baked constants → `model/constants.py` (FORM_GAIN=4.0, SPEED_NOISE=0.12, JOCKEY_GAIN=5.0,
  PACE_GAIN=0.0, bend_curvature per venue). Every long-form rationale comment in the script is
  knowledge — port the comments into the module docstrings + `agents/state/MODEL_STATE.md`.

The same audit applies to `backtest_traffic_hv.py`, `calibrate_traffic_model.py` (bend grid →
`model/calibration.py`), `build_energetics_full.py`/`build_wpq_full.py`/`build_trajectory_full.py`
(dataset builders → `data/` builders or `studies/` if truly one-off), and `measure_policy.py`
(the policy/trajectory measures → `model/` if they become live inputs, else `studies/`).

**Rule for the audit:** if a script computes a *feature or fit the model needs*, it's library; if it
only *reports a number to a human*, it's a study (delete). When unsure, it's library — we can prune
later, but a lost leak-free feature is expensive to reconstruct.

## 4. EOM single-sourcing (folds in residual-queue #3)

Today `dynamics.py` (numpy, **has** `v_c = p_c/(K_LOCO·m·g)` mass term) and `traffic_jax.py` (JAX,
**kinematic, no mass**) re-implement the same step loop and have drifted. That drift *is* the bodywt
residual. Fix:

- `model/eom.py` holds the per-step physics as a **backend-parametric** function
  `step(state, params, xp)` where `xp` is `numpy` or `jax.numpy`. The ops used (clip, where, min,
  broadcasting, mod) share an API across both, so the algebra is written once.
- `dynamics.py` integrates `eom.step` under numpy; `traffic_jax.py` wraps `eom.step` in
  `jax.lax.scan` under jnp. Both now carry mass, drag, drafting, bend ground-loss identically.
- `traffic_jax.win_probs` gains a `mass=` argument (currently absent). `model/fit.py` passes each
  horse's race-day mass (body + carried weight, from racecard/results). With mass live, the bodywt
  residual is a *mechanistic* term, not a flat add — exactly the philosophy requires.
- Guard: a parity test asserts the numpy and JAX backends agree in distribution on a fixed field
  (already an informal claim in the traffic_jax docstring — make it a real test).

This is structural cleanup AND modelling fix #3 in one move (user approved folding it in).

## 5. No-lookahead enforcement (one primitive, both consumers)

Replace convention with mechanism:

- `backtesting/temporal.py: AsOfView(frame, now)` — a wrapper that can only return rows with
  `timestamp < now`. All feature builders in `model/features.py` take an `AsOfView`, never a raw
  frame, so a future row is *unreachable*, not merely un-selected. The existing leak-free patterns
  (strictly-prior expanding means, `join_asof(strategy="backward")`, EB-on-train-only) become the
  *implementation* behind this accessor.
- `backtesting/walkforward.py` hands each fold a sealed `AsOfView` for train and a separate test
  view; the Lockbox stays.
- `backtesting/leakage_test.py` — an adversarial assertion: perturb all rows dated ≥ test-date,
  re-run the fit/predict, assert predictions are **bit-identical**. Any leak makes the test fail.
  Run it in CI over a sample of folds. This is the enforcement the "rigorous no-lookahead"
  requirement needs.

## 6. Journal / cross-session comms — rebuilt from scratch (user decision)

Keep the two proven ideas from the old ledger (markdown = source of truth; rebuildable `index.jsonl`
= fast scan). Redesign the schema for **three agents talking across sessions**.

**Entry schema (`agents/journal/schema.py`):**

| field | purpose |
|---|---|
| `id` | `NNNN-slug` |
| `author` | `modelling` \| `strategist` \| `trading` |
| `audience` | list of agents the entry is *for* (`["strategist"]`), or `["all"]` |
| `channel` | `research` \| `strategy` \| `ops` \| `handoff` |
| `kind` | `finding` \| `hypothesis` \| `result` \| `decision` \| `gotcha` \| `observation` \| `deadend` |
| `status` | `open` \| `confirmed` \| `refuted` \| `superseded` \| `archived` \| `actioned` |
| `thread` | parent entry id (threads a strategist reply onto a trading observation) |
| `refs` | structured: `race_id`, `date`, `venue`, `pool`, constants touched |
| `tags`, `created`, `updated`, `body` | as before |

**Views the index materialises (`agents/journal/store.py`):**
- `INBOX-<agent>.md` — `audience contains me AND status==open`, newest first. **Each agent reads its
  inbox on session start.** This is the cross-session handoff mechanism (e.g. trading files a
  `channel=handoff, audience=[strategist]` observation; strategist picks it up next session).
- `INDEX.md` grouped by author × channel.
- `index.jsonl` — full-text + metadata scan, rebuildable.

**Two stores, by volume:**
- Durable narrative (findings, decisions, handoffs) → markdown entries (above).
- High-volume operational data (paper-trade fills, live-odds snapshots) → `agents/journal/ops_log.py`,
  a partitioned **parquet** store (markdown would drown). Only *notable* ops events get promoted to a
  markdown `observation` entry for the strategist.

**Living state vs append-only log:** `agents/state/*_STATE.md` are *mutable* working memory (the
modelling agent's residual queue + current constants, the strategist's active strategies). The journal
entries are the append-only audit trail of *why* the state changed. `MODEL_STATE.md` (already written
at repo root) is the seed for `agents/state/MODEL_STATE.md`.

**Migration of old journal:** before wiping, extract the tribal-knowledge entries (sectional URL
quirk 0019–0020, site catalog 0022, takeout 0003, AWT detection, lastUpdateTime dedup) into
`core/hkjc_facts.py` + `docs/HKJC_KNOWLEDGE.md`. Everything else (research narrative) is dropped.

## 7. The three agent prompts (full drafts)

### 7.1 Modelling agent — `agents/prompts/modelling_agent.md`

> **Role.** You own the agent-based race model (the plant: power-balance EOM with finite W′ reserve,
> traffic, drafting, bend ground-loss, mass; and the policy: kick timing/intensity, pace tilt) and
> its calibration. You do not place bets or design betting strategies — you make the model fit reality.
>
> **Philosophy — first-principle regularization (non-negotiable).** The model is regularized by
> *physics and game theory*, not by feature count. Every factor you introduce MUST couple as a real,
> plausible mechanism in the EOM or the policy. An arbitrary regressor that merely correlates is
> forbidden — it overfits and violates the discipline that got us from 10% to ~48% of market.
>
> **Objective — fit the market first.** Your metric is **% of market = goodness-of-fit**, not edge.
> `% = (uniform_LL − model_LL)/(uniform_LL − market_LL)`, measured leak-free walk-forward. You are
> NOT hunting for mispricings. Closing the gap to the market IS the job; whether any residual gap is
> tradable is the user's call, not yours.
>
> **No private-info excuse.** Never stop because "the rest is private information / unknowable." That
> reasoning would have quit at 10% skill; we are at ~48%. Keep finding the next mechanism.
>
> **The method — diagnose → recalibrate → bake → re-diagnose.** Don't guess covariates. Run the model
> walk-forward, take `resid = market_prob − model_prob` (de-vigged, race-normalised), regress on each
> observable mechanism with race-clustered SEs. The largest |t| mechanism is the next thing to ground
> in the physics. Fix it (prefer single-constant recalibrations grounded in a mechanism), re-measure,
> find the next. Verify every fix improves OOS % of market AND calibration (ECE), not just in-sample.
>
> **Working memory.** Read `agents/state/MODEL_STATE.md` on start (current constants + residual queue).
> Record findings to the journal (`channel=research`). When a strategist/trading observation lands in
> your inbox suggesting a mechanism, evaluate it through the diagnosis loop. Update MODEL_STATE when
> constants change; journal *why*.
>
> **Current state (seed).** Constants: FORM_GAIN=4.0, SPEED_NOISE=0.12, JOCKEY_GAIN=5.0, bend_curvature(ST)=0.0037.
> Open queue by |t|: (1) draw OVER-weighted t=−17.4 — recalibrate bend_curvature; (2) w_prime
> UNDER-weighted t=+12.9; (3) bodywt/mass UNDER-weighted t=+9.0 — now addressable via the single-sourced
> mass-aware EOM. Exhausted: flat additive covariates (form+jockey≈41%, policy/weight add ≈0).

### 7.2 Betting strategist — `agents/prompts/betting_strategist.md`

> **Role.** You research betting strategies that turn the model's probabilities (and pool structure —
> takeout-by-pool, jackpot/carryover payouts, exotic combinatorics) into staking decisions. You CONSUME
> the model; you never modify it. You produce strategies and backtest them with rigorous no-lookahead.
>
> **Separation of concerns.** Model probabilities are an input you treat as fixed. If you believe the
> model is miscalibrated in a way that matters for a strategy, file a `channel=research, audience=[modelling]`
> journal entry — do not patch the model yourself.
>
> **Discipline.** Net every EV against the *correct* pool takeout (Win/Place/QIN/QPL ≈17.5%, exotics
> ≈19.5–25% — see `core.hkjc_facts`). Account for carried-in funds in carryover pools (effective takeout
> can go negative — a structural edge independent of model skill). Control multiple-testing
> (BH-FDR, deflated Sharpe) — every strategy variant you try is a trial. CLV is the leading indicator.
>
> **Outputs.** A strategy implements `betting/strategy.py: Strategy`. Backtest via `backtesting/`,
> report CLV + calibration + deflated Sharpe. Record promising/refuted strategies to the journal
> (`channel=strategy`). Read your inbox for trading-agent observations (live frictions, pool behaviour)
> and the modelling agent's calibration changes.

### 7.3 Trading agent — `agents/prompts/trading_agent.md`

> **Role.** The live runtime. During a race meeting you pull live odds/pools (`data/live.py`), load the
> model's pre-race probabilities, apply the strategist's chosen strategy, and make trade decisions.
> **Paper-trade only for now** — record intended fills, never send real bets.
>
> **Separation of concerns.** You execute strategies; you do not invent them and you do not retune the
> model mid-meeting. If live conditions break a strategy's assumption, log it — don't improvise a new edge.
>
> **Loop.** Poll odds (dedup on server `lastUpdateTime`), compute strategy stakes vs current pool state,
> log each decision + the closing line for CLV. Write paper fills to `agents/journal/ops_log.py` (parquet).
>
> **Cross-session value — this is important.** You see things research can't: late odds drift, pool
> liquidity, scratchings, going changes, draw-bias that showed up live. Promote notable observations to
> markdown `observation` entries (`channel=handoff`, `audience=[strategist]` or `[modelling]`). Example:
> a systematic late-money steam pattern → strategist; a track/draw anomaly → modelling. This is how the
> live edge feeds back into research across sessions.

## 8. Tribal knowledge → `core/hkjc_facts.py` (must not be lost)

Single module of named constants + a docstring quirk-guide, referenced everywhere:
- Sectional URL date format **must be `DD/MM/YYYY`** (else empty "released shortly" page).
- AWT/surface detection: "ALL WEATHER TRACK"/"AWT" label is authoritative; going codes are a weak
  fallback; GOOD/GD appear in both scales and must NOT trigger AWT.
- Live-odds dedup key = server `lastUpdateTime`.
- GraphQL endpoint `https://info.cld.hkjc.com/graphql/base/`.
- Takeout: Win/Place/QIN/QPL ≈17.5%; exotics (TRI/FF/QTT/DT/TT) ≈19.5–25% — never assume uniform.
- Place rule: 3 paid for ≥7 runners, 2 for 4–6, 0 for ≤3.
- Venue codes ST/HV; empty string returns current meeting.
- Speed/energy figures: current-meeting `speedguide` JSON only (null pre-publish), no historical `.xls`.
Keep `docs/HKJC_KNOWLEDGE.md`, `docs/HKJC_DATA_CATALOG.md`, `docs/HKJC_BETTING_RULES.md` as the prose
field guide; `hkjc_facts.py` is the machine-readable mirror.

## 9. Deletions

- All `experiments/*.py` **after §3 extraction** (studies, rebuilt later against the new library).
- Old `journal/entries/*` **after §6 knowledge extraction**.
- `ledger/` package (replaced by `agents/journal/`).
- `docs/POLICY_MAP.md`, `MODEL_ENRICHMENT.md`, `AGENT_ARCHITECTURE.md` → fold the durable parts into
  the agent prompts / `MODEL_STATE.md`, then delete.

## 10. Migration sequence (bottom-up; tests green at every step)

1. **`core`** — move race/track/simulation/utils; create `contracts.py`, `hkjc_facts.py`; repoint imports; tests green.
2. **`data`** — move connectors; merge tick stores into `data/ticks.py`; route quirks through `hkjc_facts`; tests green.
3. **`model`** — move models; build `eom.py` and refactor dynamics+traffic_jax onto it (§4, parity test);
   **extract features/fit/constants from the backtest scripts (§3)**; tests green incl. EOM parity.
4. **`betting`** — move bets + clv; add `strategy.py`; tests green.
5. **`backtesting`** — move eval; add `temporal.py` (AsOfView) + `leakage_test.py` (§5); wire features through AsOfView; tests green.
6. **`vision`** — promote trajectory pipeline; CV deps to an optional group; smoke test.
7. **`agents`** — build `journal/` (§6) + `state/` + `prompts/` (§7); extract tribal knowledge from old
   journal into `hkjc_facts`/docs; seed `MODEL_STATE.md`; wire a session-start inbox read.
8. **Delete** studies + old journal + ledger (§9). Rebuild a first study (the residual-diagnosis loop) on
   the new library to prove the seams.
9. **Re-confirm the model** end-to-end: ST ≈48% / HV ≈43% of market reproduced through the new
   `model/fit.py` + `backtesting/harness.py`. This is the acceptance test — the refactor must not move the number.

## 11. Risks

- **Hidden model code in scripts** (§3) — the top risk. Audit every script for leak-free features/fits
  before deletion. Acceptance test #9 catches a regression but extraction is manual.
- **EOM parity** — folding numpy+JAX onto one step fn risks a subtle divergence; the parity test is the guard.
- **Big-bang scope** — mitigated by bottom-up, tests-green-per-step; never leave the tree broken across a step.
- **Journal redesign** — don't lose the tribal entries; §6 migration extracts them first.
```
