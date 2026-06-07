---
title: "Stage-2 / RQ-pool (statespace): the POOL-BOOK gate is built (WIN/PLA/QIN/FCT/TCE/TRI, de-vigged on the realized combination) — and `a` does NOT convert on the ordering pools either. The reformulation's central wager FAILS its decisive probe: `a` HURTS QIN/FCT monotonically, worst in the sprint band where it lives (sprint QIN −1.01%, FCT −0.86% at g=.075). `a` lifts only WIN. An honest, important negative."
id: 0043-statespace-stage2-poolbook-gate-acceleration-does-not-convert-on-ordering-pools
author: modelling
audience: ["modelling"]
channel: research
kind: result
status: open
thread:
tags: ["statespace", "generative", "state-space", "acceleration", "pool-book", "pool-gate", "forecast", "quinella", "trifecta", "tierce", "place", "de-vig", "devig", "pct-of-market", "order-distribution", "ordering-pools", "leak-free", "stage-2", "rq-pool", "negative-result", "draft", "#40", "#41", "0042"]
refs: {"design": "docs/STATESPACE_DESIGN.md", "program": "docs/RESEARCH_PROGRAM.md", "rq_a1": "journal/entries/0042-statespace-stage1-rq-a1-acceleration-gate-byte-exact-core-no-promote.md", "study": "studies/statespace_poolbook_gate.py", "win_gate": "studies/statespace_a_gate.py", "pricer": "src/hkjc/statespace/pricer.py", "gate": "src/hkjc/statespace/gate.py", "facts": "src/hkjc/core/hkjc_facts.py", "dividends": "data_cache/dividends_st.parquet", "test": "tests/test_statespace_skeleton.py"}
created: 2026-06-06T00:00:00+00:00
updated: 2026-06-06T00:00:00+00:00
---
DRAFT — RQ-pool result. Production `hkjc.model` UNTOUCHED (imported/reused only). All work is in
`src/hkjc/statespace/` (pricer + gate) + `studies/` + a test. Dividends are realized post-race
outcomes used ONLY for scoring, never fitting. `a` is NOT promoted into any pool model.

## THE DECISIVE TEST
RQ-a1 (journal 0042) found acceleration `a` does NOT convert on the WIN order-statistic (+0.14%
%market overall, ECE strictly worse) — the win bit could not reward a latent that shapes the ORDER
but not the winner (`a` = a fast opener that leads early, shapes the order, but cruises lower so does
not win more). The reformulation's CENTRAL WAGER was that the broader POOL BOOK — the ORDERING pools
(forecast/quinella) that read the orderings and separations win discards — rewards latents win cannot.
This entry builds that gate and runs `a`'s decisive probe on it.

## WHAT WAS BUILT (the pool-book gate)
1. **PoolPricer — realized-combination probabilities** (`statespace/pricer.py`). From the ONE joint
   `OrderDistribution` MC sample (the object every pool price is a functional of), the model's
   Laplace-smoothed probability of each race's REALIZED winning combination, per pool: `prob_win`
   (P 1st), `prob_place` / `place(k)` (top-k, k per HKJC `places_paid`: 3 for field≥7 else 2),
   `prob_quinella` ({i,j}=top-2 unordered), `prob_forecast` ((i,j)=top-2 ordered), `prob_tierce`
   ((i,j,k)=top-3 ordered = TCE), `prob_trio` ({i,j,k}=top-3 unordered = TRI). All Laplace-smoothed
   `(hits+1)/(n_samples+n)` so a zero-sample combo never gives infinite log-loss. Consistency proven
   (quinella = fc(i,j)+fc(j,i); tierce ⊆ trio; trio = place-set for k=3) — they are marginals of ONE law.
2. **De-vigged market benchmark** (`studies/statespace_poolbook_gate.py`). The dividend's `decimal` on
   the realized combination → market implied prob = (1 − takeout)/decimal, takeout per pool IMPORTED
   from `hkjc.core.hkjc_facts.takeout_for` (WIN/PLA/QIN 17.5%, FCT/TCE 20.5%, TRI 25%). PLA scored
   PER-LEG (each realized placer is its own "horse i in top-k" event priced by its own place dividend;
   the set-as-product over-counts via spurious independence — leg-by-leg is the clean anchor).
3. **Per-pool skill metric — `gate.pool_book_score`**. Per pool: mean per-race model / market / uniform
   log-loss on the realized combination and the exact `%market = (uniform_LL − model_LL)/(uniform_LL −
   market_LL)` — EXACTLY the win %market definition, but on the exotic combination. Uniform baseline =
   P(realized combo | uniform random order): WIN 1/n, PLA k/n (per-leg), QIN 1/C(n,2), FCT 1/(n(n−1)),
   TCE 1/(n(n−1)(n−2)), TRI 1/C(n,3).
4. **Coverage.** ST dividends only (no `dividends_hv.parquet` — ST has the strongest sprint-`a` signal).
   170 ST races (2026-03-01..05-31), all six pools on all 170 (169 after field/combo validity). The
   model frame is the full 1952-race leak-free `backtest_st.load()`; the gate scores the 170-race
   dividend subset. Bands: sprint n=76, mile n=71, middle n=20 (the middle cell is small/noisy).

## (1) THE GATE TABLE — a=OFF vs the RQ-a1 `a_start_gain` grid, per pool (ST, %market | model_LL)
PAIRED fixed-seed (PYTHONHASHSEED=0), full-scale, N_SAMPLES=20000. Per-race seed shared a-OFF/a-ON.

  pool   n    a=OFF      g=.05      g=.075     g=.1       g=.2       Δ(best≠0 vs OFF)
  WIN   169  74.14%     74.65%     74.80%     74.97%     75.53%     +1.39% (g=.2)   ↑ CONVERTS
  PLA   170  75.63%     75.24%     75.01%     74.80%     73.66%     −1.97%          ↓ hurts (monotone)
  QIN   169  67.91%     67.66%     67.44%     67.28%     66.33%     −1.58%          ↓ hurts (monotone)
  FCT   169  70.50%     70.06%     69.92%     69.72%     68.65%     −1.85%          ↓ hurts (monotone)
  TCE   169  70.15%     70.13%     69.84%     70.17%     68.42%     ~0 / −1.74%     ↓ (soft, noisy)
  TRI   169  79.21%     78.91%     78.44%     78.31%     77.24%     −1.97%          ↓ hurts (soft)

  BEST non-zero gain per pool: WIN g=.2 (+1.39%); PLA/QIN/FCT/TRI best = a=OFF (every gain hurts);
  TCE best g=.1 (+0.02%, pure MC noise). Only WIN improves; every ordering/set pool degrades monotonically.

## (2) THE SPRINT BAND — the #41/0040 prediction, the decisive cell (a=OFF vs forced g=.075)
`a` lives in the SPRINT (start_frac = min(1, L_ref/L), L_ref≈1200m). RQ-a1's ONLY win lift was the
sprint band (+0.48% sprint %market). The wager: that sprint lift should be BIGGER on the ORDERING pools.
It is the opposite sign:

  QIN  sprint(76)  73.62% → 72.61%   Δ −1.01%      mile(71) −0.19%   middle(20) +0.56%
  FCT  sprint(76)  73.54% → 72.68%   Δ −0.86%      mile(71) −0.46%   middle(20) +0.08%

The ordering pools lose the MOST in the exact band where `a` produced its only win lift. (The middle
+0.56%/+0.08% is n=20, inside the MC/soft-pool noise band.)

## (3) THE VERDICT — does `a` convert on the ordering pools where it failed on win?
**NO.** The reformulation's central wager FAILS its decisive probe. `a` does not merely fail to convert
on the ordering pools — it actively HURTS them, monotonically in the gain, worst in the sprint band
where the mechanism is strongest. The only pool `a` lifts is WIN (+1.4% at g=.2 here vs +0.14% in
RQ-a1's full-scale win gate — the larger figure is the 170-race March–May subsample, not the full
walk-forward, so it overstates; the SIGN, win-only-positive, is what matters).

Why this is coherent, not a bug: the win marginal is the ONE pool where seeding the front of the order
helps (a faster opener wins the close-finish tie a touch more often). The QIN/FCT/TRI combinations are
dominated by the SAME minor-placing reshuffles that hurt RQ-a1's calibration — `a` redistributes mass
in the body of the order (exactly the top-2/top-3 region these pools price), and that redistribution is
NET WRONG against the realized orderings. The richer validation surface does NOT reward the
path-identified `a`; it PENALISES it more sharply than win did. An honest, important negative.

## (4) WHAT THIS IMPLIES FOR THE PROGRAM
- **`a` via the start-phase drift coupling is REFUTED on the whole pool book, not just win.** It is now
  DON'T-PROMOTE on WIN (0042) AND DON'T-PROMOTE on PLA/QIN/FCT/TCE/TRI (here). The drift channel is
  exhausted for `a`. (The D-variance channel also failed on win in 0042; not re-run per-pool — the
  drift-channel pool-book sign is decisive enough.) `a` remains validated for PROFILE FIT / the Stage-2
  MARGIN surface (#40/#41) — but it does not earn a place in the PRICE of any pool we can de-vig.
- **The wager itself is the lesson.** The hypothesis "the order book rewards what win discards" is
  FALSIFIED for THIS latent through THIS coupling: the ordering pools are NOT a softer judge that
  rewards path-honest order structure — they are a HARDER one (they penalise the same minor-placing
  error win only diluted). This tempers the program's optimism about the pool book as a rescue surface
  for win-invisible latents. It does NOT refute the pool-book gate as INFRASTRUCTURE — that is built,
  leak-free, and reusable, and IS the right Stage-2 judge.
- **Proceed, but recalibrated.** The gate is the deliverable; the next latents (ρ recovery → PLACE/QIN
  margins per §4 RQ-pool-place; a control/ground-saved latent → HV pool book per §3) should be tested
  on it — but with the prior that a latent must improve the SHARP anchors (WIN/PLA) to be a net win;
  "it'll convert on the exotics" is now an evidenced-against escape hatch. Recommend: do NOT spend more
  on `a`→win/pool; move to a latent that moves MARGINS (W′/ρ), which the place/quinella body actually
  prices, and gate it here. ρ stays population-fixed (Stage-1 scope); per-horse ρ a Stage-3 CV target.

## (5) LEAK-FREE + DE-VIG CAVEATS
- **Leak-free.** The order distribution is the leak-free model (θ strictly-prior: production `v_c_eb`/
  `w_prime_eb` from the train-only EB fit per fold; trailing `a` shifted-by-one). Dividends are realized
  post-race outcomes used ONLY for scoring, never fitting — the boundary is explicit (the pricer never
  sees a dividend; the gate joins them only to SCORE). `assert_no_lookahead` on the predict path
  (a_start_gain=0.2): drift **0.00e+00**. Paired fixed-seed full-scale (PYTHONHASHSEED=0) — not the
  ~4×-overstating quick_eval.
- **De-vig is HONEST but PARTIAL.** The dividend gives ONLY the WINNING combination's market prob
  ((1−takeout)/decimal); the market's full distribution over all combinations is unobserved. So %market
  here is on the REALIZED combination — a valid but partial benchmark (it tests whether the model puts
  the right mass on what actually happened vs what the market priced it at). Stated plainly so it is not
  over-read as a full distributional %market.
- **Exotic softness honoured.** WIN/PLA/QIN are the sharp anchors (17.5% takeout, liquid); FCT/TCE
  (20.5%) and TRI (25%) are SOFTER, higher-vig, noisier de-vig — flagged `(soft)` and treated as
  confirmatory, not decisive. The verdict rests on the sharp QIN + the medium FCT, which agree with the
  soft pools and the sprint band: every ordering/set pool says the same thing — `a` hurts.
- Tests green (`tests/test_statespace_skeleton.py` 9/9, incl. the new pricer + pool_book_score guards;
  full suite **157 passed**); `ruff` clean. Production `hkjc.model` untouched; nothing baked.
