---
title: "S4 MULTI-POOL GATE: score the mid-rebuild model's skill% across the EFFICIENCY-WEIGHTED POOL BOOK (win/place/quinella/quinella-place/forecast/tierce/trio), not just win, as functionals of the ONE joint OrderDistribution. The pricer exotics are completed (QPL added; place() now sources k from hkjc_facts; dead-heat/refund routed to the settlement seam, not the order-prob functionals). NEW studies/pool_book_gate.py scores every pool's exact %-of-market on the realized winning combination vs HONEST baselines — the de-vigged DIVIDEND (market = (1−takeout)/decimal, takeout sourced from hkjc_facts) and the UNIFORM no-skill order law (same pricer) — leak-free (assert_no_lookahead drift = 0), paired fixed-seed (PYTHONHASHSEED=0), both venues. MODEL = path-energetics θ (v_c, W') pooled STRICTLY-PRIOR by the train-only EB population (calibration.horse_energetics_trailing), NO meta-signal, via build_fit_predict + the DEFAULT θ-builder (does NOT depend on the in-flight ability-READ fusion, 0052). VERDICT (HONEST, mid-rebuild): the model captures a SMALL but UNIFORMLY POSITIVE slice of the market across the whole book — ST WIN 10.9% / PLA 12.4% / QIN 12.2% / QPL 13.0% / FCT 12.3% / TCE 9.9% / TRI 13.0% of market; HV WIN 9.9% (dividends ST-only, so HV exotics are not scorable from disk). The joint/ordering pools (QIN/QPL/FCT/TRI ≈12–13%) track the sharp win anchor (≈11%) rather than collapsing relative to it — the joint structure of the order distribution is no WORSE per-pool than its win marginal, but it does not yet ADD skill beyond win either. This is a MEASUREMENT of the rebuild's GoF floor (consistent with 0050's win fall), not a fitting target; the path-energetics generator is far below the deleted model's 65%/60% as expected."
id: 0053-s4-multi-pool-gate-model-skill-across-pool-book-honest-10-13-percent-of-market-joint-structure-tracks-win
author: modelling
audience: ["modelling"]
channel: research
kind: result
status: open
thread: 0050-causal-payoff-test-pure-mechanism-hill-physiology-only-form-does-not-reemerge-gradient-not-recovered-gof-falls-to-floor
tags: ["s4", "pool-book", "multi-pool", "gate", "pricer", "order-distribution", "win", "place", "quinella", "quinella-place", "qpl", "forecast", "tierce", "trio", "exotics", "percent-of-market", "de-vig", "dividends", "market-baseline", "uniform-baseline", "efficiency-weighted", "joint-structure", "leak-free", "strictly-prior", "paired-fixed-seed", "both-venues", "gof-fall", "last-yardstick", "first-principles", "no-regression", "hkjc-facts", "places-paid", "takeout", "dead-heat", "0043", "0050", "0052"]
refs: {"study": "studies/pool_book_gate.py", "pricer": "src/hkjc/model/pricer.py (PoolPricer.prob_quinella_place, place k-rule, dead-heat note)", "gate": "src/hkjc/model/gate.py (pool_book_score, _pct_of_market)", "pipeline": "src/hkjc/model/pipeline.py (build_fit_predict, _default_theta_builder, predict_race)", "generative": "src/hkjc/model/generative.py (OrderDistribution)", "calibration": "src/hkjc/model/calibration.py (horse_energetics_trailing, fit_energetics_population)", "facts": "src/hkjc/core/hkjc_facts.py (places_paid, takeout_for)", "test": "tests/test_model_skeleton.py (test_pricer_qpl_and_place_field_size_rule)", "dividends": "data_cache/dividends_st.parquet", "wpq": "data_cache/wpq_dataset_full.parquet", "energetics": "data_cache/energetics_dataset_full.parquet", "thread_journal": "journal/entries/0050-causal-payoff-test-pure-mechanism-hill-physiology-only-form-does-not-reemerge-gradient-not-recovered-gof-falls-to-floor.md", "design": "docs/RESEARCH_PROGRAM.md (§4 RQ-pool-*)"}
created: 2026-06-06T00:00:00+00:00
updated: 2026-06-06T00:00:00+00:00
---

S4 — the multi-pool gate (RESEARCH_PROGRAM §4, RQ-pool-*). The win gate scores only the order
distribution's left tail; this scores the WHOLE joint object. Every HKJC pool is a functional of the
ONE `OrderDistribution` (`hkjc.model.pricer`), so a single Monte-Carlo sample prices the whole book
consistently. This is the LAST yardstick (PHILOSOPHY §6) — a MEASUREMENT of the mid-rebuild model,
NEVER a fitting target.

## What was built

1. **Pricer exotics completed as functionals of the ONE joint order distribution**
   (`src/hkjc/model/pricer.py`, owned). Added `prob_quinella_place(i,j,k)` (QPL — both runners in the
   top-k SET, the pairwise top-k analogue of the existing pairwise top-2 `prob_quinella`). `place()`
   now SOURCES `k` from `hkjc.core.hkjc_facts.places_paid` (3 for 7+, 2 for 4–6, 0 for tiny fields)
   instead of a hard-coded default. Dead-heat / refund are documented as a SETTLEMENT-seam concern
   (they split the payout, not the order-probability the gate scores) and routed there, not into the
   order-prob functionals — sourced from `hkjc_facts`, not hard-coded. The dense `(n,n,n)` trifecta
   tensor stays a deliberate `NotImplementedError` (the gate needs only `prob_trio`/`prob_tierce` on
   the realized combination, never the full tensor). Win/place/forecast/quinella/tierce/trio were
   already correct Monte-Carlo functionals; left intact.

2. **`studies/pool_book_gate.py`** (NEW, owned). Per-pool %-of-market on the realized winning
   combination vs two HONEST baselines, leak-free, paired fixed-seed, both venues.

## The model and the baselines (honest, leak-free)

- **MODEL.** Per-runner drift θ = (v_c, W') is the horse's PATH-INFERRED energetics
  (`data_cache/energetics_dataset_full.parquet`), pooled STRICTLY-PRIOR by the train-only EB
  population (`calibration.fit_energetics_population` on TRAIN dates only +
  `calibration.horse_energetics_trailing`, an expanding window over each horse's EARLIER runs). NO
  meta-signal column is read. The order distribution is the existing generative Wald first-passage law
  via `build_fit_predict` + the DEFAULT θ-builder + `predict_race` — so this does NOT depend on the
  in-flight ability-READ fusion (0052). A horse with no path history falls back to the band population
  mean. `assert_no_lookahead` drift = **0.00e+00** on both venues (strictly-prior, outcome-blind).
- **MARKET (honest baseline).** The de-vigged DIVIDEND: market prob of the realized combination =
  `(1 − takeout) / decimal`, takeout SOURCED from `hkjc_facts.takeout_for` (WIN/PLA/QIN/QPL 17.5%,
  FCT/TCE 20.5%, TRI 25%). PARTIAL by construction: the dividend prices only the realized (winning)
  combination, so %market is on that one combination (the 0043 caveat).
- **UNIFORM (honest baseline).** The no-skill order law — every runner equally fast
  (independent-uniform argmin) — priced through the SAME `PoolPricer`, so the baseline carries the
  honest no-information joint structure, not a hand-set number.
- **Metric.** The EXACT win %market definition (`gate.pool_book_score` / `_pct_of_market`),
  per pool, on the realized-combination NLL:
  `%market = (uniform_LL − model_LL) / (uniform_LL − market_LL)`.

## RESULT (PYTHONHASHSEED=0, n_samples=8000, paired fixed-seed, bit-reproducible run-to-run)

Per-pool skill% on the dividend test window (ST 170 dividend races, Mar–May 2026; population fit on
all ST dates before the window):

```
  pool      n   %market    model_LL market_LL  unif_LL   p_model    p_mkt
  WIN     170    10.9%      2.4802    2.0493   2.5329    0.0865   0.1818
  PLA     170    12.4%      1.3986    1.1212   1.4379    0.2517   0.3780   (per placed-runner leg)
  QIN     170    12.2%      4.1663    3.3859   4.2748    0.0170   0.0573
  QPL     168    13.0%      3.1121    2.4933   3.2050    0.0475   0.1188
  FCT     170    12.3%      4.8576    4.0806   4.9661    0.0086   0.0313
  TCE     170     9.9%      7.0691    6.0621   7.1797    0.0011   0.0066
  TRI     170    13.0%      5.3651    4.4469   5.5027    0.0059   0.0249

  HV  WIN  206    9.9%      2.4244    2.0725   2.4631    0.0927   0.1619   (WPQ holdout; HV dividends not on disk)
```

## Reading it (honest)

- The model captures a SMALL but UNIFORMLY POSITIVE slice of the market across the entire book —
  ~10–13% of market on every pool, ST and HV. Positive = it beats uniform everywhere; far below 100%
  = far below the market, and far below the deleted model's recorded 65%/60% win benchmark, exactly as
  the 0050 GoF fall predicts for a meta-signal-free path generator.
- **The joint/ordering pools do NOT collapse relative to the sharp win anchor.** QIN/QPL/FCT/TRI sit
  at ~12–13%, slightly ABOVE the win anchor's ~11% — i.e. the joint structure of the order
  distribution is no worse per-pool than its win marginal. But this is NOT yet evidence the joint
  carries skill win cannot: the clean RQ-pool-quin test (joint vs win-marginals×independence) is not
  run here, and the uniform baseline's joint is also independent-argmin, so the ~12% is consistent with
  the model's per-pool edge being driven by the same v_c spread that drives win. TCE (ordered top-3) is
  the weakest at 9.9% — the deepest ordered structure is where the coarse energetics θ has least to
  say, as expected.
- This is a GoF-floor MEASUREMENT of the rebuild, recorded per PHILOSOPHY §6 — not optimized, not a
  gate the build must pass.

## Caveats / NOTES for the orchestrator

- **Dividends are ST-only on disk** (`data_cache/dividends_st.parquet`); HV exotics are not scorable
  until an `dividends_hv.parquet` lands. The study auto-detects `dividends_{venue}.parquet` and slots
  HV in when present; meanwhile HV reports the WIN anchor from the WPQ panel (real market_win_prob).
- **PARTIAL benchmark (0043).** %market is on the realized winning combination only (the dividend does
  not expose the market's full distribution over losing combinations). Valid for a per-pool skill read,
  not a full cross-entropy against the market book.
- **The order distribution had everything the exotics need** — the joint MC sample is sufficient; no
  missing field in `OrderDistribution`. No request to edit `generative.py`/`field.py`/etc.
- **Efficiency weighting** (§4: a single weighted pool-book number, sharp anchors full weight, exotics
  down-weighted) is reported here as the per-pool table rather than collapsed to one scalar, so the
  decisive mass (win/place) is legible separately from the soft exotics; collapsing to one weighted
  number is a trivial post-step if a scalar gate is wanted.

## Discipline

Leak-free (drift = 0, both venues), strictly-prior θ, paired fixed-seed (bit-identical run-to-run),
NO meta-signal in the model input, NO win/pool optimization (pure MEASUREMENT). `PYTHONHASHSEED=0
uv run pytest -q` green (172 passed, incl. the new `test_pricer_qpl_and_place_field_size_rule`);
`uv run ruff check` clean.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
