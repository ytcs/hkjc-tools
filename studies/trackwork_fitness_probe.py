"""Banister fitness-fatigue from REAL daily training gallops — conversion test.

SELF-CONTAINED scratch study (per the brief: do NOT edit fit.py/constants/backtest_st).
It (a) defines the leak-free Banister fitness feature inline (the exact function to add to
hkjc.model.features), (b) reuses backtest_st's loader to build the base modelling frame,
(c) as-of-joins the fitness trace, and (d) runs the PAIRED fixed-seed walk-forward
conversion test (base vs base+fitness), full slice + the layoff / recent-gallop subslices
where the signal should matter most.

THE FEATURE (Banister 1975 fitness-fatigue, driven by real dated gallops):
  Each dated gallop is a training IMPULSE with dose w_i (intensity × worked distance, the
  gallop's training load). At race day T the model integrates two exponential traces over
  the horse's STRICTLY-PRIOR gallops:
      Fitness(T) = Σ_i w_i · exp(-(T - t_i)/TAU_FIT)     slow  (TAU_FIT ~ 42 d)
      Fatigue(T) = Σ_i w_i · exp(-(T - t_i)/TAU_FAT)     fast  (TAU_FAT ~ 7  d)
      banister   = K_FIT·Fitness(T) - K_FAT·Fatigue(T)   (readiness = fitness minus fatigue)
  Strictly prior: only sessions with session_date < race_date enter, so it is leak-free
  exactly like form/collateral — and it is signal race history cannot give (race-derived
  condition is collinear with ability; this is the daily training load itself).

Run: PYTHONHASHSEED=0 uv run --group bayes python studies/trackwork_fitness_probe.py
     PYTHONHASHSEED=0 VENUE=HV uv run --group bayes python studies/trackwork_fitness_probe.py
"""
from __future__ import annotations

import os
from pathlib import Path

import polars as pl

from hkjc.backtesting import harness, walk_forward_folds
from hkjc.model.fit import fit_predict_traffic

import sys
sys.path.insert(0, str(Path(__file__).parent))
from backtest_st import load  # reuse the exact base modelling frame  # noqa: E402

CACHE = Path("data_cache")
VENUE = os.environ.get("VENUE", "ST")
TW = CACHE / "trackwork.parquet"

# Banister constants (days). Slow fitness / fast fatigue — the classic ~6:1 separation.
TAU_FIT = float(os.environ.get("TAU_FIT", "42"))
TAU_FAT = float(os.environ.get("TAU_FAT", "7"))
K_FIT = float(os.environ.get("K_FIT", "1.0"))
K_FAT = float(os.environ.get("K_FAT", "2.0"))
HORIZON = int(os.environ.get("HORIZON", "90"))   # ignore impulses older than this (negligible weight)
FIT_GAIN = float(os.environ.get("FIT_GAIN", "0.0"))  # set by the sweep below


def banister_fitness_table(trackwork: pl.DataFrame, race_dates: pl.DataFrame, *,
                           tau_fit: float = TAU_FIT, tau_fat: float = TAU_FAT,
                           k_fit: float = K_FIT, k_fat: float = K_FAT,
                           horizon: int = HORIZON) -> pl.DataFrame:
    """Leak-free Banister fitness-fatigue trace per (horse, race_date) from real gallops.

    ``trackwork`` is the data_cache/trackwork.parquet panel (per horse × session_date, with
    gallop_velocity / worked_distance_m). ``race_dates`` is the unique (horse_id, race_date)
    pairs we need the trace for (horse_id is the SHORT brand, matching trackwork.horse_brand).
    For each race, sums impulse contributions over the horse's gallops STRICTLY before the race
    day. Returns (horse_id, d, banister, fit_trace, fat_trace, days_since_gallop, n_gallops_28d).

    Impulse dose w_i = (gallop_velocity z-scored across the panel, floored at 0) × worked
    furlongs — a fast, far gallop is a big load; a slow short pipe-opener a small one. The
    z-score is a panel-level condition constant (not the horse's own outcome), so it is leak-free.
    """
    tw = trackwork.filter(pl.col("session_date").is_not_null()
                          & pl.col("gallop_velocity").is_not_null()).with_columns(
        pl.col("session_date").str.strptime(pl.Date, "%Y-%m-%d", strict=False).alias("t"),
        pl.col("worked_distance_m").fill_null(201.168).alias("dist"))
    vmean = float(tw["gallop_velocity"].mean())
    vstd = float(tw["gallop_velocity"].std()) or 1.0
    tw = tw.with_columns(
        (((pl.col("gallop_velocity") - vmean) / vstd).clip(lower_bound=0.0)
         * (pl.col("dist") / 201.168)).alias("w"))
    # build per-horse session lists once
    sess: dict[str, list[tuple]] = {}
    for hid, t, w in zip(tw["horse_brand"].to_list(), tw["t"].to_list(), tw["w"].to_list()):
        if hid is None or t is None:
            continue
        sess.setdefault(hid, []).append((t, w))
    for hid in sess:
        sess[hid].sort()

    out_h, out_d, out_b, out_f, out_g, out_dsg, out_n = [], [], [], [], [], [], []
    for hid, d in zip(race_dates["horse_id"].to_list(), race_dates["d"].to_list()):
        lst = sess.get(hid)
        out_h.append(hid)
        out_d.append(d)
        if not lst:
            out_b.append(0.0)
            out_f.append(0.0)
            out_g.append(0.0)
            out_dsg.append(None)
            out_n.append(0)
            continue
        fit = fat = 0.0
        dsg = None
        n28 = 0
        for t, w in lst:
            age = (d - t).days
            if age <= 0 or age > horizon:     # STRICTLY prior only (leak-free)
                continue
            if dsg is None or age < dsg:
                dsg = age
            if age <= 28:
                n28 += 1
            fit += w * 2.718281828 ** (-age / tau_fit)
            fat += w * 2.718281828 ** (-age / tau_fat)
        out_f.append(fit)
        out_g.append(fat)
        out_b.append(k_fit * fit - k_fat * fat)
        out_dsg.append(dsg)
        out_n.append(n28)
    return pl.DataFrame({
        "horse_id": out_h, "d": out_d, "banister": out_b,
        "fit_trace": out_f, "fat_trace": out_g,
        "days_since_gallop": out_dsg, "n_gallops_28d": out_n})


def attach_fitness(df: pl.DataFrame) -> pl.DataFrame:
    if not TW.exists():
        return df.with_columns(pl.lit(0.0).alias("banister"),
                               pl.lit(None, dtype=pl.Int64).alias("days_since_gallop"),
                               pl.lit(0, dtype=pl.Int64).alias("n_gallops_28d"))
    tw = pl.read_parquet(TW)
    rd = df.select("horse_id", "d").unique()
    fit = banister_fitness_table(tw, rd)
    # within-race centring happens in the model term; here just attach
    df = df.join(fit.select("horse_id", "d", "banister", "days_since_gallop", "n_gallops_28d"),
                 on=["horse_id", "d"], how="left")
    return df.with_columns(pl.col("banister").fill_null(0.0),
                           pl.col("n_gallops_28d").fill_null(0))


def _pct(summary):
    uni, mkt, mdl = (summary[k]["race_win_log_loss"] for k in ("uniform", "market", "traffic"))
    return (uni - mdl) / (uni - mkt) if uni > mkt else float("nan"), mdl, mkt


def _eval(df, folds, jky_mean, fit_gain, subset_mask=None):
    """Paired model: base term-stack + fit_gain·banister injected straight into v_c.

    Faithful injection: the production model already reads a ``speedfig`` velocity channel via
    the ``speedfig_gain`` arg (baked 0.0, unused in production). We OVERWRITE that column with
    the standardized banister readiness and set ``speedfig_gain = fit_gain`` — so the fitness
    term enters v_c exactly as a real baked velocity feature would (within-race centred by the
    model's own ``cfm``-style handling is not applied to speedfig, so we centre it here). Setting
    fit_gain=0 reproduces the untouched base (the speedfig column is multiplied by 0). This is the
    same paired structure as backtest_st; PYTHONHASHSEED=0 keeps base/candidate on shared MC seeds.

    If ``subset_mask`` is given, %market is computed only over races containing >=1 masked runner
    (the subslice read), but the model still fits/normalises on the full field.
    """
    bstd = float(df.filter(pl.col("banister") != 0.0)["banister"].std()) or 1.0
    d2 = df.with_columns(
        ((pl.col("banister") - pl.col("banister").mean().over("race_id")) / bstd).alias("speedfig"))

    def _model(train, test):
        return fit_predict_traffic(train, test, jky_mean=jky_mean, sim="wald",
                                   speedfig_gain=fit_gain)

    def _market(train, test):
        return test.with_columns((pl.col("market_win_prob")
                                 / pl.col("market_win_prob").sum().over("race_id")).alias("p"))

    def _uniform(train, test):
        return test.with_columns((1.0 / pl.col("field_size")).alias("p"))

    eval_df = d2
    if subset_mask is not None:
        keep_races = d2.filter(subset_mask)["race_id"].unique().implode()
        eval_df = d2.filter(pl.col("race_id").is_in(keep_races))

    summary = {}
    for name, fp in {"traffic": _model, "market": _market, "uniform": _uniform}.items():
        fm = harness.walk_forward_evaluate(eval_df, folds, fp, date_col="date",
                                           prob_col="p", outcome_col="won", race_col="race_id")
        summary[name] = harness.aggregate(fm)
    return _pct(summary)


def main():
    df, jky_mean = load(VENUE)
    df = attach_fitness(df)
    cov = df.filter(pl.col("banister") != 0.0).height
    print(f"{VENUE}: {df['race_id'].n_unique()} races, {df.height} runners; "
          f"banister!=0 on {cov} ({100*cov/df.height:.1f}%) runners")
    print(f"  days_since_gallop: median={df['days_since_gallop'].median()} "
          f"null(no gallop)={df.filter(pl.col('days_since_gallop').is_null()).height}")

    dates = df["date"].unique().to_list()
    folds = walk_forward_folds(dates, n_folds=5, embargo=1, min_train=1)

    gains = [float(x) for x in os.environ.get("GAINS", "1,2,4,8").split(",")]

    print("\n  PAIRED fixed-seed conversion (FULL slice):")
    base_pct, base_mdl, mkt = _eval(df, folds, jky_mean, 0.0)
    print(f"    base                 % of market = {base_pct:.2%}   (mdl {base_mdl:.4f}, mkt {mkt:.4f})")
    best = (0.0, base_pct)
    for g in gains:
        pct, mdl, _ = _eval(df, folds, jky_mean, g)
        print(f"    +banister gain={g:<5g} % of market = {pct:.2%}   (Δ {pct-base_pct:+.2%})")
        if pct > best[1]:
            best = (g, pct)

    # SUBSLICE: races with a freshly-worked / layoff horse — where readiness should matter most.
    # Layoff proxy = banister dominated by the fast fatigue trace is absent but fitness present,
    # i.e. a horse that worked recently after a gap. We slice races containing >=2 gallops in 28d.
    mask = (pl.col("n_gallops_28d") >= 2) & (pl.col("banister") != 0.0)
    n_fresh = df.filter(mask).height
    print(f"\n  SUBSLICE — races with a fresh-work runner (>=2 gallops/28d): "
          f"{n_fresh} runners ({100*n_fresh/df.height:.1f}%)")
    sb_base, _, _ = _eval(df, folds, jky_mean, 0.0, subset_mask=mask)
    print(f"    base (subslice)      % of market = {sb_base:.2%}")
    for g in gains:
        pct, _, _ = _eval(df, folds, jky_mean, g, subset_mask=mask)
        print(f"    +banister gain={g:<5g} % of market = {pct:.2%}   (Δ {pct-sb_base:+.2%})")
    print(f"\n  BEST full-slice gain = {best[0]} ({best[1]:.2%}, Δ {best[1]-base_pct:+.2%})")


if __name__ == "__main__":
    main()
