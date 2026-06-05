"""Tests for the DataSource protocol, feature store and W/P/Q outcome schema."""
import polars as pl

from hkjc.core.contracts import CallableSource, DataSource, make_race_id
from hkjc.data.feature_store import (
    FeatureStore,
    add_win_place_outcomes,
    market_features_from_win,
    paying_places,
    results_to_features,
    results_to_outcomes,
    trailing_form_features,
)


# ------------------------------------------------------------------- outcomes
def test_paying_places_rule():
    assert paying_places(12) == 3
    assert paying_places(7) == 3
    assert paying_places(6) == 2
    assert paying_places(4) == 2
    assert paying_places(3) == 0


def test_add_win_place_outcomes_field_size_dependent():
    # one 8-runner race (3 paid) and one 5-runner race (2 paid)
    rows = [{"date": "d", "venue": "ST", "race_no": 1, "runner": i, "finish_pos": i}
            for i in range(1, 9)]
    rows += [{"date": "d", "venue": "ST", "race_no": 2, "runner": i, "finish_pos": i}
             for i in range(1, 6)]
    df = add_win_place_outcomes(pl.DataFrame(rows))

    r1 = df.filter(pl.col("race_no") == 1)
    assert r1["field_size"].unique().to_list() == [8]
    assert r1["paying_places"].unique().to_list() == [3]
    assert r1.filter(pl.col("finish_pos") == 1)["won"].item() == 1
    assert r1.filter(pl.col("finish_pos") == 3)["placed"].item() == 1
    assert r1.filter(pl.col("finish_pos") == 4)["placed"].item() == 0

    r2 = df.filter(pl.col("race_no") == 2)
    assert r2["paying_places"].unique().to_list() == [2]
    assert r2.filter(pl.col("finish_pos") == 3)["placed"].item() == 0   # only 2 paid


def test_results_to_outcomes_maps_hkjc_columns():
    results = pl.DataFrame({
        "Date": ["2026/05/31"] * 4,
        "Venue": ["ST"] * 4,
        "RaceNumber": [1] * 4,
        "HorseNo": [1, 2, 3, 4],
        "Pla": [2, 1, 4, 3],
        "WinOdds": [3.5, 2.1, 12.0, 6.0],
    })
    out = results_to_outcomes(results)
    assert out.filter(pl.col("runner") == 2)["won"].item() == 1     # Pla 1
    assert out.filter(pl.col("runner") == 1)["won"].item() == 0     # Pla 2
    assert out["paying_places"].unique().to_list() == [2]           # 4 runners -> 2 paid
    assert out.filter(pl.col("runner") == 1)["placed"].item() == 1  # Pla 2 is paid
    assert out.filter(pl.col("runner") == 4)["placed"].item() == 0  # Pla 3 not paid


def test_results_to_features_devig_and_columns():
    results = pl.DataFrame({
        "Date": ["2026/05/31"] * 3, "Venue": ["ST"] * 3, "RaceNumber": [1] * 3,
        "HorseNo": [1, 2, 3], "WinOdds": [2.0, 4.0, 4.0],
        "Dr": [5, 1, 8], "ActWt": [126, 120, 133], "DeclarHorseWt": [1100, 1050, 1080],
    })
    feat = results_to_features(results)
    assert abs(feat["market_win_prob"].sum() - 1.0) < 1e-9
    assert feat.filter(pl.col("runner") == 1)["market_win_prob"].item() == 0.5
    # log_market_prob is monotone in market prob (favourite has the largest)
    assert feat["log_market_prob"].arg_max() == 0
    assert {"draw", "act_wt", "decl_wt"} <= set(feat.columns)


def test_market_features_devig_sums_to_one():
    out = market_features_from_win([2.0, 4.0, 4.0], "2026-05-31", "ST", 1)
    probs = out["market_win_prob"].to_numpy()
    assert abs(probs.sum() - 1.0) < 1e-9
    assert out["runner"].to_list() == [1, 2, 3]


def test_trailing_form_features_no_leakage():
    import datetime as dt
    # one horse, 4 runs: finishes 4,1,2,3 on consecutive weeks
    panel = pl.DataFrame({
        "horse_id": ["H1"] * 4,
        "date": [dt.date(2026, 1, d) for d in (1, 8, 15, 22)],
        "finish_pos": [4, 1, 2, 3],
        "won": [0, 1, 0, 0],
    })
    out = trailing_form_features(panel).sort("date")
    # debut row: no prior info
    assert out["prior_runs"].to_list() == [0, 1, 2, 3]
    assert out["days_since_last"].to_list()[1:] == [7, 7, 7]
    assert out["days_since_last"][0] is None
    # win rate before run 3 (after a win in 2 starts) = 1/2; uses only earlier runs
    assert out["win_rate_prior"][0] is None
    assert out["win_rate_prior"][2] == 0.5
    # avg of last-3 finishes before run 4 = mean(4,1,2)
    assert abs(out["avg_finish_prior3"][3] - (4 + 1 + 2) / 3) < 1e-9


def test_make_race_id():
    assert make_race_id("2026/05/31", "ST", 3) == "20260531ST3"
    assert make_race_id("2026-05-31", "HV", 11) == "20260531HV11"


# --------------------------------------------------------------- feature store
def _src(name, languages, provides, rows, role="feature"):
    return CallableSource(name, list(languages), list(provides),
                          lambda d, v, _rows=rows: pl.DataFrame(_rows), role=role)


def test_assemble_joins_sources_on_runner_keys():
    def keys(r):
        return {"date": "d", "venue": "ST", "race_no": 1, "runner": r}
    outcomes = _src("results", ["en-us"], ["won"],
                    [{**keys(1), "won": 1}, {**keys(2), "won": 0}], role="outcome")
    market = _src("market", ["en-us"], ["market_win_prob"],
                  [{**keys(1), "market_win_prob": 0.6}, {**keys(2), "market_win_prob": 0.4}])
    cn = _src("tipsters_zh", ["zh-hk"], ["tip_score"],
              [{**keys(1), "tip_score": 0.9}])   # only covers runner 1

    fs = FeatureStore().register(outcomes).register(market).register(cn)
    frame = fs.assemble("d", "ST")
    assert frame.height == 2
    assert set(["won", "market_win_prob", "tip_score"]) <= set(frame.columns)
    # union universe keeps runner 2 even though the zh source skipped it
    assert frame.filter(pl.col("runner") == 2)["tip_score"].item() is None


def test_coverage_reports_per_source_and_language():
    def keys(r):
        return {"date": "d", "venue": "ST", "race_no": 1, "runner": r}
    market = _src("market", ["en-us"], ["market_win_prob"],
                  [{**keys(1), "market_win_prob": 0.6}, {**keys(2), "market_win_prob": 0.4}])
    cn = _src("tipsters_zh", ["zh-hk"], ["tip_score"],
              [{**keys(1), "tip_score": 0.9}])   # 1 of 2 rows
    fs = FeatureStore().register(market).register(cn)
    frame = fs.assemble("d", "ST")

    cov = fs.coverage_report(frame)
    assert cov.filter(pl.col("source") == "market")["coverage_pct"].item() == 1.0
    assert cov.filter(pl.col("source") == "tipsters_zh")["coverage_pct"].item() == 0.5

    lang = fs.language_coverage(frame)
    # the Chinese blind spot is visible as a number: zh-hk covers half, en-us all
    assert lang.filter(pl.col("language") == "zh-hk")["coverage_pct"].item() == 0.5
    assert lang.filter(pl.col("language") == "en-us")["coverage_pct"].item() == 1.0


def test_callable_source_satisfies_protocol():
    s = _src("x", ["en-us"], ["a"], [])
    assert isinstance(s, DataSource)
