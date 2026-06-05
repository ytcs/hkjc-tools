"""Tests for the walk-forward evaluation harness."""
import numpy as np
import polars as pl

from hkjc.backtesting import harness
from hkjc.backtesting import walkforward as wf


def _make_data() -> pl.DataFrame:
    """8 meetings, one 4-runner race each, a rotating winner."""
    rows = []
    for di, date in enumerate([f"2026-03-{d:02d}" for d in range(1, 9)]):
        winner = di % 4
        for r in range(4):
            rows.append(dict(date=date, race_id=di + 1, runner=r, won=int(r == winner)))
    return pl.DataFrame(rows)


def _uniform_fit_predict(train: pl.DataFrame, test: pl.DataFrame) -> pl.DataFrame:
    # model ignores train and assigns uniform 1/(field size) per race
    return test.with_columns((1.0 / pl.len().over("race_id")).alias("p"))


def test_walk_forward_evaluate_and_aggregate():
    data = _make_data()
    dates = data["date"].unique().to_list()
    folds = wf.walk_forward_folds(dates, n_folds=2, embargo=1, min_train=2)

    res = harness.walk_forward_evaluate(data, folds, _uniform_fit_predict, race_col="race_id")
    assert res.height == len(folds)
    assert {"fold", "n_train", "n_test", "log_loss", "brier", "race_win_log_loss"} <= set(res.columns)

    # uniform 0.25 on the winner -> race_win_log_loss == -log(0.25)
    assert np.allclose(res["race_win_log_loss"].to_numpy(), -np.log(0.25))

    agg = harness.aggregate(res)
    assert agg["folds"] == len(folds)
    assert abs(agg["race_win_log_loss"] - (-np.log(0.25))) < 1e-9


def test_fit_predict_sees_only_test_rows():
    data = _make_data()
    folds = wf.walk_forward_folds(data["date"].unique().to_list(), n_folds=2, embargo=1, min_train=2)
    captured = {}

    def fp(train, test):
        captured["train_dates"] = set(train["date"].to_list())
        captured["test_dates"] = set(test["date"].to_list())
        return test.with_columns(pl.lit(0.25).alias("p"))

    harness.walk_forward_evaluate(data, [folds[0]], fp, race_col="race_id")
    # the model's train and test never overlap (no leakage)
    assert not (captured["train_dates"] & captured["test_dates"])


def test_empty_metrics_when_no_folds():
    assert harness.aggregate(pl.DataFrame()) == {"folds": 0}
