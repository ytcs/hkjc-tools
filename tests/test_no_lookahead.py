"""The no-lookahead enforcement: AsOfView gates the past; the leak check catches peeking."""
import polars as pl
import pytest

from hkjc.backtesting import AsOfView, assert_no_lookahead


def _frame():
    return pl.DataFrame({
        "date": ["2025-01-01", "2025-01-08", "2025-01-15"],
        "race_id": ["r1", "r2", "r3"], "runner": [1, 1, 1],
        "feat": [0.1, 0.2, 0.3], "won": [1, 0, 1],
    })


def test_asof_view_hides_future():
    v = AsOfView(_frame(), cutoff="2025-01-08")     # strictly before the 8th
    vis = v.visible()
    assert vis.height == 1 and vis["date"][0] == "2025-01-01"
    assert len(v.at("2025-01-16")) == 3


def test_leak_check_passes_clean_pipeline():
    # Clean: prediction is a function of a pre-race feature only.
    def clean(train, test):
        return test.with_columns((pl.col("feat") * 0 + 0.5).alias("p"))
    drift = assert_no_lookahead(clean, _frame().head(2), _frame().tail(1))
    assert drift == 0.0


def test_leak_check_catches_outcome_peeking():
    # Leaky: prediction reads the realised outcome ('won') — must be detected.
    def leaky(train, test):
        return test.with_columns(pl.col("won").cast(pl.Float64).alias("p"))
    with pytest.raises(AssertionError, match="look-ahead detected"):
        # need >1 test row so shuffling 'won' actually changes it
        assert_no_lookahead(leaky, _frame().head(1), _frame().tail(2))
