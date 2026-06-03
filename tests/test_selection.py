"""Tests for multiple-testing control."""
import numpy as np

from hkjc.eval import selection as sel


def test_bh_rejects_all_when_uniformly_small():
    p = [0.01, 0.02, 0.03, 0.04, 0.05]
    reject, adj = sel.benjamini_hochberg(p, alpha=0.05)
    assert reject.all()
    assert adj.shape == (5,)
    assert (adj <= 1.0).all()


def test_bh_rejects_only_strong():
    p = [0.001, 0.5, 0.5]
    reject, _ = sel.benjamini_hochberg(p, alpha=0.05)
    assert reject.tolist() == [True, False, False]


def test_bh_preserves_input_order():
    p = [0.5, 0.001, 0.5]
    reject, _ = sel.benjamini_hochberg(p, alpha=0.05)
    assert reject.tolist() == [False, True, False]


def test_psr_half_at_benchmark():
    assert abs(sel.probabilistic_sharpe_ratio(0.5, 0.5, n=100) - 0.5) < 1e-9


def test_psr_increases_with_n():
    a = sel.probabilistic_sharpe_ratio(0.3, 0.0, n=50)
    b = sel.probabilistic_sharpe_ratio(0.3, 0.0, n=500)
    assert b > a > 0.5


def test_expected_max_sharpe_grows_with_trials():
    assert sel.expected_max_sharpe(100) > sel.expected_max_sharpe(10) > 0
    assert sel.expected_max_sharpe(1) == 0.0   # guarded


def test_deflated_sharpe_haircut():
    # Same observed SR is less convincing after more trials.
    dsr_few = sel.deflated_sharpe_ratio(1.0, n=250, n_trials=5, sr_trials_std=0.5)
    dsr_many = sel.deflated_sharpe_ratio(1.0, n=250, n_trials=500, sr_trials_std=0.5)
    assert dsr_few > dsr_many


def test_min_track_record_length():
    assert sel.min_track_record_length(0.0, 0.0) == float("inf")
    assert sel.min_track_record_length(0.5, 0.0) > 1.0
