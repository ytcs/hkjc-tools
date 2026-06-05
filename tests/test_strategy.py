"""The Strategy seam: positive-EV filter and fractional Kelly behave as specified."""
import numpy as np

from hkjc.betting import FractionalKelly, Strategy, ValueThreshold


def test_value_threshold_only_stakes_positive_edge():
    s = ValueThreshold(edge_threshold=0.05, stake_frac=0.01)
    assert isinstance(s, Strategy)
    model_prob = np.array([0.5, 0.2, 0.3])
    market_odds = np.array([2.5, 2.0, 3.0])     # edges: 0.25, -0.6, -0.1
    st = s.stakes(model_prob=model_prob, market_odds=market_odds, bankroll=1000.0)
    assert st[0] == 10.0                          # only runner 0 clears the threshold
    assert st[1] == 0.0 and st[2] == 0.0


def test_fractional_kelly_caps_and_skips_negative_edge():
    s = FractionalKelly(fraction=0.5, cap=0.05)
    model_prob = np.array([0.6, 0.1])
    market_odds = np.array([3.0, 2.0])            # runner0 +EV (b=2,p=.6), runner1 -EV
    st = s.stakes(model_prob=model_prob, market_odds=market_odds, bankroll=1000.0)
    # full kelly f* = (2*0.6 - 0.4)/2 = 0.4; half-kelly 0.2 -> capped to 0.05 -> 50.0
    assert abs(st[0] - 50.0) < 1e-9
    assert st[1] == 0.0
