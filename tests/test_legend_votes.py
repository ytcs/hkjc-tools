"""The running-order list on a field smaller than the 8-slot legend."""

import numpy as np

from hkjc.recon import core


def test_seven_runner_field_reads_seven_slots_without_indexing_past_the_field():
    """A 7-runner field fills 7 slots: a row reading all 7 is full, its facts are the 6 adjacent pairs, and there is
    no unlisted horse."""
    runners = [1, 2, 3, 4, 5, 6, 7]
    legend = np.array([[10, 3, 1, 4, 2, 7, 5, 6, 0]])  # 7 numbers shown, slot 8 empty
    L = core.list_rows(legend, lambda f: np.asarray(f, float) * 0.1, runners)
    assert L.full.tolist() == [True] and L.slots[0, 7] == -1
    St = core.statements(L, len(runners))
    num = np.array(runners)
    assert list(zip(num[St.a], num[St.b])) == [(3, 1), (1, 4), (4, 2), (2, 7), (7, 5), (5, 6)]
    assert not St.unl.any()


def test_legend_slots_spread_for_small_fields():
    from hkjc.recon.ocr import _legend_slot_x

    W = 945
    assert _legend_slot_x(W, 8) == [int(0.064 * W + k * W / 8.8) for k in range(8)]
    assert _legend_slot_x(W, 14) == _legend_slot_x(W, 8)
    for n, pitch in ((7, 120), (6, 133)):  # measured digit spacing of 7- and 6-runner legends
        xs = _legend_slot_x(W, n)
        assert len(xs) == n and abs(np.diff(xs).mean() - pitch) <= 1 and abs(np.mean(xs) - 438) <= 1
