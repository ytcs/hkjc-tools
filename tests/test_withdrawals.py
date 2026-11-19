import numpy as np
import polars as pl

from hkjc.data.historical import _clean_race_data
from hkjc.recon import identity


def _field(numbers, F=40, extra=0):
    """len(numbers) + extra tracklets, one detection each per frame, tracklet k always k-th; legend lists numbers[:8]."""
    n = len(numbers) + extra
    frame_of_det = np.tile(np.arange(F), n)
    members = [np.arange(k * F, (k + 1) * F) for k in range(n)]
    d = np.repeat(np.arange(n, dtype=float) * 3.0, F)
    legend = np.c_[np.arange(F), np.tile(np.array(numbers[:8]), (F, 1))]
    return members, frame_of_det, d, legend


def test_legend_votes_withdrawn_numbers_match_the_renumbered_field():
    """A field with withdrawn numbers (gaps) casts the same slot and elimination votes as the same field
    numbered 1..N: the whole-field test counts runners, not the highest saddlecloth number."""
    gapped = [1, 3, 5, 6, 7, 8, 9, 10, 11, 12]  # #2 and #4 withdrawn
    dense = list(range(1, 11))
    a = identity.legend_votes(*_field(gapped), gapped, parts=True)
    b = identity.legend_votes(*_field(dense), dense, parts=True)
    assert a[0].sum() > 0 and a[3].sum() > 0
    for x, y in zip(a[:3], b[:3]):
        assert np.array_equal(x[:, gapped], y[:, dense])
        assert x[:, [2, 4]].sum() == 0  # no evidence for a withdrawn number
    assert np.array_equal(a[3], b[3])


def test_legend_votes_non_runner_in_the_legend_casts_no_elimination():
    numbers = [1, 3, 5, 6, 7, 8, 9, 10, 11, 12]
    members, frame_of_det, d, legend = _field(numbers)
    legend[:, 8] = 2  # the legend shows a withdrawn number in slot 8
    votes, rank, elim, n_elim = identity.legend_votes(members, frame_of_det, d, legend, numbers, parts=True)
    assert elim.sum() == 0 and n_elim.sum() == 0
    assert rank[:, 2].sum() == 0 and rank[0, 1] > 0


def test_legend_votes_more_detections_than_runners_still_vote():
    """Pins current behaviour: a frame with an extra detection (a false positive) passes the whole-field test."""
    numbers = list(range(1, 11))
    rank = identity.legend_votes(*_field(numbers, extra=1), numbers, parts=True)[1]
    assert rank.sum() > 0


def test_clean_race_data_drops_withdrawn_rows_and_keeps_the_numbers():
    """Withdrawn runners (WV-A, HV 2026-04-15 R6) are dropped; the remaining saddlecloth numbers keep their gaps."""
    raw = pl.DataFrame(
        {
            "Pla": ["1", "2", "3", "WV-A", "WV-A"],
            "HorseNo": ["3", "9", "1", "2", "4"],
            "ActWt": ["120", "126", "135", "130", "128"],
            "DeclarHorseWt": ["1100", "1050", "1150", "1080", "1090"],
            "Dr": ["11", "9", "6", "---", "---"],
            "WinOdds": ["4.1", "6.2", "3.3", "---", "---"],
            "FinishTime": ["1:09.76", "1:09.92", "1:10.00", "---", "---"],
        }
    )
    out = _clean_race_data(raw)
    assert out["HorseNo"].to_list() == [3, 9, 1] and out["Dr"].to_list() == [11, 9, 6]
