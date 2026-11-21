"""Circle render: tracklets split where the disc number read changes and stays changed."""

import numpy as np

from hkjc.recon.tracking import split_by_number


def _run(seq):
    t = np.arange(len(seq), dtype=float)
    return split_by_number([np.arange(len(seq))], t, np.array(seq))


def test_swap_mid_tracklet_splits_at_the_first_new_read():
    seq = [2] * 30 + [0] * 3 + [9] * 30  # unread frames between are fine
    parts = _run(seq)
    assert len(parts) == 2 and parts[1][0] == 33


def test_dropped_digits_and_stray_misreads_do_not_split():
    seq = [11] * 20 + [1, 11, 1, 1, 11] * 6 + [11] * 10  # '1' = '11' with a digit dropped
    assert len(_run(seq)) == 1
    seq = [5] * 20 + [7, 7, 7] + [5] * 20  # 3 stray reads: below the window share
    assert len(_run(seq)) == 1


def test_two_switches():
    parts = _run([12] * 25 + [11] * 25 + [3] * 25)
    assert [len(p) for p in parts] == [25, 25, 25]


def test_short_burst_and_tail_burst_do_not_split():
    assert len(_run([2] * 40 + [3] * 9 + [2] * 30)) == 1  # 9-read burst mid-tracklet
    assert len(_run([2] * 40 + [3] * 12)) == 1  # burst at the tracklet's end (discs overlap at the line)


def test_a_longer_number_on_a_shorter_tracklet_is_a_switch():
    assert len(_run([2] * 30 + [12] * 30)) == 2  # digits are dropped, never added
