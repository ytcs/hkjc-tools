"""identity.legend_votes on a field smaller than the 8-slot legend."""

import numpy as np

from hkjc.recon.identity import legend_votes


def test_seven_runner_field_votes_without_indexing_past_the_field():
    runners = [1, 2, 3, 4, 5, 6, 7]
    members = [[k] for k in range(7)]  # one detection per tracklet, all in frame 10
    frame_of_det = [10] * 7
    d = np.arange(7.0)  # tracklet k is k-th from the front
    legend = [[10, 3, 1, 4, 2, 7, 5, 6, 0]]  # 7 numbers shown, slot 8 empty
    v = legend_votes(members, frame_of_det, d, legend, runners)
    assert [int(np.argmax(v[t])) for t in range(7)] == [3, 1, 4, 2, 7, 5, 6]


def test_field_argument_lets_frames_missing_one_runner_vote():
    # 10 runners, #7 never rendered: frames track 9; the strict whole-field test casts nothing
    runners = list(range(1, 11))
    members = [[k] for k in range(9)]
    frame_of_det = [10] * 9
    d = np.arange(9.0)
    legend = [[10, 2, 6, 3, 5, 4, 9, 10, 1]]
    assert not legend_votes(members, frame_of_det, d, legend, runners).any()
    v, rank, elim, n_elim = legend_votes(members, frame_of_det, d, legend, runners, parts=True, field=9)
    assert [int(np.argmax(rank[t])) for t in range(8)] == [2, 6, 3, 5, 4, 9, 10, 1]
    assert n_elim[8] == 1 and elim[8, 7] == 1 and elim[8, 8] == 1  # the 9th: one of the unlisted #7, #8


def test_legend_slots_spread_for_small_fields():
    from hkjc.recon.ocr import _legend_slot_x

    W = 945
    assert _legend_slot_x(W, 8) == [int(0.064 * W + k * W / 8.8) for k in range(8)]
    assert _legend_slot_x(W, 14) == _legend_slot_x(W, 8)
    for n, pitch in ((7, 120), (6, 133)):  # measured digit spacing of 7- and 6-runner legends
        xs = _legend_slot_x(W, n)
        assert len(xs) == n and abs(np.diff(xs).mean() - pitch) <= 1 and abs(np.mean(xs) - 438) <= 1
