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
