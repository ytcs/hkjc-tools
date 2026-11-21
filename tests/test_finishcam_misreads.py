"""Finish camera: misreads on a well-read cloth must not merge with another number's real reads."""

import numpy as np

from hkjc.recon import finishcam


def _reads():
    rows = []
    for u in range(0, 60):  # #7, well read, moving left across the line (x 1600 -> 1050), lane y 470
        rows.append((u, 1000 + u, 1600 - 9.0 * u, 470, 7, 0.9))
    for u in (40, 41, 42):  # '3' misread ON the #7 cloth
        rows.append((u, 1000 + u, 1600 - 9.0 * u, 468, 3, 0.7))
    for u in range(150, 160):  # the real #3, much later, lane y 495 (within the 40 px y tolerance of 468)
        rows.append((u, 1000 + u, 2400 - 9.0 * (u - 150) * 6, 495, 3, 0.99))
    return np.array(rows, float)


def test_overhead_camera_drops_misreads_on_a_well_read_cloth():
    sc = dict(line_x=960.0, reads=_reads(), overhead=True)
    u3 = finishcam.tracks(sc)[3][0]
    assert u3.min() >= 150  # only the real #3's reads build its track


def test_side_on_camera_is_unchanged():
    sc = dict(line_x=960.0, reads=_reads(), overhead=False)
    u3 = finishcam.tracks(sc)[3][0]
    assert u3.min() < 150
