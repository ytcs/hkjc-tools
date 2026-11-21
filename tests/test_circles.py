"""Numbered-disc (2026/27) render: disc centres."""

import numpy as np

from hkjc.recon import circles

BLUE = (210, 110, 20)  # BGR, HSV hue ~107: the render's disc blue


def _panel(centres, size=(300, 500)):
    import cv2

    p = np.full((*size, 3), (60, 140, 70), np.uint8)  # turf green
    for x, y in centres:
        cv2.circle(p, (x, y), circles.R, BLUE, -1)
        cv2.putText(p, "7", (x - 8, y + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return p


def test_centres_find_isolated_and_overlapping_discs():
    pts = [(80, 80), (200, 150), (225, 150), (400, 220)]  # two discs 25 px apart (~1.3 m)
    d = circles.centres(_panel(pts))
    assert len(d) == 4 and d.shape[1] == 6
    for x, y in pts:
        assert np.min(np.hypot(d[:, 0] - x, d[:, 1] - y)) <= 2


def test_discs_closer_than_the_split_limit_merge():
    assert len(circles.centres(_panel([(200, 150), (214, 150)]))) == 1


def test_no_discs_on_plain_turf():
    assert len(circles.centres(_panel([]))) == 0
