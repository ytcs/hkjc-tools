"""Render-panel box: robust to a crossfade first frame."""

import numpy as np


def test_crossfade_frame_takes_the_layout_box():
    from hkjc.recon.video import PanelBox, find_panel_box, frame_box

    img = np.zeros((1080, 1920, 3), np.uint8)
    img[271:807, 963:1908] = (60, 140, 70)  # 2-up panel inside a black letterbox
    layout = find_panel_box(img)
    assert (layout.y0, layout.y1) == (272, 806)
    fade = img.copy()
    fade[:271, 960:] = fade[807:, 960:] = 90  # crossfade: the live picture still fills the letterbox
    assert find_panel_box(fade).y0 < 10
    assert frame_box(fade, layout) == layout
    jitter = PanelBox(layout.x0, layout.x1 + 2, layout.y0 - 1, layout.y1)  # antialiased-edge jitter
    assert frame_box(img, jitter) == layout  # within the jitter the frame's own box is kept
