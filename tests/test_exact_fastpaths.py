"""Fast paths that must give exactly the reference results: seeking decode, render-gate and finish-line fast paths."""

import numpy as np
import pytest


def _h264(path, n=90, gop=20):
    av = pytest.importorskip("av")
    try:
        av.codec.Codec("libx264", "w")
    except Exception:
        pytest.skip("no libx264 encoder")
    rng = np.random.default_rng(0)
    base = rng.integers(0, 255, (72, 128, 3), np.uint8)
    with av.open(str(path), "w") as c:
        st = c.add_stream("libx264", rate=25)
        st.width, st.height, st.pix_fmt = 128, 72, "yuv420p"
        st.codec_context.gop_size = gop
        st.codec_context.options = {"bf": "2", "keyint_min": str(gop)}
        for k in range(n):
            img = np.roll(base, 3 * k, axis=1)
            for pk in st.encode(av.VideoFrame.from_ndarray(img, format="bgr24")):
                c.mux(pk)
        for pk in st.encode():
            c.mux(pk)


def test_decode_from_matches_the_decode_from_frame_0(tmp_path):
    import av

    from hkjc.recon import video

    p = tmp_path / "v.mp4"
    _h264(p)
    with av.open(str(p)) as c:
        ref = [fr.to_ndarray(format="bgr24") for fr in c.decode(video=0)]
    for start in (0, 1, 19, 20, 21, 47, 60, 89):
        with av.open(str(p)) as c:
            assert start == 0 or video._seek_point(c, c.streams.video[0], start) is not None
        with av.open(str(p)) as c:
            got = list(video.decode_from(c, start))
        assert [i for i, _ in got] == list(range(start, len(ref)))
        assert all(np.array_equal(fr.to_ndarray(format="bgr24"), ref[i]) for i, fr in got)


def test_texture_gate_equals_the_float64_laplacian_variance():
    import cv2

    from hkjc.recon import video

    rng = np.random.default_rng(1)
    for sd in (2.0, 5.0, 6.0, 8.0, 15.0):
        g = np.clip(128 + rng.normal(0, sd, (90, 160)), 0, 255).astype(np.uint8)
        v = float(cv2.Laplacian(g, cv2.CV_64F).var())
        for thr in (240.0, v, np.nextafter(v, 0), np.nextafter(v, np.inf)):  # thresholds at the value itself
            assert video._texture_var_at_least(g, thr) == (v >= thr)


def test_green_fraction_and_legend_top_match_the_numpy_forms():
    import cv2

    from hkjc.recon import video

    rng = np.random.default_rng(2)
    for _ in range(20):
        p = rng.integers(0, 255, (120, 200, 3), np.uint8)
        p[90:100, rng.integers(0, 150) :][:, :40] = (0, 0, 255)  # a vivid legend box low in the panel
        hsv = cv2.cvtColor(p, cv2.COLOR_BGR2HSV)
        h, s = hsv[..., 0], hsv[..., 1]
        assert video._green_frac(hsv) == float(((h > 35) & (h < 90) & (s > 60)).mean())
        vivid = (hsv[..., 1] > 110) & (hsv[..., 2] > 110) & ((hsv[..., 0] < 25) | (hsv[..., 0] > 80))
        band = vivid.sum(1).astype(float)
        band[: int(0.65 * p.shape[0])] = 0
        want = int(0.80 * p.shape[0]) if band.max() < 0.05 * p.shape[1] else int(np.where(band > 0.3 * band.max())[0].min())
        assert video.legend_top(p) == want


def test_finish_line_x_matches_the_numpy_form():
    import cv2

    from hkjc.recon import finishcam

    rng = np.random.default_rng(3)
    for k in range(10):
        img = rng.integers(0, 255, (1080, 400, 3), np.uint8)
        img[:, 100 + 20 * k] = 250  # a white column
        hsv = cv2.cvtColor(img[300:1000], cv2.COLOR_BGR2HSV)
        white = ((hsv[..., 1] < 70) & (hsv[..., 2] > 160)).mean(0)
        x = int(np.argmax(white))
        assert finishcam.line_x(img) == (x if white[x] > 0.6 else None)
