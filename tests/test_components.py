import numpy as np

from hkjc.recon import detect, finishcam


def test_features_job_masks_the_detector_pool_boxes(monkeypatch):
    from hkjc.recon import pipeline
    from hkjc.recon import video as video_mod

    raw = np.array([[10.0, 20.0, 30.0, 44.0], [5.0, 400.0, 25.0, 420.0]]), np.array([0.9, 0.8])

    class Cache:
        idx = np.array([7])

        def panel(self, k):
            return np.random.default_rng(0).integers(0, 255, (535, 945, 3), np.uint8)

    monkeypatch.setattr(pipeline, "_CACHE", Cache())
    monkeypatch.setattr(pipeline, "_STYLE", "silk")
    lrow = video_mod.legend_top(Cache().panel(0))
    i, lt, f, shape, rs, hist = pipeline._feat_job((0, 500, raw, lrow))
    assert lt == lrow
    assert i == 7 and shape == (535, 945)
    assert np.array_equal(rs, detect.AvatarDetector.drop_legend(raw, lt)) and len(hist) == len(rs)
    assert len(f.pts) == len(f.des)


def test_bgsub_finds_horses_and_merges_fragments():
    back = np.full((200, 300, 3), (60, 80, 100), np.uint8)  # dirt-like
    panel = back.copy()
    for x in (60, 150, 240):  # three "horses": head + rump fragments 12 px apart, silk in between
        panel[95:105, x - 12 : x - 4] = (200, 40, 40)  # head/rump fragments 12-14 px apart
        panel[95:105, x + 2 : x + 10] = (40, 200, 40)
    blobs = detect.bgsub_blobs(panel, back, np.ones((200, 300), np.uint8))
    assert len(blobs) == 3
    assert np.allclose(np.sort(blobs[:, 0]), [59, 149, 239], atol=4)


def test_finish_gaps_spatial_and_occlusion_recovery():
    # three horses moving left at the same speed, 100 px apart; horse 3 is never read at the line
    lx, reads = 900.0, []
    for u in range(60):
        for num, off in ((1, 0.0), (2, 100.0), (3, 200.0)):
            x = 1500 - 20 * u + off
            if num == 3 and 850 < x < 950:
                continue
            reads.append((u, u, x, 300 + 50 * num, num, 0.9))
    order, gaps, _u = finishcam.finish_gaps(dict(line_x=lx, reads=np.array(reads, float)))
    assert order == [1, 2, 3]
    assert abs(gaps[2] - 100) < 5 and abs(gaps[3] - 200) < 5


def test_start_phase_prior_allows_standing_start():
    """The core's state from the gate (grid from t = 0): the standing-start prior lets a horse seen only after 4 s
    accelerate out of the draw."""
    from types import SimpleNamespace

    from hkjc.recon import core

    t = np.arange(0, 20, 0.1)
    s_true = np.where(t < 3, 0.5 * 5.5 * t**2, 0.5 * 5.5 * 9 + 16.5 * (t - 3))
    m = core.Model(SimpleNamespace(horses=np.array([1])), core.list_rows(None, None, [1]), t, 22.5)
    k = t > 4
    obs = core.Obs.cat([core.Obs(np.zeros(1, int), np.zeros(1), np.zeros(1), np.full(1, 0.3)),
                        core.Obs(np.zeros(k.sum(), int), t[k], s_true[k], np.full(k.sum(), 0.5))])
    X = m.solve_s(obs)[0]
    assert abs(X[0, 0]) < 1.0 and np.abs(X[0] - s_true)[k].max() < 1.0  # no data in 0..4 s


def test_rail_despike_removes_false_ridge_hump():
    from hkjc.recon import rails

    n = np.arange(2000.0)
    R = 3000.0  # gently curved rail (world px)
    th = n * 2.0 / R
    W = np.c_[R * np.sin(th), R * (1 - np.cos(th))]
    Nrm = np.c_[-np.sin(th), np.cos(th)]
    bad = W.copy()
    bad[900:940] += 30.0 * Nrm[900:940]  # supported false ridge ~1.7 m off the rail
    out, ok = rails.despike_world(bad, np.ones(len(W), bool))
    assert not ok[905:935].any() and ok[:800].all() and ok[1100:].all()
    assert np.abs(out - W).max() < 2.0


def test_tick_seeded_marker_grid_ignores_edge_artefact():
    from hkjc.recon import rails

    ppm = 20.0
    ss = np.arange(0.0, 9000.0)
    resp = np.zeros_like(ss)
    finish = 8000.0
    for k in (0, 2, 3, 4, 6):  # FINISH, 100, 150, 200, 300 m
        resp[int(finish - k * 50 * ppm)] = 0.6
    resp[8700] = 1.0  # strong coverage-edge artefact downstream of the finish
    ev_v = np.arange(10.0, 400.0, 10.0)
    ev_s = finish - ev_v * ppm * 1.035  # HUD counts ~3.5 % more metres than the paint
    fit = rails.fit_marker_grid_seeded(ss, resp, ev_s, ev_v)
    assert fit is not None and abs(fit[1] - finish) < 20 and abs(fit[0] - ppm) < 0.5


def test_film_marker_crossings_need_a_sweeping_line():
    """A painted marker sweeps UP the front-on film; a static stripe at the bottom never counts.
    The leader's hooves (y=380) pass a line moving up 10 px per sample -> one crossing, labelled by
    the render odometer (lagging 10 m) as the 300 m marker; t = race clock."""
    from hkjc.recon import film

    rows = []
    for k in range(40):
        yc = 520.0 - 10.0 * k  # marker line: 520 -> 130
        lines = [(yc, 0, yc, 900, yc)] if yc > 210 else []
        lines.append((528.0, 0, 528.0, 900, 528.0))  # static bottom stripe (always present)
        rows.append(dict(f=1000 + 2 * k, diff=5.0, lines=lines, lead=(380.0, 400.0)))
    # hoof - line changes sign between k=14 (yc 380) ... crossing at frame 1028
    got = film.crossings_from_rows(rows, lambda f: 310.0, f0=0.0)
    assert list(got) == [300]
    assert abs(got[300] - 1028 / 25.0) < 0.05
    # only the static stripe: nothing
    static = [dict(r, lines=[(528.0, 0, 528.0, 900, 528.0)]) for r in rows]
    assert film.crossings_from_rows(static, lambda f: 310.0, f0=0.0) == {}


def test_film_anchor_offset_recovers_constant():
    """Render leader 6 m 'behind' reality at constant 16 m/s: the film crossings give delta = +6 m
    (d_true = d - delta); fewer than FILM_MIN matched markers -> None."""
    from hkjc.recon import film

    t = np.arange(0.0, 40.0, 0.1)
    d_true = 600.0 - 16.0 * t
    d_render = d_true + 6.0
    film_t = {m: (600.0 - m) / 16.0 for m in (400, 300, 200, 100)}
    delta, per = film.anchor_offset(t, d_render, film_t)
    assert abs(delta - 6.0) < 0.05 and len(per) == 4
    assert film.anchor_offset(t, d_render, {400: 12.5, 300: 18.75})[0] is None


def test_time_local_shift_recovers_per_frame_slip():
    """A 4-s per-frame pose slip (leader 25 m 'behind' the odometer) is returned as a shift inside the
    window and zero elsewhere; small tick noise alone triggers nothing."""
    from hkjc.recon import rails

    tf = np.arange(0.0, 60.0, 0.1)
    d_true = 1000.0 - 16.0 * tf
    odo = np.ceil(d_true / 10.0) * 10.0  # round-up display: ticks V+10 -> V as the leader reaches V
    slip = np.where((tf > 30) & (tf < 34), 25.0, 0.0)
    shift, n = rails.time_local_shift(tf, d_true + slip, odo)
    assert n == 1
    assert np.all(np.abs(shift[(tf > 31) & (tf < 33)] - 25.0) < 3.0)
    assert np.all(shift[(tf < 28) | (tf > 36)] == 0.0)
    shift0, n0 = rails.time_local_shift(tf, d_true + np.sin(tf) * 2.0, odo)
    assert n0 == 0 and not shift0.any()


def test_finishcam_digit_drop_prefers_longer_number():
    """'1' reads lying on the '11' cloth (a dropped digit) outnumber the real '11' reads: the winner
    #11 must survive, and the real #1 (elsewhere) keeps its own track."""
    rows = []
    for u in range(20, 80, 6):  # the real "11" cloth, moving left
        rows.append([u, u, 1300 - 10.0 * u, 300.0, 11, 0.9])
    for u in range(21, 80, 3):  # the same cloth misread as "1" (more reads than "11")
        rows.append([u, u, 1300 - 10.0 * u, 302.0, 1, 0.8])
    for u in range(100, 140, 4):  # the real #1, another horse, another lane
        rows.append([u, u, 1700 - 10.0 * u, 520.0, 1, 0.9])
    T = finishcam.tracks(dict(line_x=960.0, reads=np.array(rows, float)))
    assert 11 in T
    assert 1 in T and np.median(T[1][2]) > 500  # #1's track is the real #1, not the "11" cloth


def test_finishcam_digit_drop_keeps_a_real_short_number():
    """A few "2" reads on the "12" cloth (dropped digit) must not delete the REAL #2, read crossing
    later after #12 has left (HV 0603 R3: 3 of 37 "2" reads on the "12" cloth; the real #2 was
    8th). A "3" read almost only on the "13" cloth (ST 0419 R4: 9 of 10) is still dropped."""
    rows = [[u, u, 1460 - 20.0 * u, 300.0, 12, 0.9] for u in range(10, 60)]  # #12 crosses u 25
    rows += [[u, u, 1460 - 20.0 * u + 5, 302.0, 2, 0.8] for u in (20, 21, 40)]  # "2" on the "12"
    rows += [[u, u, 2400 - 20.0 * u, 500.0, 2, 0.9] for u in range(62, 110)]  # real #2, crosses u 72
    order, gaps, _u = finishcam.finish_gaps(dict(line_x=960.0, reads=np.array(rows, float)))
    assert 2 in gaps and order.index(12) < order.index(2)
    rows = [[u, u, 1460 - 20.0 * u, 300.0, 13, 0.9] for u in range(10, 60)]
    rows += [[u, u, 1460 - 20.0 * u + 5, 302.0, 3, 0.8] for u in range(12, 60, 5)]  # "3" on "13"
    rows += [[u, u, 1900 - 20.0 * u, 500.0, 7, 0.9] for u in range(30, 80)]
    order, gaps, _u = finishcam.finish_gaps(dict(line_x=960.0, reads=np.array(rows, float)))
    assert 3 not in gaps and 13 in gaps and 7 in gaps


def test_finishcam_isolated_first_crosser_keeps_the_field():
    """A winner far ahead (its cloth gone before the next horse crosses) must not wipe the gaps of
    the field, which chain among themselves."""
    rows = []
    for u in range(10, 40):  # winner #5 crosses at u ~ 22 and leaves the read window by u 40
        rows.append([u, u, 1400 - 20.0 * u, 300.0, 5, 0.9])
    for h, x0, y in [(2, 3600.0, 420.0), (7, 3750.0, 520.0), (9, 3900.0, 620.0)]:
        for u in range(120, 180):  # the field, 150 px apart, in view together, crossing u ~ 132-147
            # (the winner's extrapolated cloth is then > max_dx past the line: it links nobody)
            rows.append([u, u, x0 - 20.0 * u, y, h, 0.9])
    order, gaps, _u = finishcam.finish_gaps(dict(line_x=960.0, reads=np.array(rows, float)))
    assert order[:3] == [2, 7, 9]
    assert abs(gaps[7] - gaps[2] - 150.0) < 1 and abs(gaps[9] - gaps[2] - 300.0) < 1


def test_register_mask_boxes_zeroes_grown_avatar_boxes():
    """Avatar boxes (+ shadow margin) leave the ORB mask; the rest is untouched."""
    from hkjc.recon import register

    m = np.full((100, 200), 255, np.uint8)
    out = register.mask_boxes(m, np.array([[50.0, 50.0, 20.0, 10.0, 0.9, 0.0]]), grow=1.6)
    assert out[50, 50] == 0 and out[50, 50 + 15] == 0  # inside the grown box (half-width 16)
    assert out[50, 50 + 20] == 255 and out[10, 150] == 255
    assert m[50, 50] == 255  # input not modified


def test_video_grid_keep_drops_off_grid_reencodes():
    """10 Hz render in 25 fps video: ticks at two fixed residues of frame % 5. A keyframe re-encode
    at 400 (its tick started at 399, kept) is dropped. The first frame after a render-gate dropout
    (on-grid start not yielded) is the only copy of its tick and is kept."""
    from hkjc.recon import video

    ticks = [k * 5 + o for k in range(60, 100) for o in (2, 4)]  # residues {2, 4} (HV 04-22 phase)
    idx = sorted(ticks + [400, 500])
    keep = video.grid_keep(idx)
    assert [i for i, k in zip(idx, keep) if not k] == [400, 500]
    assert all(keep[idx.index(t)] for t in (399, 402, 404))
    gap = [i for i in ticks if not 450 <= i <= 474] + [473]  # dropout 450-474; 473 off-grid, its tick start 472 missing
    assert video.grid_keep(sorted(gap)).all()
    assert video.grid_keep([1, 2, 3, 4, 5, 6, 7]).all()  # no dominant phase -> keep everything


def test_video_keyframe_reencode_after_live_film_gets_its_tick_time():
    """ST 2024-12-29 R8: 93 frames of live start film pass the render gate (every frame differs), then
    the render ticks at residues {1, 3} and every 50th frame (a video keyframe, residue 0) re-encodes
    the previous tick past the de-dup tolerance. The film outvotes the grid over all frames; the
    cadence vote still finds it, the re-encodes are dropped, and every kept render frame is stamped
    with the time of the tick it shows (tick k starts at 1 + floor(2.5 k) and shows 0.75 + 2.5 k)."""
    from hkjc.recon import video

    film = list(range(121, 214))
    starts = [1 + (5 * k) // 2 for k in range(100, 610)]  # tick starts 251 .. 1523
    keys = [f for f in range(300, 1524, 50) if f not in starts]
    idx = np.array(sorted(film + starts + keys))
    shown = {f: max(k for k, s in enumerate(starts) if s <= f) for f in starts + keys}  # tick on screen
    kept = idx[video.grid_keep(idx)]
    grid = video.tick_grid(kept)  # the stages downstream vote on the stored (kept) frames
    render = kept[kept >= starts[0]]
    np.testing.assert_allclose(
        video.tick_frame(render, grid), [0.75 + 2.5 * (100 + shown[int(f)]) for f in render]
    )
    assert video.tick_grid(idx) == grid == (1, 3)
    assert not set(keys) & set(kept.tolist()) and set(starts) <= set(kept.tolist())


def _render_stream(lo, hi, res=(1, 3), keys=50):
    """De-dup stream of a 10 Hz render: tick starts at residues ``res`` of frame % 5 in [lo, hi),
    plus a keyframe re-encode every ``keys`` frames where that is not a tick start."""
    a, b = res
    starts = [f for f in range(lo, hi) if f % 5 in (a, b)]
    return sorted(starts + [f for f in range(lo, hi) if f % keys == 0 and f not in starts])


def _build(tmp_path, monkeypatch, idx):
    """Run ``FrameCache.build`` on a synthetic de-dup stream; returns the stored frame indices."""
    from hkjc.recon import video

    full = np.full((400, 1000, 3), 100, np.uint8)  # panel box = the right half

    def fake(path, **kw):
        for i in idx:
            yield i, np.full((8, 8, 3), i % 256, np.uint8), full

    monkeypatch.setattr(video, "iter_unique_render", fake)
    n = video.FrameCache.build("unused.mp4", tmp_path)
    stored = np.load(tmp_path / "frames_idx.npz")["idx"]
    assert n == len(stored) == len(np.load(tmp_path / "frames_clock.npy"))
    return stored


def test_video_live_film_before_the_render_is_not_cached(tmp_path, monkeypatch):
    """Live start film that passes the render gate (5 of 329 replays) changes on every frame; the
    render changes every 2-3 frames. Film frames are never cached: ST 2024-12-29 R8 (93 film frames,
    a render-gate dropout, the render at {1, 3} from an off-grid first frame), HV 2023-10-29 R6 (a
    static gate shot that the de-dup thins to 2-3 frame gaps, a dropout inside the film) and
    ST 2025-01-05 R6 (only 8 frames between the film and the render)."""
    st8_film = list(range(121, 204)) + [205, 206, 207, 208, 210, 211, 212, 213]
    st8_render = [272] + _render_stream(273, 900)
    hv6_film = [5, 6, 8, 11, 13, 15, 18, 20, 22, 24, 26] + list(range(27, 88)) + list(range(95, 130))
    hv6_render = [333] + _render_stream(334, 900, (2, 4))
    st6_film = list(range(172, 307))
    st6_render = [314] + _render_stream(315, 900, (3, 0))
    for film, render in ((st8_film, st8_render), (hv6_film, hv6_render), (st6_film, st6_render)):
        stored = _build(tmp_path, monkeypatch, film + render)
        assert stored[0] == render[0] and not set(film) & set(stored.tolist())


def test_video_film_keep_leaves_render_streams_alone():
    """Streams without a leading film run keep every frame: keyframe re-encodes every 50 or 100
    frames, an off-grid first frame, a render-gate dropout, a short dissolve at the render start,
    film after the render has started, and streams without the 10 Hz cadence."""
    from hkjc.recon import video

    cases = [
        _render_stream(250, 1500),
        _render_stream(250, 1500, (2, 4), keys=100),
        [303] + _render_stream(304, 1500, (2, 4)),
        [f for f in _render_stream(300, 1500) if not 600 <= f < 650] + [651],
        [295, 296, 297, 298] + _render_stream(299, 1500, (3, 0)),
        list(range(0, 9)) + _render_stream(300, 1500),  # 8 one-frame gaps < FILM_RUN
        _render_stream(250, 1200) + list(range(1250, 1350)),
        list(range(0, 200)),
        [1, 2, 3, 4, 5, 6, 7],
        [],
    ]
    for idx in cases:
        assert video.film_keep(idx).all(), idx[:12]


def test_video_tick_frame_puts_ticks_on_the_uniform_grid():
    """Ticks at residues {2, 4}: 2 -> 4 is a 2-frame step, 4 -> 7 a 3-frame step. Tick times move
    the first residue -0.25 and the second +0.25 frame, so every interval is 2.5 frames; off-grid
    frames and non-10 Hz streams are unchanged."""
    from hkjc.recon import video

    ticks = [k * 5 + o for k in range(60, 100) for o in (2, 4)]
    grid = video.tick_grid(ticks)
    assert grid == (2, 4)
    assert video.tick_grid([k * 5 + o for k in range(60, 100) for o in (4, 1)]) == (4, 1)
    tf = video.tick_frame(ticks, grid)
    np.testing.assert_allclose(np.diff(tf), 2.5)
    assert abs(float(np.mean(tf - np.asarray(ticks)))) < 1e-12  # mean shift zero
    assert video.tick_frame([473], grid)[0] == 473.0  # off-grid: unchanged
    assert video.tick_grid([1, 2, 3, 4, 5, 6, 7]) is None
    np.testing.assert_array_equal(video.tick_frame([1, 2, 3], None), [1.0, 2.0, 3.0])


def test_rails_fixed_rail_gap_finds_the_line_not_the_band_edge():
    """A thin bright line 9.5 m inside the racing rail (ppm 10) is measured at 9.5 m; a bright apron
    beyond the band's far edge must not pull the estimate to the band edge."""
    from hkjc.recon import rails

    ppm = 10.0
    img = np.full((260, 500, 3), 90, np.uint8)
    img[143:148, :] = 125  # faint fixed rail: y = 50 + 9.5 m * ppm
    img[:125, :] = 40  # dark ground at the band's near end (band 6.7-11.7 m -> y 117-167)
    img[158:, :] = 150  # bright apron from inside the band's far end outward
    # (with a wrapping side comparison, offsets near the far edge "see" the dark near end and fire)
    cnt = np.ones((260, 500), np.float32)
    rail = np.c_[np.arange(60.0, 440.0), np.full(380, 50.0)]
    gap, sup = rails.fixed_rail_gap(img, cnt, lambda W: np.asarray(W, float), rail, 1, ppm, 9.2)
    assert abs(gap - 9.5) < 0.2 and sup > 0.5


def test_rails_inner_side_is_right_on_a_straight():
    """A straight camera path's turning sign is noise; the rail side must not follow it."""
    from hkjc.recon import rails

    rng = np.random.default_rng(3)
    x = np.arange(0, 2000, 2.0)
    for k in range(6):  # tiny wobbles of either sign (|turn| << an oval's)
        y = 0.5 * np.sin(x / (150 + 40 * k)) * (-1) ** k + rng.normal(0, 0.05, len(x))
        _P, T, _s = rails.resample_path(np.c_[x, y], step=2.0, smooth=80.0)
        assert rails.inner_side(T) == 1
        assert rails.inner_side(T, "left") == -1  # a straight takes the side from the race record
    # a decisive turn uses the MEASURED sign, whatever the record says: clockwise and anti-clockwise
    a = np.linspace(0, np.pi, 400)
    for sgn in (1, -1):
        P = np.c_[500 * np.cos(-a), -500 * sgn * np.sin(-a)]
        _P, T, _s = rails.resample_path(P, 2.0, 80.0)
        assert rails.inner_side(T, "right") == rails.inner_side(T, "left") == rails.inner_sign(T)
    assert {rails.inner_side(rails.resample_path(np.c_[500 * np.cos(-a), -500 * g * np.sin(-a)],
                                                 2.0, 80.0)[1]) for g in (1, -1)} == {1, -1}


def test_rails_smooth_rail_bridges_gaps_smoothly_and_resists_snaps():
    from hkjc.recon import rails

    rng = np.random.default_rng(0)
    a = np.linspace(0, 1.2, 3000)  # an arc of radius 2000 px (stations ~0.8 px apart)
    W = np.c_[2000 * np.cos(a), 2000 * np.sin(a)]
    noisy = W + rng.normal(0, 1.0, W.shape)
    ok = np.ones(len(W), bool)
    ok[800:1700] = False  # a long UNSUPPORTED gap whose DP points still lie near the rail (ST 05-24) ...
    noisy[790:800] += np.linspace(0, 6, 10)[:, None]  # ... with noisy end slopes
    noisy[2000:2005] += 40.0  # a burst of false ridge snaps
    Z = rails.smooth_rail(noisy, ok)
    err = np.hypot(*(Z - W).T)
    assert err[800:1700].max() < 4.0  # no bulge across the gap (weight 1e-4 overshot ~150 px)
    assert np.percentile(err, 99) < 4.0 and err[2000:2005].max() < 4.0


def test_tracking_split_by_appearance_cuts_identity_switches_only():
    from hkjc.recon import tracking

    rng = np.random.default_rng(2)
    nb = 26
    a, b = rng.dirichlet(np.ones(nb)), rng.dirichlet(np.ones(nb))

    def dets(p, n):
        return np.array([rng.multinomial(400, p) for _ in range(n)], float)

    F = np.r_[dets(a, 120), dets(b, 120), dets(a, 200)]  # trk 0: horse A then B; trk 1: pure A
    t = np.r_[np.arange(240) * 0.1, np.arange(200) * 0.1]
    members = [np.arange(240), np.arange(240, 440)]
    out = sorted(tracking.split_by_appearance(members, t, F), key=lambda m: m[0])
    assert [len(m) for m in out] == [120, 120, 200]


def test_rails_robust_offset_ignores_a_long_outlier_run():
    from hkjc.recon import rails

    rng = np.random.default_rng(4)
    h = 30 + 5 * np.sin(np.arange(6000) / 900.0) + rng.normal(0, 2, 6000)
    h[3000:3110] = 240.0  # a 110-station run of outliers (sparse detections far inside the field)
    z = rails.robust_offset(h)
    assert np.abs(z[3000:3110] - 30 - 5 * np.sin(np.arange(3000, 3110) / 900.0)).max() < 6.0


def test_finishcam_misread_before_the_real_track_is_the_same_cloth():
    """HV 04-15 R1: the "7" cloth read as "1" on the 4 unique frames BEFORE the "7" track starts
    (no shared frame) lies on #7's line continued back -> one cloth, kept as #7; a real #1
    elsewhere is unaffected."""
    rows = [[u, u, 1486 - 38.0 * (u - 135), 310.0, 7, 0.95] for u in range(135, 168)]
    rows += [[u, u, 1486 - 38.0 * (u - 135) + 2, 328.0, 1, 0.95] for u in range(131, 135)]
    T = finishcam.tracks(dict(line_x=956.0, reads=np.array(rows, float)))
    assert 7 in T and 1 not in T
    rows += [[u, u, 1700 - 30.0 * (u - 100), 600.0, 1, 0.95] for u in range(100, 130)]
    T = finishcam.tracks(dict(line_x=956.0, reads=np.array(rows, float)))
    assert 7 in T and 1 in T and np.median(T[1][2]) == 600.0


def test_finishcam_static_object_does_not_outvote_a_moving_cloth():
    """ST8: a rail post read as "1" ~60x at one spot (plus 2 stray reads that pass the raw ptp test)
    must not outvote the real #1 cloth moving across the frame in another lane."""
    rows = [[u, u, 1566.0, 1052.0, 1, 0.8] for u in range(7, 127, 2)]
    rows += [[30, 30, 405.0, 1037.0, 1, 0.6], [80, 80, 1278.0, 1029.0, 1, 0.7]]
    rows += [[u, u, 1532 - 43.0 * (u - 58), 356.0, 1, 0.99] for u in range(58, 88)]
    T = finishcam.tracks(dict(line_x=960.0, reads=np.array(rows, float)))
    assert 1 in T and np.median(T[1][2]) == 356.0 and np.ptp(T[1][1]) > 1000


def test_rails_straight_datum_measures_the_fence_where_it_is_seen():
    """Straight course: the rail polyline is the field-edge scaffold; the fence offset is the median
    over the DP's LONGEST contiguous supported run (the fence where the mosaic shows it), not the
    scattered texture support elsewhere."""
    from hkjc.recon import rails

    n = 3000
    Ps = np.c_[np.arange(n) * 2.0, np.zeros(n)]
    Nin = np.tile([0.0, 1.0], (n, 1))
    hin = np.full(n, 5.0)
    rail_off = np.full(n, np.nan)
    rail_off[::37] = hin[::37] + np.random.default_rng(0).uniform(-10, 50, len(rail_off[::37]))
    rail_off[1200:1700] = hin[1200:1700] + np.linspace(105, 115, 500)  # the seen fence
    C = Ps[::10] + [0.0, -30.0]
    out = rails.straight_datum(Ps, Nin, hin, rail_off, 1, C)
    assert abs(out["lat0_px"] - 110.0) < 0.5
    assert out["lat0_range"][0] >= 104.9 and out["lat0_range"][1] <= 115.1
    assert np.allclose(out["rail"][:, 1], 5.0, atol=0.5) and out["support"].all()
    assert rails.is_straight(np.tile([1.0, 0.0], (100, 1)))
    th = np.linspace(0, np.pi, 200)
    assert not rails.is_straight(np.c_[np.cos(th), np.sin(th)])


def test_film_lazy_leads_match_eager_reads():
    """fill_leads runs the detector on a track only up to the crossing crossings_from_rows takes:
    an unreadable odometer at the first crossing keeps the track open for the next one."""
    from hkjc.recon import film

    rows, truth = [], {}
    for k in range(40):
        yc = 520.0 - 10.0 * k
        lines = [(yc, 0, yc, 900, yc)] if yc > 210 else []
        rows.append(dict(f=1000 + 2 * k, diff=5.0, lines=lines, lead=None))
        # hooves cross at k=14, fall back above the line for k=15..19, cross again at k=20
        truth[1000 + 2 * k] = (300.0 if 15 <= k <= 19 else 380.0 if k < 15 else 330.0, 400.0)
    for odo_at in (lambda f: 310.0, lambda f: np.nan if f < 1030 else 310.0):
        calls = []

        def lead_of(f):
            calls.append(f)
            return truth[f]

        def frames(need):
            return ((f, f) for f in sorted(need))

        lazy = film.fill_leads([dict(r) for r in rows], frames, lead_of, odo_at)
        eager = [dict(r, lead=truth[r["f"]]) for r in rows]
        n_need = sum(1 for r in rows if r["lines"])
        got = film.crossings_from_rows(lazy, odo_at, f0=0.0)
        assert got == film.crossings_from_rows(eager, odo_at, f0=0.0) and list(got) == [300]
        assert len(calls) < n_need
    assert abs(got[300] - 1039.0 / 25.0) < 0.05  # the second crossing, the first had no odometer


def test_stable_legend_rows_replace_only_far_rows_of_frames_that_show_the_strip():
    """The race's list row (median over frames showing the strip) replaces a frame's row more than LEGEND_TOL px off
    it (a vivid band below the list pushed it under the icons); rows within the tolerance and frames without a strip
    keep their own."""
    from hkjc.recon import video

    rows = np.array([437, 436, 438, 437, 505, 459, 428, 437])
    found = np.array([True, True, True, True, True, True, False, True])
    out = video.stable_legend_rows(rows, found)
    assert out.tolist() == [437, 436, 438, 437, 437, 437, 428, 437]
    assert video.stable_legend_rows(rows, np.zeros(8, bool)).tolist() == rows.tolist()


def test_a_wider_vivid_band_below_the_list_moves_legend_scan_but_not_the_race_row():
    """legend_scan alone follows a wider vivid band below the list (its 30 % threshold rises above the icon rows);
    the race-level rule restores the list row."""
    from hkjc.recon import video

    p = np.full((536, 945, 3), (40, 140, 60), np.uint8)  # turf (BGR)
    p[437:500, 100:300] = (0, 0, 230)  # icon strip: vivid red over 200 px
    plain = video.legend_scan(p)
    assert plain == (437, True)
    q = p.copy()
    q[520:536, :] = (0, 0, 230)  # a full-width vivid band below the list
    pushed = video.legend_scan(q)
    assert pushed[0] >= 500 and pushed[1]
    rows = [plain[0]] * 9 + [pushed[0]]
    assert video.stable_legend_rows(rows, [True] * 10).tolist() == [437] * 10
