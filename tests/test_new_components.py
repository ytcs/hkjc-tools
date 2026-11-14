import numpy as np

from hkjc.recon import detect, finishcam, fusion, identity


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
    t = np.arange(0, 20, 0.1)
    s_true = np.where(t < 3, 0.5 * 5.5 * t**2, 0.5 * 5.5 * 9 + 16.5 * (t - 3))
    obs = [
        fusion.Obs(np.array([0.0]), np.array([0.0]), np.array([3.0]), 0.3, 0.4, "draw", 1, 1),
        fusion.Obs(t[t > 4], s_true[t > 4], np.full((t > 4).sum(), 3.0), 0.5, 0.3, "render", 1, 2),
    ]
    s, _sd, lat, _sl, _c = fusion.smooth(obs, fusion.Prior(width=22.5), t)
    assert abs(s[0]) < 1.0 and np.abs(s - s_true)[t > 4].max() < 1.0  # no data in 0..4 s


def test_completed_legend_votes():
    # 3 horses; frame where only horse B (middle) is detected: rank completed from the posterior
    members = [np.array([0])]
    legend = np.array([[10, 7, 4, 9, 0, 0, 0, 0, 0]])  # legend: #7 leads, #4 2nd, #9 3rd

    def field(t):
        return np.array([100.0, 104.0, 110.0]), np.array([0.5, 0.5, 0.5])

    v = identity.legend_votes_completed(
        members, np.array([10]), np.array([5.0]), np.array([104.2]), legend, field, nmax=9
    )
    assert v[0, 4] == 1  # middle detection -> rank 1 -> legend slot 2 = #4


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


def test_em_assign_frames_uses_motion_and_exclusion():
    from hkjc.recon import assoc

    tg = np.arange(0, 10.01, 0.1)
    paths = {
        1: fusion.Path(1, tg, 15.0 * tg, np.full_like(tg, 2.0), np.full_like(tg, 0.3),
                       np.full_like(tg, 0.2), np.ones_like(tg)),
        2: fusion.Path(2, tg, 15.0 * tg - 3.0, np.full_like(tg, 4.0), np.full_like(tg, 0.3),
                       np.full_like(tg, 0.2), np.ones_like(tg)),
    }
    t = np.array([5.0, 5.0, 5.0])
    s = np.array([75.0, 72.0, 40.0])  # horse 1, horse 2, clutter far away
    lat = np.array([2.0, 4.0, 3.0])
    lab, _chi = assoc.assign_frames(np.array([7, 7, 7]), t, s, lat, paths, np.zeros((0, 9), int))
    assert lab.tolist() == [1, 2, 0]


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
    t, v = film.leader_finish({200: 60.0, 100: 66.0})
    assert abs(v - 100 / 6.0) < 1e-9 and abs(t - 72.0) < 1e-9


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


def test_em_assign_negative_legend_evidence():
    """A horse missing from a fully read top-8 legend runs behind all 8 listed horses: it must not
    take a detection at the front, even when its own (vague) posterior points there."""
    from hkjc.recon import assoc

    tg = np.arange(0, 10.01, 0.1)

    def path(h, s0, sd):
        return fusion.Path(h, tg, 15.0 * tg + s0, np.full_like(tg, 2.0), np.full_like(tg, sd),
                           np.full_like(tg, 0.3), np.ones_like(tg))

    paths = {h: path(h, -2.0 * h, 0.3) for h in range(1, 9)}  # listed, tight, 2 m apart
    paths[9] = path(9, 5.0, 6.0)  # NOT listed; vague posterior sitting ahead of the field
    t = np.full(9, 5.0)
    s = np.array([80.0] + [75.0 - 2.0 * h for h in range(1, 9)])  # a front detection + the 8
    lat = np.full(9, 2.0)
    leg = np.array([[7, 1, 2, 3, 4, 5, 6, 7, 8]])
    lab, _ = assoc.assign_frames(np.full(9, 7), t, s, lat, paths, leg, n_field=9)
    assert lab[0] != 9  # the front detection is not the unlisted horse


def test_em_assign_rank_attraction_final_stretch():
    """Near the line, a listed horse whose (overconfident, extrapolated) path sits 7 m ahead of its
    real detection takes the detection at its legend rank; away from the line it does not."""
    from hkjc.recon import assoc

    tg = np.arange(0, 10.01, 0.1)

    def path(h, s0, sd):
        return fusion.Path(h, tg, 15.0 * tg + s0, np.full_like(tg, 2.0), np.full_like(tg, sd),
                           np.full_like(tg, 0.3), np.ones_like(tg))

    D = 1000.0
    # at t = 5 s (15 m/s): #1's path at D - 70 (on its detection), #2's at D - 68: 7 m AHEAD of
    # #2's real detection at D - 75 (and 2 m ahead of #1: overconfident constant-speed extrapolation)
    paths = {1: path(1, D - 70.0 - 75.0, 0.3), 2: path(2, D - 68.0 - 75.0, 1.3)}
    t = np.full(2, 5.0)
    s = np.array([D - 70.0, D - 75.0])  # #1's detection leads; #2's real detection 7 m behind its path
    lat = np.full(2, 2.0)
    leg = np.array([[7, 1, 2, 0, 0, 0, 0, 0, 0]])
    lab, _ = assoc.assign_frames(np.full(2, 7), t, s, lat, paths, leg, n_field=2, s_line=D)
    assert lab[1] == 2  # rank-2 detection goes to the rank-2 horse
    lab, _ = assoc.assign_frames(np.full(2, 7), t, s, lat, paths, leg, n_field=2, s_line=D + 500)
    assert lab[1] == 0  # 575 m out: no attraction, chi^2 too large -> clutter


def test_em_assign_silk_arbitrates_between_close_candidates(monkeypatch):
    """Two horses side by side, positionally interchangeable: the official-silk signature decides
    (bounded), and an empty detection signature is no evidence. (The term is off by default.)"""
    from hkjc.recon import assoc

    monkeypatch.setenv("HKJC_SILK_W", "20")

    tg = np.arange(0, 10.01, 0.1)

    def path(h, lat):
        return fusion.Path(h, tg, 15.0 * tg, np.full_like(tg, lat), np.full_like(tg, 0.3),
                           np.full_like(tg, 0.3), np.ones_like(tg))

    paths = {1: path(1, 2.0), 2: path(2, 2.2)}
    t = np.full(2, 5.0)
    s = np.full(2, 75.0)
    lat = np.array([2.2, 2.0])  # positions slightly favour the SWAPPED assignment
    red, blue = np.zeros(26), np.zeros(26)
    red[12], blue[12 + 8] = 1.0, 1.0  # vivid hue bins
    F = np.array([red, blue])  # detection 0 wears red, detection 1 blue
    proto = {1: red, 2: blue}
    leg = np.zeros((0, 9), int)
    lab, _ = assoc.assign_frames(np.full(2, 7), t, s, lat, paths, leg, silk=(F, proto))
    assert list(lab) == [1, 2]
    lab, _ = assoc.assign_frames(np.full(2, 7), t, s, lat, paths, leg, silk=(np.zeros((2, 26)), proto))
    assert list(lab) == [2, 1]  # no silk evidence: position decides


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


def test_fusion_anonymous_association_respects_forbid():
    """An anonymous tracklet that the E-step's legend evidence excluded for a horse must not be
    handed to that horse by fusion's chi^2 association."""
    tg = np.arange(0.0, 10.0, 0.2)
    lab = fusion.Obs(tg[:20], 15.0 * tg[:20], np.full(20, 2.0), 0.8, 0.35, "render", horse=1, tid=1)
    anon = fusion.Obs(tg[20:], 15.0 * tg[20:], np.full(30, 2.0), 0.8, 0.35, "render", 0, 2)
    prior = fusion.Prior(width=30.0)
    _p, a = fusion.fuse_race([lab, anon], [1], prior, anchor_check=False)
    assert a.get(2) == 1  # continuation of horse 1: associated
    anon_no = fusion.Obs(tg[20:], 15.0 * tg[20:], np.full(30, 2.0), 0.8, 0.35, "render", 0, 2,
                         frozenset({1}))
    _p, a = fusion.fuse_race([lab, anon_no], [1], prior, anchor_check=False)
    assert 2 not in a
