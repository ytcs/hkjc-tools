"""recon.core: segments, list fault chains, the objective and the solve on synthetic races."""

import numpy as np

from hkjc.recon import core


def _race(n_h=3, T=20.0, gap_lat=1.5, speeds=None, start_s=None):
    """Synthetic detections: n_h horses at constant speeds, one box each per 0.1 s tick."""
    speeds = speeds or [16.0 + 0.1 * k for k in range(n_h)]
    start_s = start_s or [100.0 - 3.0 * k for k in range(n_h)]
    t = np.arange(1.0, T, 0.1)
    rows, tt, ss, ll, who = [], [], [], [], []
    for f, ti in enumerate(t):
        for h in range(n_h):
            s = start_s[h] + speeds[h] * ti
            lat = 2.0 + gap_lat * h
            rows.append([f, 800 - 8 * (s - 16.0 * ti - 90.0), 200 + 30 * lat, 20, 20, 0.9])
            tt.append(ti)
            ss.append(s)
            ll.append(lat)
            who.append(h)
    return (np.array(rows, float), np.array(tt), np.array(ss), np.array(ll), np.array(who))


def _evidence(rows, t, s, lat, members, horses, legend=None, n_cam=()):
    pose = dict(idx=np.unique(rows[:, 0]).astype(int), edges=np.zeros((0, 3)), kinds=np.zeros(0, str),
                resid=np.zeros(0))
    pose["edges"] = np.array([[i, i + 1, 1.0] for i in range(len(pose["idx"]) - 1)])
    pose["kinds"] = np.array(["seq"] * len(pose["edges"]))
    pose["resid"] = np.array([5.0 if int(pose["idx"][i + 1]) in n_cam else 0.2 for i in range(len(pose["edges"]))])
    T = dict(rows=rows, t=t, s=s, lat=lat)
    return core.build_evidence(horses, T, members, pose, 1000.0, True, panel_w=1600.0)


def test_segments_cut_at_gaps_camera_transitions_and_link_them():
    rows, t, s, lat, who = _race(n_h=1)
    keep = ~((t > 5.0) & (t < 6.5))  # a 1.5 s gap inside the tracklet
    rows, t, s, lat = rows[keep], t[keep], s[keep], lat[keep]
    fr = rows[:, 0].astype(int)
    seg, links = core.segments(t, s, lat, fr, [np.arange(len(t))], cam_frames=[150])
    assert len(np.unique(seg)) == 3  # gap cut + camera cut
    rew = sorted(links[:, 2].tolist())
    assert core.LINK_WEAK in rew and core.LINK_TRACK in rew


def test_omission_chain_absorbs_a_long_omission_and_frozen_chain_a_held_list():
    H, R = 10, 300
    slots = np.tile(np.arange(8), (R, 1))
    L = core.ListRows(np.arange(R) * 0.1, slots, np.ones(R, bool), np.r_[False, np.ones(R - 1, bool)])
    St = core.statements(L, H)
    # horse 9 (unlisted) is truly 5 m ahead of the last listed horse on every row: an omission
    d = np.where(St.unl & (St.b == 9), -5.0, 3.0)
    _J, pf, om = core.list_chains(L, St, core._p_order(d)[1], H, d)
    assert om[9, 50:250].min() > 0.9
    assert om[8, 50:250].max() < 0.1
    # every adjacent pair contradicted on a list held identical: frozen
    d2 = np.where(St.unl, 3.0, -3.0)
    _J2, pf2, _om2 = core.list_chains(L, St, core._p_order(d2)[1], H, d2)
    assert pf2[20:280].min() > 0.9 and pf.max() < 0.1


def test_solve_recovers_identities_across_a_tracker_swap_from_the_list():
    """Two horses abreast-ish; the tracker swaps them at t = 10 s (one tracklet jumps between horses). The list states
    the order throughout; the solve labels the pieces by horse, not by tracklet."""
    rows, t, s, lat, who = _race(n_h=2, speeds=[16.0, 16.0], start_s=[100.0, 96.0], gap_lat=1.5)
    a = np.flatnonzero(((who == 0) & (t < 10)) | ((who == 1) & (t >= 10)))
    b = np.flatnonzero(((who == 1) & (t < 10)) | ((who == 0) & (t >= 10)))
    horses = np.array([3, 7])
    ev = _evidence(rows, t, s, lat, [a, b], horses)
    fr = np.unique(rows[:, 0])
    leg = np.c_[fr, np.tile([3, 7, 0, 0, 0, 0, 0, 0], (len(fr), 1))]
    Lr = core.list_rows(leg, lambda f: np.asarray(f, float) * 0.1 + 1.0, horses)
    tg = np.arange(1.0, 20.0, core.DT)
    m = core.Model(ev, Lr, tg, width=20.0)
    res = core.solve(m, log=lambda *_: None)
    lab = res.seg_lab[ev.seg]
    assert (lab[who == 0] == 0).mean() > 0.95 and (lab[who == 1] == 1).mean() > 0.95
    J = [h[1] for h in res.log]
    assert all(b <= a + 1e-6 for a, b in zip(J[1:], J[2:]))  # every accepted step lowers J


def test_unseen_horse_in_a_block_without_list_does_not_take_a_visible_horse():
    """Three horses; horse 2 is never detected after 5 s (off the panel behind). In a block without list rows
    (masked), the rear visible horse keeps its own label: the unseen horse's vague posterior does not steal it."""
    rows, t, s, lat, who = _race(n_h=3, speeds=[16.0, 16.0, 15.0], start_s=[100.0, 97.0, 94.0])
    keep = ~((who == 2) & (t > 5.0))
    rows, t, s, lat, who = rows[keep], t[keep], s[keep], lat[keep], who[keep]
    members = [np.flatnonzero(who == h) for h in range(3)]
    horses = np.array([1, 2, 3])
    ev = _evidence(rows, t, s, lat, members, horses)
    fr = np.unique(rows[:, 0])
    tt = fr * 0.1 + 1.0
    leg = np.c_[fr, np.tile([1, 2, 3, 0, 0, 0, 0, 0], (len(fr), 1))]
    leg[(tt > 8) & (tt < 16), 1:] = 0  # withheld block
    Lr = core.list_rows(leg, lambda f: np.asarray(f, float) * 0.1 + 1.0, horses)
    m = core.Model(ev, Lr, np.arange(1.0, 20.0, core.DT), width=20.0)
    res = core.solve(m, log=lambda *_: None)
    lab = res.seg_lab[ev.seg]
    blk = (t > 8) & (t < 16)
    assert (lab[blk & (who == 1)] == 1).mean() > 0.95
    assert (lab[blk & (who == 0)] == 0).mean() > 0.95


def test_an_unlisted_horse_level_with_the_last_listed_one_is_not_omitted():
    """The omitted state needs positive evidence: a horse abreast of the 8th listed one (gap 0) stays in the ok state,
    so the list keeps ordering it; one a length and more ahead is omitted."""
    H, R = 10, 300
    slots = np.tile(np.arange(8), (R, 1))
    L = core.ListRows(np.arange(R) * 0.1, slots, np.ones(R, bool), np.zeros(R, bool))
    St = core.statements(L, H)
    d = np.where(St.unl & (St.b == 9), 0.0, 3.0)
    _J, _pf, om = core.list_chains(L, St, core._p_order(d)[1], H, d)
    assert om[9].max() < 0.1
    d = np.where(St.unl & (St.b == 9), -4.0, 3.0)
    _J, _pf, om = core.list_chains(L, St, core._p_order(d)[1], H, d)
    assert om[9, 50:250].min() > 0.9


def test_a_box_wholly_inside_the_list_band_is_an_icon():
    """Silk render: a box whose top lies below the list band's top is one of the list's icons (P_GHOST), whatever its
    confidence or tracklet length; a box straddling the band's edge is P_EDGE; a box in open grass P_HORSE0."""
    H = 535.0
    field = [[f, 100 + 40 * k, 330 + 10 * k, 30, 30, 0.9] for f in range(5) for k in range(8)]  # cy 330-400
    icon = [[f, 300, 470, 40, 50, 0.9] for f in range(5)]  # top 445
    edge = [[f, 500, 430, 30, 40, 0.9] for f in range(5)]  # top 410, bottom 450
    rows = np.array(field + icon + edge, float)
    p = core.p_horse(rows, np.zeros(len(rows)), np.full(len(rows), 500), H, True)
    lt = core.legend_top(rows[:, 2], H)
    assert 300 < lt < 440
    assert (p[:40] == core.P_HORSE0).all() and (p[40:45] == core.P_GHOST).all() and (p[45:] == core.P_EDGE).all()
