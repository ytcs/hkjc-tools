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
    _J, pf, om, _pw = core.list_chains(L, St, core._p_order(d)[1], H, d)
    assert om[9, 50:250].min() > 0.9
    assert om[8, 50:250].max() < 0.1
    # every adjacent pair contradicted on a list held identical: frozen
    d2 = np.where(St.unl, 3.0, -3.0)
    _J2, pf2, _om2, _pw2 = core.list_chains(L, St, core._p_order(d2)[1], H, d2)
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
    _J, _pf, om, _pw = core.list_chains(L, St, core._p_order(d)[1], H, d)
    assert om[9].max() < 0.1
    d = np.where(St.unl & (St.b == 9), -4.0, 3.0)
    _J, _pf, om, _pw = core.list_chains(L, St, core._p_order(d)[1], H, d)
    assert om[9, 50:250].min() > 0.9


def test_a_pair_the_list_orders_wrongly_for_a_stretch_costs_its_entry_not_every_row():
    """Rows that keep the order of every pair except one (horses 4 and 5 stated 4 ahead of 5 while 5 is 3 m ahead):
    the pair's order-wrong chain switches on, so each further row costs a coin flip, not a misread, and no other pair
    is called wrong."""
    H = 8

    def cost(R):
        slots = np.tile(np.arange(8), (R, 1))
        L = core.ListRows(np.arange(R) * 0.1, slots, np.ones(R, bool), np.zeros(R, bool))
        St = core.statements(L, H)
        d = np.where((St.a == 4) & (St.b == 5), -3.0, 3.0)
        J, _pf, _om, pw = core.list_chains(L, St, core._p_order(d, St.eps)[1], H, d)
        J0, *_ = core.list_chains(L, St, core._p_order(np.abs(d), St.eps)[1], H, np.abs(d))
        return J - J0, pw, St

    j100, pw, St = cost(100)
    j300, _, _ = cost(300)
    bad = (St.a == 4) & (St.b == 5)
    assert pw[bad][20:].min() > 0.9 and pw[~bad].max() < 0.1
    flip, misread = -2 * np.log(0.5 / (1 - 0.5 * core.EPS_LIST)), -2 * np.log(0.5 * core.EPS_LIST)
    assert (j300 - j100) / 200 < 1.1 * flip  # a coin flip per further row
    assert j100 < 0.3 * 100 * misread  # far below the iid misread cost of every row


def test_a_read_with_a_dropped_digit_partner_is_a_weaker_fact_and_does_not_end_an_omission():
    """Runners 1 and 11: a slot read '1' may be 11 with a digit dropped, so its facts carry P_DROP more misread
    probability, and runner 1 (truly far ahead, unlisted) stays omitted through rows that read '1'."""
    horses = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 11])
    P = core.drop_partners(horses)
    assert P[0, 9] and P[9, 0] and not P[1, 9]
    R = 300
    slots = np.tile(np.arange(1, 9), (R, 1))  # columns 1..8 (numbers 2..9) listed; column 0 (#1) and 9 (#11) not
    slots[100:300:20, 7] = 0  # ten isolated rows read "1" in the last slot (where 11 would be)
    L = core.ListRows(np.arange(R) * 0.1, slots, np.ones(R, bool), np.zeros(R, bool), P)
    St = core.statements(L, len(horses))
    assert St.eps[(St.row == 100) & (St.b == 0)].min() > core.EPS_LIST
    d = np.where(St.unl & (St.b == 0), -6.0, 3.0)
    _J, _pf, om, _pw = core.list_chains(L, St, core._p_order(d, St.eps)[1], len(horses), d)
    assert om[0, 50:250].min() > 0.5


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


def test_pair_exchanges_cover_every_pair_over_the_whole_race():
    """Any two labelled horses get a whole-race exchange, however far apart they run; every exchange keeps the
    labels' multiset."""
    rows, t, s, lat, who = _race(n_h=3, gap_lat=3.0, start_s=[100.0, 90.0, 80.0])
    members = [np.flatnonzero(who == h) for h in range(3)]
    horses = np.array([1, 2, 3])
    ev = _evidence(rows, t, s, lat, members, horses)
    m = core.Model(ev, core.list_rows(None, lambda f: f, horses), np.arange(1.0, 20.0, core.DT), width=20.0)
    lab = np.array([who[ev.seg == k][0] for k in range(ev.n_seg)])
    st = core.evaluate(m, lab)
    cands = core.pair_exchanges(ev, lab, st)
    assert {d.split(" over ")[0] for d, _ in cands} == {"exchange #1/#2", "exchange #1/#3", "exchange #2/#3"}
    for _d, new in cands:
        assert sorted(new.tolist()) == sorted(lab.tolist())


def test_a_chain_move_carries_a_tracklets_clutter_continuation():
    """A tracklet cut at a camera transition is one tracker chain; when its tail is clutter, the chain move carrying the
    chain's label over its continuation labels the tail too."""
    rows, t, s, lat, who = _race(n_h=2, gap_lat=3.0, start_s=[100.0, 90.0])
    members = [np.flatnonzero(who == h) for h in range(2)]
    horses = np.array([1, 2])
    cam = int(rows[np.searchsorted(t, 10.0), 0])
    ev = _evidence(rows, t, s, lat, members, horses, n_cam=(cam,))
    succ, chains = core.tracker_chains(ev)
    assert all(len(c) == 2 for c in chains)  # each horse's tracklet: two segments joined across the transition
    m = core.Model(ev, core.list_rows(None, lambda f: f, horses), np.arange(1.0, 20.0, core.DT), width=20.0)
    lab = np.array([who[ev.seg == k][0] for k in range(ev.n_seg)])
    tail = chains[0][1]
    lab[tail] = -1
    st = core.evaluate(m, lab)
    full = lab.copy()
    full[tail] = lab[chains[0][0]]
    assert any(np.array_equal(new, full) for _d, new in core.chain_moves(ev, lab, st))


def test_refit_scores_track_the_exact_change_of_j():
    """At a labelling with the tracker swap left in (each tracklet one horse throughout), the one-horse-refit and
    closed-form scores of undoing the swap agree in sign with the exact J change, and the refit score of a relabel
    that leaves the labelling unchanged is 0."""
    rows, t, s, lat, who = _race(n_h=2, speeds=[16.0, 16.0], start_s=[100.0, 96.0], gap_lat=1.5)
    a = np.flatnonzero(((who == 0) & (t < 10)) | ((who == 1) & (t >= 10)))
    b = np.flatnonzero(((who == 1) & (t < 10)) | ((who == 0) & (t >= 10)))
    horses = np.array([3, 7])
    ev = _evidence(rows, t, s, lat, [a, b], horses)
    fr = np.unique(rows[:, 0])
    leg = np.c_[fr, np.tile([3, 7, 0, 0, 0, 0, 0, 0], (len(fr), 1))]
    Lr = core.list_rows(leg, lambda f: np.asarray(f, float) * 0.1 + 1.0, horses)
    m = core.Model(ev, Lr, np.arange(1.0, 20.0, core.DT), width=20.0)
    right = np.array([np.bincount(who[ev.seg == k], minlength=2).argmax() for k in range(ev.n_seg)])
    late = np.array([ev.t[ev.seg == k].min() >= 10 for k in range(ev.n_seg)])
    wrong = np.where(late, 1 - right, right)
    st = core.evaluate(m, wrong)
    rf = core.Refit(m, st, wrong)
    exact = core.evaluate(m, right).J - st.J
    assert exact < -core.MIN_GAIN
    assert rf.score(right) < -core.MIN_GAIN and rf.score_quick(right) < -core.MIN_GAIN
    assert rf.score(wrong) == 0.0
