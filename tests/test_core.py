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


def test_refit_label_and_detection_terms_are_exact_increments():
    """Refit's incremental label and detection terms equal label_terms / miss_terms of the new labelling at the same
    paths."""
    rows, t, s, lat, who = _race(n_h=3, speeds=[16.0, 16.0, 15.0], start_s=[100.0, 97.0, 94.0])
    members = [np.flatnonzero(who == h) for h in range(3)]
    horses = np.array([1, 2, 3])
    ev = _evidence(rows, t, s, lat, members, horses)
    m = core.Model(ev, core.list_rows(None, lambda f: f, horses), np.arange(1.0, 20.0, core.DT), width=20.0)
    lab = np.array([who[ev.seg == k][0] for k in range(ev.n_seg)])
    st = core.evaluate(m, lab)
    rf = core.Refit(m, st, lab)
    new = lab.copy()
    new[lab == 0], new[lab == 1] = 1, 0
    new[np.flatnonzero(lab == 2)[:1]] = -1
    ch = np.flatnonzero(new != lab)
    assert abs(rf.d_lab(ch, new) - (core.label_terms(ev, new) - core.label_terms(ev, lab))) < 1e-6
    if st.pin.shape[1]:
        hs = np.array([0, 1, 2])
        S, L = st.S[[1, 0, 2]], st.L[[1, 0, 2]]
        pin = core.in_view(ev, S, L, st.sdS, st.sdL, m.tg)
        ref = core.miss_terms(ev, pin, new)[0] - st.parts["J_det"]
        assert abs(rf.d_det(hs, new, S, L) - ref) < 1e-6


def test_probabilities_come_from_the_alternative_set_alone():
    """A segment's probability is its kept label's weight 1 against exp(-dJ / (2 T_PROB)) per alternative; a segment
    no alternative changes keeps P 1."""
    lab = np.array([0, 1, -1])
    alt = np.array([1, 0, -1])  # exchanges the first two segments
    P = core.alternative_posterior(lab, [(alt, 2 * core.T_PROB * np.log(3.0))], 2)
    assert np.allclose(P[0], [0.75, 0.25, 0.0]) and np.allclose(P[1], [0.25, 0.75, 0.0])
    assert np.allclose(P[2], [0.0, 0.0, 1.0])



def test_an_assignment_with_the_same_inputs_is_solved_once():
    """core.milp memoises by a digest of every input: the same inputs return the first solve's labelling without a new
    solve; any changed input (here one exclusion) is solved afresh."""
    from types import SimpleNamespace
    from unittest import mock

    ev = SimpleNamespace(links=np.zeros((0, 3)), seg_lo=np.array([0, 0]), seg_hi=np.array([9, 9]),
                         forbid=np.zeros((2, 2), bool))
    U = np.array([[-5.0, -50.0], [-40.0, -1.0]])
    core._MILP_MEMO.clear()
    with mock.patch.object(core, "_milp_solve", wraps=core._milp_solve) as solve:
        a = core.milp(U, ev)
        b = core.milp(U.copy(), ev)
        assert solve.call_count == 1 and a.tolist() == b.tolist() == [1, 0]
        ev.forbid = np.array([[False, True], [False, False]])
        assert core.milp(U, ev).tolist() == [-1, 0] and solve.call_count == 2



def test_every_solved_assignment_records_its_highs_status_gap_nodes_and_wall_time():
    """core.MILP_STATS gets one (status, gap, nodes, wall s) per MILP solved (a memo hit adds none); status 0 is an
    optimum, 1 a stop at MILP_NODE_LIMIT."""
    from types import SimpleNamespace

    ev = SimpleNamespace(links=np.zeros((0, 3)), seg_lo=np.array([0, 0]), seg_hi=np.array([9, 9]),
                         forbid=np.zeros((2, 2), bool))
    U = np.array([[-6.0, -51.0], [-41.0, -2.0]])
    core._MILP_MEMO.clear()
    n0 = len(core.MILP_STATS)
    core.milp(U, ev)
    core.milp(U, ev)
    assert len(core.MILP_STATS) == n0 + 1
    status, gap, nodes, wall = core.MILP_STATS[-1]
    assert status == 0 and gap == 0.0 and nodes >= 0 and wall >= 0.0



def test_the_assignment_reaches_the_exhaustive_optimum_with_branching_links():
    """core.milp (with the link-aggregation rows) returns a labelling whose objective equals the exhaustive optimum of
    the base problem (one horse per segment, one segment per horse per frame, a link's reward when both ends hold the
    same horse) on an instance where a segment links to two overlapping successors."""
    import itertools
    from types import SimpleNamespace

    lo, hi = np.array([0, 10, 10, 20, 20]), np.array([9, 19, 19, 29, 29])
    links = np.array([[0, 1, 4.0], [0, 2, 4.0], [1, 3, 4.0], [2, 4, 4.0], [1, 4, 1.0], [2, 3, 1.0]])
    ev = SimpleNamespace(links=links, seg_lo=lo, seg_hi=hi, forbid=np.zeros((5, 2), bool))
    U = np.array([[-1.0, -0.5], [-0.2, -0.9], [-0.8, -0.1], [-0.3, -0.6], [-0.7, -0.2]])

    def obj(lab):
        o = sum(U[k, h] for k, h in enumerate(lab) if h >= 0)
        return o - sum(w for a, b, w in links if lab[int(a)] >= 0 and lab[int(a)] == lab[int(b)])

    best = np.inf
    for lab in itertools.product((-1, 0, 1), repeat=5):
        if any(lab[a] >= 0 and lab[a] == lab[b] and lo[a] <= hi[b] and lo[b] <= hi[a]
               for a in range(5) for b in range(a + 1, 5)):
            continue
        best = min(best, obj(lab))
    core._MILP_MEMO.clear()
    assert np.isclose(obj(core.milp(U, ev).tolist()), best)



def _milp_problem_reference(U, ev, allow=None):
    """The segment MILP's arrays built row by row (the plain loops core._milp_problem vectorises)."""
    from scipy.sparse import coo_matrix

    S, H = U.shape
    links = ev.links
    reward = np.zeros(S)
    if len(links):
        np.add.at(reward, links[:, 0].astype(int), links[:, 2])
        np.add.at(reward, links[:, 1].astype(int), links[:, 2])
    ok = np.isfinite(U) & (U < core.INF / 2) & (~ev.forbid) & (U <= reward[:, None])
    if allow is not None:
        ok &= allow
    vs, vh = np.nonzero(ok)
    nx = len(vs)
    vid = np.full((S, H), -1)
    vid[vs, vh] = np.arange(nx)
    rows, cols = [vs.copy()], [np.arange(nx)]
    r = S
    lo, hi = ev.seg_lo, ev.seg_hi
    for x in np.unique(lo):
        cl = np.flatnonzero((lo <= x) & (hi >= x))
        if len(cl) > 1:
            sub = vid[cl]
            for h in np.flatnonzero((sub >= 0).sum(0) > 1):
                v = sub[:, h][sub[:, h] >= 0]
                rows.append(np.full(len(v), r))
                cols.append(v)
                r += 1
    n_ub1 = r
    zc, za, zb = [], [], []
    succ, pred = {}, {}
    for a, b, w in links:
        a, b = int(a), int(b)
        for h in np.flatnonzero((vid[a] >= 0) & (vid[b] >= 0)):
            succ.setdefault((a, int(h)), []).append((b, len(zc)))
            pred.setdefault((b, int(h)), []).append((a, len(zc)))
            zc.append(-w)
            za.append(vid[a, h])
            zb.append(vid[b, h])
    nz = len(zc)
    zi = nx + np.arange(nz)
    if nz:
        rr = r + np.arange(2 * nz)
        rows += [rr, rr]
        cols += [np.repeat(zi, 2), np.ravel(np.c_[za, zb])]
        r += 2 * nz
    rows = np.concatenate(rows)
    cols = np.concatenate(cols)
    vals = np.ones(len(rows))
    if nz:
        vals[-2 * nz:] = -1.0
    ar, ac, av = [], [], []
    for groups in (succ, pred):
        for (a, h), lst in groups.items():
            rest = sorted(lst, key=lambda p: (lo[p[0]], p[1]))
            while len(rest) > 1:
                cl, keep = [rest[0]], []
                for p in rest[1:]:
                    if all(lo[p[0]] <= hi[q[0]] and lo[q[0]] <= hi[p[0]] for q in cl):
                        cl.append(p)
                    else:
                        keep.append(p)
                rest = keep
                if len(cl) > 1:
                    ar += [r] * (len(cl) + 1)
                    ac += [nx + k for _b, k in cl] + [vid[a, h]]
                    av += [1.0] * len(cl) + [-1.0]
                    r += 1
    if ar:
        rows, cols, vals = np.r_[rows, ar], np.r_[cols, ac], np.r_[vals, av]
    A = coo_matrix((vals, (rows, cols)), shape=(r, nx + nz)).tocsr()
    return np.r_[U[vs, vh], np.asarray(zc, float)], A, np.r_[np.ones(n_ub1), np.zeros(r - n_ub1)], vs, vh


def test_the_vectorised_milp_builder_gives_the_row_by_row_arrays_exactly():
    """core._milp_problem returns the same c, CSR A (indptr, indices, data) and ub, element for element and in the same
    order, as the row-by-row construction, on random instances with overlapping segments and branching links."""
    from types import SimpleNamespace

    rng = np.random.default_rng(0)
    for _ in range(20):
        S, H = int(rng.integers(5, 40)), int(rng.integers(2, 8))
        lo = rng.integers(0, 60, S)
        hi = lo + rng.integers(0, 15, S)
        L = int(rng.integers(0, 3 * S))
        links = np.c_[rng.integers(0, S, (L, 2)), rng.choice([1.0, 2.0, 4.0], L)] if L else np.zeros((0, 3))
        ev = SimpleNamespace(links=links, seg_lo=lo, seg_hi=hi, forbid=rng.random((S, H)) < 0.1)
        U = rng.normal(-1.0, 3.0, (S, H))
        allow = rng.random((S, H)) < 0.8 if rng.random() < 0.5 else None
        got, ref = core._milp_problem(U, ev, allow), _milp_problem_reference(U, ev, allow)
        if got is None:
            assert len(ref[3]) == 0
            continue
        for g, f in zip(got[:1] + got[2:], ref[:1] + ref[2:]):
            assert np.array_equal(g, f)
        assert np.array_equal(got[1].indptr, ref[1].indptr) and np.array_equal(got[1].indices, ref[1].indices)
        assert np.array_equal(got[1].data, ref[1].data) and got[1].shape == ref[1].shape
