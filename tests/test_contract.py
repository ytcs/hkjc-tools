"""Output contract: free-running flag, E-step margins."""

import numpy as np
import polars as pl

from hkjc.recon import assoc, contract, fusion


def test_node_free_uses_one_length_and_the_abreast_spacing():
    t = np.array([0.0, 0.1])
    P = pl.DataFrame({"horse_no": [1, 1, 2, 2, 3, 3], "t": np.tile(t, 3),
                      "s": [10.0, 11.0, 11.0, 20.0, 30.0, 30.0], "lat": [1.0, 1.0, 3.5, 3.5, 1.0, 9.0]})
    f = contract.node_free(P)
    # 1 and 2 at t = 0: 1 m along and 2.5 render units across (abreast) -> neither is free; 9 m apart at t = 0.1
    assert f[1][1].tolist() == [False, True] and f[2][1].tolist() == [False, True]
    assert f[3][1].tolist() == [True, True]


def test_margins_are_read_only_and_rank_the_ambiguous_detection_lowest():
    tg = np.arange(0, 10.01, 0.1)
    paths = {
        1: fusion.Path(1, tg, 15.0 * tg, np.full_like(tg, 2.0), np.full_like(tg, 0.3),
                       np.full_like(tg, 0.2), np.ones_like(tg)),
        2: fusion.Path(2, tg, 15.0 * tg - 3.0, np.full_like(tg, 4.0), np.full_like(tg, 0.3),
                       np.full_like(tg, 0.2), np.ones_like(tg)),
    }
    t = np.array([5.0, 5.0, 5.0, 6.0])
    s = np.array([75.0, 72.0, 40.0, 88.5])  # horse 1, horse 2, clutter; at t = 6 one detection between the two
    lat = np.array([2.0, 4.0, 3.0, 3.0])
    fr = np.array([7, 7, 7, 8])
    lab0, chi0 = assoc.assign_frames(fr, t, s, lat, paths, np.zeros((0, 9), int))
    m = np.full(4, np.nan)
    lab, chi = assoc.assign_frames(fr, t, s, lat, paths, np.zeros((0, 9), int), margin_out=m)
    assert np.array_equal(lab, lab0) and np.array_equal(chi, chi0)
    assert lab.tolist()[:3] == [1, 2, 0]
    assert np.isnan(m[2]) and (m[[0, 1]] > 0).all()
    assert m[3] < m[0] and m[3] < m[1]


# ------------------------------------------------------------------------------------------- frozen running-order list
def _held(read, t):
    return np.tile(np.asarray(read, int), (len(t), 1))


def test_legend_frozen_marks_a_held_near_cloth_read_only():
    t = np.arange(0.0, 30.0, 0.1)
    live = np.array([(2, 11, 12, 10, 5, 6, 1, 4), (2, 11, 10, 12, 5, 6, 1, 4)] * 150)  # changes every 0.1 s
    S = live.copy()
    w = (t >= 10) & (t < 20)
    S[w] = (2, 1, 3, 4, 5, 6, 7, 8)  # one adjacent swap off cloth order, held 10 s
    S[(t >= 14) & (t < 14.5), 3] = 0  # an unread slot does not break the hold
    m, wins = contract.legend_frozen(S, t, 12)
    assert m.tolist() == w.tolist()
    assert len(wins) == 1 and wins[0]["read"] == [2, 1, 3, 4, 5, 6, 7, 8] and abs(wins[0]["t1"] - t[w][-1]) < 1e-9
    # a strung-out field: a non-cloth read held 30 s is a race reading
    assert not contract.legend_frozen(_held((5, 9, 2, 12, 1, 7, 3, 10), t), t, 12)[0].any()
    # a near-cloth read shorter than the hold, or one interrupted by another full read, is not marked
    S2 = live.copy()
    S2[(t >= 10) & (t < 10 + contract.FROZEN_HOLD - 0.5)] = (1, 2, 3, 4, 5, 6, 7, 8)
    assert not contract.legend_frozen(S2, t, 12)[0].any()
    S3 = _held((2, 1, 3, 4, 5, 6, 7, 8), t)
    S3[::30] = (2, 1, 3, 4, 5, 6, 7, 9)  # another (also near-cloth) full read every 3 s
    assert not contract.legend_frozen(S3, t, 12)[0].any()
    # a small field: a list of < FROZEN_MIN_READ slots is never judged
    assert not contract.legend_frozen(_held((1, 2, 3, 4, 5, 0, 0, 0), t), t, 5)[0].any()
    assert contract.legend_frozen(_held((1, 2, 3, 4, 5, 6, 0, 0), t), t, 6)[0].all()


def _silk_run(tmp_path, read, t_end=10.0):
    """One horse (#1) seen alone from 2 s to ``t_end`` at 10 Hz on the silk render; ``read`` the legend list at every
    frame (one list, or rows x 8)."""
    from types import SimpleNamespace

    D = 1000.0
    t = np.round(np.arange(2.0, t_end, 0.1), 2)
    fr = np.round(t * 30).astype(int)
    rows = np.zeros((len(t), 2), int)
    rows[:, 0] = fr
    members = np.empty(1, dtype=object)
    members[0] = np.arange(len(t))
    np.savez(tmp_path / "tracks.npz", t=t, d=D - (100 + 15 * t), rows=rows, members=members, lat=np.full(len(t), 2.0))
    np.savez(tmp_path / "det_labels.npz", labels=np.ones(len(t), int))
    np.savez(tmp_path / "det_labels_final.npz", labels=np.ones(len(t), int))
    read = np.asarray(read, int)
    leg = np.c_[fr, read if read.ndim == 2 else np.tile(read, (len(t), 1))]
    info = SimpleNamespace(distance=D, runners=tuple(range(1, 13)))
    run = SimpleNamespace(p=lambda n: tmp_path / n, legend_rows=lambda: leg, info=info)
    tg = np.round(np.arange(0.0, t_end + 2.0, 0.1), 2)
    P = pl.DataFrame({"horse_no": 1, "t": tg, "s": 100 + 15 * tg, "lat": 2.0, "sd_s": 0.3})
    return run, P


def test_frozen_list_reads_do_not_confirm_ranks(tmp_path, monkeypatch):
    # a live list that ranks #1 first confirms its stretch; the guard finds nothing to mark
    run, P = _silk_run(tmp_path, (1, 7, 3, 12, 5, 6, 2, 4))
    st, _ = contract.flag_stretches(run, P)
    assert [(s["horse"], s["n_leg_conf"]) for s in st] == [(1, 70)]
    monkeypatch.setattr(contract, "FROZEN_HOLD", np.inf)  # no guard: the same stretches
    assert contract.flag_stretches(run, P)[0] == st
    # a list frozen in cloth order also "ranks" #1 first; with the guard its frames are not reads
    run, P = _silk_run(tmp_path, (1, 2, 3, 4, 5, 6, 7, 8))
    assert [s["n_leg_conf"] for s in contract.flag_stretches(run, P)[0]] == [70]
    monkeypatch.setattr(contract, "FROZEN_HOLD", 5.0)
    assert contract.flag_stretches(run, P)[0] == []
    assert contract.frozen_windows(run)[0]["read"] == [1, 2, 3, 4, 5, 6, 7, 8]


def test_no_stretch_spans_a_frozen_window(tmp_path, monkeypatch):
    # a live list confirms #1 from 2 to 30 s except where it freezes in cloth order (10-20 s). Without the guard one
    # stretch runs through the window; with it the sides stand alone, and the 1.4 s after the window is too short
    t = np.round(np.arange(2.0, 30.0, 0.1), 2)
    reads = np.tile((1, 7, 3, 12, 5, 6, 2, 4), (len(t), 1))
    reads[(t >= 10) & (t < 20)] = (1, 2, 3, 4, 5, 6, 7, 8)
    keep = (t < 21.5) | (t > 26.0)  # #1 unseen at 21.5-26.0 s
    run, P = _silk_run(tmp_path, reads, t_end=30.0)
    T = dict(np.load(tmp_path / "tracks.npz", allow_pickle=True))
    members = np.empty(1, dtype=object)
    members[0] = np.arange(int(keep.sum()))
    np.savez(tmp_path / "tracks.npz", t=T["t"][keep], d=T["d"][keep], rows=T["rows"][keep], members=members,
             lat=T["lat"][keep])
    for f in ("det_labels.npz", "det_labels_final.npz"):
        np.savez(tmp_path / f, labels=np.ones(int(keep.sum()), int))
    monkeypatch.setattr(contract, "FROZEN_HOLD", np.inf)
    assert [(s["t0"], s["t1"]) for s in contract.flag_stretches(run, P)[0]] == [(3.0, 21.4), (26.1, 29.9)]
    monkeypatch.setattr(contract, "FROZEN_HOLD", 5.0)
    assert [(s["t0"], s["t1"]) for s in contract.flag_stretches(run, P)[0]] == [(3.0, 9.9), (26.1, 29.9)]


# -------------------------------------------------------------------------------------------------- position flag
def _pos_case(withheld=None):
    from types import SimpleNamespace

    D, v = 1200.0, 16.0
    bnd = [800.0, 400.0, 0.0]
    off = {1: 0.0, 2: 0.5, 3: 1.0, 4: 1.5}  # official time behind horse 1 at every boundary (s)
    cum = {h: np.array([(D - b) / v + o for b in bnd]) for h, o in off.items()}
    if withheld is not None:
        for c in cum.values():
            c[withheld] = np.nan
    info = SimpleNamespace(distance=D, official_boundaries=lambda: (bnd, cum))
    tg = np.round(np.arange(0.0, 80.0, 0.1), 2)
    vid_shift = {1: 0.0, 2: 0.0, 3: 0.0, 4: 3.0}  # horse 4's video-only path runs 3 m ahead of its official position
    P = pl.concat([pl.DataFrame({"horse_no": h, "t": tg, "s": np.maximum(v * (tg - o), 0.0), "sd_s": 0.4})
                   for h, o in off.items()])
    Pv = pl.concat([pl.DataFrame({"horse_no": h, "t": tg, "s": np.maximum(v * (tg - o), 0.0) + vid_shift[h]})
                    for h, o in off.items()])
    dets = {h: tg[(tg > 5) & ~((h == 2) & (tg > 49.8) & (tg < 60))] for h in off}  # horse 2 unseen at 49.9-59.9 s
    return SimpleNamespace(info=info), P, Pv, dets


def test_position_crossings_check_identity_field_relative():
    run, P, Pv, dets = _pos_case()
    cr = contract.position_crossings(run, P, Pv, dets)
    assert len(cr) == 12 and all(c["firm"] for c in cr)
    by = {(c["horse"], c["j"]): c for c in cr}
    # the boundary median (three horses at 0, one at -3 m) is removed: horses 1-3 sit at 0; horse 4 is 3 m ahead of its
    # official position, i.e. early: r_v = -3 m (late / behind is +)
    assert all(abs(by[(h, j)]["r_v"]) < 1e-6 and by[(h, j)]["pass"] for h in (1, 3) for j in range(3))
    assert all(abs(by[(4, j)]["r_v"] + 3.0) < 1e-6 and not by[(4, j)]["pass"] for j in range(3))
    # horse 2 is not observed at its 400 m to go crossing (50.5 s; nearest detection 49.8 s)
    assert not by[(2, 1)]["observed"] and by[(2, 1)]["id_ok"] and not by[(2, 1)]["pass"]
    # a withheld boundary is not seen
    run, P, Pv, dets = _pos_case(withheld=1)
    assert {c["j"] for c in contract.position_crossings(run, P, Pv, dets)} == {0, 2}


def test_node_pos_ok_reads_the_node_and_the_bounding_boundaries():
    run, P, Pv, dets = _pos_case()
    cr = contract.position_crossings(run, P, Pv, dets)
    for c in cr:  # horse 1 fails the identity check at the last boundary (the line) only
        if c["horse"] == 1 and c["j"] == 2:
            c["id_ok"] = False
    pos = contract.node_pos_ok(cr, P, dets)
    t, s, h = P["t"].to_numpy(), P["s"].to_numpy(), P["horse_no"].to_numpy()
    seen, unseen = t > 5.0, t < 4.5  # first detection 5.1 s
    # horse 3: observed and firm from 5 s on, identity good at every boundary
    assert pos[(h == 3) & seen].all() and not pos[(h == 3) & unseen].any()
    # horse 4: identity fails everywhere -> never pos_ok
    assert not pos[h == 4].any()
    # horse 1: pos_ok up to the 400 m to go boundary; between it and the line, and beyond the line, the line's check fails
    m1 = (h == 1) & seen
    assert (pos[m1] == (s[m1] < 800.0)).all()
    # horse 2: not observed at 50-60 s -> not pos_ok there, though its boundaries' identity checks hold
    m2 = h == 2
    assert not pos[m2 & (t > 50.6) & (t < 59.4)].any() and pos[m2 & (t > 20) & (t < 49)].all()



def test_axis_core_var_is_pinned_at_anchors_and_grows_away_from_them():
    from hkjc.recon import contract as ct

    a = [400.0, 800.0, 1200.0]
    v = ct.axis_core_var([400.0, 500.0, 600.0, 1300.0, 1600.0], a)
    assert abs(v[0]) < 1e-9 and 0 < v[1] < v[2] and 0 < v[3] < v[4] < ct.AXIS_S2
    assert np.isnan(ct.axis_core_var([500.0], [])).all()
    mix = 1 - ct.AXIS_PI + ct.AXIS_PI * ct.AXIS_K ** 2
    assert np.allclose(ct.axis_var([600.0], a), ct.axis_core_var([600.0], a) * mix)


def test_uncertainty_columns_add_in_quadrature_and_the_gap_grows_with_time_since_the_last_label():
    from hkjc.recon import contract as ct

    t = np.array([10.0, 11.0, 14.0, 20.0])
    s = 15.0 * t
    out = ct.uncertainty_columns(s, t, np.full(4, 0.2), [150.0, 300.0], "silk", np.array([10.0, 11.0]))
    assert np.allclose(ct.gap_dt(t, [10.0, 11.0]), [0.0, 0.0, 3.0, 9.0])
    v = ct.GAP_SA ** 2 * np.array([0.0, 0.0, 27.0, 729.0]) / 3
    assert np.allclose(out["sd_s_gap"] ** 2, v / (1 + v / ct.GAP_CAP ** 2))
    assert np.allclose(out["sd_s_ident"], 0.0)
    assert np.allclose(out["sd_s_rel"] ** 2, 0.04 + ct.HORSE_VAR["silk"] + out["sd_s_gap"] ** 2)
    assert np.all(np.diff(out["sd_s_gap"]) >= 0) and ct.gap_var([1e4])[0] < ct.GAP_CAP ** 2
    idn = ct.uncertainty_columns(s, t, np.full(4, 0.2), [150.0, 300.0], "silk", np.array([10.0, 11.0]), np.full(4, 0.5))
    assert np.allclose(idn["sd_s_rel"] ** 2, out["sd_s_rel"] ** 2 + 0.5)
    assert np.allclose(out["sd_s_total"] ** 2, out["sd_s_rel"] ** 2 + out["sd_s_axis"] ** 2)
    assert np.isnan(ct.uncertainty_columns(s, t, np.full(4, 0.2), [150.0], "circle", np.zeros(0))["sd_s_total"]).all()


# ------------------------------------------------------------------------------------------------- identity robustness
def test_id_thresholds_are_the_median_of_the_two_folds():
    from hkjc.recon import contract as ct

    for k, (a, b) in ct.ID_FOLDS.items():
        assert ct.ID_THRESHOLDS[k] == ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
    assert abs(ct.ID_THRESHOLDS[("silk", "section")][0] - 0.0496) < 1e-4
    assert abs(ct.ID_THRESHOLDS[("circle", "bend")][0] - 0.8697) < 1e-4
    lv, lf = np.array([1, 1, 1, 2, 0]), np.array([1, 1, 2, 2, 1])
    # horse 1: labelled by either at rows 0, 1, 2, 4; by exactly one at rows 2, 4
    assert ct.relabel_share(lv, lf, 1, np.ones(5, bool)) == 0.5 and ct.relabel_share(lv, lf, 3, np.ones(5, bool)) == 0.0


def test_id_robust_reads_neighbours_relabels_and_read_support(tmp_path):
    from types import SimpleNamespace

    from hkjc.recon import contract as ct

    D = 800.0
    tg = np.round(np.arange(0.0, 60.0, 0.1), 3)
    s = {1: 15.0 * tg, 2: 15.0 * tg, 3: 15.0 * tg - 20.0}
    lat = {1: 1.0, 2: 2.0, 3: 1.0}
    P = pl.DataFrame({"horse_no": np.repeat([1, 2, 3], len(tg)), "t": np.tile(tg, 3),
                      "s": np.concatenate([s[h] for h in (1, 2, 3)]),
                      "lat": np.concatenate([np.full(len(tg), lat[h]) for h in (1, 2, 3)]), "n_obs": np.ones(3 * len(tg))})
    live = np.concatenate([(s[h] > 0) & (s[h] < D) for h in (1, 2, 3)])
    t = P["t"].to_numpy()[live]
    d = D - P["s"].to_numpy()[live]
    h = P["horse_no"].to_numpy()[live]
    lf = h.copy()
    sec0_2 = np.flatnonzero((h == 2) & (d >= 400))
    lf[sec0_2[::3]] = 0  # a third of horse 2's first-section detections lose their label in the final pass
    num = h.copy()
    num[(h == 3) & (d < 400)] = 0  # no disc read of horse 3 in the second section
    np.savez(tmp_path / "tracks.npz", t=t, d=d)
    np.savez(tmp_path / "det_labels.npz", labels=h)
    np.savez(tmp_path / "det_labels_final.npz", labels=lf)
    (tmp_path / "numbers.npz").write_bytes(b"")
    info = SimpleNamespace(distance=D, official_boundaries=lambda: ([400.0, 0.0], {}),
                           geometry=SimpleNamespace(bend_segments=((100.0, 300.0),), circumference=1000.0))
    run = SimpleNamespace(info=info, p=lambda n: tmp_path / n, _number_reads=lambda T: num)
    node, secs, bends, yld = ct.id_robust(run, P)
    v = {(r["horse"], r["section"]): r["robust"] for r in secs}
    # horse 1 runs abreast of the relabelled horse 2 -> not robust in section 0; horse 3 is 20 m back -> robust
    assert v == {(1, 0): False, (2, 0): False, (3, 0): True, (1, 1): True, (2, 1): True, (3, 1): False}
    r30 = next(r for r in secs if (r["horse"], r["section"]) == (3, 0))
    assert r30["Rmax"] == 0.0 and r30["P"] == 1.0
    assert next(r for r in secs if (r["horse"], r["section"]) == (1, 0))["Rmax"] > 0.3
    # the column: the node's section verdict; no unit (before the render start, after the line) -> False
    hh, ss, tt = P["horse_no"].to_numpy(), P["s"].to_numpy(), P["t"].to_numpy()
    inside = (tt > 0.5) & (ss < D)
    assert node[(hh == 1) & inside & (ss > 400)].all() and not node[(hh == 1) & (ss <= 400)].any()
    assert not node[~inside].any()
    # bends: horse 2's relabels are all before the bend -> every horse robust there (circle read support 1)
    assert [b["robust"] for b in bends] == [True, True, True]
    assert yld["section_units"] == 6 and yld["section_units_robust"] == 3 and yld["rows_robust"] == int(node.sum())
    js = ct.describe_id_robust(secs, bends, yld)
    assert js["yield"] == yld and len(js["thresholds"]) == 4


def test_ident_var_is_eps_times_the_mean_squared_offset_to_the_two_nearest_neighbours_across_within_the_lane():
    import polars as pl

    from hkjc.recon import contract as ct

    t = np.array([1.0, 2.0])
    rows = [(1, 100.0, 0.0), (2, 103.0, 1.0), (3, 96.0, 0.5), (4, 101.0, 9.0)]  # 4 is beyond FREE_DLAT across
    P = pl.DataFrame({"horse_no": np.repeat([r[0] for r in rows], 2), "t": np.tile(t, 4),
                      "s": np.repeat([r[1] for r in rows], 2), "lat": np.repeat([r[2] for r in rows], 2)})
    v = ct.ident_var(P)
    assert np.allclose(v[1][1], ct.IDENT_EPS * (9.0 + 16.0) / 2)
    assert np.allclose(v[4][1], 0.0)  # no neighbour within the lane
    assert np.allclose(v[2][1], ct.IDENT_EPS * (9.0 + 49.0) / 2)


# ------------------------------------------------------------------------------------------------ lateral-metre error
def test_lat_m_err_is_provisional_sha_tin_bend_and_home_only_and_adds_in_quadrature():
    from hkjc.data import Mask, MaskedRace, RaceInfo, Runner
    from hkjc.recon import contract as ct
    from hkjc.recon.pipeline import RaceRun

    assert ct.LAT_M_ERR == {("ST", "bend"): (0.98 / 1.645, 0.25), ("ST", "home"): (0.95 / 1.645, 0.0)}
    seg = np.array(["start", "straight", "bend", "home", "bend", ""])
    sd = np.array([0.03, 0.03, 0.04, 0.05, np.nan, np.nan])  # lat_m undefined at the last two nodes
    e = ct.lat_m_err("ST", seg, sd)
    hb, hh = 0.98 / 1.645, 0.95 / 1.645
    assert round(hb, 2) == 0.60 and round(hh, 2) == 0.58
    assert np.allclose(e["sd_lat_m_horse"][2:4], [hb, hh]) and np.allclose(e["sd_lat_m_node"][2:4], [0.25, 0.0])
    assert np.allclose(e["sd_lat_m_total"][2:4] ** 2,
                       sd[2:4] ** 2 + e["sd_lat_m_horse"][2:4] ** 2 + e["sd_lat_m_node"][2:4] ** 2)
    for k in e:  # start and straight: no value although lat_m is defined; NaN where lat_m is NaN
        assert np.isnan(e[k][[0, 1, 4, 5]]).all()
    assert all(np.isnan(v).all() for v in ct.lat_m_err("HV", seg, sd).values())  # Happy Valley: none
    # segments as the pipeline derives them (ST disc, home straight 430 m): start, straight, bend, home, past the post
    info = RaceInfo("20260913", "ST", 1, 1200.0, "B", "turf", "GOOD", "Class 4", (Runner(1, "H1", "X001", 1),),
                    pl.DataFrame(), pl.DataFrame())
    run = RaceRun.__new__(RaceRun)
    run.info = MaskedRace(info, Mask())
    s2 = run.lat_m_segment(np.array([1100.0, 1000.0, 700.0, 200.0, -0.1]), np.array([12.0, 20.0, 40.0, 62.0, 71.0]), 11.0)
    assert s2.tolist() == ["start", "straight", "bend", "home", ""]
    e2 = ct.lat_m_err("ST", s2, np.full(5, 0.03))
    assert np.isfinite(e2["sd_lat_m_total"]).tolist() == [False, False, True, True, False]


def test_describe_lists_the_provisional_lat_m_err_values():
    from hkjc.recon import contract as ct

    d = ct.describe([])["lat_m_err"]
    assert d["sd_m"] == {"ST bend": {"horse": 0.98 / 1.645, "node": 0.25}, "ST home": {"horse": 0.95 / 1.645, "node": 0.0}}
    assert "provisional" in d["status"]
