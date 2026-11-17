"""Visual validation — look at the pixels (numbers hide CV failures).

  reproject(run, frames)   the world reconstruction drawn back onto raw render frames: racing rail
                           (magenta) + 5/10 m offset guides (cyan) + each detection with its fused
                           horse number and lateral metres
  posecheck(run, frames)   the static world map warped back through each frame's pose, blended 50/50
                           with the raw panel (misregistration shows as doubled rails/markers)
  straightened(run)        the map straightened along the rail stations (s -> x, inward -> y) with
                           the fitted rail and all horse points, in rows
  paths_plot(run)          fused paths: lateral vs distance-to-go, and gap-behind-leader vs time
  sync_video(run, out, offsets)  the four replay feeds time-synced on the race clock beside the
                           reconstruction (course overview from the measured rail + a straightened
                           field view: metres behind the leader x lateral off the rail)
Outputs go to ``<run dir>/viz/``.
"""

from __future__ import annotations

import numpy as np

from . import rails, video


def _out(run, name):
    d = run.dir / "viz"
    d.mkdir(exist_ok=True)
    return str(d / name)


def reproject(run, want):
    import cv2

    P = np.load(run.p("pose.npz"))
    z = np.load(run.p("rail.npz"))
    m = np.load(run.p("metric.npz"))
    T = np.load(run.p("tracks.npz"), allow_pickle=True)
    lab = np.load(run.p("labels.npz"))["labels"]
    ppm = float(m["ppm"])
    rf = rails.RailFrame.from_points(z["rail"], int(z["inner"]))
    k_of = {int(f): k for k, f in enumerate(P["idx"])}
    det_lab = np.zeros(len(T["d"]), int)
    for k, mem in enumerate(T["members"]):
        det_lab[np.asarray(mem, int)] = lab[k]
    if run.p("det_labels.npz").exists():  # detection-level EM identity supersedes fragment labels
        det_lab = np.load(run.p("det_labels.npz"))["labels"]
    want = sorted(want)
    outs = []
    for i, p in video.iter_unique_render(run.video(), start=min(want) - 3, stop=max(want) + 4):
        if not any(abs(i - w) <= 2 for w in want) or i not in k_of:
            continue
        want = [w for w in want if abs(i - w) > 2]
        k = k_of[i]
        Gi = np.linalg.inv(P["G"][k])
        img = p.copy()
        st = z["st_of_frame"][k]
        sel = slice(max(0, st - 500), min(len(rf.R), st + 500))
        # straight courses (Z6): the polyline is the field-edge DATUM; the racing rail drawn (and the
        # lateral reference) is the fence measured lat0_px inside it (rails.straight_datum)
        lat0 = float(z["lat0_px"]) if "lat0_px" in z.files else 0.0
        if lat0:
            W = rf.R[sel]
            uv = (np.c_[W, np.ones(len(W))] @ Gi.T)[:, :2].astype(np.int32)
            cv2.polylines(img, [uv], False, (160, 160, 160), 1)
        for off, col in ((0, (255, 0, 255)), (5 * ppm, (255, 255, 0)), (10 * ppm, (255, 255, 0))):
            W = rf.R[sel] + (off - lat0) * rf.Nout[sel]
            uv = (np.c_[W, np.ones(len(W))] @ Gi.T)[:, :2].astype(np.int32)
            cv2.polylines(img, [uv], False, col, 2 if off == 0 else 1)
        rows = T["rows"]
        for j in np.where(rows[:, 0] == i)[0]:
            cx, cy = int(rows[j, 1]), int(rows[j, 2])
            cv2.circle(img, (cx, cy), 4, (0, 0, 255), -1)
            tag = f"#{det_lab[j]}" if det_lab[j] else "?"
            cv2.putText(
                img,
                f"{tag} {T['lat'][j]:.1f}m",
                (cx + 6, cy - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
        dd = T["d"][rows[:, 0] == i]
        cv2.putText(
            img,
            f"f{i} leader d={dd.min():.0f}m" if len(dd) else f"f{i}",
            (10, 140),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
        )
        cv2.imwrite(_out(run, f"reproj_f{i}.png"), img)
        outs.append(img)
    return outs


def posecheck(run, want):
    import cv2

    P = np.load(run.p("pose.npz"))
    img_map, _cnt, _to = run._map()
    meta = np.load(run.p("map_meta.npz"))
    x0, y0 = meta["bounds"]
    sc = float(meta["sc"])
    A = np.array([[sc, 0, -x0 * sc], [0, sc, -y0 * sc], [0, 0, 1]])
    k_of = {int(f): k for k, f in enumerate(P["idx"])}
    out = []
    want = sorted(want)
    for i, p in video.iter_unique_render(run.video(), start=min(want) - 3, stop=max(want) + 4):
        if not any(abs(i - w) <= 2 for w in want) or i not in k_of:
            continue
        want = [w for w in want if abs(i - w) > 2]
        back = cv2.warpAffine(
            img_map, (A @ P["G"][k_of[i]])[:2], (p.shape[1], p.shape[0]), flags=cv2.WARP_INVERSE_MAP
        )
        out.append(np.hstack([p, cv2.addWeighted(p, 0.5, back, 0.5, 0)]))
    if out:
        cv2.imwrite(_out(run, "posecheck.jpg"), np.vstack(out))


def straightened(run, row: int = 1800, half: int = 420):
    import cv2

    z = np.load(run.p("rail.npz"))
    Ps, Nin, sup = z["stations"], z["Nin"], z["support"]
    img, _cnt, to_px = run._map()
    offs = np.arange(-half, half + 1, 1.0)
    X = Ps[:, None, :] + offs[None, :, None] * Nin[:, None, :]
    uv = to_px(X.reshape(-1, 2)).astype(np.float32)
    st = cv2.remap(
        img, uv[:, 0].reshape(len(Ps), -1), uv[:, 1].reshape(len(Ps), -1), cv2.INTER_LINEAR
    )
    st = np.ascontiguousarray(st.transpose(1, 0, 2))
    from scipy.spatial import cKDTree

    T = np.load(run.p("tracks.npz"), allow_pickle=True)
    j = cKDTree(Ps).query(T["X"])[1]
    ho = ((T["X"] - Ps[j]) * Nin[j]).sum(1)
    for jj, o in zip(j, ho):
        y = int(round(o + half))
        if 0 <= y < st.shape[0]:
            st[y, jj] = (0, 0, 255)
    ro = ((z["rail"] - Ps) * Nin).sum(1)
    for n in range(len(Ps)):
        y = int(round(ro[n] + half))
        if 0 <= y < st.shape[0]:
            st[y, n] = (255, 0, 255) if sup[n] else (0, 255, 255)
    for k, a in enumerate(range(0, len(Ps), row)):
        cv2.imwrite(_out(run, f"straight_{k}.png"), st[:, a : a + row])


def paths_plot(run, name="paths.parquet"):
    """The paths as time series: distance to finish d(t) and lateral(t); plus lateral vs d."""
    import matplotlib
    import polars as pl

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    P = pl.read_parquet(run.p(name))
    fig, ax = plt.subplots(3, 1, figsize=(16, 12))
    for h in sorted(P["horse_no"].unique().to_list()):
        g = P.filter(P["horse_no"] == h).sort("t")
        t, d, lat = g["t"].to_numpy(), g["d"].to_numpy(), g["lat"].to_numpy()
        ok = (g["sd_s"] < 1.5).to_numpy()
        ax[0].plot(t[ok], d[ok], ".", ms=1.2, label=f"#{h}")
        ax[1].plot(t[ok], lat[ok], ".", ms=1.2)
        ax[2].plot(d[ok], lat[ok], ".", ms=1.2)
    ax[0].set_ylabel("distance to finish (m)")
    ax[0].legend(ncol=14, fontsize=7, markerscale=6)
    ax[1].set_ylabel("lateral off racing rail (m)")
    ax[1].set_xlabel("race time (s)")
    ax[2].invert_xaxis()
    ax[2].set_xlabel("distance to finish (m)")
    ax[2].set_ylabel("lateral (m)")
    fig.suptitle(f"{run.info.key} {name} (posterior sd < 1.5 m)")
    plt.tight_layout()
    plt.savefig(_out(run, name.replace(".parquet", ".png")), dpi=70)
    plt.close(fig)


def _silk_colour(horse_id):
    """Dominant body colour (BGR) of the official silk image (non-white, non-transparent pixels)."""
    from PIL import Image

    from ..data.silks import silk_path

    p = silk_path(horse_id)
    if not p:
        return (200, 200, 200)
    g = np.array(Image.open(p).convert("RGB")).astype(int)
    h, w = g.shape[:2]
    body = g[int(0.45 * h) : int(0.8 * h), int(0.3 * w) : int(0.7 * w)].reshape(-1, 3)
    keep = body[(body.max(1) - body.min(1) > 25) | (body.max(1) < 200)]
    c = np.median(keep if len(keep) else body, axis=0)
    return (int(c[2]), int(c[1]), int(c[0]))


def _map_geometry(run):
    """Track map from the run's own measured rail: (rail points px, outward normals, rail arclength px,
    ppm, d -> rail arclength mapping). d <-> s_px is taken from the observe stage's own projection of
    the detections (matched to tracks.npz), so the map uses exactly the reconstruction's geometry."""
    T = np.load(run.p("tracks.npz"), allow_pickle=True)
    z = np.load(run.p("rail.npz"))
    m = np.load(run.p("metric.npz"))
    rf = rails.RailFrame.from_points(z["rail"], int(z["inner"]))
    rows, X, _h, _P = run.world_dets()
    k_of = {int(f): k for k, f in enumerate(z["frame_idx"])}
    hint = rf.s[z["st_of_frame"][[k_of[int(f)] for f in rows[:, 0]]]]
    s_px, _l = rf.project(X, s_hint=hint, window=1200.0)
    key = {tuple(np.round(r[:3], 2)): s for r, s in zip(rows, s_px)}
    sT = np.array([key.get(tuple(np.round(r[:3], 2)), np.nan) for r in T["rows"]])
    ok = np.isfinite(sT)
    o = np.argsort(-T["d"][ok])  # d decreasing <-> s_px increasing
    dd, ss = T["d"][ok][o], sT[ok][o]
    # monotone d -> s_px: bin by d and take medians (projection noise), then interpolate
    bins = np.arange(np.floor(dd.min()) - 5, dd.max() + 10, 5.0)
    idx = np.digitize(dd, bins)
    bd = np.array([np.median(dd[idx == i]) for i in np.unique(idx)])
    bs = np.array([np.median(ss[idx == i]) for i in np.unique(idx)])
    order = np.argsort(bd)
    bd, bs = bd[order], bs[order]
    lat0 = float(z["lat0_px"]) if "lat0_px" in z.files else 0.0
    ppm = float(m["ppm"])

    def d_to_s(d):
        return np.interp(d, bd, bs)  # beyond the observed range: clamped (drawn at the edge)

    return rf, ppm, lat0, d_to_s


def full_offset(run, windows=((20.0, 32.0), (40.0, 52.0))):
    """Video seconds at race clock 0 for the FULL (TV) feed: the aerial feed's left panel is a scaled
    copy of the TV picture, so the lag maximising thumbnail correlation between the two, added to the
    aerial's clock offset. Measured on two windows; returns (offset, [lags], [correlations])."""
    import av
    import cv2

    def thumbs(path, t0, t1, crop=None):
        out = []
        with av.open(str(path)) as c:
            st = c.streams.video[0]
            c.seek(int(max(t0 - 1, 0) / st.time_base), stream=st)
            for fr in c.decode(st):
                t = float(fr.pts * st.time_base)
                if t < t0:
                    continue
                if t > t1:
                    break
                g = fr.to_ndarray(format="gray")
                if crop:
                    g = g[crop[0] : crop[1], crop[2] : crop[3]]
                g = cv2.resize(g, (64, 36), interpolation=cv2.INTER_AREA).astype(float)
                out.append((t, ((g - g.mean()) / (g.std() + 1e-6)).ravel()))
        return out

    a0 = run.clock_f0() / 25.0
    lags, cors = [], []
    for t0, t1 in windows:
        A = thumbs(run.video("aerial"), t0, t1, crop=(270, 807, 0, 955))
        F = thumbs(run.video("full"), t0 - 2, t1 + 2)
        if not A or not F:
            continue
        fa = np.array([x[1] for x in A])
        ff = np.array([x[1] for x in F])
        best = None
        for lag in range(-50, 51):
            i0 = int(round((A[0][0] - F[0][0]) * 25)) + lag
            if i0 < 0 or i0 + len(fa) > len(ff):
                continue
            r = float(np.mean(np.sum(fa * ff[i0 : i0 + len(fa)], 1) / fa.shape[1]))
            if best is None or r > best[0]:
                best = (r, lag / 25.0)
        if best:
            cors.append(best[0])
            lags.append(best[1])
    return (a0 + float(np.median(lags)) if lags else None), lags, cors


def sync_video(
    run,
    out_path,
    offsets: dict,
    *,
    paths_name="paths_video.parquet",
    t0=None,
    t1=None,
    fps=25.0,
    tile=(640, 360),
):
    """Four replay feeds, time-synced on the race clock, beside the reconstructed positions on a track
    map drawn from the run's own measured rail. ``offsets`` = {feed: video seconds at race clock 0}
    (aerial: ``run.clock_f0() / 25``; full: ``full_offset(run)``; patrol/drone from the gate opening,
    see RECON; ``None`` = feed not published, drawn blank). Horses = official silk colour +
    saddlecloth number; a faint ring marks posterior sd 1.5-3 m, a hollow marker sd 3-10 m, and a
    position with sd > 10 m is left out of the field view (the leader and window use sd < 3 m only).
    Writes an H.264 MP4 via ffmpeg."""
    import subprocess

    import av
    import cv2
    import polars as pl

    feeds = ["aerial", "full", "drone", "patrol"]
    P = pl.read_parquet(run.p(paths_name))
    horses = sorted(P["horse_no"].unique().to_list())
    G = {h: P.filter(pl.col("horse_no") == h).sort("t") for h in horses}
    G = {h: {c: g[c].to_numpy() for c in ("t", "d", "lat", "sd_s")} for h, g in G.items()}
    col = {r.no: _silk_colour(r.horse_id) for r in run.info.runners}
    rf, ppm, lat0, d_to_s = _map_geometry(run)
    width_m = run.info.width if getattr(run.info, "width", None) else 30.0
    # map frame: rotate the rail's principal axis horizontal, fit the panel
    D = run.info.distance
    s_lo, s_hi = d_to_s(np.array([D, -10.0]))
    sel = (rf.s >= min(s_lo, s_hi) - 50 * ppm) & (rf.s <= max(s_lo, s_hi) + 20 * ppm)
    Rin = rf.R[sel] + (0.0 - lat0) * rf.Nout[sel]
    Rout = rf.R[sel] + (width_m * ppm - lat0) * rf.Nout[sel]
    allp = np.vstack([Rin, Rout])
    mu = allp.mean(0)
    _u, _s, Vt = np.linalg.svd(allp - mu, full_matrices=False)
    Wm, Hm = tile[0], 2 * tile[1]
    Ho = int(0.5 * Hm)  # overview on top, straightened field view below
    rot = lambda p: (np.atleast_2d(p) - mu) @ Vt.T  # noqa: E731
    q = rot(allp)
    sc = 0.88 * min(Wm / np.ptp(q[:, 0]), (Ho - 50) / np.ptp(q[:, 1]))  # long axis horizontal
    cq = (q.max(0) + q.min(0)) / 2

    def to_px(p):
        r = rot(p) - cq
        return np.c_[Wm / 2 + r[:, 0] * sc, (Ho + 40) / 2 + r[:, 1] * sc].astype(np.int32)

    base = np.full((Hm, Wm, 3), (40, 90, 40), np.uint8)
    base[Ho:] = (30, 30, 30)
    poly = np.vstack([to_px(Rin), to_px(Rout)[::-1]])
    cv2.fillPoly(base, [poly], (60, 140, 60))
    cv2.polylines(base, [to_px(Rin)], False, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.polylines(base, [to_px(Rout)], False, (220, 220, 220), 1, cv2.LINE_AA)
    sf = d_to_s(0.0)
    jf = int(np.argmin(np.abs(rf.s - sf)))
    fin = to_px(np.vstack([rf.R[jf] - lat0 * rf.Nout[jf], rf.R[jf] + (width_m * ppm - lat0) * rf.Nout[jf]]))
    cv2.line(base, tuple(fin[0]), tuple(fin[1]), (0, 0, 255), 2, cv2.LINE_AA)

    def pos(h, t):
        g = G[h]
        if t < g["t"][0] or t > g["t"][-1]:
            return None
        d = np.interp(t, g["t"], g["d"])
        lat = np.interp(t, g["t"], g["lat"])
        sd = np.interp(t, g["t"], g["sd_s"])
        s = d_to_s(d)
        j = np.clip(np.searchsorted(rf.s, s), 1, len(rf.s) - 1)
        w = (s - rf.s[j - 1]) / max(rf.s[j] - rf.s[j - 1], 1e-9)
        R = rf.R[j - 1] * (1 - w) + rf.R[j] * w
        N = rf.Nout[j - 1] * (1 - w) + rf.Nout[j] * w
        return to_px(R + (lat * ppm - lat0) * N)[0], d, sd

    # one field-view window per race (a per-frame window rescaled every horse at once: 8-11 jumps per race)
    spreads = []
    for tt in np.arange(min(g["t"][0] for g in G.values()), max(g["t"][-1] for g in G.values()), 0.5):
        ds = [np.interp(tt, g["t"], g["d"]) for g in G.values()
              if g["t"][0] <= tt <= g["t"][-1] and np.interp(tt, g["t"], g["sd_s"]) < 3.0]
        ds = [x for x in ds if x > 0]
        if len(ds) > 1:
            spreads.append(max(ds) - min(ds))
    win_race = max(30.0, 5.0 * np.ceil((np.percentile(spreads, 95) + 6.0) / 5.0)) if spreads else 40.0
    tmin = min(g["t"][0] for g in G.values())
    tmax = max(g["t"][-1] for g in G.values())
    t0 = tmin if t0 is None else t0
    t1 = tmax + 1.0 if t1 is None else t1
    # decoders, positioned at race time t0
    conts, its, nxt = {}, {}, {}
    feeds_ok = [f for f in feeds if offsets.get(f) is not None]
    for f in feeds_ok:
        c = av.open(str(run.video(f)))
        st = c.streams.video[0]
        st.thread_type = "AUTO"
        conts[f] = (c, st)
        tv = t0 + offsets[f]
        c.seek(int(max(tv - 2.0, 0) / st.time_base), stream=st)
        its[f] = c.decode(st)
        nxt[f] = None
    cur = dict.fromkeys(feeds_ok)

    def frame_at(f, tv):
        c, st = conts[f]
        while True:
            if nxt[f] is None:
                try:
                    fr = next(its[f])
                except StopIteration:
                    return cur[f]
                nxt[f] = (float(fr.pts * st.time_base), fr)
            ts, fr = nxt[f]
            if ts <= tv + 0.5 / fps:
                cur[f] = cv2.resize(fr.to_ndarray(format="bgr24"), tile, interpolation=cv2.INTER_AREA)
                nxt[f] = None
            else:
                return cur[f]

    W, H = 2 * tile[0] + Wm, Hm
    enc = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}",
         "-r", str(fps), "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", "24",
         "-pix_fmt", "yuv420p", str(out_path)],
        stdin=subprocess.PIPE,
    )
    n = int(round((t1 - t0) * fps))
    for k in range(n):
        t = t0 + k / fps
        canvas = np.zeros((H, W, 3), np.uint8)
        for i, f in enumerate(feeds):
            img = frame_at(f, t + offsets[f]) if f in feeds_ok else None
            x, y = (i % 2) * tile[0], (i // 2) * tile[1]
            if img is not None:
                canvas[y : y + tile[1], x : x + tile[0]] = img
            label = f if f in feeds_ok else f"{f} (not published)"
            cv2.putText(canvas, label, (x + 8, y + tile[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        mp = base.copy()
        cur_pos = {}
        for h in horses:
            r = pos(h, t)
            if r is not None:
                cur_pos[h] = r
        # honest display: the leader / window come from well-determined horses only (sd < 3 m); a
        # vaguer position is drawn hollow, and one with sd > 10 m is left out of the field view
        firm = {h: r for h, r in cur_pos.items() if r[2] < 3.0}
        lead = min((r[1] for r in (firm or cur_pos).values()), default=None)
        for h, (p, d, sd) in cur_pos.items():
            cv2.circle(mp, tuple(p), 4, col.get(h, (200, 200, 200)), -1 if sd < 3.0 else 1, cv2.LINE_AA)
        # straightened field view: x = metres behind the leader, y = lateral off the rail
        y0f, hf = Ho + 40, Hm - Ho - 70
        if cur_pos:
            win = win_race
            xs = lambda gap: int(Wm - 30 - gap / win * (Wm - 60))  # noqa: E731
            ys = lambda lat: int(y0f + hf - lat / width_m * hf)  # noqa: E731
            cv2.rectangle(mp, (30, y0f), (Wm - 30, y0f + hf), (60, 140, 60), -1)
            cv2.line(mp, (30, y0f + hf), (Wm - 30, y0f + hf), (255, 255, 255), 2)
            for gm in np.arange(0, win + 0.1, 5.0):
                cv2.line(mp, (xs(gm), y0f + hf), (xs(gm), y0f + hf + 6), (200, 200, 200), 1)
                if gm % 10 == 0:
                    cv2.putText(mp, f"{gm:.0f}", (xs(gm) - 8, y0f + hf + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            unplaced = sorted(set(col) - set(cur_pos))
            for h, (p, d, sd) in sorted(cur_pos.items(), key=lambda x: -x[1][1]):
                if sd > 10.0 or not (-2.0 <= d - lead <= win + 2.0):
                    unplaced.append(h)
                    continue
                g = G[h]
                lat = float(np.interp(t, g["t"], g["lat"]))
                c = (xs(d - lead), ys(lat))
                if sd < 3.0:
                    if sd > 1.5:
                        cv2.circle(mp, c, 15, (180, 180, 180), 1, cv2.LINE_AA)
                    cv2.circle(mp, c, 11, col.get(h, (200, 200, 200)), -1, cv2.LINE_AA)
                    cv2.circle(mp, c, 11, (0, 0, 0), 1, cv2.LINE_AA)
                else:  # extrapolated: hollow
                    cv2.circle(mp, c, 11, col.get(h, (200, 200, 200)), 2, cv2.LINE_AA)
                tw = cv2.getTextSize(str(h), cv2.FONT_HERSHEY_SIMPLEX, 0.45, 2)[0][0]
                cv2.putText(mp, str(h), (c[0] - tw // 2, c[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(mp, str(h), (c[0] - tw // 2, c[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(mp, "metres behind leader  (rail at bottom)", (30, y0f - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 230), 1)
            if unplaced:  # no firm position in this window: listed, never silently dropped
                txt = "not placed: " + " ".join(f"#{h}" for h in sorted(unplaced))
                cv2.putText(mp, txt, (36, y0f + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(mp, f"{run.info.key}  t = {t:6.2f} s", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        if lead is not None:
            cv2.putText(mp, f"leader {max(lead, 0):5.0f} m to go", (Wm - 230, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        prod = "reconstruction, video only" if "video" in paths_name else "final product (+ official records)"
        cv2.putText(mp, prod, (Wm - 8 - cv2.getTextSize(prod, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0][0], Ho + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        canvas[:, 2 * tile[0] :] = mp
        enc.stdin.write(canvas.tobytes())
    enc.stdin.close()
    enc.wait()
    for c, _st in conts.values():
        c.close()
    return str(out_path)
