"""Visual validation — look at the pixels (numbers hide CV failures).

  reproject(run, frames)   the world reconstruction drawn back onto raw render frames: racing rail
                           (magenta) + 5/10 m offset guides (cyan) + each detection with its fused
                           horse number and lateral metres
  posecheck(run, frames)   the static world map warped back through each frame's pose, blended 50/50
                           with the raw panel (misregistration shows as doubled rails/markers)
  straightened(run)        the map straightened along the rail stations (s -> x, inward -> y) with
                           the fitted rail and all horse points, in rows
  paths_plot(run)          fused paths: lateral vs distance-to-go, and gap-behind-leader vs time
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
