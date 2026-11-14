"""Race film (left half of the aerial replay, frame-synchronised with the render): leader crossings of
the REAL painted distance markers.

The aerial replay is a split screen: the live race film on the left (with the broadcast race clock,
the same time base as ours) and the AVV render on the right, on the same frames. In the final
straight the film is a front-on camera that sees each painted marker (400m/300m/200m/150m/100m/50m)
as a long near-horizontal paint stripe sweeping UP the image as the camera moves. MEASURED (Sha Tin
2026-04-19 R4, pixels checked): at t = 63.00 s (film clock 1:03.00) the leader's hooves are on the
300m line, at 69.00 s on the 200m line.

Per sampled frame (final ~26 s): paint = brighter + less saturated than the local grass; all long
near-horizontal Hough segments; horses from the COCO detector; the leader = the box lowest in the
image (closest to the front-on camera). Lines are TRACKED across frames and a track counts as a
marker only if it sweeps up the image (>= SWEEP_PX): REJECTED single-frame "longest line" — a STATIC
paint/edge stripe at the bottom of the film (ST4: 104 of 351 frames) and the running rail of the
side-on camera after the camera change both won it. A camera CUT (thumbnail difference > CUT_DIFF)
ends every track — but on ST4 the change to the side-on camera is a pan/dissolve (max diff 15) and it
never fires; the sweep test is what rejects the side-on views, where "lowest box" is not the leader.
Crossing = the leader's hooves passing the line. Which marker: the render HUD odometer at that
frame, at the marker or up to 40 m behind it (the render lags reality); then speed consistency
between successive markers.

MEASURED (this detector, pixels checked on ST4): 400 m crossings vs official -0.12 (ST4), -0.08
(ST9), -0.05 s (ST1 0426); ST4 300/200/150/100 m at 63.06/68.89/71.79/74.67 s. Constant-speed
extrapolation from the last two markers to the line is BIASED EARLY (ST4 80.43 vs official 81.04,
-0.61 s: the clear leader eased; ST9 -0.46, ST3 -0.49, ST1 -0.28) — a bound, not an observation.
These are REAL-VIDEO observations (no official data)."""

from __future__ import annotations

import os

import numpy as np

MARKERS = np.array([400, 300, 200, 150, 100, 50])
FILM_BOX = (270, 810, 0, 960)  # y0, y1, x0, x1 of the race film in the 1080p aerial replay
SWEEP_PX = 80.0  # a marker line moves at least this far up the image while tracked
CUT_DIFF = 40.0  # mean |diff| of 96x54 grey thumbnails between sampled frames: a camera cut


def paint_lines(film, *, min_len=300):
    """Near-horizontal painted line segments on the turf -> [(yc, x1, y1, x2, y2)], one per stripe
    (yc = y at the image centre column), longest first."""
    import cv2

    hsv = cv2.cvtColor(film, cv2.COLOR_BGR2HSV)
    h, s, v = [hsv[..., i].astype(np.float32) for i in range(3)]
    grass = (h > 30) & (h < 90)
    paint = (
        (v - cv2.blur(v, (41, 41)) > 14) & (cv2.blur(s, (41, 41)) - s > 12) & grass
    ).astype(np.uint8)
    paint[:200] = 0  # rail / stands / clock overlay
    lines = cv2.HoughLinesP(paint * 255, 1, np.pi / 360, 120, minLineLength=min_len, maxLineGap=25)
    if lines is None:
        return []
    xc = film.shape[1] / 2
    segs = []
    for x1, y1, x2, y2 in lines[:, 0]:
        if abs(y2 - y1) > 0.25 * abs(x2 - x1):
            continue
        yc = y1 + (y2 - y1) * (xc - x1) / max(x2 - x1, 1)
        segs.append((np.hypot(x2 - x1, y2 - y1), yc, x1, y1, x2, y2))
    segs.sort(key=lambda q: -q[0])
    out = []
    for _ln, yc, *xy in segs:
        if all(abs(yc - o[0]) > 12 for o in out):
            out.append((yc, *xy))
    return out


def _detector():
    from ultralytics import YOLO

    dev = os.environ.get("HKJC_DEVICE", "cpu")
    ov = "models/coco_yolov8x_544x960_openvino_model"
    if dev.startswith("intel:") and os.path.exists(ov):
        return YOLO(ov, task="detect"), dict(imgsz=(544, 960), device=dev)
    return YOLO("models/coco_yolov8x.pt"), dict(imgsz=960, device=dev if dev != "cpu" else None)


def scan(video, f_start, f_stop, *, step=2):
    """-> per sampled frame dict(f, diff, lines, lead): diff = thumbnail change vs the previous
    sample (camera cuts), lead = (hoof_y, cx) of the lowest horse."""
    import av
    import cv2

    model, kw = _detector()
    y0, y1, x0, x1 = FILM_BOX
    out, prev = [], None
    with av.open(str(video)) as c:
        c.streams.video[0].thread_type = "AUTO"
        for i, fr in enumerate(c.decode(video=0)):
            if i < f_start or (i - f_start) % step:
                continue
            if i > f_stop:
                break
            film = fr.to_ndarray(format="bgr24")[y0:y1, x0:x1]
            th = cv2.resize(cv2.cvtColor(film, cv2.COLOR_BGR2GRAY), (96, 54)).astype(np.float32)
            diff = float(np.abs(th - prev).mean()) if prev is not None else 0.0
            prev = th
            r = model.predict(film, conf=0.35, classes=[17], verbose=False, **kw)[0]
            b = r.boxes.xyxy.cpu().numpy()
            j = int(np.argmax(b[:, 3])) if len(b) else -1
            lead = (float(b[j, 3]), float((b[j, 0] + b[j, 2]) / 2)) if j >= 0 else None
            out.append(dict(f=i, diff=diff, lines=paint_lines(film), lead=lead))
    return out


def line_tracks(rows, *, max_gap=3, up_max=60.0, down_max=6.0):
    """Link lines across sampled frames (a marker moves UP: yc decreases by 0..up_max per sample);
    a cut ends all tracks. -> list of tracks, each a list of (row index, line)."""
    tracks, live = [], []
    for k, r in enumerate(rows):
        if r["diff"] > CUT_DIFF:
            live = []
        used, new = set(), []
        for ln in r["lines"]:
            best = None
            for ti, tr in enumerate(live):
                kl, ll = tr[-1]
                dy = ll[0] - ln[0]  # > 0: moved up
                if ti not in used and -down_max <= dy <= up_max * (k - kl):
                    if best is None or abs(dy) < best[0]:
                        best = (abs(dy), ti)
            if best is not None:
                used.add(best[1])
                live[best[1]].append((k, ln))
            else:
                new.append([(k, ln)])
        tracks += new
        live = [tr for tr in live + new if k + 1 - tr[-1][0] <= max_gap]
    return [tr for tr in tracks if len(tr) >= 4 and tr[0][1][0] - tr[-1][1][0] >= SWEEP_PX]


def crossings_from_rows(rows, odo_at, f0, fps=25.0):
    """Leader crossings on sweeping marker tracks -> {marker_m: race-clock s}."""
    got = {}
    for tr in line_tracks(rows):
        seq = []
        for k, (_yc, x1, y1, x2, y2) in tr:
            ld = rows[k]["lead"]
            if ld is None:
                continue
            hy, cx = ld
            seq.append((rows[k]["f"], hy - (y1 + (y2 - y1) * (cx - x1) / max(x2 - x1, 1))))
        for (fa, da), (fb, db) in zip(seq[:-1], seq[1:]):
            if da < 0 <= db and fb - fa <= 6:
                fc = fa + (0 - da) / (db - da) * (fb - fa)
                o = odo_at(fc)
                if not np.isfinite(o):
                    continue
                cand = MARKERS[(o - MARKERS >= -5) & (o - MARKERS <= 40)]
                if len(cand):
                    mk = int(cand.max())
                    if mk not in got:
                        got[mk] = (fc - f0) / fps
                break  # one crossing per marker line
    return _speed_consistent(got)


def crossings(video, hud_rows, t_last, *, span_s=26.0, fps=25.0):
    """Leader crossings of the painted markers -> {marker_m: race-clock s}. ``hud_rows`` = the HUD
    stage rows (frame, clock, odometer); ``t_last`` = approximate race-clock time of the finish."""
    ok = ~np.isnan(hud_rows[:, 1])
    f0 = float(np.median(hud_rows[ok, 0] - fps * hud_rows[ok, 1]))
    hf, ho = hud_rows[:, 0], hud_rows[:, 2]
    g = np.isfinite(ho)
    hf, ho = hf[g], ho[g]

    def odo_at(f):
        k = int(np.argmin(np.abs(hf - f)))
        return ho[k] if abs(hf[k] - f) <= 6 else np.nan

    rows = scan(video, int(f0 + fps * (t_last - span_s)), int(f0 + fps * (t_last + 2)))
    return crossings_from_rows(rows, odo_at, f0, fps)


def _speed_consistent(got, vmin=12.0, vmax=22.0):
    """Drop crossings whose implied speeds to BOTH neighbours are implausible (spurious lines)."""
    mk = sorted(got, reverse=True)
    keep = dict(got)
    for i, m in enumerate(mk):
        bad = []
        for j in (i - 1, i + 1):
            if 0 <= j < len(mk):
                dt = abs(got[m] - got[mk[j]])
                v = abs(m - mk[j]) / dt if dt > 0 else np.inf
                bad.append(not (vmin <= v <= vmax))
        if bad and all(bad):
            keep.pop(m)
    return keep


def leader_finish(film_t, *, vmin=12.0, vmax=22.0):
    """Extrapolate the film leader to the line from the last two markers (constant speed) ->
    (t_line, speed) or None. BIASED EARLY (leaders slow in the last 100 m): MEASURED -0.28..-0.61 s
    on 4 Sha Tin races, so use it only as a physical bound."""
    mk = sorted(film_t)
    if len(mk) < 2:
        return None
    a, b = mk[0], mk[1]  # nearest the line, next
    dt = film_t[a] - film_t[b]
    if dt <= 0:
        return None
    v = (b - a) / dt
    if not (vmin <= v <= vmax):
        return None
    return film_t[a] + a / v, v


FILM_MIN = 3  # crossings needed to anchor the metric on the film (else: the render-paint pin)


def anchor_offset(t, d_lead, film_t, *, win=1.0):
    """Constant metre offset that puts the render leader on the film's marker crossings.

    ``t``/``d_lead``: the render leader's distance-to-go per frame (any metric that is right up to a
    constant, e.g. the odometer ticks with no convention offset); ``film_t`` = {marker_m: race-clock
    s}. Per marker: delta_k = v_k * (t_render(marker) - t_film), v_k = the leader's local speed over
    +-``win`` s. -> (median delta in m, per-marker deltas) or (None, []) with < FILM_MIN crossings.
    d_true = d - delta. MEASURED on 15 turf races (recon_v3 artifacts, journal 0137): with the ticks
    unpinned, recon - film is one flat offset per race through the straight (+2.4..+11 m, spread
    <= 3 m on most races) and it continues through the line; the render-paint pin made it a step in
    the last 50 m (leader +0.2 s late at 400-100 m, -0.16 s early at the line). Used only with
    HKJC_FILM=1 (it destabilises finish-camera identity on some races; see RaceRun.observe)."""
    t, d_lead = np.asarray(t, float), np.minimum.accumulate(np.asarray(d_lead, float))
    out = []
    for mk, tf in film_t.items():
        k = np.where(d_lead <= mk)[0]
        if len(k) == 0 or k[0] == 0:
            continue
        a, b = k[0] - 1, k[0]
        tc = t[a] + (d_lead[a] - mk) / max(d_lead[a] - d_lead[b], 1e-9) * (t[b] - t[a])
        w = (t > tc - win) & (t < tc + win)
        if w.sum() < 3:
            continue
        v = -np.polyfit(t[w], d_lead[w], 1)[0]
        if 10.0 <= v <= 22.0:
            out.append(v * (tc - tf))
    if len(out) < FILM_MIN:
        return None, out
    return float(np.median(out)), out
