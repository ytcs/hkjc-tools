"""TASK #99 — DIAGNOSTIC WALKTHROUGH VIDEO for the 4-feed fusion reconstruction.

Builds an mp4 where EACH FRAME is a composite at one shared-clock timestamp ``race_t``:

  * CENTER (largest) panel — the TOP-DOWN RECONSTRUCTION. The track is drawn from the rail
    geometry + the 22.5 m C-course width box (#96). Every horse is plotted at its fused
    (arclen -> distance-along-track, y_lat) position, number- + silk-coloured, with its
    uncertainty ELLIPSE (sd_arclen x sd_ylat straight from the fused parquet) and a fading
    tail of recent positions. A HUD names race_t / arclen / race PHASE (gate/straight/BEND/
    finish from the #96 geometry map) and WHICH INSTRUMENTS ARE LIVE — and, on a bend, WHY the
    ellipses balloon (the absolute-lateral-metre gap, CV_RECON §7). That is the payload.

  * FOUR side panels — the synced SOURCE frames, each pulled at ITS OWN video time via the
    shared-clock offset (CV_RECON §3), overlays drawn on the actual pixels:
      - AERIAL render (right half of the split-screen): freshly-fit rail spline + detected
        ground-contacts + odometer/arclen readout, AND a clear homography-grid breakdown when
        the arclen is on a BEND (under-determined -> drawn broken/red). This is where the key
        difficulty shows.
      - PATROL (head-on): the fixed rail line + a metric lane grid + freshly-detected silk
        ground-contacts; greyed with a "NO COVERAGE — camera panning" banner outside the
        head-on window.
      - RF / race-film (side-on): cached longitudinal ORDER (legend) + finish-MARGIN readout.
      - DRONE: freshly-fit rail + cached numbered ordering cross-check.
    Any feed not covering this race_t -> greyed panel + a one-line reason banner.

HONESTY: overlays that are freshly DETECTED at the exact frame are labelled "(live detect)";
overlays read from a cached per-feed panel at the nearest race_t are labelled "(cached)". The
bend homography is drawn BROKEN where the algorithm genuinely has none — never a confident grid
on a bend. Feeds whose clock/detector cannot be reconstructed are greyed with the honest reason.

Reuses (does not reinvent): the fused physkernel trajectories (#96+#97), the #96 geometry phase
map (cv_fusion_constrained.bend_fraction_map + W=22.5), the aerial render detectors
(cv_s0d_aerial_render), the patrol fixed-rail + silk detector (cv_patrol_lateral), and the
cached per-feed numbered parquets for RF/drone.

Study-only scratch. Official data is VALIDATION-ONLY; the draw is the allowed pre-race initial
condition. ``PYTHONHASHSEED=0``.

Regenerate:
    PYTHONHASHSEED=0 uv run python studies/cv_walkthrough_video.py R01
    PYTHONHASHSEED=0 uv run python studies/cv_walkthrough_video.py R08
Output: data_cache/figs/cv_recon_walkthrough_{R01,R08}.mp4
Per-feed overlay caches: data_cache/cv_walkthrough_cache_{R01,R08}_{aerial,patrol,drone}.npz
"""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import av
import cv2
import numpy as np
import pyarrow.parquet as pq

# Reuse the #96 geometry phase map + width box and the aerial/patrol detectors.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import cv2 as _cv2  # noqa: E402,F401  (ensure cv2 importable before study imports)

from cv_fusion_constrained import (  # noqa: E402
    BEND_RADIUS_M,
    HOME_STRAIGHT_C,
    W_C_COURSE,
    bend_fraction_map,
)
from cv_patrol_lateral import (  # noqa: E402
    COURSE_WIDTH_M as PATROL_W,
)
from cv_patrol_lateral import (  # noqa: E402
    _fit_white_rail,
    _turf_hue,
    detect_jockeys,
)
from cv_s0d_aerial_render import (  # noqa: E402
    _Shift,
    fit_rail_spline,
    ground_contact,
    track_band,
)

VIDEO_DIR = Path("data_cache/video")
CACHE_DIR = Path("data_cache")
FIG_DIR = Path("data_cache/figs")
FPS_VID = 25.0  # all feeds 1920x1080 @ 25 fps

# Output mp4 frame rate: generate on a subsampled fused grid, play ~12 fps (watchable).
OUT_FPS = 12.0
GRID_STRIDE = 2  # the fused grid is 5 Hz; take every 2nd node -> 2.5 Hz sampling, 12 fps -> ~5x


# --------------------------------------------------------------------------------------------
# Per-race config — clock anchors (CV_RECON §3) + geometry, all REUSED from the per-race studies.
# --------------------------------------------------------------------------------------------
@dataclass
class RaceCfg:
    name: str
    suffix: str                       # video suffix: "01" / "08"
    dist: float
    field_size: int
    sections: list[float]
    d_bounds: np.ndarray
    winner_finish_t: float
    # clock offsets:  race_t -> the feed's own video time
    rf_offset: float                  # race_t = RF_vid - rf_offset
    patrol_offset: float              # race_t = patrol_vid - patrol_offset
    drone_t0: float                   # drone gate-jump vid time -> race_t 0
    drone_finish_vid: float           # drone winner-cross vid time -> race_t winner_finish
    patrol_headon: tuple[float, float]  # patrol head-on VIDEO-time window (rail steady)
    rail_horses: set[int]
    wide_horses: set[int]
    t_cum: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.t_cum = np.concatenate([[0.0], np.cumsum(self.sections)])

    def arclen_to_race_t(self, arclen: np.ndarray) -> np.ndarray:
        return np.interp(np.asarray(arclen, float), self.d_bounds, self.t_cum)

    def race_t_to_arclen(self, race_t: np.ndarray) -> np.ndarray:
        return np.interp(np.asarray(race_t, float), self.t_cum, self.d_bounds)

    def drone_vid_to_race_t(self, vid_t: float) -> float:
        rate = self.winner_finish_t / (self.drone_finish_vid - self.drone_t0)
        return (vid_t - self.drone_t0) * rate

    def drone_race_t_to_vid(self, race_t: float) -> float:
        rate = self.winner_finish_t / (self.drone_finish_vid - self.drone_t0)
        return race_t / rate + self.drone_t0


def make_cfg(name: str) -> RaceCfg:
    if name == "R01":
        # offsets from studies/cv_rf_longitudinal.py (1.34), cv_fusion_spine.py (patrol 117.5 cross),
        # cv_drone_revisit.py (gate 3.0 / cut 113.68); sections/bounds from cv_fusion_4feed.py.
        return RaceCfg(
            name="R01", suffix="01", dist=1650.0, field_size=12,
            sections=[27.65, 23.30, 23.75, 24.15],
            d_bounds=np.array([0.0, 450.0, 850.0, 1250.0, 1650.0]),
            winner_finish_t=98.85,
            rf_offset=1.34,
            patrol_offset=117.5 - 98.85,      # 18.65
            drone_t0=3.0, drone_finish_vid=113.68,
            patrol_headon=(83.0, 96.0),
            rail_horses={12, 8, 4, 11}, wide_horses={1, 2, 6, 9},
        )
    if name == "R08":
        # offsets from studies/cv_r08_rf.py (cross 109.7), cv_r08_patrol.py (cross 120.7),
        # cv_r08_drone.py (gate 3.0 / winner-cross 120.5); sections/bounds from cv_r08_common.py.
        return RaceCfg(
            name="R08", suffix="08", dist=1800.0, field_size=11,
            sections=[13.78, 23.56, 24.56, 23.62, 23.10],
            d_bounds=np.array([0.0, 200.0, 600.0, 1000.0, 1400.0, 1800.0]),
            winner_finish_t=108.62,
            rf_offset=109.7 - 108.62,          # 1.08
            patrol_offset=120.7 - 108.62,      # 12.08
            drone_t0=3.0, drone_finish_vid=120.5,
            patrol_headon=(98.0, 112.0),       # route head-on window (auto-detected ~95-120)
            rail_horses={4, 7, 11}, wide_horses={8, 2, 6},  # R08 stewards trip calls (loose)
        )
    raise SystemExit(f"unknown race {name}")


# --------------------------------------------------------------------------------------------
# Fused trajectories + a fixed number-keyed colour palette (the recon silk colours).
# --------------------------------------------------------------------------------------------
def load_fused(cfg: RaceCfg) -> dict:
    p = CACHE_DIR / f"cv_fusion_trajectories_{cfg.name}_physkernel.parquet"
    d = pq.read_table(str(p)).to_pydict()
    return {k: np.asarray(v) for k, v in d.items()}


# 14 visually-distinct BGR colours keyed by saddlecloth number (schematic — the green render
# washes the real silks, so we key colour to NUMBER, which is accurate + legible). HKJC-ish.
_PALETTE = [
    (60, 60, 220), (220, 120, 40), (40, 180, 60), (200, 60, 200), (40, 200, 220),
    (160, 80, 220), (80, 160, 240), (200, 200, 60), (120, 220, 120), (60, 120, 200),
    (200, 120, 160), (140, 200, 240), (180, 140, 80), (90, 90, 90),
]


def runner_bgr(n: int) -> tuple[int, int, int]:
    return _PALETTE[(n - 1) % len(_PALETTE)]


# --------------------------------------------------------------------------------------------
# RACE PHASE from the #96 geometry map (gate / straight / BEND / finish).
# --------------------------------------------------------------------------------------------
def race_phase(cfg: RaceCfg, arclen: float, race_t: float) -> str:
    if race_t <= 3.0 or arclen < 60.0:
        return "GATE"
    if arclen >= cfg.dist - HOME_STRAIGHT_C and arclen <= cfg.dist + 5:
        # in the home straight; the last ~60 m is the finish run-in.
        if arclen >= cfg.dist - 60.0:
            return "FINISH"
        return "STRAIGHT (home)"
    bf = float(bend_fraction_map(np.array([arclen]), cfg.dist)[0])
    if bf >= 0.5:
        return "BEND"
    return "STRAIGHT"


def is_bend(cfg: RaceCfg, arclen: float) -> bool:
    return float(bend_fraction_map(np.array([arclen]), cfg.dist)[0]) >= 0.5


# ============================================================================================
# PER-FEED OVERLAY CACHES — decode each video once, run cheap detectors at the sampled race_t
# grid, cache the overlay primitives (pixel polylines/points) to NPZ. Re-render is then cheap.
# ============================================================================================
def _grid_race_t(cfg: RaceCfg, fused: dict) -> np.ndarray:
    """The shared sampling grid: the fused-grid race_t values, subsampled by GRID_STRIDE."""
    rt = np.unique(fused["race_t"])
    rt.sort()
    return rt[::GRID_STRIDE]


def _decode_at(container, target_frames: set[int]):
    """Yield (frame_idx, bgr_img) for the requested frame indices in one sequential pass."""
    want = sorted(target_frames)
    if not want:
        return
    wi = 0
    nxt = want[0]
    for i, fr in enumerate(container.decode(video=0)):
        if i < nxt:
            continue
        yield i, fr.to_ndarray(format="bgr24")
        wi += 1
        if wi >= len(want):
            break
        nxt = want[wi]


# ---- AERIAL ------------------------------------------------------------------------------
def build_aerial_cache(cfg: RaceCfg, grid_rt: np.ndarray) -> None:
    """Per-grid-frame aerial overlays: rail polyline + ground-contacts + arclen + phase.

    The aerial split-screen render carries the odometer; frame i video-time = i/fps, but the
    SHARED CLOCK comes from the odometer arclen. We therefore map each grid race_t -> arclen
    (official curve) -> the aerial frame whose own odometer arclen is closest. To avoid a second
    OCR pass we use the cached aerial-render parquet (frame, x_arclen) to build frame<->arclen,
    then pick the aerial frame nearest each grid arclen, decode it, and re-fit rail+detect.
    """
    from hkjc.vision.mode_b.detect import detect_blobs
    from hkjc.vision.mode_b.layout import find_layout

    out = CACHE_DIR / f"cv_walkthrough_cache_{cfg.name}_aerial.npz"
    if out.exists():
        print(f"[aerial] cache exists -> {out.name}")
        return
    apq = (CACHE_DIR / ("cv_s0d_aerial_render.parquet" if cfg.name == "R01"
                        else "cv_r08_aerial_render.parquet"))
    at = pq.read_table(str(apq)).to_pydict()
    afr = np.asarray(at["frame"])
    axa = np.asarray(at["x_arclen"], float)
    ok = np.isfinite(axa)
    fr_u, idx = np.unique(afr[ok], return_index=True)
    arc_u = axa[ok][idx]
    order = np.argsort(arc_u)
    arc_u, fr_u = arc_u[order], fr_u[order]
    if len(arc_u) < 2:
        print("[aerial] no odometer arclen frames -> aerial cache empty (will grey panel)")
        np.savez(out, race_t=grid_rt, have=np.zeros(len(grid_rt), bool))
        return
    # grid race_t -> arclen -> nearest aerial frame with a known odometer arclen
    grid_arc = cfg.race_t_to_arclen(grid_rt)
    sel_frames = []
    for a in grid_arc:
        k = int(np.argmin(np.abs(arc_u - a)))
        sel_frames.append(int(fr_u[k]))
    sel_frames = np.array(sel_frames)

    video = VIDEO_DIR / f"aerial_20260603_{cfg.suffix}.mp4"
    container = av.open(str(video))
    fps = float(container.streams.video[0].average_rate)

    n = len(grid_rt)
    rails = [np.empty((0, 2)) for _ in range(n)]
    gcs = [np.empty((0, 2)) for _ in range(n)]  # ground-contact pixels
    arcs = np.full(n, np.nan)
    have = np.zeros(n, bool)
    half_w = [0] * n

    frame_to_rows: dict[int, list[int]] = {}
    for ri, f in enumerate(sel_frames):
        frame_to_rows.setdefault(int(f), []).append(ri)

    print(f"[aerial] decoding {len(frame_to_rows)} unique frames for {n} grid points...")
    for fi, img in _decode_at(container, set(frame_to_rows)):
        layout = find_layout(img)
        if not layout.aerial_active:
            continue
        half = img.shape[1] // 2
        panel = img[:, half:]
        ty0, ty1 = track_band(panel)
        from dataclasses import replace as dc_replace
        layout = dc_replace(layout, track_y0=ty0, track_y1=ty1)
        dets = detect_blobs(img, layout, frame=fi, t=fi / fps, max_blobs=cfg.field_size + 2)
        if not dets:
            continue
        cx = float(np.mean([d.cx for d in dets]))
        cy = float(np.mean([d.cy for d in dets]))
        rail = fit_rail_spline(panel, ty0, ty1, (cx - half, cy))
        gc_pts = []
        for d in dets:
            # _Shift presents det.cx/cy in PANEL coords (subtracts half); ground_contact returns
            # the shadow base in panel coords -> shift back to absolute image coords for drawing.
            gx, gy, _u = ground_contact(panel, _Shift(d, half), ty0, ty1)
            gc_pts.append((gx + half, gy))
        rail_xy = (np.column_stack([rail.px + half, rail.py]) if rail.valid
                   else np.empty((0, 2)))
        for ri in frame_to_rows[fi]:
            rails[ri] = rail_xy
            gcs[ri] = np.array(gc_pts, float)
            arcs[ri] = float(grid_arc[ri])
            have[ri] = rail.valid or len(gc_pts) > 0
            half_w[ri] = half
    container.close()

    np.savez(
        out, race_t=grid_rt, have=have, arclen=arcs, half_w=np.array(half_w),
        sel_frame=sel_frames,
        **{f"rail_{i}": rails[i] for i in range(n)},
        **{f"gc_{i}": gcs[i] for i in range(n)},
    )
    print(f"[aerial] cached {int(have.sum())}/{n} grid points -> {out.name}")


# ---- PATROL ------------------------------------------------------------------------------
def build_patrol_cache(cfg: RaceCfg, grid_rt: np.ndarray) -> None:
    """Fixed rail (fit once on the head-on window) + per-grid-frame silk ground-contacts."""
    out = CACHE_DIR / f"cv_walkthrough_cache_{cfg.name}_patrol.npz"
    if out.exists():
        print(f"[patrol] cache exists -> {out.name}")
        return
    video = VIDEO_DIR / f"patrol_20260603_{cfg.suffix}.mp4"
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or FPS_VID
    t0, t1 = cfg.patrol_headon
    # fit fixed rail once across the head-on window
    A, B = [], []
    for t in np.arange(t0, t1 + 0.01, 0.5):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
        ok, fr = cap.read()
        if not ok:
            continue
        r = _fit_white_rail(fr)
        if r:
            A.append(r[0])
            B.append(r[1])
    if A:
        a_med, b_med = float(np.median(A)), float(np.median(B))
        slope_cv = float(np.std(A) / abs(a_med)) if a_med else float("nan")
    else:
        a_med = b_med = slope_cv = float("nan")
    print(f"[patrol] fixed rail a={a_med:.3f} b={b_med:.1f} slope_cv={slope_cv:.3f}")

    n = len(grid_rt)
    in_win = np.zeros(n, bool)
    gcs = [np.empty((0, 2)) for _ in range(n)]
    sat = [np.empty((0, 3)) for _ in range(n)]  # silk centroids (cx, cy, none) for colour
    for i, rt in enumerate(grid_rt):
        vid_t = rt + cfg.patrol_offset
        if not (t0 <= vid_t <= t1):
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(vid_t * fps))
        ok, fr = cap.read()
        if not ok:
            continue
        in_win[i] = True
        turf = _turf_hue(fr)
        det = detect_jockeys(fr, turf)
        pts = [(d["cx"], d["y_ground"]) for d in det]
        scen = [(d["cx"], d["cy"], 0.0) for d in det]
        gcs[i] = np.array(pts, float)
        sat[i] = np.array(scen, float)
    cap.release()
    np.savez(
        out, race_t=grid_rt, in_win=in_win, a=a_med, b=b_med, slope_cv=slope_cv,
        headon=np.array(cfg.patrol_headon),
        **{f"gc_{i}": gcs[i] for i in range(n)},
        **{f"silk_{i}": sat[i] for i in range(n)},
    )
    print(f"[patrol] cached {int(in_win.sum())}/{n} in-window grid points -> {out.name}")


# ---- DRONE -------------------------------------------------------------------------------
def build_drone_cache(cfg: RaceCfg, grid_rt: np.ndarray) -> None:
    """Per-grid-frame drone rail (simple white/turf-boundary line) + cached numbered ordering."""
    out = CACHE_DIR / f"cv_walkthrough_cache_{cfg.name}_drone.npz"
    if out.exists():
        print(f"[drone] cache exists -> {out.name}")
        return
    video = VIDEO_DIR / f"drone_20260603_{cfg.suffix}.mp4"
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or FPS_VID
    n = len(grid_rt)
    have = np.zeros(n, bool)
    rails = [np.empty((0, 2)) for _ in range(n)]
    for i, rt in enumerate(grid_rt):
        vid_t = cfg.drone_race_t_to_vid(float(rt))
        if vid_t < 0 or vid_t > cfg.drone_finish_vid + 1:
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(vid_t * fps))
        ok, fr = cap.read()
        if not ok:
            continue
        rail = _drone_rail(fr)
        if rail is not None:
            rails[i] = rail
            have[i] = True
        else:
            have[i] = True  # frame exists even if rail not found (ordering still cached)
    cap.release()
    np.savez(
        out, race_t=grid_rt, have=have,
        **{f"rail_{i}": rails[i] for i in range(n)},
    )
    print(f"[drone] cached {int(have.sum())}/{n} grid points -> {out.name}")


def _drone_rail(fr: np.ndarray) -> np.ndarray | None:
    """Crude inner-rail / turf-boundary polyline for the drone (a moving cam, ordering only).

    The drone is metric-null (CV_RECON §2); we draw the rail only as a visual reference, fit as
    the lower turf/non-turf boundary on the left-mid of the frame. Returns an (N,2) polyline or
    None. (Labelled 'reference only' in the panel — never used for metres.)
    """
    h, w = fr.shape[:2]
    hsv = cv2.cvtColor(fr, cv2.COLOR_BGR2HSV)
    ss, vv = hsv[..., 1], hsv[..., 2]
    white = ((vv > 175) & (ss < 55)).astype(np.uint8)
    pts = []
    for y in range(int(0.30 * h), h, 4):
        row = np.where(white[y, : int(0.6 * w)] > 0)[0]
        if len(row) > 3:
            pts.append((float(np.median(row)), float(y)))
    if len(pts) < 12:
        return None
    pts = np.array(pts, float)
    a, b = np.polyfit(pts[:, 1], pts[:, 0], 1)
    yy = np.linspace(pts[:, 1].min(), pts[:, 1].max(), 60)
    xx = a * yy + b
    return np.column_stack([xx, yy])


# ============================================================================================
# CACHED FEED READOUTS (RF / drone numbered) — keyed by race_t, drawn as honest readout strips.
# ============================================================================================
def load_rf_readout(cfg: RaceCfg) -> dict:
    p = CACHE_DIR / ("cv_rf_longitudinal.parquet" if cfg.name == "R01"
                     else "cv_r08_rf_longitudinal.parquet")
    d = pq.read_table(str(p)).to_pydict()
    return {k: np.asarray(v) for k, v in d.items()}


def load_drone_numbered(cfg: RaceCfg) -> dict:
    p = CACHE_DIR / ("cv_drone_numbered.parquet" if cfg.name == "R01"
                     else "cv_r08_drone_numbered.parquet")
    d = pq.read_table(str(p)).to_pydict()
    out = {k: np.asarray(v) for k, v in d.items()}
    # R01 drone parquet is vid_t-keyed; add race_t.
    if "race_t" not in out and "vid_t" in out:
        out["race_t"] = np.array([cfg.drone_vid_to_race_t(float(v)) for v in out["vid_t"]])
    return out


# ============================================================================================
# RENDERING — the composite frame at one race_t.
# ============================================================================================
CW, CH = 1920, 1080          # full composite canvas
CENTER_W = 1100              # center recon panel width
PANEL_W = (CW - CENTER_W) // 2
PANEL_H = CH // 2


def _put(img, text, org, scale=0.6, color=(235, 235, 235), thick=1, bg=True):
    if bg:
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
        x, y = org
        cv2.rectangle(img, (x - 2, y - th - 3), (x + tw + 2, y + 3), (20, 20, 20), -1)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def _grey_panel(w: int, h: int, title: str, reason: str) -> np.ndarray:
    p = np.full((h, w, 3), 45, np.uint8)
    _put(p, title, (12, 26), 0.7, (180, 180, 180), 2)
    # diagonal hatch to read as "no coverage"
    for x in range(-h, w, 40):
        cv2.line(p, (x, 0), (x + h, h), (60, 60, 60), 1)
    _put(p, "NO COVERAGE", (12, h // 2 - 10), 0.9, (90, 90, 230), 2)
    for k, line in enumerate(_wrap(reason, 44)):
        _put(p, line, (12, h // 2 + 20 + k * 24), 0.55, (200, 200, 200), 1)
    return p


def _wrap(s: str, n: int) -> list[str]:
    words = s.split()
    out, cur = [], ""
    for wd in words:
        if len(cur) + len(wd) + 1 > n:
            out.append(cur)
            cur = wd
        else:
            cur = (cur + " " + wd).strip()
    if cur:
        out.append(cur)
    return out


def _fit_panel(img: np.ndarray, w: int, h: int) -> np.ndarray:
    """Letterbox a source frame (16:9) into the panel (w x h), padding to fill."""
    ih, iw = img.shape[:2]
    sc = min(w / iw, h / ih)
    nw, nh = int(iw * sc), int(ih * sc)
    rz = cv2.resize(img, (nw, nh))
    out = np.full((h, w, 3), 0, np.uint8)
    ox, oy = (w - nw) // 2, (h - nh) // 2
    out[oy:oy + nh, ox:ox + nw] = rz
    return out, sc, ox, oy


# ---- CENTER: top-down reconstruction ----------------------------------------------------
def render_center(cfg: RaceCfg, fused: dict, rt: float, live: dict, tails: dict) -> np.ndarray:
    """Top-down recon: track box (rail spline straight + 22.5 m width) + horses + ellipses."""
    P = np.full((CH, CENTER_W, 3), 18, np.uint8)
    margin_top, margin_bot = 70, 90
    plot_h = CH - margin_top - margin_bot
    pad_x = 90
    plot_w = CENTER_W - 2 * pad_x
    W = W_C_COURSE  # 22.5 m

    # vertical axis = distance-along-track (arclen 0..dist, finish at top); horizontal = y_lat 0..W.
    def to_px(arclen, ylat):
        fx = pad_x + (ylat / W) * plot_w
        fy = margin_top + (1.0 - arclen / cfg.dist) * plot_h
        return int(fx), int(fy)

    # track box + rail
    x0, _ = to_px(0, 0)
    x1, _ = to_px(0, W)
    cv2.rectangle(P, (x0, margin_top), (x1, margin_top + plot_h), (70, 70, 70), 1)
    # inner rail (y_lat=0) bold
    cv2.line(P, (x0, margin_top), (x0, margin_top + plot_h), (200, 200, 200), 2)
    _put(P, "RAIL", (x0 - 70, margin_top + plot_h // 2), 0.5, (200, 200, 200), 1)
    _put(P, f"outer (+{W:.0f} m)", (x1 - 30, margin_top - 12), 0.45, (140, 140, 140), 1)
    # finish line + start
    _put(P, "FINISH", (x0 + 6, margin_top - 12), 0.55, (90, 200, 255), 2)
    _put(P, "START", (x0 + 6, margin_top + plot_h + 22), 0.55, (160, 160, 160), 1)
    # BEND shading — fine 25 m cells, each cell on a bend gets a faint maroon fill (the geometry
    # phase map from #96). The shaded zones are exactly where the absolute lateral metre is lost.
    cell = 25
    ov = P.copy()
    for a in range(0, int(cfg.dist), cell):
        if is_bend(cfg, a + cell / 2):
            _, y_top = to_px(a + cell, 0)
            _, y_bot = to_px(a, 0)
            cv2.rectangle(ov, (x0 + 1, y_top), (x1 - 1, y_bot), (60, 30, 70), -1)
    cv2.addWeighted(ov, 0.5, P, 0.5, 0, P)
    # arclen gridlines + labels every 200 m
    for a in range(0, int(cfg.dist) + 1, 200):
        _, gy = to_px(a, 0)
        cv2.line(P, (x0, gy), (x1, gy), (55, 55, 55), 1)
        _put(P, f"{a}m", (x1 + 6, gy + 4), 0.4, (130, 130, 130), 1, bg=False)
    # legend tag for the shading
    _put(P, "maroon = BEND (lateral metre lost)", (x0 + 6, margin_top + plot_h + 44),
         0.42, (150, 110, 170), 1)
    # redraw rail/box over shading
    cv2.rectangle(P, (x0, margin_top), (x1, margin_top + plot_h), (70, 70, 70), 1)
    cv2.line(P, (x0, margin_top), (x0, margin_top + plot_h), (200, 200, 200), 2)

    # horses
    msk = np.isclose(fused["race_t"], rt, atol=1e-6)
    runners = fused["runner"][msk]
    arc = fused["arclen"][msk]
    yl = fused["y_lat_m"][msk]
    sda = fused["sd_arclen"][msk]
    sdl = fused["sd_ylat"][msk]
    # px-per-metre scales for ellipse
    ppm_arc = plot_h / cfg.dist
    ppm_lat = plot_w / W
    for k in np.argsort(arc):
        n = int(runners[k])
        a = float(arc[k])
        y = float(np.clip(yl[k], 0, W))
        bgr = runner_bgr(n)
        # fading tail
        tl = tails.get(n, [])
        for j, (ta, ty) in enumerate(tl[-8:]):
            alpha = (j + 1) / 9.0
            tx, tyy = to_px(ta, np.clip(ty, 0, W))
            cv2.circle(P, (tx, tyy), 2, tuple(int(c * alpha) for c in bgr), -1)
        px, py = to_px(a, y)
        # uncertainty ellipse (sd_arclen x sd_ylat) — the honest band
        ea = max(2, int(sda[k] * ppm_arc))
        el = max(2, int(sdl[k] * ppm_lat))
        ov = P.copy()
        cv2.ellipse(ov, (px, py), (el, ea), 0, 0, 360, bgr, -1)
        cv2.addWeighted(ov, 0.22, P, 0.78, 0, P)
        cv2.ellipse(P, (px, py), (el, ea), 0, 0, 360, bgr, 1)
        cv2.circle(P, (px, py), 9, bgr, -1)
        cv2.circle(P, (px, py), 9, (15, 15, 15), 1)
        _put(P, str(n), (px - 6, py + 5), 0.5, (15, 15, 15), 2, bg=False)

    # ---- HUD ----
    arc_field = float(np.median(arc)) if len(arc) else 0.0
    phase = race_phase(cfg, arc_field, rt)
    _put(P, f"{cfg.name}  TOP-DOWN RECONSTRUCTION  (fused 4-feed physics-kernel v2)",
         (16, 26), 0.62, (255, 255, 255), 1)
    _put(P, f"race_t = {rt:6.2f}s   arclen(field) = {arc_field:6.0f} m   phase = {phase}",
         (16, 50), 0.6, (120, 230, 255), 2)
    # live instruments line
    inst = "  ".join(live["labels"])
    _put(P, f"live: {inst}", (16, CH - 60), 0.5, (210, 210, 210), 1)
    # ellipse explanation (the pedagogical payload) when on a bend
    if phase == "BEND":
        med_sd = float(np.median(sdl)) if len(sdl) else 0.0
        msg = (f"BEND: aerial render foreshortens -> homography UNDER-DETERMINED (CV_RECON s7); "
               f"absolute lateral metre is unconstrained -> sd_ylat ~{med_sd:.1f} m "
               f"(box-capped at ~W/sqrt12). Only the SIGN/order survives.")
        for k, line in enumerate(_wrap(msg, 78)):
            _put(P, line, (16, CH - 38 + k * 18), 0.42, (150, 200, 255), 1)
    else:
        med_sd = float(np.median(sdl)) if len(sdl) else 0.0
        anchor = "draw" if rt < 5 else ("patrol metric anchor" if phase.startswith("STRAIGHT")
                                        or phase == "FINISH" else "feeds")
        _put(P, f"STRAIGHT: lateral metre anchored ({anchor}); sd_ylat ~{med_sd:.1f} m",
             (16, CH - 36), 0.45, (150, 230, 180), 1)
    return P


# ---- AERIAL panel ------------------------------------------------------------------------
def render_aerial(cfg: RaceCfg, ac: dict, i: int, rt: float) -> np.ndarray:
    have = bool(ac["have"][i])
    sel_frame = int(ac["sel_frame"][i]) if "sel_frame" in ac else -1
    if not have:
        return _grey_panel(PANEL_W, PANEL_H, "AERIAL render",
                           "render odometer/rail not resolved at this race_t "
                           "(single-cam lead-in or OCR gap).")
    video = VIDEO_DIR / f"aerial_20260603_{cfg.suffix}.mp4"
    img = _read_frame(str(video), sel_frame)
    if img is None:
        return _grey_panel(PANEL_W, PANEL_H, "AERIAL render", "frame decode failed.")
    half = int(ac["half_w"][i]) if int(ac["half_w"][i]) else img.shape[1] // 2
    rail = ac[f"rail_{i}"]
    gc = ac[f"gc_{i}"]
    arclen = float(ac["arclen"][i])
    bend = is_bend(cfg, arclen)
    # draw on the RIGHT (render) half only
    panel = img[:, half:].copy()
    px_off = half
    if len(rail):
        pts = (rail - [px_off, 0]).astype(np.int32)
        cv2.polylines(panel, [pts], False, (90, 220, 255), 2)
    # ground contacts
    for (gx, gy) in gc:
        cv2.circle(panel, (int(gx - px_off), int(gy)), 5, (60, 255, 120), -1)
        cv2.circle(panel, (int(gx - px_off), int(gy)), 5, (10, 10, 10), 1)
    # homography lane grid: valid on straights, BROKEN on bends (the key difficulty)
    _draw_aerial_homography(panel, rail, bend)
    out, _, _, _ = _fit_panel(panel, PANEL_W, PANEL_H)
    _put(out, "AERIAL render (split-screen RIGHT)  (live detect)", (10, 22), 0.5, (255, 255, 255), 1)
    _put(out, f"odometer arclen = {arclen:.0f} m", (10, PANEL_H - 14), 0.5, (90, 220, 255), 1)
    if bend:
        _put(out, "HOMOGRAPHY UNDER-DETERMINED (bend foreshortening)",
             (10, 44), 0.5, (90, 90, 240), 2)
    else:
        _put(out, "homography lane grid (straight: rail+width)", (10, 44), 0.45, (120, 230, 160), 1)
    return out


def _draw_aerial_homography(panel, rail, bend):
    """Draw a few lateral lane lines parallel to the rail. On a straight, solid green (the single
    rail + known width gives a workable map). On a BEND, draw them BROKEN/red — the algorithm has
    NO valid per-frame homography there (CV_RECON s7); we never draw a confident grid on a bend."""
    if len(rail) < 5:
        return
    # lateral lane lines = the rail polyline shifted +off px in x (schematic; only a VALID,
    # metric map on a straight — on a bend the foreshortening is non-linear so this constant
    # px shift is geometrically WRONG, which is exactly the point we visualise).
    step = 26
    for lane in (1, 2, 3):
        off = lane * step
        line = rail.copy()
        line[:, 0] = rail[:, 0] + off
        pts = line.astype(np.int32)
        if bend:
            # broken red dashes -> "no valid homography"
            for k in range(0, len(pts) - 4, 8):
                cv2.line(panel, tuple(pts[k]), tuple(pts[k + 3]), (70, 70, 235), 1)
        else:
            cv2.polylines(panel, [pts], False, (110, 210, 150), 1)


_CAP_CACHE: dict[str, cv2.VideoCapture] = {}


def _read_frame(video: str, frame_idx: int) -> np.ndarray | None:
    """Random-access a single frame, reusing a persistent VideoCapture per video (the open +
    seek dominates; re-opening per panel per frame is what made the first pass slow)."""
    cap = _CAP_CACHE.get(video)
    if cap is None:
        cap = cv2.VideoCapture(video)
        _CAP_CACHE[video] = cap
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
    ok, fr = cap.read()
    return fr if ok else None


# ---- PATROL panel ------------------------------------------------------------------------
def render_patrol(cfg: RaceCfg, pc: dict, i: int, rt: float) -> np.ndarray:
    in_win = bool(pc["in_win"][i])
    t0, t1 = pc["headon"]
    if not in_win:
        why = ("camera panning (route): head-on window is "
               f"race_t {t0 - cfg.patrol_offset:.0f}-{t1 - cfg.patrol_offset:.0f}s")
        return _grey_panel(PANEL_W, PANEL_H, "PATROL (head-on)", why)
    vid_t = rt + cfg.patrol_offset
    video = VIDEO_DIR / f"patrol_20260603_{cfg.suffix}.mp4"
    img = _read_frame(str(video), int(round(vid_t * FPS_VID)))
    if img is None:
        return _grey_panel(PANEL_W, PANEL_H, "PATROL (head-on)", "frame decode failed.")
    a, b = float(pc["a"]), float(pc["b"])
    h, w = img.shape[:2]
    # fixed rail line
    if np.isfinite(a):
        y0, y1 = int(0.24 * h), h - 1
        p0 = (int(a * y0 + b), y0)
        p1 = (int(a * y1 + b), y1)
        cv2.line(img, p0, p1, (255, 255, 255), 2)
        # parallel metric lane grid (perspective lane lines converging toward the rail's VP)
        for lane_px in (120, 240, 360, 480):
            lx0 = int(a * y0 + b) + int(lane_px * (y0 / h))
            lx1 = int(a * y1 + b) + lane_px
            cv2.line(img, (lx0, y0), (lx1, y1), (120, 200, 150), 1)
    gc = pc[f"gc_{i}"]
    for (cx, yg) in gc:
        cv2.circle(img, (int(cx), int(yg)), 6, (60, 255, 120), -1)   # ground contact
        cv2.line(img, (int(cx), int(yg) - 60), (int(cx), int(yg)), (60, 255, 120), 1)  # plumb
    out, sc, ox, oy = _fit_panel(img, PANEL_W, PANEL_H)
    _put(out, "PATROL head-on (fixed rail + metric lanes)  (live detect)",
         (10, 22), 0.5, (255, 255, 255), 1)
    _put(out, f"silk ground-contacts: {len(gc)}   cv(slope)={float(pc['slope_cv']):.3f}",
         (10, PANEL_H - 14), 0.5, (120, 230, 160), 1)
    return out


# ---- RF panel ----------------------------------------------------------------------------
def render_rf(cfg: RaceCfg, rf: dict, rt: float) -> np.ndarray:
    video = VIDEO_DIR / f"full_20260603_{cfg.suffix}.mp4"
    vid_t = rt + cfg.rf_offset
    if vid_t < 0 or vid_t > (cfg.winner_finish_t + cfg.rf_offset + 8):
        return _grey_panel(PANEL_W, PANEL_H, "RF / race-film", "outside RF broadcast window.")
    img = _read_frame(str(video), int(round(vid_t * FPS_VID)))
    if img is None:
        return _grey_panel(PANEL_W, PANEL_H, "RF / race-film", "frame decode failed.")
    out, sc, ox, oy = _fit_panel(img, PANEL_W, PANEL_H)
    _put(out, "RF / race-film side-on  (cached order/margins)", (10, 22), 0.5, (255, 255, 255), 1)
    # cached legend ORDER nearest this race_t
    kind = rf["kind"]
    legend = (kind == "legend_order") if "kind" in rf else np.zeros(len(rf["race_t"]), bool)
    rts = rf["race_t"]
    if legend.any():
        lt = rts[legend]
        tnear = lt[np.argmin(np.abs(lt - rt))]
        if abs(tnear - rt) < 6.0:
            sel = legend & np.isclose(rts, tnear)
            runs = rf["runner"][sel]
            gaps = rf["arclen_or_gap"][sel]
            order = runs[np.argsort(-gaps)]  # higher arclen_or_gap = nearer the lead
            txt = "order " + ">".join(str(int(x)) for x in order[:8])
            _put(out, f"{txt}  (legend @{tnear:.0f}s)", (10, 46), 0.5, (90, 220, 255), 1)
    # finish-margin gauge near the line
    fin = (kind == "finish_margin") if "kind" in rf else np.zeros(len(rf["race_t"]), bool)
    if fin.any() and rt > cfg.winner_finish_t - 12:
        runs = rf["runner"][fin]
        mg = rf["margin_len"][fin]
        ordr = np.argsort(mg)
        gy = PANEL_H - 80
        _put(out, "finish margins (len):", (10, gy), 0.45, (255, 230, 120), 1)
        for j, k in enumerate(ordr[:6]):
            _put(out, f"#{int(runs[k]):2d} +{mg[k]:.1f}", (10, gy + 18 + j * 16), 0.42,
                 runner_bgr(int(runs[k]))[::-1], 1)
    return out


# ---- DRONE panel -------------------------------------------------------------------------
def render_drone(cfg: RaceCfg, dc: dict, dn: dict, i: int, rt: float) -> np.ndarray:
    have = bool(dc["have"][i])
    if not have:
        return _grey_panel(PANEL_W, PANEL_H, "DRONE", "outside drone coverage (pre-gate / post-cut).")
    vid_t = cfg.drone_race_t_to_vid(float(rt))
    video = VIDEO_DIR / f"drone_20260603_{cfg.suffix}.mp4"
    img = _read_frame(str(video), int(round(vid_t * FPS_VID)))
    if img is None:
        return _grey_panel(PANEL_W, PANEL_H, "DRONE", "frame decode failed.")
    rail = dc[f"rail_{i}"]
    if len(rail):
        cv2.polylines(img, [rail.astype(np.int32)], False, (90, 220, 255), 2)
    out, sc, ox, oy = _fit_panel(img, PANEL_W, PANEL_H)
    _put(out, "DRONE (moving cam; ordering only)  rail=reference", (10, 22), 0.5, (255, 255, 255), 1)
    # cached numbered ordering cross-check near this race_t
    drt = dn["race_t"]
    near = np.abs(drt - rt) < 1.0
    if near.any():
        nums = dn["number"][near]
        ys = dn["y_lat"][near]
        # one read per number (nearest)
        seen = {}
        for nm, yv in zip(nums, ys):
            seen[int(nm)] = float(yv)
        order = sorted(seen, key=lambda k: seen[k])  # rail->wide by y_lat ordering
        txt = "lat order rail->wide: " + ">".join(str(x) for x in order[:8])
        _put(out, txt, (10, 46), 0.45, (120, 230, 160), 1)
        _put(out, f"{len(seen)} numbered (cached)", (10, PANEL_H - 14), 0.45, (120, 230, 160), 1)
    else:
        _put(out, "no numbered drone read near this race_t (cached)",
             (10, PANEL_H - 14), 0.45, (160, 160, 160), 1)
    return out


# ---- live-instruments line --------------------------------------------------------------
def live_instruments(cfg: RaceCfg, ac, pc, dc, dn, rf, i, rt) -> dict:
    labels = []
    arc = float(cfg.race_t_to_arclen(np.array([rt]))[0])
    bend = is_bend(cfg, arc)
    # aerial
    if bool(ac["have"][i]):
        labels.append("aerial:ORDERING-only(bend)" if bend else "aerial:live")
    else:
        labels.append("aerial:OUT")
    # patrol
    if bool(pc["in_win"][i]):
        labels.append("patrol:live(metric)")
    else:
        labels.append("patrol:OUT(panning)")
    # drone
    labels.append("drone:live" if bool(dc["have"][i]) else "drone:OUT")
    # RF
    rf_in = (rt >= 0 and rt <= cfg.winner_finish_t + 8)
    labels.append("RF:live" if rf_in else "RF:OUT")
    # draw anchor only meaningful at the gate
    if rt < 4.0:
        labels.append("draw-anchor")
    return {"labels": labels}


# ============================================================================================
# MAIN — assemble PNG sequence + ffmpeg to mp4.
# ============================================================================================
def render_race(name: str) -> None:
    cfg = make_cfg(name)
    fused = load_fused(cfg)
    grid_rt = _grid_race_t(cfg, fused)
    print(f"\n=== {name} — {len(grid_rt)} composite frames "
          f"(race_t {grid_rt[0]:.1f}..{grid_rt[-1]:.1f}s) ===")

    # 1) build / load per-feed overlay caches (decode-once detector passes)
    build_aerial_cache(cfg, grid_rt)
    build_patrol_cache(cfg, grid_rt)
    build_drone_cache(cfg, grid_rt)
    ac = dict(np.load(CACHE_DIR / f"cv_walkthrough_cache_{name}_aerial.npz", allow_pickle=True))
    pc = dict(np.load(CACHE_DIR / f"cv_walkthrough_cache_{name}_patrol.npz", allow_pickle=True))
    dc = dict(np.load(CACHE_DIR / f"cv_walkthrough_cache_{name}_drone.npz", allow_pickle=True))
    rf = load_rf_readout(cfg)
    dn = load_drone_numbered(cfg)

    # 2) composite each grid race_t -> PNG
    png_dir = FIG_DIR / f"_walkthrough_{name}_frames"
    png_dir.mkdir(parents=True, exist_ok=True)
    for f in png_dir.glob("*.png"):
        f.unlink()

    tails: dict[int, list] = {}
    for i, rt in enumerate(grid_rt):
        rt = float(rt)
        # update tails from this frame's fused positions
        msk = np.isclose(fused["race_t"], rt, atol=1e-6)
        for n, a, y in zip(fused["runner"][msk], fused["arclen"][msk], fused["y_lat_m"][msk]):
            tails.setdefault(int(n), []).append((float(a), float(y)))

        live = live_instruments(cfg, ac, pc, dc, dn, rf, i, rt)
        center = render_center(cfg, fused, rt, live, tails)
        a_pan = render_aerial(cfg, ac, i, rt)
        p_pan = render_patrol(cfg, pc, i, rt)
        r_pan = render_rf(cfg, rf, rt)
        d_pan = render_drone(cfg, dc, dn, i, rt)

        canvas = np.full((CH, CW, 3), 12, np.uint8)
        canvas[:, PANEL_W:PANEL_W + CENTER_W] = center
        canvas[0:PANEL_H, 0:PANEL_W] = a_pan                       # top-left  AERIAL
        canvas[PANEL_H:CH, 0:PANEL_W] = p_pan                      # bot-left  PATROL
        canvas[0:PANEL_H, PANEL_W + CENTER_W:CW] = r_pan           # top-right RF
        canvas[PANEL_H:CH, PANEL_W + CENTER_W:CW] = d_pan          # bot-right DRONE
        cv2.imwrite(str(png_dir / f"f{i:04d}.png"), canvas)
        if i % 20 == 0:
            print(f"  frame {i:4d}/{len(grid_rt)}  race_t {rt:6.1f}s")

    # 3) ffmpeg -> mp4
    out_mp4 = FIG_DIR / f"cv_recon_walkthrough_{name}.mp4"
    if out_mp4.exists():
        out_mp4.unlink()
    cmd = [
        "ffmpeg", "-y", "-framerate", str(OUT_FPS),
        "-i", str(png_dir / "f%04d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23",
        "-vf", "scale=1920:1080", str(out_mp4),
    ]
    print("[ffmpeg]", " ".join(cmd))
    subprocess.run(cmd, check=True, capture_output=True)
    dur = len(grid_rt) / OUT_FPS
    print(f"[done] {out_mp4}  ({len(grid_rt)} frames, {OUT_FPS:.0f} fps, {dur:.1f}s)")


def main() -> None:
    races = sys.argv[1:] or ["R01", "R08"]
    os.environ.setdefault("PYTHONHASHSEED", "0")
    print(f"geometry: W={W_C_COURSE} m, home straight={HOME_STRAIGHT_C} m, bend R~{BEND_RADIUS_M:.0f} m, "
          f"patrol W={PATROL_W} m")
    for r in races:
        render_race(r)


if __name__ == "__main__":
    main()
