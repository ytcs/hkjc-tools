"""Race reconstruction pipeline: race info + replay videos -> per-horse paths.

``RaceRun(info).run()`` executes the stages below, each cached under ``data_cache/recon/<key>/`` and
skipped when its outputs exist (``force=`` re-runs named stages). Stages:

  features  AVV render: unique 10 Hz frames -> turf ORB features + consecutive steps
  pose      camera pose graph (sequential/long-baseline/loop edges) -> world poses
  detect    avatar detections (self-supervised YOLO) + silk histograms
  hud       race clock, odometer (distance-to-go), legend running order (recognition-only OCR)
  worldmap  mosaic of the registered frames (static world: turf, rails, painted markers)
  rails     racing rail on the map; marker grid -> px/m + finish anchor
  observe   detections -> (d, lateral) metres (odometer-tick distance calibration) -> world tracklets
  identify  tracklet -> saddlecloth number (legend order + elimination + silk)
  finish    RF/drone finish-line camera: saddlecloth OCR -> finish order + spatial gaps
  fuse      Bayesian fusion of all feeds' proposals with physics/geometry priors -> paths.parquet
  validate  official comparison (validation only)

Memory: run under a hard cap (``scripts/capped.sh``) — the MP4s and feature caches are large.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
from dataclasses import replace

from . import (
    detect as _detect,
    film,
    finishcam,
    fusion,
    identity,
    ocr,
    pose,
    rails,
    register,
    tracking,
    video,
    worldmap,
)
from ..data import RaceInfo, download_replay, replay_path

FPS = 25.0


def track_app_enabled() -> bool:
    """Appearance-aware tracking (HKJC_TRACK_APP=1; needs the silk stage)."""
    return os.environ.get("HKJC_TRACK_APP", "0") != "0"


def film_enabled() -> bool:
    """The race-film metric anchor: ON by default, HKJC_FILM=0 disables (``RaceRun.observe``)."""
    return os.environ.get("HKJC_FILM", "1") != "0"


STAGES = (
    "frames",
    "features",
    "pose",
    "worldmap",
    "detect",
    "silk",  # v2 silk signatures + official prototypes (E-step appearance term; HKJC_SILK_W > 0 only)
    "hud",
    "rails",
    "film",  # race-film marker crossings -> the metric's constant offset (HKJC_FILM=0: skipped)
    "observe",
    "finish",  # independent of the render; before identify (finish-camera identity votes)
    "identify",
    "fuse",
    "validate",
)
OUTPUTS = {
    "frames": ["frames.bin", "frames_idx.npz", "frames_clock.npy"],
    "features": ["feats.npz"],
    "pose": ["pose.npz"],
    "detect": ["dets.npz"],
    "silk": ["silk.npz"],
    "hud": ["hud.npz", "legend.npz"],
    "film": ["film.npz"],
    "worldmap": ["map.png", "map_cnt.npy", "map_meta.npz"],
    "rails": ["rail.npz", "metric.npz"],
    "observe": ["tracks.npz"],
    "identify": ["labels.npz"],
    "finish": ["finish.npz"],
    "fuse": ["paths.parquet", "paths_video.parquet"],
    "validate": ["validation.json"],
}


def _nproc() -> int:
    """Worker processes per race (env HKJC_NPROC; default: all cores / parallel races)."""
    return max(1, int(os.environ.get("HKJC_NPROC", "0")) or (os.cpu_count() or 4))


_CACHE = None
_FEATS_PAR = None


def _feat_job(args):
    k, nfeat = args
    i, p = _CACHE.idx[k], _CACHE.panel(k)
    lt = video.legend_top(p)
    m = register.world_mask_sel(p, video.hud_mask(p, lt))
    if _DET is not None:  # HKJC_WORLD_MASK=yolo: also mask the avatar detector's boxes (+ shadows)
        m = register.mask_boxes(m, _DET.panel(p, lt))
    f = register.extract(p, m, nfeat)
    return int(i), lt, f, p.shape[:2]


_DET = None


def _det_init():
    global _DET
    _DET = _detect.AvatarDetector()


def _det_job(k):
    i, p = int(_CACHE.idx[k]), _CACHE.panel(k)
    rs = _DET.panel(p, video.legend_top(p))
    return i, rs, [_detect.silk_hist(p, r) for r in rs]


def _step_job(k):
    m = register.match(_FEATS_PAR[k - 1], _FEATS_PAR[k])
    return np.r_[m.S.ravel(), m.n_inl, m.resid] if m else None


class RaceRun:
    def __init__(self, info: RaceInfo, root: str | Path = "data_cache/recon", log=print):
        self.info = info
        self.dir = Path(root) / info.key
        self.dir.mkdir(parents=True, exist_ok=True)
        self.log = log

    # ------------------------------------------------------------------ plumbing
    def p(self, name: str) -> Path:
        return self.dir / name

    def video(self, feed: str = "aerial") -> Path:
        path = replay_path(self.info.date, self.info.race_no, feed)
        if not path.exists():
            download_replay(self.info.date, self.info.race_no, feed)
        return path

    def done(self, stage: str) -> bool:
        return all(self.p(o).exists() for o in OUTPUTS[stage])

    # stages that read only the videos/frame cache run CONCURRENTLY with the render chain; the
    # chain waits for them right before their first consumer (results identical, wall time shorter)
    BACKGROUND = {"hud": ("frames", ("film", "observe")), "finish": (None, ("identify",))}

    def run(self, stages=STAGES, force=()):
        import multiprocessing as mp

        todo = [st for st in stages if st not in self.done_set() or st in force]
        if not film_enabled():  # HKJC_FILM=0 (see observe)
            todo = [st for st in todo if st != "film"]
        if not (float(os.environ.get("HKJC_SILK_W", 0)) > 0 or track_app_enabled()):  # off by default
            todo = [st for st in todo if st != "silk"]
        bg = {}

        launched = set()

        def launch(st):
            launched.add(st)

            def work():
                t0 = time.time()
                getattr(self, st)()
                self.log(f"{self.info.key} {st}: {time.time() - t0:.0f}s (background)")

            pr = mp.get_context("fork").Process(target=work)
            pr.start()
            bg[st] = pr

        def wait(st):
            pr = bg.pop(st)
            pr.join()
            if pr.exitcode != 0:
                raise RuntimeError(f"{self.info.key}: background stage {st} failed")

        def ready(b):  # prerequisite done (or never in this run's todo list)
            after = self.BACKGROUND[b][0]
            return after is None or after not in todo or after in finished

        finished = set()
        for b in self.BACKGROUND:
            if b in todo and ready(b):
                launch(b)
        for st in todo:
            for b in list(bg):
                if st in self.BACKGROUND[b][1]:
                    wait(b)
            if st in self.BACKGROUND:
                continue  # runs (or ran) in the background
            t0 = time.time()
            getattr(self, st)()
            self.log(f"{self.info.key} {st}: {time.time() - t0:.0f}s")
            finished.add(st)
            for b in self.BACKGROUND:
                if b in todo and b not in launched and ready(b):
                    launch(b)
        for b in list(bg):
            wait(b)

    def done_set(self):
        return {st for st in STAGES if self.done(st)}

    # ------------------------------------------------------------------ stages
    def frames(self):
        n = video.FrameCache.build(self.video(), self.dir)
        self.log(f"   frames: {n} unique render frames cached")

    def cache(self):
        return video.FrameCache(self.dir)

    def features(self, nfeat: int = 3000):
        """ORB features of every unique render frame + the sequential step matches. Per-frame work
        is independent and OpenCV's RANSAC seeds its RNG per call, so a process pool gives results
        IDENTICAL to the sequential loop (86 s -> a few s on 14 cores)."""
        import multiprocessing as mp

        cache = self.cache()
        global _CACHE, _FEATS_PAR
        _CACHE = cache
        yolo = os.environ.get("HKJC_WORLD_MASK", "v1") == "yolo"  # off by default (not adopted: run Y)
        nw = _nproc()
        if yolo:  # one detector per worker; one accelerator context (as in ``detect``)
            accel = os.environ.get("HKJC_DEVICE", "cpu").startswith(("intel:", "cuda"))
            nw = 1 if accel else min(nw, 4)
        with mp.get_context("fork").Pool(nw, initializer=_det_init if yolo else None) as pool:
            res = pool.map(_feat_job, [(k, nfeat) for k in range(len(cache))], chunksize=8)
        # a NEW pool: workers fork the parent's state, so the features must exist before forking
        _FEATS_PAR = [r[2] for r in res]
        pairs = [k for k in range(1, len(res)) if res[k][0] - res[k - 1][0] <= 6]
        with mp.get_context("fork").Pool(_nproc()) as pool:
            ms = dict(zip(pairs, pool.map(_step_job, pairs, chunksize=8)))
        _CACHE = _FEATS_PAR = None
        steps = [
            ms.get(k) if ms.get(k) is not None else np.full(11, np.nan) for k in range(len(res))
        ]
        np.savez_compressed(
            self.p("feats.npz"),
            idx=np.array([r[0] for r in res]),
            lrow=np.array([r[1] for r in res]),
            pts=np.concatenate([r[2].pts for r in res]),
            des=np.concatenate([r[2].des for r in res]),
            off=np.cumsum([0] + [len(r[2].pts) for r in res]),
            steps=np.array(steps),
            shape=np.array(res[0][3]),
        )

    def pose(self):
        z = dict(np.load(self.p("feats.npz")))
        G, info = pose.solve_race(z)
        np.savez(self.p("pose.npz"), idx=z["idx"], G=G, shape=z["shape"], **info)
        self.log(f"   pose: {info['summary']}")

    def detect(self):
        rows, hists, box = [], [], None
        import multiprocessing as mp

        import cv2

        cache = self.cache()
        x0, x1, y0, y1 = cache.box
        box = video.PanelBox(x0, x1, y0, y1)
        P = np.load(self.p("pose.npz"))
        k_of = {int(f): k for k, f in enumerate(P["idx"])}
        img_map, _cnt, _to = self._map()
        meta = np.load(self.p("map_meta.npz"))
        mx0, my0 = meta["bounds"]
        sc = float(meta["sc"])
        A = np.array([[sc, 0, -mx0 * sc], [0, sc, -my0 * sc], [0, 0, 1]])
        ry, rb, hy, hb = [], [], [], []
        # frames are independent: detect in a process pool (one ONNX session per worker); each
        # frame's detections are computed exactly as in the sequential loop
        global _CACHE
        _CACHE = cache
        nw = min(_nproc(), 4)  # ONNX Runtime threads inside each worker too
        if os.environ.get("HKJC_DEVICE", "cpu").startswith(("intel:", "cuda")):
            nw = 1  # one accelerator context; the device serialises work anyway
        with mp.get_context("fork").Pool(nw, initializer=_det_init) as pool:
            for i, rs, hs in pool.imap(_det_job, range(len(cache)), chunksize=8):
                for r, h in zip(rs, hs):
                    ry.append([i, *r[:5]])
                    hy.append(h)
        _CACHE = None
        nfr = len(cache)
        yolo_med = (
            float(
                np.median(
                    np.bincount(np.searchsorted(cache.idx, np.array(ry)[:, 0]), minlength=nfr)
                )
            )
            if ry
            else 0.0
        )
        if yolo_med < 0.6 * self.info.n:  # YOLO blind here (e.g. AWT dirt): map-difference pass
            for i, p in cache:
                if i not in k_of:
                    continue
                lt = video.legend_top(p)
                back = cv2.warpAffine(
                    img_map,
                    (A @ P["G"][k_of[i]])[:2],
                    (p.shape[1], p.shape[0]),
                    flags=cv2.WARP_INVERSE_MAP,
                )
                for r in _detect.bgsub_blobs(p, back, video.hud_mask(p, lt)):
                    rb.append([i, *r[:5]])
                    hb.append(_detect.silk_hist(p, r))
        ry, rb = np.array(ry).reshape(-1, 6), np.array(rb).reshape(-1, 6)
        nfr = len(cache)
        yolo_med = (
            float(np.median(np.bincount(np.searchsorted(cache.idx, ry[:, 0]), minlength=nfr)))
            if len(ry)
            else 0.0
        )
        # Race-level switch: on turf the self-supervised YOLO is precise and complete, and adding
        # background-subtraction blobs only adds duplicates/false positives (shattered R08 tracking);
        # where YOLO is blind (AWT dirt: median ~1/frame) the map-difference detector takes over.
        if yolo_med >= 0.6 * self.info.n:
            rows, hists, src = ry, np.array(hy), "yolo"
        else:
            nb = 14  # silk_hist length
            hy = np.array(hy).reshape(len(ry), nb) if len(ry) else np.zeros((0, nb))
            hb = np.array(hb).reshape(len(rb), nb) if len(rb) else np.zeros((0, nb))
            keep = np.ones(len(rb), bool)
            for f in np.unique(rb[:, 0]):
                jb = np.where(rb[:, 0] == f)[0]
                a_ = ry[ry[:, 0] == f]
                if len(a_):
                    dmin = np.hypot(
                        rb[jb, 1][:, None] - a_[:, 1][None], rb[jb, 2][:, None] - a_[:, 2][None]
                    ).min(1)
                    keep[jb[dmin < 10.0]] = False
            rows = np.vstack([ry, rb[keep]])
            hists = np.vstack([hy, hb[keep]]) if len(hy) else hb[keep]
            src = "yolo+bgsub"
        self.log(
            f"   detect: YOLO median {yolo_med:.1f}/frame (field {self.info.n}) -> {src}; "
            f"{len(rows)} detections"
        )
        np.savez_compressed(
            self.p("dets.npz"),
            rows=rows,
            hist=hists,
            box=np.array([box.x0, box.x1, box.y0, box.y1]),
            source=src,
        )

    def silk(self):
        """v2 silk signature of every detection (``detect.silk_feature``, from the aerial video's render
        panel) + the OFFICIAL silk prototype of every runner (``detect.silk_prototype``; pre-race
        input). Used only by the detection-level E-step (``assoc.assign_frames``)."""
        import av

        from ..data.silks import silk_path

        rows = np.load(self.p("dets.npz"))["rows"]
        x0, x1, y0, y1 = [int(b) for b in np.load(self.p("frames_idx.npz"))["box"]]
        feat = np.zeros((len(rows), 2 * _detect.SILK_NH + 2), np.float32)
        order = np.argsort(rows[:, 0], kind="stable")
        fr_sorted = rows[order, 0].astype(int)
        want = set(fr_sorted.tolist())
        last = max(want) if want else -1
        with av.open(str(self.video())) as c:
            c.streams.video[0].thread_type = "AUTO"
            for i, fr in enumerate(c.decode(video=0)):
                if i > last:
                    break
                if i not in want:
                    continue
                panel = fr.to_ndarray(format="bgr24")[y0:y1, x0:x1]
                a, b = np.searchsorted(fr_sorted, [i, i + 1])
                for j in order[a:b]:
                    feat[j] = _detect.silk_feature(panel, rows[j, 1:5])
        nos, proto = [], []
        for r in self.info.runners:
            pth = silk_path(r.horse_id)
            if pth is not None:
                nos.append(r.no)
                proto.append(_detect.silk_prototype(pth))
        np.savez(self.p("silk.npz"), feat=feat, proto_no=np.array(nos), proto=np.array(proto))
        self.log(f"   silk: {len(rows)} detections, {len(nos)}/{len(self.info.runners)} official silks")

    def hud(self):
        rows, leg = ocr.read_render_hud_cached(self.cache(), n_field=max(14, self.info.n))
        np.savez(self.p("hud.npz"), rows=rows)
        np.savez(self.p("legend.npz"), rows=leg)

    def worldmap(self, scale: float = 1.0):
        import cv2

        z = np.load(self.p("pose.npz"))
        G, shape = z["G"], tuple(z["shape"])
        k_of = {int(f): k for k, f in enumerate(z["idx"])}
        mo = worldmap.Mosaic(worldmap.bounds_of(G, shape), scale)
        for i, p in self.cache():
            if i in k_of:
                mo.add(p, video.hud_mask(p), G[k_of[i]])
        cv2.imwrite(str(self.p("map.png")), mo.image())
        np.save(self.p("map_cnt.npy"), mo.cnt.astype(np.float16))
        np.savez(self.p("map_meta.npz"), bounds=np.array([mo.x0, mo.y0]), sc=scale)

    def _final_pass_map(self, P, out, rf, span):
        """Mosaic of only the frames whose rail station lies in the last ``span`` world px of the
        trip (the final pass over the finish area), cropped to those frames' bounds (memory: a
        full-extent float mosaic OOM'd R03 under a 3 GB cap). -> (img, cnt, to_px)."""
        from . import worldmap

        G = P["G"]
        k_of = {int(f): k for k, f in enumerate(P["idx"])}
        s_of_k = rf.s[np.clip(out["st_of_frame"], 0, len(rf.s) - 1)]
        sel = np.where(s_of_k >= rf.s[-1] - span - 1500)[0]
        if len(sel) < 20:
            return self._map()
        mo = worldmap.Mosaic(worldmap.bounds_of(G[sel], tuple(P["shape"])), 1.0)
        keep = set(sel.tolist())
        for i, p in self.cache():
            k = k_of.get(i)
            if k is not None and k in keep:
                mo.add(p, video.hud_mask(p), G[k])
        img, cnt = mo.image(), mo.cnt
        del mo
        x0, y0 = worldmap.bounds_of(G[sel], tuple(P["shape"]))[:2]
        return img, cnt, (lambda W: np.asarray(W) - [x0, y0])

    def _map(self):
        import cv2

        img = cv2.imread(str(self.p("map.png")))
        cnt = np.load(self.p("map_cnt.npy")).astype(np.float32)
        meta = np.load(self.p("map_meta.npz"))
        x0, y0 = meta["bounds"]
        sc = float(meta["sc"])
        return img, cnt, (lambda W: (np.asarray(W) - [x0, y0]) * sc)

    def world_dets(self):
        """Detections (gated) mapped to world px through the camera poses."""
        P = np.load(self.p("pose.npz"))
        D = np.load(self.p("dets.npz"))
        k_of = {int(f): k for k, f in enumerate(P["idx"])}
        rows = D["rows"]
        keep = np.array([int(f) in k_of for f in rows[:, 0]], bool)
        # HUD gate: the detector fires on disclaimer glyphs; inside that band only confident boxes
        keep &= ~((rows[:, 2] < video.DISCLAIMER_ROWS) & (rows[:, 5] < 0.45))
        # static-overlay gate: a "detection" recurring at the same PANEL position in many frames is
        # glyph furniture (e.g. the disclaimer word "phones" in R02A, which got numbered as a horse);
        # horses move through the panel
        q = np.round(rows[:, 1:3] / 8.0).astype(int)
        key = q[:, 0] * 10_000 + q[:, 1]
        uk, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
        nfr = max(1, len(np.unique(rows[:, 0])))
        # (only inside the HUD band: the render camera FOLLOWS the field, so real horses also sit
        #  near-static in the panel for long stretches)
        keep &= ~((cnt[inv] >= 0.04 * nfr) & (rows[:, 2] < video.DISCLAIMER_ROWS + 20))
        rows, hist = rows[keep], D["hist"][keep]
        self._wd_idx = np.where(keep)[0]  # dets.npz row index of each kept detection (silk lookup)
        G = P["G"][[k_of[int(f)] for f in rows[:, 0]]]
        X = np.einsum("nij,nj->ni", G, np.c_[rows[:, 1:3], np.ones(len(rows))])[:, :2]
        return rows, X, hist, P

    def rails(self):
        img, cnt, to_px = self._map()
        rows, X, _h, P = self.world_dets()
        out = rails.extract_racing_rail(img, cnt, to_px, P["G"], tuple(P["shape"]), X)
        g = self.info.geometry
        exp_gap = g.width("A") - self.info.width if self.info.surface == "turf" else 0.0
        shift, ssup = rails.movable_rail_check(
            img, cnt, to_px, out["rail"], int(out["inner"]), X, exp_gap
        )
        if shift > 0:  # extracted the FIXED rail: move out to the movable racing rail and re-snap
            T = np.gradient(out["rail"], axis=0)
            T /= np.linalg.norm(T, axis=1, keepdims=True)
            nout = -int(out["inner"]) * rails.right_normal(T)
            moved = out["rail"] + shift * nout
            out["rail"], out["support"] = rails.refine_rail(
                img, cnt, to_px, moved, int(out["inner"]), out["support"]
            )
            out["rail"], out["support"] = rails.despike_world(out["rail"], out["support"])
            self.log(
                f"   rails: extracted ridge was the FIXED rail -> moved {shift:.0f} px out to the "
                f"movable rail (course gap {exp_gap:.1f} m, ridge support {ssup:.0%})"
            )
        np.savez(self.p("rail.npz"), frame_idx=P["idx"], **out)
        rf = rails.RailFrame.from_points(out["rail"], int(out["inner"]))
        # painted markers are drawn only near the END of the race: profile them on a FINAL-PASS
        # mosaic (frames whose time-ordered rail station lies in the last 9000 px). On trips that
        # pass the finish area twice (HV 2200) the all-frames mean faded them below detection.
        mimg, mcnt, mto_px = self._final_pass_map(P, out, rf, 9000.0)
        ss, resp = rails.marker_profile(mimg, mcnt, mto_px, rf, rf.s[-1] - 9000, rf.s[-1])
        del mimg, mcnt
        np.savez(self.p("marker_profile.npz"), ss=ss, resp=resp)  # observe: tick-seeded grid fit
        fit = rails.fit_marker_grid(ss, resp)
        if fit is None:
            # markers are optional: the odometer ticks carry the metric (observe stage); give a
            # nominal guess that the marker/tick consistency check will reject
            self.log("   rails: painted markers NOT found -> distance from odometer ticks only")
            ppm, sF, ks, ps = 18.0, float(rf.s[-1]), np.zeros(0, int), np.zeros(0)
        else:
            ppm, sF, ks, ps = fit
        g = self.info.geometry
        expected = g.width("A") - self.info.width if self.info.surface == "turf" else 0.0
        gap, gsup = rails.fixed_rail_gap(
            img, cnt, to_px, out["rail"], int(out["inner"]), ppm, expected
        )
        np.savez(
            self.p("metric.npz"),
            ppm=ppm,
            sF=sF,
            ks=ks,
            ps=ps,
            fixed_gap=gap,
            fixed_gap_expected=expected,
            fixed_gap_support=gsup,
        )
        self.log(
            f"   rails: support {out['support'].mean():.3f}; markers k={ks.tolist()} ppm {ppm:.3f}"
        )

    def clock_f0(self) -> float:
        r = np.load(self.p("hud.npz"))["rows"]
        ok = ~np.isnan(r[:, 1])
        return float(np.median(r[ok, 0] - FPS * r[ok, 1]))

    def film(self):
        """Leader crossings of the REAL painted markers in the race film (``recon.film``); the
        finish time is located from the HUD odometer reaching its minimum (no official data)."""
        hud = np.load(self.p("hud.npz"))["rows"]
        g = np.isfinite(hud[:, 2])
        mk, tt = np.zeros(0, int), np.zeros(0)
        if g.sum() >= 10:
            t_last = (hud[g, 0][np.argmin(hud[g, 2])] - self.clock_f0()) / FPS
            g_ = self.info.geometry
            mx = g_.home_straight(self.info.course) - film.STRAIGHT_MARGIN
            got = film.crossings(self.video(), hud, t_last, max_marker=mx)
            mk = np.array(sorted(got, reverse=True), int)
            tt = np.array([got[m] for m in mk], float)
        np.savez(self.p("film.npz"), markers=mk, t=tt)
        self.log(f"   film: {len(mk)} marker crossings {dict(zip(mk.tolist(), tt.round(2)))}")

    def observe(self):
        rows, X, hist, _P = self.world_dets()
        seed = os.environ.get("HKJC_PERTURB_SEED")  # DIAGNOSTIC: robustness ensembles
        if seed:  # drop 1 % of detections (seeded) -> spread of the downstream gate metrics
            keep = np.random.default_rng(int(seed)).random(len(rows)) >= 0.01
            rows, X, hist = rows[keep], X[keep], hist[keep]
            self._wd_idx = self._wd_idx[keep]
        z = np.load(self.p("rail.npz"))
        m = np.load(self.p("metric.npz"))
        rf = rails.RailFrame.from_points(z["rail"], int(z["inner"]))
        k_of = {int(f): k for k, f in enumerate(z["frame_idx"])}
        hint = rf.s[z["st_of_frame"][[k_of[int(f)] for f in rows[:, 0]]]]
        s_px, lat_px = rf.project(X, s_hint=hint, window=1200.0)
        ppm, sF = float(m["ppm"]), float(m["sF"])
        fu = np.unique(rows[:, 0])
        # leader position per frame. A front detection > 8 m clear of the next is a false positive
        # ONLY if that gap is not persistent: a genuinely clear leader keeps its margin frame after
        # frame (R02A: the winner was 13 m clear near the line; dropping it made the odometer ticks
        # lag 15 m and the finish came out 1.06 s early). REJECTED: the per-frame 8 m cut alone.
        from scipy.ndimage import median_filter

        s0 = np.empty(len(fu))
        s1 = np.full(len(fu), np.nan)
        for n, f in enumerate(fu):
            ss = np.sort(s_px[rows[:, 0] == f])[::-1]
            s0[n] = ss[0]
            if len(ss) > 1:
                s1[n] = ss[1]
        gap = np.nan_to_num(s0 - s1, nan=0.0)
        persistent = median_filter(gap, 11, mode="nearest") > 8 * ppm
        s_lead = np.where((gap > 8 * ppm) & ~persistent, s1, s0)
        hud = {int(f): o for f, _c, o in np.load(self.p("hud.npz"))["rows"]}
        odo = np.array([hud.get(int(f), np.nan) for f in fu])
        # marker grid SEEDED by the ticks (exact 50 m spacing; tick scale/finish as the prior)
        have_markers = len(m["ks"]) >= 4 and 0 in m["ks"]
        if self.p("marker_profile.npz").exists():
            mp = np.load(self.p("marker_profile.npz"))
            ev_s, ev_v = rails.tick_events(fu, s_lead, odo)
            seeded = rails.fit_marker_grid_seeded(mp["ss"], mp["resp"], ev_s, ev_v)
            if seeded is not None:
                ppm, sF, ks_seed, _ps = seeded
                have_markers = True
                self.log(f"   observe: tick-seeded marker grid k={ks_seed.tolist()} ppm {ppm:.3f}")
        pin_env = os.environ.get("HKJC_PIN")  # diagnostic override: "1" force pin, "0" never
        # DEFAULT (HKJC_FILM=0 disables; journals 0137/0139): the race film's REAL painted markers
        # fix the metric's constant -- odometer ticks as the metric to the line (unpinned, no
        # convention offset), shifted so the render leader meets the film's marker crossings. MEASURED: the render-paint
        # pin + paint-derived convention put the leader +0.2 s late over 400-100 m and -0.16 s early
        # at the line (a step in the last 50 m); on the ticks the film shows one flat offset per
        # race. Leader/sectional gains are robust (6-seed ensembles: ST9 timing 2 -> 4/5, sectional
        # 0.455 -> 0.232 s; ST1 sectional 0.36 -> 0.25 s), BUT it consistently loses 2-5 horses'
        # paths on ST4/ST1 without the finish SEEDS (those horses hung on orphan-chain matching,
        # which flips with the changed line speed: the pin stretched the last 50 m, 17.0 vs 15.2
        # m/s). With seeds (26 races): timing gates 117 -> 120, leader mean 0.213 -> 0.166 s,
        # sectional 0.243 -> 0.197 s; remaining losses are structural (ST3 tail horses cross at
        # the render's end, ST4's ill-posed finish anchor) plus a 0.01 s photo finish (ST 05-24).
        # Fallback (< FILM_MIN crossings): the render-paint pin.
        fz = self.p("film.npz")
        film_t = {}
        if fz.exists() and film_enabled():
            zf = np.load(fz)
            film_t = dict(zip(zf["markers"].tolist(), zf["t"].tolist()))
        use_film = len(film_t) >= film.FILM_MIN and pin_env is None
        d_of_s, c, ppm_loc, resid, (ppm_odo, sF_odo, pinned, c_m, nsh) = (
            rails.calibrate_distance_ticks(
                fu,
                s_lead,
                odo,
                ppm,
                sF,
                pin_finish=False if use_film else (None if pin_env is None else pin_env == "1"),
                have_markers=have_markers,
            )
        )
        delta = 0.0
        if use_film:
            dlt, per = film.anchor_offset((fu - self.clock_f0()) / FPS, d_of_s(s_lead), film_t)
            if dlt is None:  # crossings not matched by the render leader: back to the paint pin
                d_of_s, c, ppm_loc, resid, (ppm_odo, sF_odo, pinned, c_m, nsh) = (
                    rails.calibrate_distance_ticks(
                        fu, s_lead, odo, ppm, sF, have_markers=have_markers
                    )
                )
            else:
                delta = dlt
                self.log(
                    f"   observe: film anchor delta {delta:+.2f} m from {len(per)} markers "
                    f"(spread {np.ptp(per):.1f} m)"
                )
        d = d_of_s(s_px) - delta
        # time-local pose slips (render camera swing on the straights): per-frame along-track shift
        if os.environ.get("HKJC_TIME_SHIFT", "1") != "0":
            tf_u = (fu - self.clock_f0()) / FPS
            shift_u, n_win = rails.time_local_shift(tf_u, d_of_s(s_lead) - delta, odo)
            if n_win:
                d = d - shift_u[np.searchsorted(fu, rows[:, 0])]
                self.log(
                    f"   observe: time-local shift in {n_win} window(s), max {np.abs(shift_u).max():.1f} m"
                )
        lat = lat_px / ppm_loc(s_px)
        g = lat > -1.0  # nothing races inside the racing rail
        rows, d, lat, hist, X = rows[g], d[g], lat[g], hist[g], X[g]
        det_idx = self._wd_idx[g]
        t = (rows[:, 0] - self.clock_f0()) / FPS
        frames = np.unique(rows[:, 0])
        fi = np.searchsorted(frames, rows[:, 0])
        times = (frames - self.clock_f0()) / FPS
        tshift = os.environ.get("HKJC_TRACK_SHIFT", "0") != "0"
        if track_app_enabled():  # appearance-aware tracking (render-domain silk continuity)
            zs = np.load(self.p("silk.npz"))
            F = zs["feat"][det_idx].astype(float)
            F = F / np.maximum(F.sum(1, keepdims=True), 1e-9)
            p1 = tracking.track(times, -d, lat, fi, shift=tshift)
            llr = tracking.calibrate_llr(p1, F, t, fi, -d, lat)
            trs = tracking.stitch(tracking.track(times, -d, lat, fi, shift=tshift, F=F, llr=llr))
        else:
            trs = tracking.stitch(tracking.track(times, -d, lat, fi, shift=tshift))
        trs.sort(key=lambda tk: -len(tk.t))
        members = [np.array(tk.det) for tk in trs]
        if os.environ.get("HKJC_SEAMS", "0") != "0":  # tracking.resolve_seams (NOT adopted)
            members = tracking.resolve_seams(members, rows[:, 0], d, lat)
            members.sort(key=lambda m: -len(m))
        np.savez(
            self.p("tracks.npz"),
            rows=rows,
            d=d,
            lat=lat,
            t=t,
            X=X,
            hist=hist,
            members=np.array(members + [None], dtype=object)[:-1],
            odo_c=c,
            film_delta=delta,
            det_idx=det_idx,
        )
        self.log(
            f"   observe: {len(resid)} odometer ticks, ppm odo/marker {ppm_odo:.3f}/{ppm:.3f}, "
            f"marker-zone c {c_m:+.1f} m (grid shift {nsh}, convention c {c:+.1f} m) -> FINISH "
            f"{'pinned to markers' if pinned else 'from ticks'}, |res| p50 "
            f"{np.median(np.abs(resid)):.2f} m); {len(members)} tracklets, top lens "
            f"{[len(m) for m in members[: self.info.n]]}"
        )

    def identify(self):
        T = np.load(self.p("tracks.npz"), allow_pickle=True)
        mem = [np.asarray(m, int) for m in T["members"]]
        leg = np.load(self.p("legend.npz"))["rows"]
        nmax = max(
            r.no for r in self.info.runners
        )  # saddlecloth numbers may have gaps (scratchings)
        valid = {r.no for r in self.info.runners}
        votes = identity.legend_votes(mem, T["rows"][:, 0], T["d"], leg, nmax)
        fin, _fit = self._finish_identity(T, mem)
        for k, num in fin.items():
            if num <= nmax:
                votes[k, num] += self.FINISH_VOTES
        if os.environ.get("HKJC_ID_GLOBAL", "0") != "0":  # global ML assignment (identity.assign_global)
            lab = identity.assign_global(mem, T["rows"][:, 0], votes, nmax)
        else:
            lab, _sig, _proto = identity.assign(mem, T["rows"][:, 0], votes, T["hist"], nmax)
        lab = np.array([x if x in valid else 0 for x in lab])
        n_elim, el_ok = 0, []
        if _fit is not None:  # ELIMINATION at the line: leftover tracklets vs leftover numbers
            at_line = {k for k in _fit["tc"] if lab[k]}
            el = finishcam.eliminate(_fit, at_line, {int(lab[k]) for k in at_line})
            for k, num in el.items():
                if num in valid and not any(
                    identity.time_overlap(T["rows"][mem[k], 0], T["rows"][mem[u], 0])
                    for u in np.where(lab == num)[0]
                ):
                    lab[k] = num
                    el_ok.append(k)
                    n_elim += 1
        fixed = np.zeros(len(mem), int)  # finish-camera names: hard constraints for the EM pass
        for k, num in fin.items():
            if lab[k] == num:
                fixed[k] = num
        for k in el_ok:
            fixed[k] = lab[k]
        np.savez(self.p("labels.npz"), labels=lab, votes=votes, fixed=fixed)
        self.log(
            f"   identify: finish-camera named {len(fin)} + {n_elim} by elimination; "
            f"numbered {int((lab > 0).sum())}/{len(mem)} tracklets "
            f"({sum(len(m) for k, m in enumerate(mem) if lab[k]) / len(T['d']):.0%} of detections)"
        )

    FINISH_VOTES = 40.0  # one finish-camera match ~ 4 s of unanimous legend votes

    def _finish_identity(self, T=None, mem=None):
        """({tracklet index: number}, label-free fit) from the finish camera (1D registration of the
        tracklets' line-crossing times to the camera's gaps; tracklets ending < 30 m short are
        extrapolated)."""
        if T is None:
            T = np.load(self.p("tracks.npz"), allow_pickle=True)
            mem = [np.asarray(m, int) for m in T["members"]]
        if not self.p("finish.npz").exists():
            return {}, None
        z = np.load(self.p("finish.npz"))
        order, gaps, _u = finishcam.finish_gaps(dict(line_x=float(z["line_x"]), reads=z["reads"]))
        gaps = {int(k): v for k, v in gaps.items()}
        if len(gaps) < 3:
            return {}, None
        tc, vs = {}, []
        for k, m in enumerate(mem):
            if len(m) < 10:
                continue
            o = np.argsort(T["t"][m])
            t, d = T["t"][m][o], T["d"][m][o]
            # a crossing must be bracketed by samples < 0.3 s apart: the render can skip ~1 s near
            # the line (R02A: crossings quantized at the first frame after the gap); otherwise
            # extrapolate from the last pre-line sample (< 30 m short) at its own speed
            pre = np.where(d > 0)[0]
            if not len(pre):
                continue
            j = pre[-1]
            w = slice(max(0, j - 9), j + 1)
            v = -np.polyfit(t[w], d[w], 1)[0] if j >= 3 else np.nan
            if j + 1 < len(t) and t[j + 1] - t[j] < 0.3:
                tc[k] = t[j] + d[j] / (d[j] - d[j + 1]) * (t[j + 1] - t[j])
            elif 0 < d[j] < 30 and np.isfinite(v) and v > 8:
                tc[k] = t[j] + d[j] / v
            else:
                continue
            vs.append(v)
        ppm_fc = float(z["ppm"]) if "ppm" in z.files else np.nan
        c_prior = (
            float(np.median(vs)) * ppm_fc * finishcam.ruler_cal("reg") if vs and np.isfinite(ppm_fc) else None
        )
        m, fit = finishcam.identity_match(tc, gaps, c_prior=c_prior)
        if fit is not None:
            fit = dict(fit, tc=tc, gaps=gaps)
        return m, fit

    def finish(self):
        # the finish replay follows the race: start scanning a little after the winner could finish
        t_fin = self.info.distance / 18.0
        sc = finishcam.scan(
            self.video("full"),
            start_s=t_fin,
            stop_s=t_fin + 70.0,
            n_field=max(14, self.info.n),
            # white-cloth reader only at Sha Tin (side-on camera, white cloths); on HV's overhead
            # camera it only added false reads (HV0527: a "#7" at 0.5 s, real #7 2.36 s behind)
            white=self.info.venue != "HV",
        )
        order, gaps, ustar = finishcam.finish_gaps(sc)
        np.savez(
            self.p("finish.npz"),
            ppm=sc.get("ppm", np.nan),
            order=np.array(order, int),
            line_x=sc["line_x"],
            nums=np.array(list(gaps), int),
            gaps=np.array(list(gaps.values()), float),
            reads=sc["reads"],
        )
        self.log(
            f"   finish: line x={sc['line_x']:.0f}, {len(sc['frames'])} unique frames, order {order}"
        )

    def finish_observations(self, paths) -> list:
        """Finish-camera proposals: each horse crosses d=0 at t_win + gap/C. The camera fixes ORDER
        and GAPS; C (px/s) and the winner's clock time come from the render-fused paths."""
        if not self.p("finish.npz").exists():
            return []
        z = np.load(self.p("finish.npz"))
        # order/gaps recomputed from the cached cloth reads (cheap; no video re-scan)
        order, gaps, _u, gsd = finishcam.finish_gaps(
            dict(line_x=float(z["line_x"]), reads=z["reads"]), with_sd=True
        )
        order = [int(n) for n in order]
        gaps = {int(k): v for k, v in gaps.items()}
        gsd = {int(k): v for k, v in gsd.items()}
        if len(order) < 3:
            return []
        from .validate import crossing

        # Only OBSERVED crossings anchor the clock: a path whose labelled tracklets ended early is
        # extrapolated by fusion at its last speed with a growing sd (ST4 0419: 8 horses' tracklets
        # ended at 432 m to go; their paths "crossed" 4.5-5.3 s early at sd ~24 m and dragged the
        # anchor to C = 800 px/s and the clock 2.2 s early; their 20 m/s extrapolated speeds also
        # inflated the C prior 47 %). REJECTED: crossing() without the sd check.
        sd_ok = self.ANCHOR_SD

        def cross_or_extrap(p):
            d = self.info.distance - p.s
            t = crossing(p.t, d, None, 0.0)
            if np.isfinite(t):
                return t if np.interp(t, p.t, p.sd_s) <= sd_ok else np.nan
            if not (0.0 < d[-1] <= 40.0) or p.sd_s[-1] > sd_ok or len(p.t) <= 11:
                return np.nan
            v = (p.s[-1] - p.s[-11]) / (p.t[-1] - p.t[-11])
            return p.t[-1] + d[-1] / v if np.isfinite(v) and v > 5 else np.nan  # render ended short

        t_cross = {h: cross_or_extrap(p) for h, p in paths.items()}
        # physical prior on C: leader speed at the line (fused) x finish-camera px/m (horse ruler)
        c_prior = None
        ppm_fc = float(z["ppm"]) if "ppm" in z.files else np.nan
        v = []
        for p in paths.values():
            d = self.info.distance - p.s
            k = np.where((d > -10) & (d < 30) & (p.sd_s <= sd_ok))[0]
            if len(k) > 5:
                v.append(np.median(np.gradient(p.s[k], p.t[k])))
        v_line = float(np.median(v)) if v else 16.5  # m/s at the line (time sd -> metres)
        if np.isfinite(ppm_fc) and v:
            c_prior = v_line * ppm_fc * finishcam.ruler_cal("anchor")
        cal = finishcam.anchor(gaps, t_cross, c_prior=c_prior)
        # label-free sanity check: the anchor fits the clock to LABELLED render paths, so wrong
        # labels drag it (R02A: C 3741-4000 vs ~1750, clock 0.5-1.2 s late). Registration of the
        # anonymous tracklet crossings to the gaps needs no labels (biased ~0.15 s early, MEASURED).
        _m, fit = self._finish_identity()
        reg_ok = fit is not None and not fit["edge"] and fit["n_close"] >= 0.5 * self.info.n
        src = "anchor"
        if cal is not None and reg_ok:
            if abs(cal[0] - fit["t_w"]) > 0.4 or abs(cal[1] / fit["C"] - 1) > 0.3:
                cal, src = None, "registration"
        if cal is None:
            if not reg_ok:
                self.log("   finish: no consistent clock/scale (anchor and registration fail)")
                return []
            g = np.array([gaps[h] for h in gaps if np.isfinite(gaps[h])])
            sd_tw = 0.15  # the registration's measured clock bias, as an honest sd
            H = np.array([[1.0, g.mean()], [g.mean(), (g**2).mean()]]) * len(g) / sd_tw**2
            cal = (fit["t_w"], fit["C"], fit["n_close"], np.linalg.inv(H + np.eye(2) * 1e-9))
            src = "registration"
        t_w, C, n_used, cov = cal
        D = self.info.distance

        def sd_t(h):
            # method-derived, CONDITIONAL on the fitted clock t_w: fusion treats each horse's obs
            # as independent, so the clock error shared by all horses (t_w is itself fitted to the
            # render arrivals) must not be added per horse — it would bury the precise relative
            # order/gaps (R08: rho 0.93 -> 0.55 with the marginal sd). Scale term + gap spread.
            var_inv_c = 1.0 / np.linalg.inv(cov)[1, 1]  # var(1/C | t_w)
            return float(np.sqrt(gaps[h] ** 2 * var_inv_c + (gsd.get(h, 15.0) / C) ** 2))

        sds = {h: sd_t(h) for h in order if np.isfinite(gaps.get(h, np.nan))}
        # DISABLED by default (2026-11-14): the clean frozen run showed it inflating the WINNER of HV
        # 0429 (9 reads, crossed before the replay segment starts; its render path was the wrong one)
        # -> no valid crossing -> wrong winner, while the misread it was built for (ST6 "#14") now has
        # 22 reads after the white-cloth recall fix and is no longer affected.
        # cross-feed robustness for WEAK camera tracks: a number read on < 10 frames that also
        # disagrees with the render-fused crossing by > 0.3 s is likely a digit misread (ST6 "#14":
        # 6 reads, 1.4 s early); inflate its sd to the disagreement. Blanket inflation (any
        # disagreement) was REJECTED: on HV the camera is the right one (R03 winner lost).
        n_reads = {
            int(k): len(v[0])
            for k, v in finishcam.tracks(dict(line_x=float(z["line_x"]), reads=z["reads"])).items()
        }
        n_infl = 0
        for h in sds:
            tc = t_cross.get(h, np.nan)
            if (
                os.environ.get("HKJC_WEAK_INFLATE", "0") == "1"  # opt-in: see the note above
                and np.isfinite(tc)
                and n_reads.get(h, 0) < 10
                and n_used >= 5  # ST8: 3-horse anchor
            ):
                dis = abs(t_w + gaps[h] / C - tc)
                if dis > 0.3:
                    sds[h] = max(sds[h], dis)
                    n_infl += 1
        self.log(
            f"   finish: C={C:.0f} px/s (physical prior {c_prior or float('nan'):.0f}), first "
            f"finisher #{order[0]} at t={t_w:.2f}s ({src}, {n_used} horses; {n_infl} inflated); "
            f"sd {min(sds.values()):.3f}..{max(sds.values()):.3f} s"
        )
        return [
            fusion.Obs(
                np.array([t_w + gaps[h] / C]),
                np.array([D]),
                None,
                v_line * sds[h],
                1.0,
                "finishcam",
                horse=h,
                tid=10_000 + h,
            )
            for h in sds
        ]

    def observations(self, lab=None) -> list:
        """All feeds' proposals as fusion.Obs. (Render now; drone/RF/patrol join here.)"""
        T = np.load(self.p("tracks.npz"), allow_pickle=True)
        lab = np.load(self.p("labels.npz"))["labels"] if lab is None else lab
        D = self.info.distance
        out = []
        for k, m in enumerate(T["members"]):
            m = np.asarray(m, int)
            if len(m) < 3:
                continue
            o = np.argsort(T["t"][m])
            m = m[o]
            out.append(
                fusion.Obs(
                    T["t"][m],
                    D - T["d"][m],
                    T["lat"][m],
                    0.8,
                    0.35,
                    "render",
                    horse=int(lab[k]),
                    tid=k,
                )
            )
        return out

    # Gate geometry MEASURED on the drone's top-down start frame (HV 2026-06-03 R8): stall labels at a
    # constant 40.2 px pitch; racing rail -> outer rail = 855 px = 22.5 m (C) at the gate's depth
    # -> 38.0 px/m: stall pitch 1.06 m; stall-1 centre 2.5 m off the racing rail (stall 1 abuts it).
    STALL_PITCH = 1.06  # m between stall centres
    STALL1_LAT = 2.5  # m: stall-1 centre off the racing rail
    DRAW_SD = 0.4  # m

    def official_observations(self) -> list:
        """HIGH-CERTAINTY records (owner directive): the gate draw fixes every runner's lateral at t=0
        (standing start, s=0), and the official per-runner sectional times fix its crossing times at
        the section boundaries (d = 400 k)."""
        from .validate import _cum_official

        D = self.info.distance
        out = []
        for r in self.info.runners:
            lat0 = self.STALL1_LAT + (r.draw - 1) * self.STALL_PITCH
            out.append(
                fusion.Obs(
                    np.array([0.0]),
                    np.array([0.0]),
                    np.array([lat0]),
                    0.3,
                    self.DRAW_SD,
                    "draw",
                    horse=r.no,
                    tid=20_000 + r.no,
                )
            )
        bnd, cum = _cum_official(self.info)
        for h, c in cum.items():
            ok = np.isfinite(c)
            out.append(
                fusion.Obs(
                    np.asarray(c)[ok],
                    D - np.asarray(bnd)[ok],
                    None,
                    0.4,
                    1.0,
                    "official",
                    horse=int(h),
                    tid=30_000 + int(h),
                )
            )
        return out

    ANCHOR_CHECK = True  # env HKJC_ANCHOR_CHECK=0 disables (on/off sweeps)

    def _fuse(self, obs, horses, prior):
        import os

        ac = self.ANCHOR_CHECK and os.environ.get("HKJC_ANCHOR_CHECK", "1") != "0"
        st = {}
        paths, assign = fusion.fuse_race(obs, horses, prior, anchor_check=ac, stats=st)
        fin = self.finish_observations(paths)  # second pass: + finish camera
        if fin:
            paths, assign = fusion.fuse_race(obs + fin, horses, prior, anchor_check=ac, stats=st)
        if st.get("demoted"):
            self.log(f"   fuse: anchor check demoted {st['demoted']} render tracklets")
        return paths, assign

    def _write_paths(self, paths, name):
        import polars as pl

        recs = [
            pl.DataFrame(
                {
                    "horse_no": h,
                    "t": p.t,
                    "s": p.s,
                    "d": self.info.distance - p.s,
                    "lat": p.lat,
                    "sd_s": p.sd_s,
                    "sd_lat": p.sd_l,
                    "n_obs": p.n_obs,
                }
            )
            for h, p in paths.items()
        ]
        pl.concat(recs).write_parquet(self.p(name))

    def _em_identity(self, paths):
        """Second identity pass: legend votes on frames with undetected horses, ranks completed by the
        first fused posterior (only frames where every horse's posterior is tight)."""
        T = np.load(self.p("tracks.npz"), allow_pickle=True)
        L = np.load(self.p("labels.npz"))
        if len(paths) < self.info.n:  # completion needs every horse's posterior
            return None
        mem = [np.asarray(m, int) for m in T["members"]]
        leg = np.load(self.p("legend.npz"))["rows"]
        hs = sorted(paths)
        D = self.info.distance

        def field(t):
            dd = np.array([D - np.interp(t, paths[h].t, paths[h].s) for h in hs])
            sd = np.array([np.interp(t, paths[h].t, paths[h].sd_s) for h in hs])
            return dd, sd

        v2 = identity.legend_votes_completed(
            mem,
            T["rows"][:, 0],
            T["t"],
            T["d"],
            leg,
            field,
            nmax=max(r.no for r in self.info.runners),
        )
        v1 = L["votes"]
        v = v1.copy()
        k = min(v.shape[1], v2.shape[1])
        v[:, :k] += v2[:, :k]
        nmax = max(r.no for r in self.info.runners)
        valid = {r.no for r in self.info.runners}
        lab, _s, _p = identity.assign(mem, T["rows"][:, 0], v, T["hist"], nmax)
        lab = np.array([x if x in valid else 0 for x in lab])
        fixed = L["fixed"] if "fixed" in L.files else np.zeros(len(mem), int)
        for k in np.where(fixed > 0)[0]:  # finish-camera names win over time-overlapping rivals
            for u in np.where(lab == fixed[k])[0]:
                if u != k and identity.time_overlap(T["rows"][mem[k], 0], T["rows"][mem[u], 0]):
                    lab[u] = 0
            lab[k] = fixed[k]
        cov = lambda lb: sum(len(m) for i, m in enumerate(mem) if lb[i]) / len(T["d"])  # noqa: E731
        self.log(
            f"   identify (EM pass): numbered {int((lab > 0).sum())}/{len(mem)} tracklets, "
            f"{cov(L['labels']):.0%} -> {cov(lab):.0%} of detections"
        )
        return lab

    def fuse(self):
        """Two posteriors: VIDEO-ONLY (what the gates validate against the official record) and the
        FINAL product, which adds the high-certainty official records (draw + sectional times)."""
        obs = self.observations()
        prior = fusion.Prior(width=self.info.width)
        horses = [r.no for r in self.info.runners]
        pv, assign = self._fuse(obs, horses, prior)
        lab2 = self._em_identity(pv)  # EM pass: legend votes completed by the first posterior
        if lab2 is not None:
            obs = self.observations(lab2)
            pv, assign = self._fuse(obs, horses, prior)
        if os.environ.get("HKJC_EM_ASSOC", "1") != "0":
            obs, pv = self._em_associate(pv, horses, prior)
        self._write_paths(pv, "paths_video.parquet")
        obs_f = self._metric_corrected(obs, pv)
        pf, _a = self._fuse(obs_f + self.official_observations(), horses, prior)
        self._write_paths(pf, "paths.parquet")
        n_anon = sum(1 for o in obs if o.horse == 0)
        n_ass = sum(1 for o in obs if o.horse == 0 and o.tid in assign)
        self.log(
            f"   fuse: video-only {len(pv)}/{len(horses)} horses, final {len(pf)}/{len(horses)}; "
            f"anonymous tracklets associated {n_ass}/{n_anon}"
        )

    def _metric_corrected(self, obs, paths):
        """FINAL product only: the render's distance metric has a smooth SHARED error b(s) (all
        horses cross a boundary early/late together: R08 +0.27 s at the first boundary, HV0527
        +0.17 s at 400 m), and many render observations treated as independent outvote the single
        high-certainty official time (final residual ~5x its stated sd). b at each official boundary
        = median over horses of (crossing delay x speed); interpolated along s; render s shifted."""
        from .validate import _cum_official, crossing

        bnd, cum = _cum_official(self.info)
        D = self.info.distance
        bs = []
        for j, bd in enumerate(bnd):
            r = []
            for h, c in cum.items():
                p = paths.get(h)
                if p is None or not np.isfinite(c[j]):
                    continue
                d = D - p.s
                tr = crossing(p.t, d, p.sd_s, bd)
                if np.isfinite(tr):
                    v = np.interp(tr, p.t[1:], np.diff(p.s) / np.diff(p.t))
                    r.append((tr - c[j]) * v)  # path late by dt -> render s short by v*dt
            bs.append((D - bd, float(np.median(r)) if len(r) >= 3 else np.nan))
        bs = [(sb, b) for sb, b in bs if np.isfinite(b)]
        if not bs:
            return obs
        sb, b = np.array(bs).T
        o = np.argsort(sb)
        sb, b = sb[o], b[o]
        self.log(f"   fuse: render metric bias at boundaries (m): {np.round(b, 2).tolist()}")
        out = []
        for ob in obs:
            if ob.feed == "render" and ob.s is not None:
                ob = replace(ob, s=ob.s + np.interp(ob.s, sb, b))
            out.append(ob)
        return out

    EM_ITERS = 8
    SEED_BACK = 3.0  # s before its finish-camera crossing that a path-less horse is seeded
    SEED_V_SD = 1.0  # m/s: speed uncertainty of a seeded horse (grows the seed's sd away from the line)

    def _finish_seeds(self, paths, prior) -> dict:
        """E-step-only SEED paths for horses the finish camera READ but that have no fused path.

        MEASURED (26 turf races, in-sample census): 10 horses were read at the line (up to 45 reads),
        crossed while the render was still running, finished 5th-12th -- and got NO path: they had no
        labelled tracklet, and a finish observation alone cannot make one (no lateral, one instant),
        so the detection-level E-step never considered them. The seed is the camera's crossing
        (t_w + gap/C, sd as in finish_observations) moving at the field's line speed, sd growing by
        SEED_V_SD per second away from the line, lateral = mid-course +- half the width; it lives
        only inside assign_frames, and whatever detection runs it wins become ordinary render obs."""
        if not paths:
            return {}
        quiet, self.log = self.log, (lambda *a, **k: None)
        try:
            fin = self.finish_observations(paths)
        finally:
            self.log = quiet
        D = self.info.distance
        v = []
        for p in paths.values():
            k = np.where((D - p.s > 0) & (D - p.s < 30) & (p.sd_s <= self.ANCHOR_SD))[0]
            if len(k) > 5:
                v.append(np.median(np.gradient(p.s[k], p.t[k])))
        v_line = float(np.median(v)) if v else 16.5
        out = {}
        for o in fin:
            h = o.horse
            if h in paths:
                continue
            tc, sd0 = float(o.t[0]), float(o.sd_s)
            tg = np.arange(tc - self.SEED_BACK, tc + 0.6, fusion.DT)
            dt = tg - tc
            sd_s = np.sqrt(sd0**2 + (self.SEED_V_SD * dt) ** 2)
            w = prior.width
            out[h] = fusion.Path(
                h,
                tg,
                D + v_line * dt,
                np.full(len(tg), 0.5 * w),
                sd_s,
                np.full(len(tg), 0.5 * w),
                np.zeros(len(tg)),
                {"seed": 1},
            )
        return out
    ANCHOR_SD = 3.0  # m: a fused crossing anchors the finish clock only where the path is tight

    @staticmethod
    def _fc_read(zf) -> set:
        """Horses the finish camera read (negative evidence / shadow-path tie-break). Default: the
        horses with a GAP; HKJC_FC_READS=reads: every read cloth (finishcam.read_numbers -- a
        back-marker crossing after the referenced cloths left the frame has reads but no gap)."""
        sc = dict(line_x=float(zf["line_x"]), reads=zf["reads"])
        if os.environ.get("HKJC_FC_READS", "gaps") == "reads":
            return finishcam.read_numbers(sc)
        return {int(k) for k in finishcam.finish_gaps(sc)[1]}

    def _em_associate(self, paths, horses, prior):
        """Detection-level EM (``assoc``): every detection re-assigned per frame against the
        continuous fused paths (+ one-sided legend rank constraint), paths re-fused; the tracker's
        fragments are only the initialisation. Unassigned detections stay anonymous (grouped by
        their tracker fragment) so association/orphan logic can still place path-less horses."""
        from . import assoc

        T = np.load(self.p("tracks.npz"), allow_pickle=True)
        leg = np.load(self.p("legend.npz"))["rows"]
        fr, t, lat = T["rows"][:, 0], T["t"], T["lat"]
        silk = None  # appearance term: per-detection v2 silk signature vs official silk prototypes
        if self.p("silk.npz").exists() and "det_idx" in T.files:
            zs = np.load(self.p("silk.npz"))
            silk = (
                zs["feat"][T["det_idx"]],
                {int(h): p for h, p in zip(zs["proto_no"], zs["proto"])},
            )
        s_run = self.info.distance - T["d"]
        frag = np.full(len(t), -1)
        for k, m in enumerate(T["members"]):
            frag[np.asarray(m, int)] = k
        obs = []
        # finish camera NEGATIVE evidence: with recall r (fraction of the field it read), a horse
        # that IS at the line goes unread with probability ~(1 - r), so detections in the last 15 m
        # cost -2 log(1 - r) for unread horses (ST7: unread #3 took a wide leader's detections and
        # "won", 1.93 s back; ST5: unread #13, last). A fixed 75 %-recall cutoff missed ST5 (71 %).
        near = None
        if self.p("finish.npz").exists():
            zf = np.load(self.p("finish.npz"))
            read = self._fc_read(zf)
            r = min(len(read) / max(self.info.n, 1), 0.95)
            if read and r >= 0.5 and os.environ.get("HKJC_NEGEV", "1") != "0":
                cost = float(-2.0 * np.log(1.0 - r))
                near = [(self.info.distance - 15.0, {h: cost for h in horses if h not in read})]
                # NO identity evidence near the line (unread by the camera AND absent from the legend
                # top 8 in the last 8 s before the line): final-stretch detections are not this horse's to take —
                # it extrapolates with honest uncertainty instead of "finishing" on another horse's
                # detections (ST7 #4, HV0415 #2 crossed with the leaders, 1.2-2.2 s early)
                # window: the 8 s before the first horse reaches the line (a horse shown in the top 8
                # earlier in the straight, then fading, has no identity evidence AT the line)
                crossings = [
                    p.t[np.argmax(self.info.distance - p.s <= 0)]
                    for p in paths.values()
                    if (self.info.distance - p.s <= 0).any()
                ]
                t_fin = min(crossings) if crossings else float(np.max(t))
                late_frames = {int(f) for f, tt in zip(fr, t) if t_fin - 8.0 <= tt <= t_fin + 1.0}
                shown = {int(x) for r_ in leg if int(r_[0]) in late_frames for x in r_[1:9] if x}
                blind = [h for h in horses if h not in read and h not in shown]
                if blind and os.environ.get("HKJC_BLIND", "1") != "0":
                    near.append((self.info.distance - 50.0, {h: assoc.CLUTTER for h in blind}))
        def em_pass(paths, targets, tag):
            """EM iterations; ``targets``: path-less finish-read horses to seed in the E-step."""
            prev = lab = obs = None
            forbid = {}
            for it in range(int(os.environ.get("HKJC_EM_ITERS", self.EM_ITERS))):  # env: diagnostic
                forbid = {}
                seeds = {}
                if targets:
                    seeds = {h: p for h, p in self._finish_seeds(paths, prior).items() if h in targets}
                lab, _chi = assoc.assign_frames(
                    fr,
                    t,
                    s_run,
                    lat,
                    {**paths, **seeds},
                    leg,
                    near_line=near,
                    n_field=len(horses),
                    forbid_out=forbid,
                    s_line=self.info.distance,
                    silk=silk,
                )
                obs = assoc.runs_to_obs(lab, t, s_run, lat)
                for k in np.unique(frag[(lab == 0) & (frag >= 0)]):
                    j = np.where((frag == k) & (lab == 0))[0]
                    if len(j) >= 3:
                        j = j[np.argsort(t[j])]
                        # the E-step's hard legend exclusions travel with the anonymous fragment,
                        # or fusion's chi^2 / orphan association hands it back (HV 0520 #8)
                        no = assoc.fragment_forbid(forbid, j)
                        obs.append(
                            fusion.Obs(t[j], s_run[j], lat[j], 0.8, 0.35, "render", 0, int(k), no)
                        )
                paths, _a = self._fuse(obs, horses, prior)
                changed = np.nan if prev is None else float((lab != prev).mean())
                self.log(
                    f"   em-assoc{tag} it{it}: {np.mean(lab > 0):.0%} of detections assigned, "
                    f"{len(paths)}/{len(horses)} horses, changed {changed:.1%}"
                    + (f", seeded {sorted(seeds)}" if seeds else "")
                )
                if prev is not None and changed < 0.002:
                    break
                prev = lab
            return paths, lab, obs, forbid

        paths, lab, obs, forbid = em_pass(paths, None, "")
        # SEED pass, CONFINED to finish-read horses still path-less after the unseeded EM (seeding
        # from it0 also fired for horses the EM places anyway and changed races with no such horse:
        # ST4 sectional 0.409 -> 0.467 on common pairs). Races without a target are unchanged.
        if os.environ.get("HKJC_SEED", "1") != "0":
            targets = set(self._finish_seeds(paths, prior))
            if targets:
                paths, lab, obs, forbid = em_pass(paths, targets, " (seed)")
        # SHADOW paths: two horses cannot occupy one position. A horse whose own detections are
        # missing can lock onto a neighbour's (R03: unread #9 rode #7's detections through the line,
        # 0.8 s early -> wrong winner). Pairs within 0.8 m along / 0.5 m laterally for >= 0.5 s (both
        # tight): the render obs of the weaker-evidence horse (no finish-camera reading, else fewer
        # obs) in that window are dropped and the paths re-fused.
        cam = set()
        if self.p("finish.npz").exists():
            zf = np.load(self.p("finish.npz"))
            cam = self._fc_read(zf)
        drop = np.zeros(len(t), bool)
        hs = sorted(paths)
        # only the finish-critical window (last 3 s before the first horse reaches the line): a
        # mid-race rule released 1323 detections of a real horse on HV0429 (coverage 94 -> 90 %)
        t_fin = min(
            (
                p.t[np.argmax(self.info.distance - p.s <= 0)]
                for p in paths.values()
                if (self.info.distance - p.s <= 0).any()
            ),
            default=np.inf,
        )
        for a_i, ha in enumerate(hs):
            for hb in hs[a_i + 1 :]:
                pa, pb = paths[ha], paths[hb]
                if len(pa.t) != len(pb.t) or pa.t[0] != pb.t[0]:
                    continue
                close = (  # physically impossible overlap (side-by-side horses are ~1 m apart)
                    (np.abs(pa.s - pb.s) < 0.8)
                    & (np.abs(pa.lat - pb.lat) < 0.5)
                    & (pa.sd_s < 1.5)
                    & (pb.sd_s < 1.5)
                    & (pa.t > t_fin - 3.0)
                )
                if close.sum() * fusion.DT < 0.5:
                    continue
                tw = pa.t[close]
                win = (t >= tw.min() - 0.2) & (t <= tw.max() + 0.2)
                na, nb = int(((lab == ha) & win).sum()), int(((lab == hb) & win).sum())
                if (ha in cam) != (hb in cam):
                    loser = hb if ha in cam else ha
                else:
                    loser = ha if na < nb else hb
                drop |= (lab == loser) & win
        if drop.any() and os.environ.get("HKJC_SHADOW", "1") != "0":
            lab = np.where(drop, 0, lab)
            obs = assoc.runs_to_obs(lab, t, s_run, lat)
            for k in np.unique(frag[(lab == 0) & (frag >= 0) & ~drop]):
                j = np.where((frag == k) & (lab == 0) & ~drop)[0]
                if len(j) >= 3:
                    j = j[np.argsort(t[j])]
                    no = assoc.fragment_forbid(forbid, j)
                    obs.append(
                        fusion.Obs(t[j], s_run[j], lat[j], 0.8, 0.35, "render", 0, int(k), no)
                    )
            paths, _a = self._fuse(obs, horses, prior)
            self.log(f"   em-assoc: shadow paths -> {int(drop.sum())} detections released")
        np.savez(self.p("det_labels.npz"), labels=lab)  # per-detection EM identity (viz, audit)
        return obs, paths

    def paths(self):
        import polars as pl

        return pl.read_parquet(self.p("paths.parquet"))

    def validate(self):
        from . import validate as _v

        res = _v.validate(self)
        self.p("validation.json").write_text(json.dumps(res, indent=1, default=float))
        self.log(_v.summary_line(self.info.key, res))
