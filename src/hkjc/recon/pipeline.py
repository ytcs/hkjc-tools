"""Race reconstruction pipeline: race info + replay videos -> per-horse paths.

``RaceRun(info).run()`` executes ``STAGES`` in order, each cached under ``data_cache/recon/<key>/``
and skipped when its ``OUTPUTS`` exist (``force=`` re-runs named stages). Stages:

  frames    AVV render: unique 10 Hz frames cached (pixels, frame index, clock)
  features  avatar detections (self-supervised YOLO) + silk histograms; turf ORB features (the
            detections masked) + consecutive step matches
  pose      camera pose graph (sequential/long-baseline/loop edges) -> world poses
  worldmap  mosaic of the registered frames (static world: turf, rails, painted markers)
  detect    the avatar detections (+ map-difference blobs where YOLO is blind)
  silk      per-detection silk signatures + official silk prototypes
  hud       race clock, odometer (distance-to-go), legend running order (recognition-only OCR)
  rails     racing rail on the map; marker grid -> px/m + finish anchor
  film      race-film leader crossings of the painted markers -> the metric's constant offset
  observe   detections -> (d, lateral) metres (odometer-tick distance calibration) -> world tracklets
  finish    RF/drone finish-line camera: saddlecloth OCR -> finish order + spatial gaps
  identify  tracklet -> saddlecloth number (legend order + finish camera + elimination + silk)
  fuse      Bayesian fusion with physics/geometry priors -> paths_video.parquet (video-only:
            no official outcome data) + paths.parquet (final: adds the official records)
  validate  gates on the video-only product vs the official record

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
from ..data import Mask, MaskedRace, RaceInfo, download_replay, replay_path

FPS = 25.0


STAGES = (
    "frames",
    "features",
    "pose",
    "worldmap",
    "detect",
    "silk",  # silk signatures + official prototypes (identify's silk LLR, observe's appearance split)
    "hud",
    "rails",
    "start",  # gate opening in the start film: re-anchors the race clock when the overlay clock lags
    "film",  # race-film marker crossings -> the metric's constant offset
    "observe",
    "finish",  # independent of the render; before identify (finish-camera identity votes)
    "identify",
    "fuse_video",
    "fuse_final",
    "validate",
)
OUTPUTS = {
    "frames": ["frames.bin", "frames_idx.npz", "frames_clock.npy"],
    "features": ["feats.npz", "yolo.npz"],  # + the avatar detections (read by detect)
    "pose": ["pose.npz"],
    "detect": ["dets.npz"],
    "silk": ["silk.npz"],
    "hud": ["hud.npz", "legend.npz"],
    "start": ["start.npz"],
    "film": ["film.npz"],
    "worldmap": ["map.png", "map_cnt.npy", "map_meta.npz"],
    "rails": ["rail.npz", "metric.npz"],
    "observe": ["tracks.npz"],
    "identify": ["labels.npz"],
    "finish": ["finish.npz"],
    "fuse_video": ["paths_video.parquet", "fuse_video.pkl"],
    "fuse_final": ["paths.parquet"],
    "validate": ["validation.json"],
}

# stages that read the official outcome (through the mask); they and their consumers depend on the mask
READS_OUTCOME = ("fuse_final",)
MASK_FILE = "mask.txt"  # the mask behind a run dir's outcome-dependent artifacts; absent = none


# stage -> the stages whose artifacts it reads (directly or through the helpers it calls)
DEPS = {
    "frames": (),
    "features": ("frames",),
    "pose": ("features",),
    "worldmap": ("frames", "pose"),
    "detect": ("frames", "features", "pose", "worldmap"),
    "silk": ("frames", "detect"),
    "hud": ("frames",),
    "rails": ("frames", "pose", "worldmap", "detect"),
    "start": ("hud",),
    "film": ("hud",),
    "observe": ("pose", "detect", "silk", "hud", "rails", "start", "film"),
    "finish": (),
    "identify": ("observe", "hud", "silk", "finish"),
    "fuse_video": ("identify", "observe", "finish", "hud", "silk", "rails"),
    "fuse_final": ("fuse_video",),
    "validate": ("fuse_video", "fuse_final"),
}


def upstream(stage: str) -> set:
    """Every stage ``stage`` depends on, transitively."""
    out, todo = set(), list(DEPS[stage])
    while todo:
        d = todo.pop()
        if d not in out:
            out.add(d)
            todo.extend(DEPS[d])
    return out


def _nproc() -> int:
    """Worker processes per race (env HKJC_NPROC; default: all cores / parallel races)."""
    return max(1, int(os.environ.get("HKJC_NPROC", "0")) or (os.cpu_count() or 4))


_CACHE = None
_FEATS_PAR = None
_STEP_NN = {}  # fork-shared bfmatch cross-check tables for the step matches (empty: OpenCV in-call)
_STYLE = "silk"  # render style of the race being processed (fork-shared with the feature workers)


def _feat_job(args):
    k, nfeat = args
    i, p = _CACHE.idx[k], _CACHE.panel(k)
    if _STYLE == "circle":  # numbered discs, no legend strip
        from . import circles

        lt = p.shape[0]
        rs = circles.centres(p)
        hist = [np.zeros(14) for _ in rs]
    else:
        lt = video.legend_top(p)
        # the avatar detector's boxes (+ shadows) are masked too: the colour mask misses brown horses
        rs = _DET.panel(p, lt)
        hist = [_detect.silk_hist(p, r) for r in rs]
    m = register.mask_boxes(register.world_mask(p, video.hud_mask(p, lt)), rs)
    f = register.extract(p, m, nfeat)
    return int(i), lt, f, p.shape[:2], rs, hist


_DET = None


def _det_init():
    global _DET
    if _STYLE != "circle":
        _DET = _detect.AvatarDetector()


def _step_job(k):
    m = register.match(_FEATS_PAR[k - 1], _FEATS_PAR[k], nn=_STEP_NN.get((k - 1, k, True)))
    return np.r_[m.S.ravel(), m.n_inl, m.resid] if m else None


class RaceRun:
    """``info``: the full race record (``truth``, read only by validation). Every stage reads the race
    through ``self.info``, the masked view: pre-race facts plus the official outcome records ``mask``
    leaves visible (default: ``Mask.from_env()``, i.e. ``HKJC_MASK`` / ``HKJC_FINAL_HOLDOUT``; no mask
    withholds nothing). A masked run writes ``fuse_final``/``validate`` artifacts like any other: give it
    its own ``root``: ``run`` refuses a dir whose outcome-dependent artifacts were made under another mask."""

    def __init__(
        self, info: RaceInfo, root: str | Path = "data_cache/recon", log=print, mask: Mask | None = None
    ):
        self.truth = info
        self.mask = Mask.from_env() if mask is None else mask
        self.info = MaskedRace(info, self.mask)
        self.dir = Path(root) / info.key
        self.dir.mkdir(parents=True, exist_ok=True)
        self.log = log
        self._style = None  # render style, read once from frames_idx.npz

    # ------------------------------------------------------------------ plumbing
    def p(self, name: str) -> Path:
        return self.dir / name

    def video(self, feed: str = "aerial") -> Path:
        path = replay_path(self.info.date, self.info.race_no, feed)
        if not path.exists():
            download_replay(self.info.date, self.info.race_no, feed)
        return path

    def out(self, name: str):
        """Context yielding the temporary path to write artifact ``name`` to (atomic rename)."""
        return video.atomic(self.p(name))

    def done(self, stage: str) -> bool:
        if stage == "features":
            self._backfill_yolo()
        return all(self.p(o).exists() for o in OUTPUTS[stage])

    def _backfill_yolo(self):
        """Run dirs made before features saved its detections: a YOLO-only dets.npz holds exactly
        the yolo.npz rows and histograms, so recover the file instead of re-running features."""
        y, d = self.p("yolo.npz"), self.p("dets.npz")
        if y.exists() or not (self.p("feats.npz").exists() and d.exists()):
            return
        z = np.load(d)
        if str(z["source"]) == "yolo":
            with self.out("yolo.npz") as fp:
                np.savez(fp, rows=z["rows"], hist=z["hist"])

    # stages that read only the videos/frame cache run CONCURRENTLY with the render chain; the
    # chain waits for one right before the first stage that depends on it (results identical)
    BACKGROUND = ("hud", "finish")

    def plan(self, stages=STAGES, force=()) -> list:
        """Stages of ``stages`` to run: those not done, those forced, and every stage that depends
        (transitively) on a forced one -- a forced stage's consumers would otherwise stay stale.
        A requested stage that has never run (none of its outputs exist) counts as forced: artifacts
        made before one of their inputs existed are stale. A stage missing only some outputs (large
        files deleted for space) is regenerated without invalidating its consumers (deterministic)."""
        never = {st for st in stages if not any(self.p(o).exists() for o in OUTPUTS[st])}
        forced = set(force) | never
        for st in STAGES:
            if any(d in forced for d in DEPS[st]):
                forced.add(st)
        done = self.done_set()
        return [st for st in stages if st not in done or st in forced]

    def outcome_stages(self) -> set:
        """READS_OUTCOME and every stage depending on one of them (transitively)."""
        dep = set(READS_OUTCOME)
        for st in STAGES:
            if any(d in dep for d in DEPS[st]):
                dep.add(st)
        return dep

    def mask_guard(self, todo):
        """One run dir holds one mask's outcome-dependent artifacts: never reuse or overwrite them under
        another mask. Records a non-empty mask before any outcome-dependent stage runs."""
        rec = self.p(MASK_FILE)
        have = rec.read_text().strip() if rec.exists() else "none"
        dep = self.outcome_stages()
        if have != str(self.mask) and any(self.p(o).exists() for st in dep for o in OUTPUTS[st]):
            raise ValueError(f"{self.info.key}: {self.dir} holds artifacts made under mask '{have}', this run's "
                             f"mask is '{self.mask}': give the run its own root")
        if dep & set(todo):
            if self.mask:
                rec.write_text(f"{self.mask}\n")
            elif rec.exists():
                rec.unlink()

    def run(self, stages=STAGES, force=()):
        import multiprocessing as mp

        todo = self.plan(stages, force)
        self.mask_guard(todo)
        bg, launched, finished = {}, set(), set()

        def launch(st):
            launched.add(st)

            def work():
                import signal
                import sys

                signal.signal(signal.SIGTERM, lambda *_: sys.exit(1))  # pools shut down cleanly
                self._timed(st, background=True)

            pr = mp.get_context("fork").Process(target=work)
            pr.start()
            bg[st] = pr

        def wait(st):
            pr = bg.pop(st)
            pr.join()
            if pr.exitcode != 0:
                raise RuntimeError(f"{self.info.key}: background stage {st} failed")

        def ready(b):  # every prerequisite in this run's todo list has finished
            return all(d not in todo or d in finished for d in DEPS[b])

        def launch_ready():
            for b in self.BACKGROUND:
                if b in todo and b not in launched and ready(b):
                    launch(b)

        current = None
        try:
            launch_ready()
            for st in todo:
                for b in list(bg):
                    if b in upstream(st):
                        current = b
                        wait(b)
                if st in self.BACKGROUND:
                    continue  # runs (or ran) in the background
                current = st
                self._timed(st)
                finished.add(st)
                launch_ready()
            for b in list(bg):
                current = b
                wait(b)
            current = None
        except Exception as e:
            self.write_status(failed=(current, f"{type(e).__name__}: {e}"))
            raise
        finally:  # a failed stage must not leave background stages running into the next race
            for pr in bg.values():
                pr.terminate()
                pr.join()
        self.write_status()

    # video-only health checks: thresholds sit between the S1/dev silk-render races (worst 5.1 ticks/s,
    # 0.64 markers per runner) and the 2026/27 numbered-circle render the pipeline cannot read (<= 3.5, ~0.08)
    HEALTH_TICKS_PER_S = 4.0  # unique render frames per expected race second (distance / 16.5 m/s)
    HEALTH_MARKERS = 0.4  # median avatar detections per render frame / runners
    HEALTH_CLOCK_READS = 3  # race-clock reads (the clock zero, film and start need them)

    def health(self) -> dict:
        """Video-only self-checks on the artifacts present: {check: (ok, value)}."""
        out = {}
        D, N = float(self.info.distance), max(int(self.info.n), 1)
        if self.p("frames_idx.npz").exists():
            v = len(np.load(self.p("frames_idx.npz"))["idx"]) / (D / 16.5)
            out["render_ticks_per_s"] = (v >= self.HEALTH_TICKS_PER_S, round(v, 2))
        if self.p("yolo.npz").exists():
            rows = np.load(self.p("yolo.npz"))["rows"]
            cnt = np.bincount(rows[:, 0].astype(int)) if len(rows) else np.zeros(1, int)
            v = float(np.median(cnt[cnt > 0])) / N if (cnt > 0).any() else 0.0
            out["markers_per_runner"] = (v >= self.HEALTH_MARKERS, round(v, 2))
        if self.p("hud.npz").exists():
            v = int(np.isfinite(np.load(self.p("hud.npz"))["rows"][:, 1]).sum())
            out["clock_reads"] = (v >= self.HEALTH_CLOCK_READS, v)
        if self.p("rail.npz").exists():
            straight = "lat0_px" in np.load(self.p("rail.npz")).files
            expect = self.info.venue == "ST" and int(D) == 1000
            out["course_shape"] = (straight == expect, "straight" if straight else "oval")
        if self.p("start.npz").exists():
            st = str(np.load(self.p("start.npz"))["status"])
            out["clock_anchor"] = (st != "none", st)
        return out

    def write_status(self, failed=None):
        """status.json (not a stage artifact): 'failed' (stage, error), 'unhealthy' (failed checks) or 'ok'."""
        try:
            h = self.health()
        except Exception as e:  # never mask the stage error being reported
            h = {"health_error": (False, f"{type(e).__name__}: {e}"[:200])}
        bad = [k for k, (ok, _v) in h.items() if not ok and k != "clock_anchor"]
        status = "failed" if failed else ("unhealthy" if bad else "ok")
        rec = dict(status=status, checks={k: {"ok": bool(ok), "value": v} for k, (ok, v) in h.items()})
        if failed:
            rec["stage"], rec["error"] = failed[0], failed[1][:300]
        if bad:
            rec["unhealthy"] = bad
        with open(self.p("status.json"), "w") as fh:
            json.dump(rec, fh, indent=1)

    def _timed(self, st, background=False):
        import resource

        def cpu():
            r = [resource.getrusage(w) for w in (resource.RUSAGE_SELF, resource.RUSAGE_CHILDREN)]
            return sum(x.ru_utime + x.ru_stime for x in r)

        t0, c0 = time.time(), cpu()
        getattr(self, st)()
        dt = time.time() - t0
        self.log(f"{self.info.key} {st}: {dt:.0f}s" + (" (background)" if background else ""))
        own, kids = (
            resource.getrusage(w).ru_maxrss // 1024
            for w in (resource.RUSAGE_SELF, resource.RUSAGE_CHILDREN)
        )
        rec = dict(
            stage=st,
            s=round(dt, 1),
            cpu_s=round(cpu() - c0, 1),  # this process + its finished workers
            background=background,
            end=round(time.time(), 1),
            maxrss_mb=own,
            maxrss_children_mb=kids,
            nproc=_nproc(),
        )
        with open(self.p("timings.jsonl"), "a") as fh:
            fh.write(json.dumps(rec) + "\n")

    def done_set(self):
        return {st for st in STAGES if self.done(st)}

    # ------------------------------------------------------------------ stages
    def frames(self):
        style, box = video.layout(self.video(), self.info.n)
        if style == "unknown":
            raise RuntimeError("unknown render style (neither the silk legend nor numbered discs)")
        n = video.FrameCache.build(self.video(), self.dir, style=style, layout_box=box)
        self._style = style
        self.log(f"   frames: {n} unique render frames cached ({style} render)")

    def render_style(self) -> str:
        """'silk' or 'circle', recorded by the frames stage (caches made before it are silk)."""
        if self._style is None:
            z = np.load(self.p("frames_idx.npz"))
            self._style = str(z["style"]) if "style" in z.files else "silk"
        return self._style

    def legend_row(self, panel) -> int:
        """Top row of the silk-legend strip (panel-relative); the circle render has none."""
        return panel.shape[0] if self.render_style() == "circle" else video.legend_top(panel)

    def cache(self):
        return video.FrameCache(self.dir)

    def features(self, nfeat: int = 3000):
        """ORB features of every unique render frame + the sequential step matches. Per-frame work
        is independent and OpenCV's RANSAC seeds its RNG per call, so the process pool gives results
        identical to a sequential loop. The avatar detections made for the mask are saved for
        ``detect`` (yolo.npz)."""
        import multiprocessing as mp

        cache = self.cache()
        from . import bfmatch

        global _CACHE, _FEATS_PAR, _STYLE, _STEP_NN
        _CACHE = cache
        _STYLE = self.render_style()
        # one avatar detector per worker; one accelerator context
        accel = os.environ.get("HKJC_DEVICE", "cpu").startswith(("intel:", "cuda"))
        nw = 1 if accel else min(_nproc(), 4)
        with mp.get_context("fork").Pool(nw, initializer=_det_init) as pool:
            res = pool.map(_feat_job, [(k, nfeat) for k in range(len(cache))], chunksize=8)
        # a NEW pool: workers fork the parent's state, so the features must exist before forking
        _FEATS_PAR = [r[2] for r in res]
        pairs = [k for k in range(1, len(res)) if res[k][0] - res[k - 1][0] <= 6]
        _STEP_NN = bfmatch.tables([f.des for f in _FEATS_PAR], [(k - 1, k, True) for k in pairs])
        with mp.get_context("fork").Pool(_nproc()) as pool:
            ms = dict(zip(pairs, pool.map(_step_job, pairs, chunksize=8)))
        _CACHE = _FEATS_PAR = None
        _STEP_NN = {}
        steps = [
            ms.get(k) if ms.get(k) is not None else np.full(11, np.nan) for k in range(len(res))
        ]
        with self.out("feats.npz") as fp:
            np.savez_compressed(
                fp,
                idx=np.array([r[0] for r in res]),
                lrow=np.array([r[1] for r in res]),
                pts=np.concatenate([r[2].pts for r in res]),
                des=np.concatenate([r[2].des for r in res]),
                off=np.cumsum([0] + [len(r[2].pts) for r in res]),
                steps=np.array(steps),
                shape=np.array(res[0][3]),
            )
        nb = 14  # silk_hist length
        with self.out("yolo.npz") as fp:
            np.savez(
                fp,
                rows=np.array([[r[0], *b[:5]] for r in res for b in r[4]]).reshape(-1, 6),
                hist=np.array([h for r in res for h in r[5]]).reshape(-1, nb),
            )
        if _STYLE == "circle":
            self._read_numbers(cache, res)
        _STYLE = "silk"

    def _read_numbers(self, cache, res):
        """numbers.npz (circle render): the number read inside each disc, row-aligned with yolo.npz."""
        from . import circles

        crops = []
        for k, r in enumerate(res):
            p = cache.panel(k)
            crops += [circles.crop(p, b[0], b[1]) for b in r[4]]
        reads = circles.read(crops)
        with self.out("numbers.npz") as fp:
            np.savez(fp, num=np.array([n for n, _c in reads], int), conf=np.array([c for _n, c in reads]))
        ok = np.mean([n > 0 and c >= 0.6 for n, c in reads]) if reads else 0.0
        self.log(f"   features: {len(reads)} discs, {100 * ok:.0f}% numbers read")

    def pose(self):
        z = dict(np.load(self.p("feats.npz")))
        G, info = pose.solve_race(z)
        with self.out("pose.npz") as fp:
            np.savez(fp, idx=z["idx"], G=G, shape=z["shape"], **info)
        self.log(f"   pose: {info['summary']}")

    def detect(self):
        """The features stage's avatar detections (YOLO); where YOLO is blind (AWT dirt) the
        map-difference blobs join."""
        import cv2

        cache = self.cache()
        x0, x1, y0, y1 = cache.box
        box = video.PanelBox(x0, x1, y0, y1)
        z = np.load(self.p("yolo.npz"))
        ry, hy = z["rows"], z["hist"]
        yolo_med = (
            float(np.median(np.bincount(np.searchsorted(cache.idx, ry[:, 0]), minlength=len(cache))))
            if len(ry)
            else 0.0
        )
        rb, hb = [], []
        if yolo_med < 0.6 * self.info.n and self.render_style() != "circle":  # YOLO blind (AWT dirt)
            P = np.load(self.p("pose.npz"))
            k_of = {int(f): k for k, f in enumerate(P["idx"])}
            img_map, _cnt, _to = self._map()
            meta = np.load(self.p("map_meta.npz"))
            mx0, my0 = meta["bounds"]
            sc = float(meta["sc"])
            A = np.array([[sc, 0, -mx0 * sc], [0, sc, -my0 * sc], [0, 0, 1]])
            for i, p in cache:
                if i not in k_of:
                    continue
                lt = self.legend_row(p)
                back = cv2.warpAffine(
                    img_map,
                    (A @ P["G"][k_of[i]])[:2],
                    (p.shape[1], p.shape[0]),
                    flags=cv2.WARP_INVERSE_MAP,
                )
                for r in _detect.bgsub_blobs(p, back, video.hud_mask(p, lt)):
                    rb.append([i, *r[:5]])
                    hb.append(_detect.silk_hist(p, r))
        rb = np.array(rb).reshape(-1, 6)
        # race-level switch: on turf YOLO alone (background-subtraction blobs add duplicates and
        # false positives); where YOLO is blind (AWT dirt: median ~1/frame) map-difference blobs join
        if self.render_style() == "circle":  # disc detections only: numbers.npz stays row-aligned
            rows, hists, src = ry, np.array(hy), "yolo"
        elif yolo_med >= 0.6 * self.info.n:
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
        with self.out("dets.npz") as fp:
            np.savez_compressed(
                fp,
                rows=rows,
                hist=hists,
                box=np.array([box.x0, box.x1, box.y0, box.y1]),
                source=src,
            )

    def silk(self):
        """Silk signature of every detection (``detect.silk_feature``, from the aerial video's render
        panel) + the OFFICIAL silk prototype of every runner (``detect.silk_prototype``; pre-race
        input). Read by identify's silk LLR (``_silk_llr``, also in the fuse-stage EM identity pass)
        and observe's appearance split (``tracking.split_by_appearance``)."""
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
        with self.out("silk.npz") as fp:
            np.savez(fp, feat=feat, proto_no=np.array(nos), proto=np.array(proto))
        self.log(f"   silk: {len(rows)} detections, {len(nos)}/{len(self.info.runners)} official silks")

    def hud(self):
        rows, leg = ocr.read_render_hud_cached(self.cache(), n_field=max(14, self.info.n))
        if self.render_style() == "circle":  # no legend strip: its slots stay unread
            leg[:, 1:] = 0
        with self.out("hud.npz") as fp:
            np.savez(fp, rows=rows)
        with self.out("legend.npz") as fp:
            np.savez(fp, rows=leg)

    def worldmap(self, scale: float = 1.0):
        import cv2

        z = np.load(self.p("pose.npz"))
        G, shape = z["G"], tuple(z["shape"])
        k_of = {int(f): k for k, f in enumerate(z["idx"])}
        mo = worldmap.Mosaic(worldmap.bounds_of(G, shape), scale)
        for i, p in self.cache():
            if i in k_of:
                mo.add(p, video.hud_mask(p, self.legend_row(p)), G[k_of[i]])
        with self.out("map.png") as fp:
            cv2.imwrite(str(fp), mo.image())
        with self.out("map_cnt.npy") as fp:
            np.save(fp, mo.cnt.astype(np.float16))
        with self.out("map_meta.npz") as fp:
            np.savez(fp, bounds=np.array([mo.x0, mo.y0]), sc=scale)

    def _final_pass_map(self, P, out, rf, span):
        """Mosaic of only the frames whose rail station lies in the last ``span`` world px of the
        trip (the final pass over the finish area), cropped to those frames' bounds (memory: a
        full-extent float mosaic exceeds a 3 GB cap). -> (img, cnt, to_px)."""
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
                mo.add(p, video.hud_mask(p, self.legend_row(p)), G[k])
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
        # glyph furniture (e.g. the disclaimer word "phones")
        q = np.round(rows[:, 1:3] / 8.0).astype(int)
        key = q[:, 0] * 10_000 + q[:, 1]
        uk, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
        nfr = max(1, len(np.unique(rows[:, 0])))
        # only inside the HUD band: the render camera follows the field, so real horses also sit
        # near-static in the panel for long stretches
        keep &= ~((cnt[inv] >= 0.04 * nfr) & (rows[:, 2] < video.DISCLAIMER_ROWS + 20))
        rows, hist = rows[keep], D["hist"][keep]
        self._wd_idx = np.where(keep)[0]  # dets.npz row index of each kept detection (silk lookup)
        G = P["G"][[k_of[int(f)] for f in rows[:, 0]]]
        X = np.einsum("nij,nj->ni", G, np.c_[rows[:, 1:3], np.ones(len(rows))])[:, :2]
        return rows, X, hist, P

    def rails(self):
        img, cnt, to_px = self._map()
        rows, X, _h, P = self.world_dets()
        out = rails.extract_racing_rail(
            img, cnt, to_px, P["G"], tuple(P["shape"]), X, self.info.geometry.handedness
        )
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
            out["rail"] = rails.smooth_rail(out["rail"], out["support"])
            self.log(
                f"   rails: extracted ridge was the FIXED rail -> moved {shift:.0f} px out to the "
                f"movable rail (course gap {exp_gap:.1f} m, ridge support {ssup:.0%})"
            )
        if "lat0_px" in out:
            lo_, hi_ = out["lat0_range"]
            self.log(
                f"   rails: STRAIGHT course -> field-edge datum; fence measured {out['lat0_px']:.0f} px "
                f"inside it (range {lo_:.0f}..{hi_:.0f})"
            )
        with self.out("rail.npz") as fp:
            np.savez(fp, frame_idx=P["idx"], **out)
        rf = rails.RailFrame.from_points(out["rail"], int(out["inner"]))
        # painted markers are drawn only near the END of the race: profile them on a FINAL-PASS
        # mosaic (rail station in the last 9000 px); on trips passing the finish area twice (HV 2200)
        # an all-frames mean fades them below detection
        mimg, mcnt, mto_px = self._final_pass_map(P, out, rf, 9000.0)
        ss, resp = rails.marker_profile(mimg, mcnt, mto_px, rf, rf.s[-1] - 9000, rf.s[-1])
        del mimg, mcnt
        with self.out("marker_profile.npz") as fp:
            np.savez(fp, ss=ss, resp=resp)  # observe: tick-seeded grid fit
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
        with self.out("metric.npz") as fp:
            np.savez(
                fp,
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

    def overlay_f0(self) -> float:
        """Video frame where the overlay race clock reads zero."""
        r = np.load(self.p("hud.npz"))["rows"]
        ok = ~np.isnan(r[:, 1])
        return float(np.median(r[ok, 0] - FPS * r[ok, 1]))

    def clock_lag(self) -> float:
        """Frames the overlay clock lags the gate opening (``start`` stage; 0 when synced or not measured)."""
        if self.p("start.npz").exists():
            return max(0.0, float(np.load(self.p("start.npz"))["lag_frames"]))
        return 0.0

    def clock_f0(self) -> float:
        """Video frame of the race-clock zero: the overlay clock's zero, moved back to the gate opening when the
        overlay clock lags it."""
        lag = self.clock_lag()
        return self.overlay_f0() - lag if lag > 0 else self.overlay_f0()

    def start(self):
        """Gate opening in the start film (``recon.start``): the overlay clock's lag behind it."""
        from . import start

        r = np.load(self.p("hud.npz"))["rows"]
        ok = ~np.isnan(r[:, 1])
        clock0 = float(np.median(r[ok, 0] - FPS * r[ok, 1])) if ok.any() else np.nan
        res = (
            start.lag(self.video(), clock0, self.info.venue)
            if np.isfinite(clock0)
            else dict(clock0=clock0, onset=np.nan, lag_frames=0.0, status="none")
        )
        with self.out("start.npz") as fp:
            np.savez(fp, **res)
        self.log(
            f"   start: gate opening {res['onset']:.1f}, clock zero {clock0:.1f} -> {res['status']}"
            f" (lag {res['lag_frames'] / FPS:.2f} s)"
        )

    def film(self):
        """Leader crossings of the REAL painted markers in the race film (``recon.film``); the
        finish time is located from the HUD odometer reaching its minimum (no official data)."""
        hud = np.load(self.p("hud.npz"))["rows"]
        g = np.isfinite(hud[:, 2])
        mk, tt = np.zeros(0, int), np.zeros(0)
        if g.sum() >= 10:
            # film.npz holds OVERLAY-clock times; observe moves them to the race clock (clock_lag)
            t_last = (hud[g, 0][np.argmin(hud[g, 2])] - self.overlay_f0()) / FPS
            g_ = self.info.geometry
            mx = g_.home_straight(self.info.course) - film.STRAIGHT_MARGIN
            got = film.crossings(self.video(), hud, t_last, max_marker=mx)
            mk = np.array(sorted(got, reverse=True), int)
            tt = np.array([got[m] for m in mk], float)
        with self.out("film.npz") as fp:
            np.savez(fp, markers=mk, t=tt)
        self.log(f"   film: {len(mk)} marker crossings {dict(zip(mk.tolist(), tt.round(2)))}")

    def observe(self):
        rows, X, hist, _P = self.world_dets()
        seed = os.environ.get("HKJC_PERTURB_SEED")  # DIAGNOSTIC: robustness perturbation
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
        # race-clock time of a render frame: its tick on the uniform 10 Hz grid (video.tick_frame)
        grid = video.tick_grid(np.load(self.p("frames_idx.npz"))["idx"])
        f0 = self.clock_f0()

        def tick_s(f):
            return (video.tick_frame(f, grid) - f0) / FPS

        # leader position per frame: a front detection > 8 m clear of the next is a false positive
        # only if that gap is not persistent (a genuinely clear leader keeps its margin)
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
        # the race film's REAL painted markers fix the metric's constant: odometer ticks are the
        # metric to the line (unpinned, no convention offset), shifted so the render leader meets the
        # film's marker crossings (one flat offset per race). Fallback (< FILM_MIN crossings, or none
        # matched by the render leader): the render-paint pin.
        fz = self.p("film.npz")
        film_t = {}
        if fz.exists():
            zf = np.load(fz)
            lag = self.clock_lag()  # film.npz is on the overlay clock
            tt = zf["t"] + lag / FPS if lag > 0 else zf["t"]
            film_t = dict(zip(zf["markers"].tolist(), tt.tolist()))
        use_film = len(film_t) >= film.FILM_MIN
        d_of_s, c, ppm_loc, resid, (ppm_odo, sF_odo, pinned, c_m, nsh) = (
            rails.calibrate_distance_ticks(
                fu,
                s_lead,
                odo,
                ppm,
                sF,
                pin_finish=False if use_film else None,
                have_markers=have_markers,
            )
        )
        delta = 0.0
        if use_film:
            dlt, per = film.anchor_offset(tick_s(fu), d_of_s(s_lead), film_t)
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
        tf_u = tick_s(fu)
        shift_u, n_win = rails.time_local_shift(tf_u, d_of_s(s_lead) - delta, odo)
        if n_win:
            d = d - shift_u[np.searchsorted(fu, rows[:, 0])]
            self.log(f"   observe: time-local shift in {n_win} window(s), max {np.abs(shift_u).max():.1f} m")
        # straight courses (the rails stage's decision, recorded as lat0_px): the rail polyline is the
        # field-edge datum and lat0_px the fence's measured offset inside it (rails.straight_datum)
        lat0 = float(z["lat0_px"]) if "lat0_px" in z.files else 0.0
        lat = (lat_px + lat0) / ppm_loc(s_px)
        g = lat > -1.0  # nothing races inside the racing rail
        rows, d, lat, hist, X = rows[g], d[g], lat[g], hist[g], X[g]
        det_idx = self._wd_idx[g]
        t = tick_s(rows[:, 0])
        frames = np.unique(rows[:, 0])
        fi = np.searchsorted(frames, rows[:, 0])
        times = tick_s(frames)
        trs = tracking.stitch(tracking.track(times, -d, lat, fi))
        trs.sort(key=lambda tk: -len(tk.t))
        members = [np.array(tk.det) for tk in trs]
        if self.p("silk.npz").exists():  # identity switches inside tracker tracklets
            Fs = np.load(self.p("silk.npz"))["feat"][det_idx].astype(float)
            n0 = len(members)
            members = tracking.split_by_appearance(members, t, Fs)
            members = sorted((m for m in members if len(m)), key=lambda m: -len(m))
            self.log(f"   observe: appearance split {n0} -> {len(members)} tracklets")
        nums = self._number_reads(dict(det_idx=det_idx))  # circle render: identity switches inside tracklets
        if nums is not None:
            n0 = len(members)
            members = tracking.split_by_number(members, t, nums)
            members = sorted((m for m in members if len(m)), key=lambda m: -len(m))
            self.log(f"   observe: number split {n0} -> {len(members)} tracklets")
        with self.out("tracks.npz") as fp:
            np.savez(
                fp,
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
        votes, rank, elim, elim_n = identity.legend_votes(
            mem, T["rows"][:, 0], T["d"], leg, sorted(valid), parts=True
        )
        fin, _fit = self._finish_identity(T, mem)
        fvotes = np.zeros_like(votes)
        for k, num in fin.items():
            if num <= nmax:
                votes[k, num] += self.FINISH_VOTES
                fvotes[k, num] += self.FINISH_VOTES
        nums = self._number_reads(T)  # circle render: the number printed on each disc
        if nums is not None:
            for k, m in enumerate(mem):
                v = nums[m]
                v = v[(v > 0) & (v <= nmax)]
                if len(v):
                    c = np.bincount(v, minlength=nmax + 1).astype(votes.dtype) * self.NUMBER_VOTE
                    votes[k] += c
                    fvotes[k] += c
        # joint ML assignment: set-valued legend evidence + silk evidence
        lab = identity.assign_global(
            mem,
            T["rows"][:, 0],
            rank,
            nmax,
            silk_llr=self._silk_llr(mem, T, nmax),
            elim=elim,
            n_elim=elim_n,
            extra=fvotes,
            valid=valid,
        )
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
        extra = dict(rank=rank, elim=elim, n_elim=elim_n, fvotes=fvotes)
        with self.out("labels.npz") as fp:
            np.savez(fp, labels=lab, votes=votes, fixed=fixed, **extra)
        self.log(
            f"   identify: finish-camera named {len(fin)} + {n_elim} by elimination; "
            f"numbered {int((lab > 0).sum())}/{len(mem)} tracklets "
            f"({sum(len(m) for k, m in enumerate(mem) if lab[k]) / len(T['d']):.0%} of detections)"
        )

    FINISH_VOTES = 40.0  # one finish-camera match ~ 4 s of unanimous legend votes
    NUMBER_VOTE = 1.0  # one disc number read (circle render) ~ one legend frame vote
    NUMBER_CONF = 0.6  # minimum OCR confidence of a disc number read

    def _number_reads(self, T):
        """Per tracks row: the disc number read (circle render; 0 = unread or low confidence), else None."""
        if not self.p("numbers.npz").exists():
            return None
        z = np.load(self.p("numbers.npz"))
        di = np.asarray(T["det_idx"], int)
        num, conf = z["num"][di], z["conf"][di]
        valid = {r.no for r in self.info.runners}
        ok = (conf >= self.NUMBER_CONF) & np.isin(num, list(valid))
        return np.where(ok, num, 0)

    def _silk_llr(self, mem, T, nmax):
        """(tracklets x numbers) silk log-likelihood ratio for identity.assign_global (None without
        silk.npz): mapped official prototype vs the tracklet's mean render signature."""
        if not self.p("silk.npz").exists():
            self.log("   identify: WARNING silk.npz missing -> no appearance term")
            return None
        from .assoc import _collapse
        from .silkmap import SILK_LLR_A, SILK_LLR_B, SILK_LLR_CLIP, SILK_MAP

        Bm = np.asarray(SILK_MAP)
        zs = np.load(self.p("silk.npz"))
        F = zs["feat"][T["det_idx"]].astype(float)
        proto = {}
        for h, p in zip(zs["proto_no"], zs["proto"]):
            q = np.clip(np.r_[_collapse(p), 1.0] @ Bm, 1e-9, None)
            proto[int(h)] = q / q.sum()
        out = np.zeros((len(mem), nmax))
        for k, m in enumerate(mem):
            f = F[m]
            f = f[f.sum(1) > 0]
            if len(f) < 10:
                continue
            sig = _collapse((f / f.sum(1, keepdims=True)).mean(0))
            for h, q in proto.items():
                if 1 <= h <= nmax:
                    tv = 0.5 * np.abs(q - sig).sum()
                    out[k, h - 1] = np.clip(SILK_LLR_A + SILK_LLR_B * tv, -SILK_LLR_CLIP, SILK_LLR_CLIP)
        return out

    def _finish_identity(self, T=None, mem=None):
        """({tracklet index: number}, label-free fit) from the finish camera (1D registration of the
        tracklets' line-crossing times to the camera's gaps; tracklets ending < 30 m short are
        extrapolated). A pure function of tracks.npz + finish.npz: called with no arguments it is
        CACHED on their mtimes (the fuse stage's EM loop calls it repeatedly)."""
        import copy

        if T is not None:
            return self._finish_identity_compute(T, mem)
        key = tuple(
            self.p(f).stat().st_mtime_ns if self.p(f).exists() else -1
            for f in ("tracks.npz", "finish.npz")
        )
        c = getattr(self, "_fin_id_cache", None)
        if c is None or c[0] != key:
            T = np.load(self.p("tracks.npz"), allow_pickle=True)
            mem = [np.asarray(m, int) for m in T["members"]]
            c = (key, self._finish_identity_compute(T, mem))
            self._fin_id_cache = c
        return copy.deepcopy(c[1])

    def _finish_identity_compute(self, T, mem):
        if not self.p("finish.npz").exists():
            return {}, None
        z = np.load(self.p("finish.npz"))
        order, gaps, _u = finishcam.finish_gaps(self.finish_sc(z))
        gaps = {int(k): v for k, v in gaps.items()}
        if len(gaps) < 3:
            return {}, None
        tc, vs = {}, []
        for k, m in enumerate(mem):
            if len(m) < 10:
                continue
            o = np.argsort(T["t"][m])
            t, d = T["t"][m][o], T["d"][m][o]
            # a crossing must be bracketed by samples < 0.3 s apart (the render can skip ~1 s near
            # the line); otherwise extrapolate from the last pre-line sample (< 30 m short) at its
            # own speed
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
            float(np.median(vs)) * ppm_fc if vs and np.isfinite(ppm_fc) else None
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
            # camera it adds false reads
            white=self.info.venue != "HV",
        )
        sc["overhead"] = self.info.venue == "HV"
        order, gaps, ustar = finishcam.finish_gaps(sc)
        with self.out("finish.npz") as fp:
            np.savez(
                fp,
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
        order, gaps, _u, gsd = finishcam.finish_gaps(self.finish_sc(z), with_sd=True)
        order = [int(n) for n in order]
        gaps = {int(k): v for k, v in gaps.items()}
        gsd = {int(k): v for k, v in gsd.items()}
        if len(order) < 3:
            return []
        from .validate import crossing

        # only OBSERVED crossings anchor the clock: a path whose labelled tracklets ended early is
        # extrapolated by fusion with a growing sd and would drag C, the clock and the C prior
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
            c_prior = v_line * ppm_fc
        cal = finishcam.anchor(gaps, t_cross, c_prior=c_prior, c_prior_rel=0.30)
        # label-free sanity check: the anchor fits the clock to LABELLED render paths, so wrong
        # labels drag it; registering the anonymous tracklet crossings to the gaps needs no labels
        # (measured bias ~0.15 s early)
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
            # CONDITIONAL on the fitted clock t_w: fusion treats each horse's obs as independent,
            # so the clock error shared by all horses must not be added per horse (it would bury
            # the precise relative order/gaps). Scale term + gap spread.
            var_inv_c = 1.0 / np.linalg.inv(cov)[1, 1]  # var(1/C | t_w)
            return float(np.sqrt(gaps[h] ** 2 * var_inv_c + (gsd.get(h, 15.0) / C) ** 2))

        sds = {h: sd_t(h) for h in order if np.isfinite(gaps.get(h, np.nan))}
        self.log(
            f"   finish: C={C:.0f} px/s (physical prior {c_prior or float('nan'):.0f}), first "
            f"finisher #{order[0]} at t={t_w:.2f}s ({src}, {n_used} horses); "
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
        """Render-tracklet proposals as fusion.Obs (horse = identify label, 0 = anonymous)."""
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

    # gate geometry from the drone's top-down start frame (HV 06-03 R8): 40.2 px stall pitch at
    # 38.0 px/m (racing -> outer rail 855 px = 22.5 m on course C); stall 1 abuts the racing rail
    STALL_PITCH = 1.06  # m between stall centres
    STALL1_LAT = 2.5  # m: stall-1 centre off the racing rail
    DRAW_SD = 0.4  # m

    def official_observations(self) -> list:
        """FINAL-product observations (official outcome data): the gate draw fixes every runner's
        lateral at t=0 (standing start, s=0), and the official per-runner sectional times fix its
        crossing times at the section boundaries (d = 400 k); what the mask withholds is absent."""
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
        bnd, cum = self.info.official_boundaries()
        for h, ft in self._official_finish().items():  # runners without sectional times: the finish only
            if h not in cum:
                out.append(fusion.Obs(np.array([ft]), np.array([D]), None, self.OFFICIAL_SD, 1.0, "official",
                                      horse=h, tid=30_000 + h))
        for h, c in cum.items():
            ok = np.isfinite(c)
            out.append(
                fusion.Obs(
                    np.asarray(c)[ok],
                    D - np.asarray(bnd)[ok],
                    None,
                    self.OFFICIAL_SD,
                    1.0,
                    "official",
                    horse=int(h),
                    tid=30_000 + int(h),
                )
            )
        return out

    ANCHOR_CHECK = True
    # official sectional/finish times are published to 0.01 s (~0.1 m at race speed): near-hard in the final
    # product, so a mislabelled render tracklet conflicts with them (and is demoted) instead of bending them
    OFFICIAL_SD = 0.1

    def _fuse(self, obs, horses, prior, t_end=None):
        """Render-fused posterior, then + finish camera. ``t_end``: the grid runs at least to this
        time (the FINAL product: past the last official finish)."""
        ac = self.ANCHOR_CHECK
        st = {}
        paths, assign = fusion.fuse_race(obs, horses, prior, anchor_check=ac, stats=st)
        fin = self.finish_observations(paths)  # second pass: + finish camera
        tr = None
        if t_end is not None:
            ts = np.concatenate([o.t for o in obs + fin])
            tr = (float(ts.min()), max(float(ts.max()), t_end))
        if fin or tr is not None:
            paths, assign = fusion.fuse_race(
                obs + fin, horses, prior, anchor_check=ac, stats=st, t_range=tr
            )
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
        with self.out(name) as fp:
            pl.concat(recs).write_parquet(fp)

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
        if "rank" not in L.files:  # stale-cache guard: labels.npz without set-valued votes
            raise RuntimeError(
                "labels.npz has no set-valued legend votes (written before baseline Z5): "
                "re-run the identify stage"
            )
        r1 = L["rank"].copy()
        r1[:, :k] += v2[:, :k]
        lab = identity.assign_global(
            mem,
            T["rows"][:, 0],
            r1,
            nmax,
            silk_llr=self._silk_llr(mem, T, nmax),
            elim=L["elim"],
            n_elim=L["n_elim"],
            extra=L["fvotes"],
            valid=valid,
        )
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

    def fuse_video(self):
        """VIDEO-ONLY posterior (no official outcome data; what the gates validate). Persists its
        observations and paths (fuse_video.pkl) for ``fuse_final``."""
        import pickle

        obs = self.observations()
        prior = fusion.Prior(width=self.info.width)
        horses = [r.no for r in self.info.runners]
        pv, assign = self._fuse(obs, horses, prior)
        lab2 = self._em_identity(pv)  # EM pass: legend votes completed by the first posterior
        if lab2 is not None:
            obs = self.observations(lab2)
            pv, assign = self._fuse(obs, horses, prior)
        obs, pv = self._em_associate(pv, horses, prior)
        self._write_paths(
            {h: fusion.with_tail(p, prior) for h, p in pv.items()}, "paths_video.parquet"
        )
        with self.out("fuse_video.pkl") as tmp, open(tmp, "wb") as fp:
            pickle.dump((obs, pv), fp, protocol=pickle.HIGHEST_PROTOCOL)
        n_anon = sum(1 for o in obs if o.horse == 0)
        n_ass = sum(1 for o in obs if o.horse == 0 and o.tid in assign)
        self.log(
            f"   fuse_video: {len(pv)}/{len(horses)} horses; anonymous tracklets associated {n_ass}/{n_anon}"
        )

    def fuse_final(self):
        """FINAL product: the video-only observations + the official records (draw + sectional
        times), with the shared metric bias, a field-relative start prior and a final identity pass.
        Its grid runs past the last official finish (else the slowest finisher ends a few cm short)."""
        import pickle

        with open(self.p("fuse_video.pkl"), "rb") as f:
            obs, pv = pickle.load(f)
        prior = fusion.Prior(width=self.info.width)
        horses = [r.no for r in self.info.runners]
        offi = self.official_observations()
        t_end = max((float(o.t.max()) for o in offi if o.feed == "official" and len(o.t)), default=None)
        t_end = None if t_end is None else t_end + 0.5
        bias = self._metric_bias(pv)
        obs_f = self._metric_corrected(obs, pv, bias)
        pf, _a = self._fuse(obs_f + offi, horses, prior, t_end=t_end)
        # the gate -> first render frame stretch has no video evidence: a field-relative start prior
        # about the field's median curve (else a horse first seen late is extrapolated back out of the
        # gate at its later speed)
        if pf:
            P0 = next(iter(pf.values()))
            if all(len(p.t) == len(P0.t) for p in pf.values()):
                t_v0 = min(float(p.t[0]) for p in pv.values()) if pv else 0.0
                prior = replace(
                    prior, ref_t=P0.t, ref_s=np.median([p.s for p in pf.values()], axis=0),
                    ref_until=t_v0 + 1.0,
                )
                pf, _a = self._fuse(obs_f + offi, horses, prior, t_end=t_end)
        # FINAL identity pass: the E-step re-run with the official records in every fuse, detections
        # carrying the final metric bias, starting from the detection paths except where a horse's
        # path conflicts with its own official skeleton (> 4 combined sigma); video-only unaffected.
        # No official times (all withheld by the mask): no skeleton
        sk = self._fuse(offi, horses, prior, t_end=t_end)[0] if t_end is not None else {}
        start = self._skeleton_merge(pf, sk)
        obs_e, _pe = self._em_associate(
            start, horses, prior, extra=offi, out_name="det_labels_final.npz", s_bias=bias
        )
        pf, _a = self._fuse(obs_e + offi, horses, prior, t_end=t_end)
        pf, warp = self._official_retime(pf)
        if warp:
            big = {h: round(w, 2) for h, w in warp.items() if w > 0.05}
            self.log(f"   fuse_final: re-timed onto the official boundary times (max shift {max(warp.values()):.3f} s"
                     + (f"; > 0.05 s: {big}" if big else "") + ")")
        self._write_paths(pf, "paths.parquet")
        self.log(f"   fuse_final: {len(pf)}/{len(horses)} horses")

    TIE_STEP = 0.002  # s between horses sharing an official finish time, in placing order

    def _official_finish(self) -> dict:
        """{horse: official finish time s} (empty where the mask withholds it)."""
        return self.info.official_finish()

    def _official_place(self, h) -> int:
        """Index of horse ``h`` in the official result (placing order; unplaced last)."""
        hs = self.info.official_order()
        return hs.index(h) if h in hs else len(hs)

    def _official_retime(self, paths):
        """FINAL product: each horse passes every official boundary at its official time. A strictly
        increasing C1 re-timing of the fused path (``_monotone_warp``; fixed at its first node) moves its
        crossings onto the official times without a speed jump at any boundary; lateral and sd follow
        the same re-timing. A boundary the mask withholds is not re-timed.
        -> (paths, {horse: largest shift s})."""
        bnd, cum = self.info.official_boundaries()
        D = self.info.distance
        fin = self._official_finish()
        targets = {h: (list(bnd), list(c)) for h, c in cum.items()}
        for h, ft in fin.items():
            targets.setdefault(h, ([0.0], [ft]))
        # equal official finish times (0.01 s resolution) keep the official placing: spaced inside the interval
        groups = {}
        for h in fin:
            if h in targets and targets[h][0] and targets[h][0][-1] == 0.0 and np.isfinite(targets[h][1][-1]):
                groups.setdefault(round(targets[h][1][-1], 2), []).append(h)
        for tie in groups.values():
            for i, h in enumerate(sorted(tie, key=self._official_place)):
                targets[h][1][-1] += i * self.TIE_STEP
        out, warp = {}, {}
        for h, p in paths.items():
            if h not in targets:
                out[h] = p
                continue
            bnd_h, c = targets[h]
            s_mono = np.maximum.accumulate(p.s)
            kt, kw = [float(p.t[0])], [float(p.t[0])]
            for b, tc in zip(bnd_h, c):
                if not np.isfinite(tc) or not (s_mono[0] < D - b <= s_mono[-1]):
                    continue
                tp = float(np.interp(D - b, s_mono, p.t))
                if tp > kt[-1] and tc > kw[-1]:
                    kt.append(tp)
                    kw.append(float(tc))
            if len(kt) < 2:
                out[h] = p
                continue
            kt, kw = np.array(kt), np.array(kw)
            w = self._monotone_warp(kt, kw, p.t)
            warp[h] = float(np.max(np.abs(kw - kt)))
            out[h] = replace(
                p,
                s=np.interp(p.t, w, p.s),
                lat=np.interp(p.t, w, p.lat),
                sd_s=np.interp(p.t, w, p.sd_s),
                sd_l=np.interp(p.t, w, p.sd_l),
            )
        return out, warp

    @staticmethod
    def _monotone_warp(kt, kw, t):
        """Strictly increasing C1 map t -> w through the knots (kt[i] -> kw[i], both strictly
        increasing): cubic Hermite with Fritsch-Butland (PCHIP) interior slopes, each <= 3x the
        adjacent secants, so every piece is strictly monotone."""
        from scipy.interpolate import CubicHermiteSpline

        h = np.diff(kt)
        m = np.diff(kw) / h
        d = np.empty_like(kt)
        # ends: the adjacent section's mean rate, continued linearly outside the knots -- C1, strictly
        # monotone, and the post-finish stretch (< ~1 s) keeps the last section's clock rate
        d[0], d[-1] = m[0], m[-1]
        if len(m) > 1:
            w1, w2 = 2 * h[1:] + h[:-1], h[1:] + 2 * h[:-1]
            d[1:-1] = (w1 + w2) / (w1 / m[:-1] + w2 / m[1:])
        w = CubicHermiteSpline(kt, kw, d)(np.clip(t, kt[0], kt[-1]))
        w = np.where(t < kt[0], kw[0] + d[0] * (t - kt[0]), w)
        return np.where(t > kt[-1], kw[-1] + d[-1] * (t - kt[-1]), w)

    def _metric_corrected(self, obs, paths, bias=None):
        """FINAL product only: the render's distance metric has a smooth SHARED error b(s) (all
        horses cross a boundary early/late together), and many render observations treated as
        independent would outvote the single official time. b at each official boundary = median
        over horses of (crossing delay x speed), interpolated along s; render s shifted by b."""
        bias = self._metric_bias(paths) if bias is None else bias
        if bias is None:
            return obs
        sb, b = bias
        out = []
        for ob in obs:
            if ob.feed == "render" and ob.s is not None:
                ob = replace(ob, s=ob.s + np.interp(ob.s, sb, b))
            out.append(ob)
        return out

    @staticmethod
    def _skeleton_merge(paths, skeleton, z_max=4.0):
        """Per horse: the detection-based path, except where it conflicts with the horse's own
        official skeleton by > z_max combined sigma -- there the skeleton (vague between boundaries)
        replaces it, so detections that dragged the path onto another horse are not re-taken."""
        out = {}
        for h, p in paths.items():
            k = skeleton.get(h)
            if k is None:
                out[h] = p
                continue
            ks = np.interp(p.t, k.t, k.s)
            ksd = np.interp(p.t, k.t, k.sd_s)
            bad = np.abs(p.s - ks) > z_max * np.hypot(p.sd_s, ksd)
            if not bad.any():
                out[h] = p
                continue
            out[h] = replace(
                p,
                s=np.where(bad, ks, p.s),
                sd_s=np.where(bad, ksd, p.sd_s),
                sd_l=np.where(bad, np.maximum(p.sd_l, np.interp(p.t, k.t, k.sd_l)), p.sd_l),
            )
        return out

    def _metric_bias(self, paths):
        """(s at the official boundaries, shared render metric bias b there, m) or None; a boundary the
        mask withholds sets no bias."""
        from .validate import crossing

        bnd, cum = self.info.official_boundaries()
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
            return None
        sb, b = np.array(bs).T
        o = np.argsort(sb)
        sb, b = sb[o], b[o]
        self.log(f"   fuse: render metric bias at boundaries (m): {np.round(b, 2).tolist()}")
        return sb, b

    EM_ITERS = 8
    SEED_BACK = 3.0  # s before its finish-camera crossing that a path-less horse is seeded
    SEED_V_SD = 1.0  # m/s: speed uncertainty of a seeded horse (grows the seed's sd away from the line)

    def _finish_seeds(self, paths, prior) -> dict:
        """E-step-only SEED paths for horses the finish camera READ but that have no fused path.

        A finish observation alone cannot make a path (no lateral, one instant), so without a seed
        the detection-level E-step never considers such a horse. The seed is the camera's crossing
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

    def finish_sc(self, zf) -> dict:
        """The finish camera's cached scan (finish.npz) for ``finishcam``: line, cloth reads, and the camera's
        geometry (Happy Valley's is overhead: one image band per lane; Sha Tin's is side-on)."""
        return dict(line_x=float(zf["line_x"]), reads=zf["reads"], overhead=self.info.venue == "HV")

    def _fc_read(self, zf) -> set:
        """Horses the finish camera read (negative evidence / shadow-path tie-break): those with a GAP."""
        return {int(k) for k in finishcam.finish_gaps(self.finish_sc(zf))[1]}

    def _em_associate(
        self, paths, horses, prior, extra=None, out_name="det_labels.npz", s_bias=None
    ):
        """Detection-level EM (``assoc``): every detection re-assigned per frame against the
        continuous fused paths (+ one-sided legend rank constraint), paths re-fused; the tracker's
        fragments are only the initialisation. Unassigned detections stay anonymous (grouped by
        their tracker fragment) so association/orphan logic can still place path-less horses."""
        from . import assoc

        T = np.load(self.p("tracks.npz"), allow_pickle=True)
        leg = np.load(self.p("legend.npz"))["rows"]
        fr, t, lat = T["rows"][:, 0], T["t"], T["lat"]
        s_run = self.info.distance - T["d"]
        if s_bias is not None:  # final product: the shared render metric bias (see _metric_corrected)
            s_run = s_run + np.interp(s_run, *s_bias)
        frag = np.full(len(t), -1)
        for k, m in enumerate(T["members"]):
            frag[np.asarray(m, int)] = k
        obs = []
        # finish camera NEGATIVE evidence: with recall r (fraction of the field it read), a horse
        # that IS at the line goes unread with probability ~(1 - r), so detections in the last 15 m
        # cost -2 log(1 - r) for unread horses
        near = None
        if self.p("finish.npz").exists():
            zf = np.load(self.p("finish.npz"))
            read = self._fc_read(zf)
            r = min(len(read) / max(self.info.n, 1), 0.95)
            if read and r >= 0.5:
                cost = float(-2.0 * np.log(1.0 - r))
                near = [(self.info.distance - 15.0, {h: cost for h in horses if h not in read})]
                # BLIND horses (unread by the camera AND absent from the legend top 8 in the 8 s
                # before the first horse reaches the line) have no identity evidence at the line:
                # final-50 m detections cost them CLUTTER, so they extrapolate with honest
                # uncertainty instead of "finishing" on another horse's detections
                crossings = [
                    p.t[np.argmax(self.info.distance - p.s <= 0)]
                    for p in paths.values()
                    if (self.info.distance - p.s <= 0).any()
                ]
                t_fin = min(crossings) if crossings else float(np.max(t))
                late_frames = {int(f) for f, tt in zip(fr, t) if t_fin - 8.0 <= tt <= t_fin + 1.0}
                shown = {int(x) for r_ in leg if int(r_[0]) in late_frames for x in r_[1:9] if x}
                nums = self._number_reads(T)  # circle render: numbers read on discs count as shown
                if nums is not None:
                    shown |= {int(nums[j]) for j, f in enumerate(fr) if int(f) in late_frames and nums[j]}
                blind = [h for h in horses if h not in read and h not in shown]
                if blind:
                    near.append((self.info.distance - 50.0, {h: assoc.CLUTTER for h in blind}))
        def em_pass(paths, targets, tag):
            """EM iterations; ``targets``: path-less finish-read horses to seed in the E-step."""
            prev = lab = obs = None
            forbid = {}
            for it in range(self.EM_ITERS):
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
                )
                lab, n_pf = assoc.release_post_finish(lab, t, s_run, self.info.distance, forbid)
                obs = assoc.runs_to_obs(lab, t, s_run, lat)
                for k in np.unique(frag[(lab == 0) & (frag >= 0)]):
                    j = np.where((frag == k) & (lab == 0))[0]
                    if len(j) >= 3:
                        j = j[np.argsort(t[j])]
                        # the E-step's hard legend exclusions travel with the anonymous fragment,
                        # or fusion's chi^2 / orphan association hands it back
                        no = assoc.fragment_forbid(forbid, j)
                        obs.append(
                            fusion.Obs(t[j], s_run[j], lat[j], 0.8, 0.35, "render", 0, int(k), no)
                        )
                paths, _a = self._fuse(obs + (extra or []), horses, prior)
                changed = np.nan if prev is None else float((lab != prev).mean())
                self.log(
                    f"   em-assoc{tag} it{it}: {np.mean(lab > 0):.0%} of detections assigned, "
                    f"{len(paths)}/{len(horses)} horses, changed {changed:.1%}"
                    + (f", seeded {sorted(seeds)}" if seeds else "")
                    + (f", {n_pf} post-finish detections released" if n_pf else "")
                )
                if prev is not None and changed < 0.002:
                    break
                prev = lab
            return paths, lab, obs, forbid

        paths, lab, obs, forbid = em_pass(paths, None, "")
        # SEED pass, CONFINED to finish-read horses still path-less after the unseeded EM; races
        # without a target are unchanged
        targets = set(self._finish_seeds(paths, prior))
        if targets:
            paths, lab, obs, forbid = em_pass(paths, targets, " (seed)")
        # SHADOW paths: two horses cannot occupy one position; a horse whose own detections are
        # missing can lock onto a neighbour's. Pairs within 0.8 m along / 0.5 m laterally for >= 0.5 s
        # (both tight): the render obs of the weaker-evidence horse (no finish-camera reading, else
        # fewer obs) in that window are dropped and the paths re-fused.
        cam = set()
        if self.p("finish.npz").exists():
            zf = np.load(self.p("finish.npz"))
            cam = self._fc_read(zf)
        drop = np.zeros(len(t), bool)
        hs = sorted(paths)
        # only the finish-critical window (last 3 s before the first horse reaches the line); a
        # mid-race window releases real horses' detections
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
        if drop.any():
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
            paths, _a = self._fuse(obs + (extra or []), horses, prior)
            self.log(f"   em-assoc: shadow paths -> {int(drop.sum())} detections released")
        with self.out(out_name) as fp:
            np.savez(fp, labels=lab)  # per-detection EM identity (viz, audit)
        return obs, paths

    def paths(self):
        import polars as pl

        return pl.read_parquet(self.p("paths.parquet"))

    def validate(self):
        from . import validate as _v

        res = _v.validate(self, self.truth)
        with self.out("validation.json") as fp:
            fp.write_text(json.dumps(res, indent=1, default=float))
        self.log(_v.summary_line(self.info.key, res))
        # label-free consistency audit (recon.audit): written beside the gates, never fails a run
        from . import audit as _a

        try:
            au = _a.audit(self)
            au["rail_blocking"] = _a.rail_blocking(au)
            with self.out("audit.json") as fp:
                fp.write_text(json.dumps(au, indent=1, default=str))
            if au["rail_blocking"]:
                self.log(f"   audit: RAIL BLOCKING {au['rail_blocking']}")
        except Exception as e:  # noqa: BLE001 -- diagnostics only
            self.log(f"   audit: failed ({e})")
