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
  start     gate opening in the start film: re-anchors the race clock when the overlay clock lags
  film      race-film leader crossings of the painted markers -> the metric's constant offset
  observe   detections -> (d, lateral) metres (odometer-tick distance calibration) -> world tracklets
  finish    RF/drone finish-line camera: saddlecloth OCR -> finish order + spatial gaps
  identify  tracklet -> saddlecloth number (legend order + finish camera + elimination + silk)
  fuse_video  Bayesian fusion with physics/geometry priors -> paths_video.parquet (no official
            outcome data)
  fuse_final  + the official records the mask leaves visible -> paths.parquet
  validate  gates vs the full official record

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
# telemetry (identify_scores.npz, identify_em_scores.npz) is written but is NOT a stage output: never a cache key
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
    "identify": ["segs.npz"],
    "finish": ["finish.npz"],
    "fuse_video": ["paths_video.parquet", "fuse_video.pkl"],
    "fuse_final": ["paths.parquet"],
    "validate": ["validation.json"],
}

# stages that read the official outcome (through the mask); they and their consumers depend on the mask
READS_OUTCOME = ("fuse_final",)
# the first stage reading the video evidence an evidence mask withholds (detections, disc-number and legend reads);
# it and its consumers depend on the mask's evidence part
READS_EVIDENCE = ("observe",)
MASK_FILE = "mask.txt"  # the mask behind a run dir's mask-dependent artifacts; absent = none


# stage -> the stages whose artifacts it reads (directly or through the helpers it calls)
DEPS = {
    "frames": (),
    "features": ("frames",),
    "pose": ("features",),
    "worldmap": ("frames", "features", "pose"),  # features: the legend rows (feats.npz lrow)
    "detect": ("frames", "features", "pose", "worldmap"),
    "silk": ("frames", "detect"),
    "hud": ("frames",),
    "rails": ("frames", "pose", "worldmap", "detect"),
    "start": ("hud",),
    "film": ("hud",),
    "observe": ("pose", "detect", "silk", "hud", "rails", "start", "film"),
    "finish": (),
    "identify": ("observe", "hud", "silk", "finish", "pose"),
    "fuse_video": ("identify", "observe", "finish", "hud", "silk", "rails", "pose"),
    "fuse_final": ("fuse_video",),
    "validate": ("fuse_video", "fuse_final"),
}


# m outside the extracted racing rail: the line whose arc length the HUD odometer's metres follow, per venue (frozen
# from the pooled HUD tick loss alone, no official data). Happy Valley only: the Sha Tin line (4.60 m) made the product
# worse at 200 m.
METRIC_W = {"HV": 3.10}


def upstream(stage: str) -> set:
    """Every stage ``stage`` depends on, transitively."""
    out, todo = set(), list(DEPS[stage])
    while todo:
        d = todo.pop()
        if d not in out:
            out.add(d)
            todo.extend(DEPS[d])
    return out


def _cpu_total() -> float:
    """CPU seconds of this process and of its reaped children (user + system)."""
    import resource

    r = [resource.getrusage(w) for w in (resource.RUSAGE_SELF, resource.RUSAGE_CHILDREN)]
    return sum(x.ru_utime + x.ru_stime for x in r)


def _nproc() -> int:
    """Worker processes per race (env HKJC_NPROC; default: all cores / parallel races)."""
    return max(1, int(os.environ.get("HKJC_NPROC", "0")) or (os.cpu_count() or 4))


_CACHE = None
_FEATS_PAR = None
_STEP_NN = {}  # fork-shared bfmatch cross-check tables for the step matches (empty: OpenCV in-call)
_STYLE = "silk"  # render style of the race being processed (fork-shared with the feature workers)


def _feat_job(args):
    k, nfeat, raw = args
    i, p = _CACHE.idx[k], _CACHE.panel(k)
    if _STYLE == "circle":  # numbered discs, no legend strip
        from . import circles

        lt = p.shape[0]
        rs = circles.centres(p)
        hist = [np.zeros(14) for _ in rs]
    else:
        lt = video.legend_top(p)
        # the avatar detector's boxes (+ shadows) are masked too: the colour mask misses brown horses
        rs = _detect.AvatarDetector.drop_legend(raw, lt)  # raw: the detector pool's boxes
        hist = [_detect.silk_hist(p, r) for r in rs]
    m = register.mask_boxes(register.world_mask(p, video.hud_mask(p, lt)), rs)
    f = register.extract(p, m, nfeat)
    return int(i), lt, f, p.shape[:2], rs, hist


_DET = None


def _det_init():
    global _DET
    if _STYLE != "circle":
        _DET = _detect.AvatarDetector()


def _det_job(k):
    return _DET.raw(_CACHE.panel(k))


def _step_job(k):
    m = register.match(_FEATS_PAR[k - 1], _FEATS_PAR[k], nn=_STEP_NN.get((k - 1, k, True)))
    return np.r_[m.S.ravel(), m.n_inl, m.resid] if m else None


class RaceRun:
    """``info``: the full race record (``truth``, read only by validation). Every stage reads the race
    through ``self.info``, the masked view: pre-race facts plus the official outcome records ``mask``
    leaves visible (default: ``Mask.from_env()``, i.e. ``HKJC_MASK`` / ``HKJC_FINAL_HOLDOUT``; no mask
    withholds nothing); the mask's evidence part withholds detections, disc reads and legend reads from ``observe``
    on.
    A masked run writes artifacts like any other: give it its own ``root``: ``run`` refuses a dir whose
    mask-dependent artifacts were made under another mask."""

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

    @staticmethod
    def _downstream(first) -> set:
        """``first`` and every stage depending on one of them (transitively)."""
        dep = set(first)
        for st in STAGES:
            if any(d in dep for d in DEPS[st]):
                dep.add(st)
        return dep

    def outcome_stages(self) -> set:
        """READS_OUTCOME and every stage depending on one of them (transitively)."""
        return self._downstream(READS_OUTCOME)

    def evidence_stages(self) -> set:
        """READS_EVIDENCE and every stage depending on one of them (transitively)."""
        return self._downstream(READS_EVIDENCE)

    def mask_guard(self, todo):
        """One run dir holds one mask's mask-dependent artifacts: never reuse or overwrite them under
        another mask. The outcome part of the mask governs the outcome stages, the evidence part the
        evidence stages. Records a non-empty mask before any mask-dependent stage runs."""
        rec = self.p(MASK_FILE)
        have_s = rec.read_text().strip() if rec.exists() else "none"
        have = Mask.parse(have_s)
        dep = set()
        for stages, a, b in (
            (self.outcome_stages(), have.outcome_part(), self.mask.outcome_part()),
            (self.evidence_stages(), have.evidence_part(), self.mask.evidence_part()),
        ):
            dep |= stages
            if a != b and any(self.p(o).exists() for st in stages for o in OUTPUTS[st]):
                raise ValueError(f"{self.info.key}: {self.dir} holds artifacts made under mask '{have_s}', this "
                                 f"run's mask is '{self.mask}': give the run its own root")
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
        self._bg, self._bg_cpu = bg, {}  # telemetry: _timed keeps background CPU out of foreground stages

        def launch(st):
            launched.add(st)
            total = mp.get_context("fork").Value("d", -1.0)

            def work():
                import signal
                import sys

                signal.signal(signal.SIGTERM, lambda *_: sys.exit(1))  # pools shut down cleanly
                self._bg = {}  # siblings are not this process's children
                self._timed(st, background=True)
                total.value = _cpu_total()  # this process + its reaped children, as the parent will receive it

            pr = mp.get_context("fork").Process(target=work)
            pr.start()
            bg[st] = pr
            self._bg_cpu[st] = total

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
        """Run stage ``st`` and append its telemetry to timings.jsonl. ``cpu_s`` = CPU of this process and of the
        children reaped during the stage, minus the background stages reaped meanwhile (multiprocessing reaps
        finished children whenever it starts a process; their CPU is in their own record)."""
        import resource

        bgs = dict(getattr(self, "_bg", {}))

        def reap():  # join finished background stages now: their CPU enters RUSAGE_CHILDREN at a known point
            for pr in bgs.values():
                pr.join(0)
            return {b for b, pr in bgs.items() if pr.exitcode is None}

        running = reap()
        t0, c0 = time.time(), _cpu_total()
        getattr(self, st)()
        dt = time.time() - t0
        reaped = running - reap()
        bg_cpu = sum(max(0.0, self._bg_cpu[b].value) for b in reaped if b in getattr(self, "_bg_cpu", {}))
        self.log(f"{self.info.key} {st}: {dt:.0f}s" + (" (background)" if background else ""))
        own, kids = (
            resource.getrusage(w).ru_maxrss // 1024
            for w in (resource.RUSAGE_SELF, resource.RUSAGE_CHILDREN)
        )
        rec = dict(
            stage=st,
            s=round(dt, 1),
            cpu_s=round(_cpu_total() - c0 - bg_cpu, 1),  # this process + its own finished workers
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

    def legend_row(self, panel, frame: int | None = None) -> int:
        """Top row of the silk-legend strip (panel-relative); the circle render has none. With the panel's
        video ``frame``, the row the features stage computed for it (feats.npz ``lrow``), when present."""
        if frame is not None:
            lr = self._lrows()
            if lr is not None and frame in lr:
                return lr[frame]
        return panel.shape[0] if self.render_style() == "circle" else video.legend_top(panel)

    def _lrows(self):
        """{video frame: legend row} from feats.npz (the same ``legend_row`` of the same cached panel), or None."""
        if getattr(self, "_lrow_cache", None) is None and self.p("feats.npz").exists():
            z = np.load(self.p("feats.npz"))
            self._lrow_cache = dict(zip(z["idx"].tolist(), z["lrow"].tolist()))
        return getattr(self, "_lrow_cache", None)

    def cache(self):
        return video.FrameCache(self.dir)

    def features(self, nfeat: int = 3000):
        """ORB features of every unique render frame + the sequential step matches. Per-frame work
        is independent and OpenCV's RANSAC seeds its RNG per call, so the process pool gives results
        identical to a sequential loop. The avatar detections made for the mask are saved for
        ``detect`` (yolo.npz). The avatar detector runs in its own pool (one process on an accelerator) and feeds up
        to 4 feature workers frame by frame."""
        import multiprocessing as mp

        cache = self.cache()
        from . import bfmatch

        bfmatch.warm()  # the CUDA matcher's interpreter starts now (closed after pose)
        global _CACHE, _FEATS_PAR, _STYLE, _STEP_NN
        _CACHE = cache
        _STYLE = self.render_style()
        accel = os.environ.get("HKJC_DEVICE", "cpu").startswith(("intel:", "cuda"))
        K = len(cache)
        nw = min(_nproc(), 4)  # detector pool feeding the feature workers; one accelerator context
        with mp.get_context("fork").Pool(nw) as pool:
            if _STYLE == "circle":  # no avatar detector
                res = pool.map(_feat_job, [(k, nfeat, None) for k in range(K)], chunksize=8)
            else:
                with mp.get_context("fork").Pool(1 if accel else nw, initializer=_det_init) as dpool:
                    raws = dpool.imap(_det_job, range(K), chunksize=8)
                    res = list(pool.imap(_feat_job, ((k, nfeat, r) for k, r in enumerate(raws)), chunksize=8))
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
        self._lrow_cache = None  # legend rows are read back from the new feats.npz
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
        from . import bfmatch

        z = dict(np.load(self.p("feats.npz")))
        try:
            G, info = pose.solve_race(z)
        finally:
            bfmatch.close()  # pose is the matcher's last consumer: free its GPU memory
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
                lt = self.legend_row(p, i)
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
            for i, fr in video.decode_from(c, min(want) if want else 0):
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
        circle = self.render_style() == "circle"  # no legend strip: its slots are not read
        rows, leg = ocr.read_render_hud_cached(
            self.cache(), n_field=max(14, self.info.n), n_slots=0 if circle else min(8, self.info.n)
        )
        if np.isfinite(rows[:, 1]).sum() < self.HEALTH_CLOCK_READS:
            # the fixed crop missed the clock (a sponsor box above it moves it down): locate it in the video,
            # on the same frames (every CLOCK_EVERY-th render frame)
            sel = np.arange(0, len(rows), ocr.CLOCK_EVERY)
            loc, dy = ocr.read_clock_located(self.video(), rows[sel, 0])
            if np.isfinite(loc).sum() > np.isfinite(rows[sel, 1]).sum():
                rows[sel, 1] = loc
            self.log(f"   hud: clock not at the fixed crop; located {dy:+d} px: {np.isfinite(loc).sum()} reads")
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
                mo.add(p, video.hud_mask(p, self.legend_row(p, i)), G[k_of[i]])
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
        cache = self.cache()
        pos = [n for n, i in enumerate(cache.idx.tolist()) if k_of.get(i) in keep]  # decode only those
        for i, p in cache.items(pos):
            mo.add(p, video.hud_mask(p, self.legend_row(p, i)), G[k_of[i]])
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
        rails.release_planes()  # the map's converted planes: not held while the final-pass mosaic is built
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
        rails.release_planes()
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
        if not ok.any():
            raise RuntimeError(f"{self.info.key}: no race-clock reads (hud): the race clock has no anchor")
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

    LAT_GATE = 1.0  # m: detections further inside the racing rail are dropped

    def fence_lat0(self, ppm: float) -> float:
        """World px from the rail polyline to the racing rail: on straight courses the fence's measured offset
        inside the field-edge datum (``rail.npz`` lat0_px), else 0. A "fence" more than LAT_GATE on the FIELD
        side of the field's inner edge is another ridge (observe's gate would drop the inner horses): unmeasured,
        0 as when no ridge is supported."""
        z = np.load(self.p("rail.npz"))
        lat0 = float(z["lat0_px"]) if "lat0_px" in z.files else 0.0
        return 0.0 if lat0 < -self.LAT_GATE * ppm else lat0

    def metric_w(self) -> float:
        """w (m) of the product's metric line (``METRIC_W`` by venue); 0 for other venues, off turf and on straight
        courses (the rails stage wrote lat0_px)."""
        if self.info.surface != "turf" or not self.p("rail.npz").exists():
            return 0.0
        if "lat0_px" in np.load(self.p("rail.npz")).files:
            return 0.0
        return float(METRIC_W.get(self.info.venue, 0.0))

    def tick_time(self):
        """-> f(frames): race-clock time (s) of render frames, each stamped with its tick on the uniform 10 Hz grid
        (video.tick_frame)."""
        grid = video.tick_grid(np.load(self.p("frames_idx.npz"))["idx"])
        f0 = self.clock_f0()

        def tick_s(f):
            return (video.tick_frame(f, grid) - f0) / FPS

        return tick_s

    def tick_metric(self, rows, s_px, rf, m, w: float = 0.0) -> dict:
        """observe's distance metric (RECON 4, observe): the render leader per frame (``s_px``, rail s of the gated
        detections ``rows``), odometer tick events, the tick-seeded marker grid (``m``: metric.npz), the tick fit
        (``rails.calibrate_distance_ticks``), the film anchor and the time-local shift. With ``w`` > 0 (m,
        ``metric_w``) the ticks are fitted against arc length along the line w m outside the rail
        (``rails.offset_arc``, the scale from the rail-s fit's tick line); lateral keeps the rail-s fit's px/m.
        -> dict(d per detection, ppm (marker or seeded), ppm_loc (lateral px/m of rail s), resid (tick residuals of
        the fit behind d), c, fit (ppm_odo, sF_odo, pinned, c_m, n_shift), delta (film anchor), w, fu, d_lead)"""
        ppm, sF = float(m["ppm"]), float(m["sF"])
        fu = np.unique(rows[:, 0])
        tick_s = self.tick_time()
        # leader position per frame: a front detection > 8 m clear of the next is a false positive
        # only if that gap is not persistent (a genuinely clear leader keeps its margin)
        from scipy.ndimage import median_filter

        s0 = np.empty(len(fu))
        s1 = np.full(len(fu), np.nan)
        # a box wholly inside the silk render's list band is a list icon, not a horse: never the leader
        s_lead_px = s_px
        if self.render_style() == "silk" and len(rows):
            from .core import legend_top

            box = np.load(self.p("frames_idx.npz"))["box"]
            icon = rows[:, 2] - rows[:, 4] / 2 >= legend_top(rows[:, 2], float(box[3] - box[2])) + 4
            if icon.any() and not icon.all():
                s_lead_px = np.where(icon, -np.inf, s_px)
        for n, f in enumerate(fu):
            ss = np.sort(s_lead_px[rows[:, 0] == f])[::-1]
            ss = ss[np.isfinite(ss)] if np.isfinite(ss).any() else np.sort(s_px[rows[:, 0] == f])[::-1]
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

        def fit_in(x_lead, x_F):
            """the tick fit in one along-track coordinate -> (calibrate_distance_ticks output, film delta, per)"""
            fit = rails.calibrate_distance_ticks(
                fu, x_lead, odo, ppm, x_F, pin_finish=False if use_film else None, have_markers=have_markers
            )
            if use_film:
                dlt, per = film.anchor_offset(tick_s(fu), fit[0](x_lead), film_t)
                if dlt is None:  # crossings not matched by the render leader: back to the paint pin
                    return rails.calibrate_distance_ticks(fu, x_lead, odo, ppm, x_F, have_markers=have_markers), 0.0, None
                return fit, dlt, per
            return fit, 0.0, None

        fit, delta, per = fit_in(s_lead, sF)
        d_of_s, c, ppm_loc, resid, info = fit
        if w:
            # the HUD's metres run along a line w m outside the extracted rail (its px/m steps where the rail turns):
            # d from the fit in that line's arc length, the marker grid and lateral stay in rail s
            sp = rails.offset_arc(rf, w * info[0])
            fit, delta, per = fit_in(sp(s_lead), float(sp(sF)))
            d_w, c, _ppm_loc_w, resid, info = fit

            def d_of_s(x):
                return d_w(sp(x))

            self.log(f"   observe: metric along the line {w:.2f} m outside the rail")
        if per is not None:
            self.log(f"   observe: film anchor delta {delta:+.2f} m from {len(per)} markers (spread {np.ptp(per):.1f} m)")
        d = d_of_s(s_px) - delta
        # time-local pose slips (render camera swing on the straights): per-frame along-track shift
        d_lead = d_of_s(s_lead) - delta
        shift_u, n_win = rails.time_local_shift(tick_s(fu), d_lead, odo)
        if n_win:
            d = d - shift_u[np.searchsorted(fu, rows[:, 0])]
            self.log(f"   observe: time-local shift in {n_win} window(s), max {np.abs(shift_u).max():.1f} m")
        return dict(d=d, ppm=ppm, ppm_loc=ppm_loc, resid=resid, c=c, fit=info, delta=delta, w=w, fu=fu, s_lead=s_lead,
                    odo=odo, d_lead=d_lead)

    def _observed_dets(self):
        """observe's detections: world_dets less the perturbation and the mask's withheld detections
        (``self._wd_idx`` follows) -> rows, X, hist"""
        rows, X, hist, _P = self.world_dets()
        seed = os.environ.get("HKJC_PERTURB_SEED")  # DIAGNOSTIC: robustness perturbation
        if seed:  # drop 1 % of detections (seeded) -> spread of the downstream gate metrics
            keep = np.random.default_rng(int(seed)).random(len(rows)) >= 0.01
            rows, X, hist = rows[keep], X[keep], hist[keep]
            self._wd_idx = self._wd_idx[keep]
        if self.mask.dets:  # evidence mask: withheld detections, as if the detector had missed them
            keep = ~np.isin(self._wd_idx, np.fromiter(self.mask.dets, int))
            rows, X, hist = rows[keep], X[keep], hist[keep]
            self._wd_idx = self._wd_idx[keep]
            self.log(f"   observe: {int((~keep).sum())} detections withheld by the mask")
        return rows, X, hist

    def _rail_projection(self, rows, X):
        """-> (rail.npz, metric.npz, RailFrame, s_px, lat_px) of observe's detections"""
        z = np.load(self.p("rail.npz"))
        m = np.load(self.p("metric.npz"))
        rf = rails.RailFrame.from_points(z["rail"], int(z["inner"]))
        k_of = {int(f): k for k, f in enumerate(z["frame_idx"])}
        hint = rf.s[z["st_of_frame"][[k_of[int(f)] for f in rows[:, 0]]]]
        s_px, lat_px = rf.project(X, s_hint=hint, window=1200.0)
        return z, m, rf, s_px, lat_px

    def observe(self):
        rows, X, hist = self._observed_dets()
        z, m, rf, s_px, lat_px = self._rail_projection(rows, X)
        mt = self.tick_metric(rows, s_px, rf, m)
        tick_s = self.tick_time()
        d, ppm, ppm_loc, resid, c = mt["d"], mt["ppm"], mt["ppm_loc"], mt["resid"], mt["c"]
        delta = mt["delta"]
        ppm_odo, sF_odo, pinned, c_m, nsh = mt["fit"]
        # straight courses (the rails stage's decision, recorded as lat0_px): the rail polyline is the
        # field-edge datum and lat0_px the fence's measured offset inside it (rails.straight_datum)
        lat0 = self.fence_lat0(ppm)
        if "lat0_px" in z.files and lat0 != float(z["lat0_px"]):
            self.log(f"   observe: straight-course fence measured {-float(z['lat0_px']):.0f} px on the field side "
                     "of its edge -> rejected")
        lat = (lat_px + lat0) / ppm_loc(s_px)
        g = lat > -self.LAT_GATE  # nothing races inside the racing rail
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
        nums = self._number_reads(dict(det_idx=det_idx, t=t))  # circle render: identity switches inside tracklets
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
        """The core's label-free evidence (``core.build_evidence``): segments, links, P(horse), camera transitions.
        Written for inspection; every fuse rebuilds it (deterministic)."""
        ev = self._evidence()
        with self.out("segs.npz") as fp:
            np.savez(fp, seg=ev.seg, links=ev.links, p_horse=ev.p_horse, cam=ev.cam)
        self.log(f"   identify: {ev.n_seg} segments, {len(ev.links)} links, "
                 f"{int((ev.p_horse < 0.5).sum())} overlay boxes (P(horse) < 0.5), {int(ev.cam.sum())} detections at "
                 "camera transitions")

    def _evidence(self, T=None, s_shift=None):
        """core.Evidence of this run (tracks.npz s = distance - d; the final product's render metric bias ``s_shift``
        moves every render s)."""
        from . import core

        T = np.load(self.p("tracks.npz"), allow_pickle=True) if T is None else T
        horses = sorted(r.no for r in self.info.runners)
        silk = None
        if self.p("silk.npz").exists():
            zs = np.load(self.p("silk.npz"))
            silk = (zs["feat"][T["det_idx"]], zs["proto_no"], zs["proto"])
        box = np.load(self.p("frames_idx.npz"))["box"]
        Td = dict(rows=T["rows"], t=T["t"], s=self.info.distance - np.asarray(T["d"], float), lat=T["lat"])
        g = self.info.geometry
        bend = np.array([g.is_bend(float(x)) for x in np.round(np.asarray(T["d"], float), 0)], bool)
        return core.build_evidence(horses, Td, [np.asarray(m, int) for m in T["members"]],
                                   np.load(self.p("pose.npz")), float(box[3] - box[2]),
                                   self.render_style() == "silk", nums=self._number_reads(T), silk=silk,
                                   panel_w=float(box[1] - box[0]), bend=bend, s_shift=s_shift)

    NUMBER_CONF = 0.6  # minimum OCR confidence of a disc number read

    def _number_reads(self, T):
        """Per tracks row: the disc number read (circle render; 0 = unread, low confidence or withheld by
        the mask), else None."""
        if not self.p("numbers.npz").exists():
            return None
        z = np.load(self.p("numbers.npz"))
        di = np.asarray(T["det_idx"], int)
        num, conf = z["num"][di], z["conf"][di]
        valid = {r.no for r in self.info.runners}
        ok = (conf >= self.NUMBER_CONF) & np.isin(num, list(valid))
        if self.mask.discs is not None:
            ok &= ~self.mask.discs_withheld(T["t"])
        return np.where(ok, num, 0)

    def legend_rows(self):
        """legend.npz rows (frame, 8 slot numbers; silk render); rows whose render tick lies in a block the
        mask's legend part withholds read as unread (slots 0, as on the circle render)."""
        leg = np.load(self.p("legend.npz"))["rows"]
        if self.mask.legend is None or not len(leg):
            return leg
        grid = video.tick_grid(np.load(self.p("frames_idx.npz"))["idx"])
        t = (video.tick_frame(leg[:, 0], grid) - self.clock_f0()) / FPS
        leg = leg.copy()
        leg[self.mask.legend_withheld(t), 1:] = 0
        return leg

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
        """Finish-camera crossings entered in a fuse: every camera crossing (``finish_camera``) except, in fuse_final,
        those of horses whose official finish is visible -- the official finish (photo-finish, OFFICIAL_SD) is the
        better measurement of the same crossing and also fills the camera's anchor role in fusion, while a camera
        crossing of another cloth, the rider's silk or scenery would pull the path and the final identity pass off it."""
        fin = self.finish_camera(paths)
        vis = {h for h, t in (self._official_clock or {}).items() if np.isfinite(t)}
        return [o for o in fin if o.horse not in vis and o.horse not in self._nonfin]

    def finish_camera(self, paths) -> list:
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
        if fit is not None and self._render_bias is not None:
            fit = dict(fit, t_w=fit["t_w"] + self.registration_shift(self._render_bias, self.info.distance, v_line))
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
        if self._official_clock:  # fuse_final: the visible official finish times own the finish clock
            dt = self.official_clock_offset({h: t_w + gaps[h] / C for h in sds}, self._official_clock)
            if dt is not None:
                t_w += dt
                self.log(f"   finish: camera clock {dt:+.3f} s onto the official finish")
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

    # gate geometry from the drone's top-down start frame (HV 06-03 R8): 40.2 px stall pitch at
    # 38.0 px/m (racing -> outer rail 855 px = 22.5 m on course C); stall 1 abuts the racing rail
    STALL_PITCH = 1.06  # m between stall centres
    STALL1_LAT = 2.5  # m: stall-1 centre off the racing rail
    DRAW_SD = 0.4  # m

    def official_observations(self) -> list:
        """FINAL-product observations (official outcome data): the gate draw fixes every runner's
        lateral at t=0 (standing start, s=0); the official finish time is a near-hard crossing at d = 0
        (OFFICIAL_SD) and the official interior sectional times are measured crossings at d = 400 k
        (OFFICIAL_INTERIOR_SD); what the mask withholds is absent."""
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
                                      horse=h, tid=40_000 + h))
        for h, c in cum.items():
            c, b = np.asarray(c, float), np.asarray(bnd, float)
            for part, sd, tid in ((b > 0, self.OFFICIAL_INTERIOR_SD, 30_000), (b == 0, self.OFFICIAL_SD, 40_000)):
                ok = part & np.isfinite(c)
                if ok.any():
                    out.append(fusion.Obs(c[ok], D - b[ok], None, sd, 1.0, "official", horse=int(h), tid=tid + int(h)))
        return out

    # official finish times are photo-finish, published to 0.01 s (~0.1 m at race speed): near-hard in the final
    # product, so a mislabelled render tracklet conflicts with them (and is demoted) instead of bending them
    OFFICIAL_SD = 0.1
    # official INTERIOR times are the leader's time plus a margin quantised to 0.04 s: against identity-verified horses
    # their horse-specific error has a robust sd of 0.58-0.59 m (disc and legend-confirmed silk races) once a per
    # race-boundary field-spread term is removed (raw, disc races: ~0.93 m); that term is not modelled
    OFFICIAL_INTERIOR_SD = 0.6

    def _write_paths(self, paths, name, contract=None):
        """``contract`` (fuse_final only): {"anchors": s of the visible boundaries the metric bias was set at}; the
        contract columns are appended after the path columns, and contract.json is written."""
        import polars as pl

        t_r0 = self.render_t0()
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
                    **self.lat_metres(p, t_r0),
                }
            )
            for h, p in paths.items()
        ]
        P = pl.concat(recs)
        if contract is not None:
            P = self._contract_columns(P, paths, contract["anchors"])
        with self.out(name) as fp:
            P.write_parquet(fp)

    def _contract_columns(self, P, paths, anchors):
        """The contract columns (``contract``) appended to the final paths ``P``; writes contract.json."""
        import polars as pl

        from . import contract as ct

        stretches, _rows = ct.flag_stretches(self, P)
        free = ct.node_free(P)
        cols = {k: [] for k in ("valid", "free")}
        for h, p in paths.items():
            cols["valid"].append(ct.node_valid(stretches, h, p.t))
            tt, fr = free[h]
            cols["free"].append(fr[np.searchsorted(tt, np.round(p.t, 2))])
        P = P.with_columns(**{k: pl.Series(np.concatenate(v)) for k, v in cols.items()})
        crossings, pos = ct.position_flag(self, P)
        P = P.with_columns(pos_ok=pl.Series(pos))
        T = np.load(self.p("tracks.npz"), allow_pickle=True)
        Lf = np.load(self.p("det_labels_final.npz"))["labels"]
        style = self.render_style()
        idv = ct.ident_var(P)
        unc = {k: [] for k in ("sd_s_axis", "sd_s_horse", "sd_s_gap", "sd_s_ident", "sd_s_rel", "sd_s_total")}
        for h, p in paths.items():
            ti, vi = idv[h]
            v = ct.uncertainty_columns(p.s, p.t, p.sd_s, anchors, style, np.sort(T["t"][Lf == h].astype(float)),
                                       vi[np.searchsorted(ti, np.round(p.t, 2))])
            for k in unc:
                unc[k].append(v[k])
        P = P.with_columns(**{k: pl.Series(np.concatenate(v)) for k, v in unc.items()})
        idr, *idr_rest = ct.id_robust(self, P)
        P = P.with_columns(id_robust=pl.Series(idr))
        seg = self.lat_m_segment(P["d"].to_numpy(), P["t"].to_numpy(), self.render_t0())
        d_ = P["d"].to_numpy()
        err = ct.lat_m_err(self.info.venue, seg, P["sd_lat_m"].to_numpy(), P["lat_m"].to_numpy() - self.LAT_MAP[0],
                           d_ - self.info.geometry.home_straight(self.info.course))  # provisional (contract.LAT_M_ERR)
        P = P.with_columns(**{k: pl.Series(v) for k, v in err.items()})
        lane, lane_used = ct.lane_columns(P["t"].to_numpy(), d_, P["lat"].to_numpy(), P["sd_lat"].to_numpy(), seg,
                                          self.info.venue, self.info.date)
        P = P.with_columns(lane_units=pl.Series(lane["lane_units"]),
                           lane=pl.Series(lane["lane"]).fill_nan(None).cast(pl.Int8))
        box = ct.box_columns(P["horse_no"].to_numpy(), P["t"].to_numpy(), P["s"].to_numpy(), d_, P["lat"].to_numpy(),
                             lane["lane_units"], seg, lane_used)
        P = P.with_columns(enclosed=pl.Series(box["enclosed"]), boxed_in=pl.Series(box["boxed_in"]),
                           enclosed_late=pl.Series(box["enclosed_late"]))
        lane_desc = ct.describe_lane(lane_used, ct.lane_period(self.info.date))
        with self.out("contract.json") as fp:
            fp.write_text(json.dumps(ct.describe(stretches, ct.frozen_windows(self), crossings, anchors, idr_rest,
                                                 lane=lane_desc, box=ct.describe_box()), indent=1, default=float))
        return P

    # Lateral in metres. The render spreads the field laterally (RECON 8), more in the compact field
    # before the home turn than in the fanned-out home straight. One map, continuous in the distance to go d, for both
    # venues and both drone-era render styles: true = a + b(x) x render lateral, x = d - home straight length,
    # b = LAT_MAP's b at x <= LAT_RAMP[0], LAT_B_FAR at x >= LAT_RAMP[1], linear between. In the home straight it is the
    # Sha Tin home-straight map.
    LAT_MAP = (-0.019562873395157165, 0.645046455287541)
    LAT_B_FAR = 0.5390915106662988
    LAT_RAMP = (0.0, 400.0)
    # segments (venue, render style) where the map passed its drone test; lat_m is NaN in every other segment
    LAT_M_SEGMENTS = {
        ("ST", "disc"): ("straight", "bend", "home"),
        ("ST", "silk"): ("start", "straight", "bend", "home"),
        ("HV", "disc"): ("start", "straight"),
        ("HV", "silk"): ("straight", "bend"),
    }
    LAT_M_START = 6.0  # s after the first render frame: the start segment
    # race dates the map applies to: drone feeds from 2026-01-07 to the end of the 2026/27 season (the drone test's
    # population ends 2026-11-23; later races are assumed to share the disc render). Disc from 2026-09-01, silk before.
    LAT_M_DATES = ("20260107", "20270731")
    LAT_M_DISC_FROM = "20260901"

    def lat_b(self, d) -> np.ndarray:
        """The map's slope b at distance-to-go ``d`` (continuous, piecewise linear in d)."""
        x = np.asarray(d, float) - self.info.geometry.home_straight(self.info.course)
        x0, x1 = self.LAT_RAMP
        w = np.clip((x1 - x) / (x1 - x0), 0.0, 1.0)
        bh, bf = self.LAT_MAP[1], self.LAT_B_FAR
        return np.where(w >= 1.0, bh, np.where(w <= 0.0, bf, bf + (bh - bf) * w))  # the end values exactly

    def lat_m_segment(self, d, t, t_r0) -> np.ndarray:
        """Segment of each node: "start" (t_r0 <= t < t_r0 + LAT_M_START), "bend", "home" (d < home straight),
        "straight"; "" before the first render frame (t < t_r0) or past the post (d < 0)."""
        d, t = np.asarray(d, float), np.asarray(t, float)
        g = self.info.geometry
        bend = np.array([g.is_bend(v) if np.isfinite(v) else False for v in d.ravel()], bool).reshape(d.shape)
        seg = np.where(bend, "bend", np.where(d < g.home_straight(self.info.course), "home", "straight"))
        seg = np.where(t < t_r0 + self.LAT_M_START, "start", seg)
        return np.where((t >= t_r0) & (d >= 0.0), seg, "")

    def lat_m_scope(self, d, t, t_r0) -> np.ndarray:
        """Nodes (distance-to-go ``d``, race time ``t``; ``t_r0`` the first render frame) where lat_m is defined: turf
        oval (not the 1000 m straight) within LAT_M_DATES, in a segment listed in LAT_M_SEGMENTS for the venue and
        render style."""
        d = np.asarray(d, float)
        info = self.info
        lo, hi = self.LAT_M_DATES
        if not (info.surface == "turf" and info.geometry.handedness != "straight" and lo <= info.date <= hi):
            return np.zeros(d.shape, bool)
        segs = self.LAT_M_SEGMENTS.get((info.venue, "disc" if info.date >= self.LAT_M_DISC_FROM else "silk"), ())
        return np.isin(self.lat_m_segment(d, t, t_r0), list(segs))

    def render_t0(self) -> float:
        """Race time of the first render frame (tracks.npz)."""
        return float(np.load(self.p("tracks.npz"))["t"].min())

    def lat_metres(self, p, t_r0) -> dict:
        """{"lat_m", "sd_lat_m"} of path ``p``: a + b(d) x lat and b(d) x sd_lat inside ``lat_m_scope``, NaN outside.
        sd_lat_m is the mapped posterior sd only; the map's own error is in the contract columns sd_lat_m_* (fuse_final)."""
        d = self.info.distance - p.s
        b = self.lat_b(d)
        m = self.lat_m_scope(d, p.t, t_r0)
        return {"lat_m": np.where(m, self.LAT_MAP[0] + b * p.lat, np.nan), "sd_lat_m": np.where(m, b * p.sd_l, np.nan)}

    def fuse_video(self):
        """VIDEO-ONLY posterior (no official outcome data; what the gates validate): the core (``core.solve``) on the
        render evidence and the list, then again with the finish camera's crossings. Persists the labels, the finish
        observations and the paths (fuse_video.pkl) for ``fuse_final``."""
        import pickle

        res, fin, paths = self._core_product(product="video")
        prior = fusion.Prior(width=self.info.width)
        self._write_paths({h: fusion.with_tail(p, prior) for h, p in paths.items()}, "paths_video.parquet")
        with self.out("fuse_video.pkl") as tmp, open(tmp, "wb") as fp:
            pickle.dump(dict(seg_lab=res.seg_lab, fin=fin, paths=paths, n_seg=len(res.seg_lab)), fp,
                        protocol=pickle.HIGHEST_PROTOCOL)
        self.log(f"   fuse_video: {len(paths)}/{self.info.n} horses")

    def _core_product(self, product, start=(), extra=(), s_shift=None, t_end=None, exclude=None):
        """One product of the core: evidence, list, grid, model; a solve without the finish camera by assignment steps
        only, the camera's crossings from that state, then the full solve with them (the first solve's labels a
        candidate start).
        -> (core.Result, finish fusion.Obs list, {horse: fusion.Path})."""
        from . import core

        T = np.load(self.p("tracks.npz"), allow_pickle=True)
        ev = self._evidence(T, s_shift)
        horses = ev.horses
        col = {int(h): c for c, h in enumerate(horses)}
        if exclude:  # horses that take no detection at s >= s_from (stopped non-finishers)
            for h, s0 in exclude.items():
                if int(h) in col:
                    hit = np.bincount(ev.seg, (ev.s >= s0).astype(float), ev.n_seg) > 0
                    ev.forbid[hit, col[int(h)]] = True
        fr, t = T["rows"][:, 0], T["t"]
        o = np.argsort(fr, kind="stable")
        tick = lambda f: np.interp(np.asarray(f, float), fr[o].astype(float), t[o])  # noqa: E731
        Lr = core.list_rows(self.legend_rows(), tick, horses)
        ts = [float(t.min()), float(t.max())]
        if len(Lr.tau):
            ts += [float(Lr.tau.min())]
        ex_s, ex_l = self._core_obs(extra, col)
        if len(ex_s.t):
            ts += [float(ex_s.t.min()), float(ex_s.t.max())]
        t0, t1 = min(ts), max(ts) if t_end is None else max(max(ts), t_end)
        tg = np.arange(t0, t1 + fusion.DT / 2, fusion.DT)
        model = core.Model(ev, Lr, tg, self.info.width)
        start = tuple(x for x in start if len(x) == ev.n_seg)  # a start from other segments cannot be compared
        res = core.solve(model, start=start, extra_s=ex_s, extra_l=ex_l, log=self.log, global_moves=False)
        paths = self._core_paths(res, ev, model)
        quiet, self.log = self.log, (lambda *a, **k: None)
        try:
            fin = self.finish_observations(paths)
        finally:
            self.log = quiet
        f_s = None
        if fin:
            from .validate import crossing

            f_s, _f_l = self._core_obs(fin, col)
            dts = {o.horse: round(float(o.t[0]) - crossing(paths[o.horse].t, self.info.distance - paths[o.horse].s,
                                                           None, 0.0), 2) for o in fin if o.horse in paths}
            self.log(f"   core ({product}): + finish camera, {len(fin)} crossings; camera - path crossing (s): {dts}")
        res = core.solve(model, start=(res.seg_lab, *start), extra_s=core.Obs.cat([ex_s, f_s]), extra_l=ex_l,
                         log=self.log)
        paths = self._core_paths(res, ev, model)
        lab = np.where(res.seg_lab[ev.seg] >= 0, horses[np.maximum(res.seg_lab[ev.seg], 0)], 0)
        p = res.marg[ev.seg, np.where(res.seg_lab[ev.seg] >= 0, res.seg_lab[ev.seg], len(horses))]
        name = "det_labels.npz" if product == "video" else "det_labels_final.npz"
        with self.out(name) as fp:
            np.savez(fp, labels=lab, prob=p, marg=res.marg[ev.seg], horses=horses)
        if product == "final":  # contract: -2 log odds of the kept label against the rest (not a stage output)
            with self.out("det_margins_final.npz") as fp:
                pc = np.clip(p, 1e-12, 1 - 1e-12)
                np.savez(fp, margin=np.where(lab > 0, 2.0 * np.log(pc / (1 - pc)), np.nan), labels=lab)
        with self.out(f"core_{product}.npz") as fp:
            np.savez(fp, seg_lab=res.seg_lab, marg=res.marg, U=res.U, J=np.array([h[1] for h in res.log]))
        return res, fin, paths

    @staticmethod
    def _core_obs(obs, col):
        """fusion.Obs (draw, official, finish camera) -> core.Obs (s, lat) over runner columns."""
        from . import core

        ps, pl_ = [], []
        for o in obs:
            if int(o.horse) not in col:
                continue
            c = col[int(o.horse)]
            if o.s is not None:
                m = np.isfinite(o.s)
                ps.append(core.Obs(np.full(m.sum(), c), o.t[m], o.s[m], np.full(m.sum(), float(o.sd_s))))
            if o.lat is not None:
                m = np.isfinite(o.lat)
                pl_.append(core.Obs(np.full(m.sum(), c), o.t[m], o.lat[m], np.full(m.sum(), float(o.sd_l))))
        return core.Obs.cat(ps), core.Obs.cat(pl_)

    def _core_paths(self, res, ev, model) -> dict:
        """{horse: fusion.Path} of every runner with an observation, from the core's state (posterior sd from the
        per-horse marginal bands)."""
        st = res.state
        sds, sdl = st.sdS, st.sdL
        lab = res.seg_lab[ev.seg]
        out = {}
        for c, h in enumerate(ev.horses):
            if not st.has_obs[c]:
                continue
            k = np.round((ev.t[lab == c] - model.tg[0]) / fusion.DT).astype(int)
            cnt = np.bincount(k[(k >= 0) & (k < len(model.tg))], minlength=len(model.tg))
            out[int(h)] = fusion.Path(int(h), model.tg, st.S[c], st.L[c], sds[c], sdl[c], cnt,
                                      {"render": int((lab == c).sum())})
        return out

    # {horse: official finish time} while fuse_final runs: the finish camera's clock is re-registered onto it
    _official_clock = None
    # (s at the official boundaries, shared render metric bias b there) while fuse_final runs with a bias
    _render_bias = None

    @staticmethod
    def registration_shift(bias, distance, v_line) -> float:
        """Shift (s) of the label-free finish registration's clock into fuse_final's bias-shifted render frame. The
        registration matches the tracklets' own line crossings (render s, unshifted), while every fuse here sees render
        s + b(s); with b(line) the bias held at the line (np.interp, flat past the last visible boundary) the shifted
        render reaches the line -b(line)/v_line later. Without it the anchor-vs-registration check compares two frames
        and, for |b(line)| above ~6 m, rejects the anchor for a clock 0.3-0.6 s off the render it is fused with."""
        return -float(np.interp(distance, *bias)) / v_line
    CLOCK_MIN_HORSES = 3

    @classmethod
    def official_clock_offset(cls, t_cam: dict, official: dict):
        """Shift (s) putting the finish camera's crossing times ``t_cam`` {horse: s} on the official clock:
        the median of official - camera over horses with both, or None with fewer than CLOCK_MIN_HORSES."""
        dt = [official[h] - t for h, t in t_cam.items() if np.isfinite(official.get(h, np.nan))]
        return float(np.median(dt)) if len(dt) >= cls.CLOCK_MIN_HORSES else None

    def fuse_final(self):
        """FINAL product: the video-only observations + the official records (draw + sectional
        times), with the shared metric bias, a field-relative start prior and a final identity pass.
        Its grid runs past the last official finish (else the slowest finisher ends a few cm short).
        The finish camera's clock is moved onto the visible official finish times (median offset over
        the horses with both, >= 3) in every fuse here; the camera keeps the order and the gaps."""
        self._official_clock = self._official_finish()
        nonfin = self.info.official_nonfinish()
        self._nonfin = self._stopped(nonfin)
        self._unseated = {h for h, c in nonfin.items() if c in self.UNSEATED}
        try:
            self._fuse_final()
        finally:
            self._official_clock = None
            self._render_bias = None
            self._nonfin = {}
            self._unseated = set()

    _nonfin: dict = {}
    _unseated: set = set()
    STOPPED = ("PU", "FE", "DNF")  # official placing codes of a horse that stops: pulled up, fell, did not finish
    UNSEATED = ("UR",)  # rider unseated: the horse may gallop on riderless, and the render may still draw it
    NONFIN_BLIND = 50.0  # m before the first boundary a stopped horse has no time for

    def _stopped(self, nonfin) -> dict:
        """{horse: s from which it takes no detection} for the runners the visible official record codes as stopping
        (``STOPPED``). The horse stops somewhere in its first section without a time (a faller at the 100 m still races
        300 m past its last time), so only from NONFIN_BLIND m before that boundary on is it barred -- there the
        detections are the finishers'."""
        bnd, cum = self.info.official_boundaries()
        out = {}
        for h, code in nonfin.items():
            if code not in self.STOPPED or not bnd:
                continue
            c = np.asarray(cum.get(h, np.full(len(bnd), np.nan)), float)
            timed = [i for i in range(len(bnd)) if any(np.isfinite(np.asarray(x, float)[i]) for x in cum.values())]
            i0 = next((i for i in timed if not np.isfinite(c[i])), len(bnd) - 1)  # a withheld boundary is no evidence
            out[h] = self.info.distance - bnd[i0] - self.NONFIN_BLIND
        return out

    def _fuse_final(self):
        import pickle

        with open(self.p("fuse_video.pkl"), "rb") as f:
            V = pickle.load(f)
        pv = V["paths"]
        horses = [r.no for r in self.info.runners]
        offi = self.official_observations()
        t_off = max((float(o.t.max()) for o in offi if o.feed == "official" and len(o.t)), default=None)
        t_vid = max((float(p.t[-1]) for p in pv.values()), default=None)
        ends = [x for x in (t_off, t_vid) if x is not None]
        t_end = max(ends) + 0.5 if ends else None
        raw_bias = self._metric_bias(pv)
        self._render_bias = raw_bias
        shift = None if raw_bias is None else (lambda s, b=raw_bias: np.interp(s, *b))
        res, _fin, pf = self._core_product("final", start=(V["seg_lab"],), extra=offi, s_shift=shift, t_end=t_end,
                                           exclude=self._nonfin)
        shape = self._shape_bias(raw_bias)
        if shape is not raw_bias and self._evidence().n_seg == len(res.seg_lab):  # labels fixed: the last state takes
            # the metric-line shape
            from . import core

            ev = self._evidence()
            st = res.state
            sh = lambda s, g=shape: np.interp(s, *g)  # noqa: E731
            col = {int(h): c for c, h in enumerate(ev.horses)}
            o_s, o_l = self._core_obs(offi + self.finish_observations(pf), col)
            model = core.Model(ev, core.list_rows(self.legend_rows(), self._tick_of_tracks(), ev.horses),
                               st.tg, self.info.width)
            res.state = core.evaluate(model, res.seg_lab, o_s, o_l, X0=st.S, s_shift=sh)
            pf = self._core_paths(res, ev, model)
        pf = self._end_at_last_label(pf)
        pf, warp = self._official_retime(pf)
        if warp:
            big = {h: round(w, 2) for h, w in warp.items() if w > 0.05}
            self.log(f"   fuse_final: re-timed onto the official finish (max shift {max(warp.values()):.3f} s"
                     + (f"; > 0.05 s: {big}" if big else "") + ")")
        ctr = {"anchors": [] if raw_bias is None else [float(x) for x in raw_bias[0]]}
        self._write_paths(pf, "paths.parquet", contract=ctr)
        self.log(f"   fuse_final: {len(pf)}/{len(horses)} horses")

    def _tick_of_tracks(self):
        T = np.load(self.p("tracks.npz"), allow_pickle=True)
        fr, t = T["rows"][:, 0], T["t"]
        o = np.argsort(fr, kind="stable")
        return lambda f: np.interp(np.asarray(f, float), fr[o].astype(float), t[o])

    def _end_at_last_label(self, paths) -> dict:
        """Stopped horses' paths end at their last labelled detection (no prior-only extension to the grid's end); one
        with no labelled detection has no path."""
        out = dict(paths)
        if not self._nonfin:
            return out
        T = np.load(self.p("tracks.npz"), allow_pickle=True)
        lab = np.load(self.p("det_labels_final.npz"))["labels"]
        for h in self._nonfin:
            if h not in out:
                continue
            p = out[h]
            tl = T["t"][lab == h]
            k = int(np.searchsorted(p.t, tl.max(), side="right")) if len(tl) else 0
            if k < 2:
                del out[h]
            elif k < len(p.t):
                out[h] = replace(p, t=p.t[:k], s=p.s[:k], lat=p.lat[:k], sd_s=p.sd_s[:k], sd_l=p.sd_l[:k],
                                 n_obs=p.n_obs[:k])
        return out

    TIE_STEP = 0.002  # s between horses sharing an official finish time, in placing order (dead heats: none)

    def _official_finish(self) -> dict:
        """{horse: official finish time s} (empty where the mask withholds it)."""
        return self.info.official_finish()

    def _official_place(self, h) -> int:
        """Official placing of horse ``h`` (dead heats share one; unplaced after every placed horse)."""
        pl_ = self.info.official_placing()
        return pl_.get(h, max(pl_.values(), default=0) + 1)

    def _official_retime(self, paths):
        """FINAL product: each horse crosses the line at its official finish time (photo-finish, exact). The
        correction is LOCAL to the last section: the re-timing is the identity up to the path's own crossing of the
        last sectional boundary before the finish, then one strictly increasing C1 segment (``_finish_warp``) moves
        the finish crossing onto the official time without a speed jump; lateral and sd follow the same re-timing.
        The interior official times are noisy measurements already in the fusion (OFFICIAL_INTERIOR_SD), so they
        are not re-timed onto. Withheld finish: no re-timing. -> (paths, {horse: largest shift s})."""
        bnd, cum = self.info.official_boundaries()
        fin = self._official_finish()
        D = self.info.distance
        targets = {h: ([0.0], [c[-1]]) for h, c in cum.items() if bnd and bnd[-1] == 0.0 and np.isfinite(c[-1])}
        for h, ft in fin.items():
            targets.setdefault(h, ([0.0], [ft]))
        # equal official finish times (0.01 s resolution) keep the official placing: spaced inside the interval
        groups = {}
        for h in fin:
            if h in targets and targets[h][0] and targets[h][0][-1] == 0.0 and np.isfinite(targets[h][1][-1]):
                groups.setdefault(round(targets[h][1][-1], 2), []).append(h)
        for tie in groups.values():
            places = sorted({self._official_place(h) for h in tie})
            for h in tie:
                targets[h][1][-1] += places.index(self._official_place(h)) * self.TIE_STEP
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
            tl = float(np.interp(D - bnd[-2], s_mono, p.t)) if len(bnd) >= 2 and s_mono[0] < D - bnd[-2] else None
            if tl is not None and tl < kt[-1] and tl < kw[-1]:
                w = self._finish_warp(tl, kt[-1], kw[-1], p.t)
            else:  # no last-section start on the path: one segment from the first node
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
    def _finish_warp(tl, tf, wf, t):
        """t -> w: the identity up to ``tl``, a cubic Hermite from (tl, tl, slope 1) to (tf, wf, slope m) with m the
        secant (wf - tl) / (tf - tl), continued at slope m after ``tf``. C1 everywhere; strictly increasing for
        m >= 1/3 (Fritsch-Carlson), else the start slope is clipped to 3 m."""
        from scipy.interpolate import CubicHermiteSpline

        m = (wf - tl) / (tf - tl)
        d0 = min(1.0, 3.0 * m)
        seg = CubicHermiteSpline([tl, tf], [tl, wf], [d0, m])(np.clip(t, tl, tf))
        return np.where(t <= tl, t, np.where(t >= tf, wf + m * (t - tf), seg))

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

    def _shape_bias(self, bias):
        """With ``metric_w`` > 0 the bias table (s, shift) becomes ds(s) + the bias left at the visible
        boundaries after ds, interpolated: ds(s) = the offset-line tick metric's s minus the rail-s metric's for the same
        detections (``tick_metric`` at w and 0 on observe's detections, median per metre of render s). The product keeps
        its values at every visible official boundary and takes the offset-line metric's shape between them. ``bias``
        unchanged when w = 0 or no boundary is visible."""
        w = self.metric_w()
        if not w or bias is None:
            return bias
        rows, X, _hist = self._observed_dets()
        _z, m, rf, s_px, _lat = self._rail_projection(rows, X)
        d0 = self.tick_metric(rows, s_px, rf, m)["d"]
        dw = self.tick_metric(rows, s_px, rf, m, w)["d"]
        s0 = self.info.distance - d0
        key, inv = np.unique(np.round(s0).astype(int), return_inverse=True)
        order = np.argsort(inv, kind="stable")
        groups = np.split((d0 - dw)[order], np.flatnonzero(np.diff(inv[order])) + 1)
        s_r, ds = key.astype(float), np.array([np.median(g) for g in groups])
        sb, b = bias
        grid = np.unique(np.r_[s_r, sb])
        shift = np.interp(grid, s_r, ds) + np.interp(grid, sb, b - np.interp(sb, s_r, ds))
        self.log(f"   fuse_final: metric line {w:.2f} m outside the rail between the boundaries (shape range "
                 f"{np.ptp(ds):.1f} m)")
        return grid, shift


    ANCHOR_SD = 3.0  # m: a fused crossing anchors the finish clock only where the path is tight

    def finish_sc(self, zf) -> dict:
        """The finish camera's cached scan (finish.npz) for ``finishcam``: line, cloth reads, and the camera's
        geometry (Happy Valley's is overhead: one image band per lane; Sha Tin's is side-on)."""
        return dict(line_x=float(zf["line_x"]), reads=zf["reads"], overhead=self.info.venue == "HV")

    def _fc_read(self, zf) -> set:
        """Horses the finish camera read (negative evidence / shadow-path tie-break): those with a GAP."""
        return {int(k) for k in finishcam.finish_gaps(self.finish_sc(zf))[1]}

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
