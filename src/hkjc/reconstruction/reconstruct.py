"""The reconstruction ORCHESTRATOR — feeds -> obs -> clock -> identity -> fusion -> trajectory.

:class:`Reconstructor` wires the validated 4-feed pipeline (the ``cv_r03_fusion`` / ``cv_r08_fusion``
study blueprint) into clean library form on top of :mod:`hkjc.vision` (perception),
:mod:`hkjc.reconstruction.geometry` (track facts from ``core.Race``), :mod:`.clock` (the shared
``race_t`` axis) and :mod:`.fusion` (the consolidated smoother v2).

The per-race EMPIRICAL sync/calibration inputs that are NOT in ``core.Race`` (the 4 video paths, the
official section table, per-feed clock offsets/anchors, coverage windows, the name->number LUT) are
gathered into :class:`RaceRecipe` — these are VALIDATION-class calibration inputs (allowed), as is
the draw, which here comes from ``core.Race`` (a pre-race initial condition).

The role each feed plays in the fused observation set (the ``build_obs`` contract, per CV_RECON §6):

- **PATROL** anchors the lateral METRIC (low-sd ``y_lat`` in metres).
- **AERIAL + DRONE** carry lateral ORDERING (inflated metric variance) + arclen.
- **RF** carries the longitudinal ORDER backbone (legend rank) + finish-MARGIN metric
  (``core.THOROUGHBRED_LENGTH_M`` per length).
- the **DRAW** anchors a low-sd lateral obs at ``race_t = 0`` via ``RaceGeom.draw_to_metres()``.

Heavy CV (running the ``vision.feeds.*`` extractors) is the IO path of ``.run()``; cv2/av/rapidocr
stay lazy (imported inside the feed modules). Official results are VALIDATION-ONLY — never consumed
in ``.run()``; an optional :func:`validate` helper is provided separately.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..core.race import Race
from ..core.track import THOROUGHBRED_LENGTH_M
from ..vision.schema import (
    RaceContext,
    RaceTrajectory,
    RunnerTrajectory,
    TrajPoint,
)
from . import fusion as fuse
from .clock import ClockAnchors, anchors_from_recipe
from .fusion import Obs
from .geometry import RaceGeom

# ============================================================================ default obs noise levels
# The metric hierarchy (CV_RECON §6) — the SAME values the validated studies used (metres).
SD_ARC_AERIAL = 7.0      # field-odometer arclen: loose per-horse (it is the FIELD mean).
SD_ARC_DRONE = 8.0
SD_ARC_PATROL = 6.0
SD_YLAT_PATROL = 1.2     # the LATERAL METRIC ANCHOR (low noise).
SD_YLAT_AERIAL = 6.0     # ordering trustworthy, metric biased -> inflated variance.
SD_YLAT_DRONE = 7.0
SD_ARC_RFFINISH = 1.0    # finish-margin metric arclen at the line (RF MAE ~1 m).
SD_ARC_RFRANK = 1.8      # legend-order soft arclen obs (imposes order, not absolute metres).
SD_ARC_RFBACK = 2.5      # absent-from-legend back-marker obs (looser, depth unknown).
RANK_GAP_M = THOROUGHBRED_LENGTH_M  # arclen offset per rank from the field-leader backbone (~1 body).
SD_YLAT_DRAW = 1.5       # draw -> y_lat(0) anchor (~half a stall).


# ============================================================================ THE PER-RACE RECIPE
@dataclass(frozen=True)
class RaceRecipe:
    """Per-race empirical inputs the pipeline needs that are NOT in :class:`core.Race`.

    Sync/calibration (VALIDATION-class) inputs only: video paths, the official section (time,
    arclen) table, per-feed clock offsets/anchors, per-feed coverage windows, and the
    name->number LUT for RF identity. The draw comes from ``core.Race`` (a pre-race initial
    condition), NOT from here.
    """

    aerial_video: str = ""
    patrol_video: str = ""
    drone_video: str = ""
    rf_video: str = ""

    sections: tuple = ()              # official per-section times (s), race order
    section_bounds: tuple = ()        # cumulative arclen at section boundaries (m); len = len+1
    #                                   (empty -> an even split over the distance)

    patrol_offset: float = 0.0        # race_t = patrol_vid - patrol_offset
    rf_offset: float = 0.0            # race_t = rf_vid - rf_offset
    drone_gate_vid_t: float = 0.0     # drone video time of the gate-jump
    drone_finishcut_vid_t: float = 0.0  # drone video time of the photo-finish cut

    windows: dict = field(default_factory=dict)   # per-feed named (t0,t1,dt) extraction windows
    name2num: dict = field(default_factory=dict)  # legend horse-name -> saddlecloth number (RF)
    cache: dict = field(default_factory=dict)      # optional per-feed cache paths (npz/parquet)

    def clock(self, distance: float) -> ClockAnchors:
        """Build the shared-clock :class:`ClockAnchors` (section curve + per-feed offsets/anchors)."""
        return anchors_from_recipe(
            distance,
            self.sections,
            self.section_bounds or None,
            patrol_offset=self.patrol_offset,
            rf_offset=self.rf_offset,
            drone_gate_vid_t=self.drone_gate_vid_t,
            drone_gate_race_t=0.0,
            drone_finishcut_vid_t=self.drone_finishcut_vid_t,
            drone_finishcut_race_t=float(np.sum(np.asarray(self.sections, float))),
        )


# ============================================================================ THE ORCHESTRATOR
class Reconstructor:
    """Reconstruct one race's per-horse trajectories from its 4 feeds + recipe.

    ``recon = Reconstructor(race, recipe); traj = recon.run()`` -> a
    :class:`hkjc.vision.schema.RaceTrajectory`.
    """

    def __init__(self, race: Race, recipe: RaceRecipe):
        self.race = race
        self.recipe = recipe
        self.geom = RaceGeom(race)
        self.clock = recipe.clock(self.geom.distance)
        self.field_size = race.n

    # ---- the build_obs adapter: feed outputs -> Obs list per runner ----
    def build_obs(self, feeds: dict) -> dict[int, list[Obs]]:
        """Adapt the per-feed detection rows (already numbered + on ``race_t``) into the fused
        :class:`Obs` set, then add the DRAW start-lateral anchor at ``race_t = 0``.

        ``feeds`` is a dict of per-feed row lists in the canonical study shapes:
          - ``aerial`` / ``drone``: rows ``{runner, race_t, arclen, y_lat}`` (lateral ORDERING).
          - ``patrol``: rows ``{runner, race_t, arclen, y_lat_m, confidence?}`` (lateral METRIC).
          - ``rf``: ``{finish: [{runner, race_t, arclen, sd?}], rank_seq: [(race_t, [lead..back])],
            field_arc: (race_t_arr, arc_arr)}`` (the longitudinal observation model).
        The geometry bend down-weight is then applied to the aerial/drone lateral sds in place.
        """
        obs_by_runner: dict[int, list[Obs]] = {}

        def add(o: Obs):
            obs_by_runner.setdefault(o.runner, []).append(o)

        for r in feeds.get("aerial", []):
            add(Obs(int(r["runner"]), float(r["race_t"]), float(r["arclen"]),
                    _opt(r.get("y_lat")), "aerial", SD_ARC_AERIAL, SD_YLAT_AERIAL))
        for r in feeds.get("drone", []):
            add(Obs(int(r["runner"]), float(r["race_t"]), float(r["arclen"]),
                    _opt(r.get("y_lat")), "drone", SD_ARC_DRONE, SD_YLAT_DRONE))
        for r in feeds.get("patrol", []):
            c = float(np.clip(r.get("confidence", 1.0), 0.25, 1.0))
            add(Obs(int(r["runner"]), float(r["race_t"]), _opt(r.get("arclen")),
                    float(r["y_lat_m"]), "patrol", SD_ARC_PATROL, SD_YLAT_PATROL / c))

        rf = feeds.get("rf")
        if rf:
            self._add_rf_obs(rf, add)

        # DRAW -> y_lat(0) anchor (a pre-race INITIAL CONDITION; from core.Race, allowed).
        for n, y in self.geom.draw_to_metres().items():
            add(Obs(int(n), 0.0, 0.0, float(y), "draw_anchor", 999.0, SD_YLAT_DRAW))

        # geometry bend down-weight of the aerial/drone lateral metric (CV_RECON §7).
        for ho in obs_by_runner.values():
            fuse.bend_downweight_obs(ho, self.geom)
        return obs_by_runner

    def _add_rf_obs(self, rf: dict, add) -> None:
        """RF -> longitudinal obs: finish-MARGIN metric + legend-ORDER soft rank-offset (CV_RECON §6).

        Mirrors ``cv_fusion_4feed.load_rf`` / ``cv_r03_fusion._load_rf_r03``: visible legend horses
        get their exact rank offset from the field-leader backbone; absent horses get a soft
        'behind the lead group' obs (keeps back-markers off the field mean). The rank gap
        (``RANK_GAP_M``) imposes ORDER; the loose sd keeps absolute mid-race metres honestly uncertain.
        """
        for r in rf.get("finish", []):
            sd = max(float(r.get("sd") or 0.0), SD_ARC_RFFINISH)
            add(Obs(int(r["runner"]), float(r["race_t"]), float(r["arclen"]), None,
                    "rf_finish", sd, 0.0))

        rank_seq = rf.get("rank_seq", [])
        field_arc = rf.get("field_arc")
        if not rank_seq or field_arc is None:
            return
        f_t, f_a = np.asarray(field_arc[0], float), np.asarray(field_arc[1], float)
        all_runners = set(range(1, self.field_size + 1))
        for race_t, order in rank_seq:
            race_t = float(race_t)
            ranks = {int(h): i + 1 for i, h in enumerate(order)}
            s_field = float(np.interp(race_t, f_t, f_a))
            max_rank = float(max(ranks.values())) if ranks else 0.0
            absent = sorted(all_runners - set(ranks))
            full = list(ranks.values()) + [max_rank + 1.0] * len(absent)
            rank_mid = float(np.mean(full)) if full else 0.0
            for h, rk in ranks.items():
                add(Obs(h, race_t, s_field + (rank_mid - rk) * RANK_GAP_M, None,
                        "rf_rank", SD_ARC_RFRANK, 0.0))
            for h in absent:
                add(Obs(h, race_t, s_field + (rank_mid - (max_rank + 1.0)) * RANK_GAP_M, None,
                        "rf_rank_back", SD_ARC_RFBACK, 0.0))

    # ---- the smoother grid ----
    def grid(self, obs_by_runner: dict[int, list[Obs]]) -> np.ndarray:
        ts = [o.race_t for ho in obs_by_runner.values() for o in ho]
        t_lo, t_hi = min(ts), max(ts)
        return np.arange(t_lo, t_hi + fuse.DT, fuse.DT)

    # ---- assemble the schema output ----
    def assemble(self, smoothed: dict[int, dict], grid: np.ndarray) -> RaceTrajectory:
        """Smoothed arrays -> a :class:`RaceTrajectory` (coords: ``d = distance - arclen``)."""
        dist = self.geom.distance
        ctx = RaceContext(
            venue=self.race.venue, date=str(self.race.date), race_no=int(self.race.race_no),
            distance=float(dist), surface=self.race.surface, course=self.race.course,
            fps=self.recipe.windows.get("fps"),
        )
        runners_out: list[RunnerTrajectory] = []
        for rn in sorted(smoothed):
            r = smoothed[rn]
            has_obs = r["node_has_obs"]
            idx = np.where(has_obs)[0]
            if not len(idx):
                continue
            lo, hi = int(idx.min()), int(idx.max())
            arclen = r["arclen"]
            ylat = r["y_lat"]
            v = r["v"]
            sd_a, sd_l = r["sd_arclen"], r["sd_ylat"]
            pts: list[TrajPoint] = []
            for i in range(lo, hi + 1):
                # conf from the joint arclen+lateral uncertainty (tight sds -> high conf).
                conf = float(np.clip(1.0 / (1.0 + 0.1 * (sd_a[i] + sd_l[i])), 0.0, 1.0))
                pts.append(TrajPoint(
                    t=float(grid[i]),
                    d=float(dist - arclen[i]),
                    off=float(ylat[i]),
                    speed=float(v[i]),
                    gap=not bool(has_obs[i]),
                    conf=conf,
                ))
            coverage = float(np.mean(has_obs[lo:hi + 1])) if hi >= lo else 0.0
            runners_out.append(RunnerTrajectory(
                runner=int(rn), horse_id=self._horse_id(rn), points=pts,
                coverage=coverage, id_confidence=1.0,
            ))
        return RaceTrajectory(context=ctx, runners=runners_out, qa={})

    def _horse_id(self, rn: int) -> str | None:
        for r in self.race.runners:
            if r.no == rn:
                return r.horse_id
        return None

    # ---- the full pipeline (heavy CV IO path) ----
    def run(self) -> RaceTrajectory:
        """Run the feeds, solve identity, clock-sync, fuse, and emit a :class:`RaceTrajectory`.

        Heavy CV lives in the feed extractors (cv2/av/rapidocr lazy). This is the wiring; a caller
        with cached per-feed panels can instead call :meth:`from_feeds` directly.
        """
        feeds = self.extract_feeds()
        return self.from_feeds(feeds)

    def from_feeds(self, feeds: dict) -> RaceTrajectory:
        """The pure (no-CV) tail: numbered feed rows -> obs -> fusion -> trajectory.

        Separated from :meth:`run` so the fusion half is testable without video.
        """
        obs_by_runner = self.build_obs(feeds)
        grid = self.grid(obs_by_runner)
        smoothed = fuse.smooth_field(
            obs_by_runner, grid, self.geom,
            race_t_to_arclen=self.clock.race_t_to_arclen,
        )
        return self.assemble(smoothed, grid)

    def extract_feeds(self) -> dict:
        """Run the ``vision.feeds.*`` perception + identity to produce numbered feed rows on ``race_t``.

        This is the heavy IO path. It builds a :class:`hkjc.vision.feeds.FeedContext` per feed from
        ``self.race`` + ``self.recipe``, runs each extractor, solves identity via
        :class:`hkjc.vision.identity.Identity` (RF anchor + silk + clock-order + the DRAW order from
        ``core.Race``), maps each feed's video time -> ``race_t`` via :attr:`clock`, and returns the
        canonical row dict :meth:`build_obs` consumes. Left as a thin, lazily-importing seam: the
        exact per-feed extractor calls are feed-specific and validated in the studies; a deployment
        supplies cached numbered panels and calls :meth:`from_feeds` directly.
        """
        raise NotImplementedError(
            "extract_feeds() is the heavy CV IO seam; supply numbered feed rows and call "
            "from_feeds(feeds) (or implement the per-feed extractor wiring for a live run)."
        )


def _opt(x) -> float | None:
    """A nan/None-tolerant optional float (drop NaN lateral reads -> a no-op for that channel)."""
    if x is None:
        return None
    xf = float(x)
    return None if np.isnan(xf) else xf
