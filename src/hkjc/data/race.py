"""One race's facts, bundled: meeting/course geometry, runners (number, brand id, draw) and the
official outcome. Pre-race facts (distance, course, draw, silks) may feed any product; the official
outcome (placings, times, sectionals, running positions) must not feed the video-only product (the
final product may use it).

The reconstruction reads the race through a :class:`MaskedRace`: the pre-race facts plus the official
outcome records a :class:`Mask` leaves visible (accuracy is measured on what the mask withholds).
Validation scores against the full :class:`RaceInfo`."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import polars as pl

from .tracks import TrackGeometry, track_for

CACHE = Path("data_cache/race_official")


@dataclass(frozen=True)
class Runner:
    no: int
    horse: str
    horse_id: str
    draw: int


@dataclass(frozen=True)
class RaceInfo:
    date: str  # YYYYMMDD
    venue: str  # "HV" | "ST"
    race_no: int
    distance: float
    course: str  # "A", "B", "C", "C+3", ... or "AWT"
    surface: str  # "turf" | "awt"
    going: str
    race_class: str
    runners: tuple  # every horse that started, finisher or not; withdrawn horses are not runners
    results: pl.DataFrame = field(repr=False, compare=False)  # official outcome: never an input to the video-only product
    sectionals: pl.DataFrame = field(repr=False, compare=False)  # official outcome (same contract)

    @property
    def key(self) -> str:
        return f"{self.date}_{self.venue}_{self.race_no}"

    @property
    def geometry(self) -> TrackGeometry:
        return track_for(self.venue, self.distance, self.surface)

    @property
    def width(self) -> float:
        """Racing-rail -> outer-rail width (m) for this race's course setting."""
        return self.geometry.width(self.course)

    @property
    def n(self) -> int:
        return len(self.runners)


def official_boundaries(info) -> tuple[list, dict]:
    """Full official record: (distance-to-go at each sectional end, {runner: cumulative times there}).
    A horse that did not finish has times up to its last complete section, NaN after."""
    sec = info.sectionals
    nsec = sec["section"].n_unique()
    bnd = [400.0 * (nsec - 1 - k) for k in range(nsec)]  # distance-to-go at section ends
    w = sec.pivot(index="runner", on="section", values="section_time").sort("runner")
    cols = sorted((c for c in w.columns if c != "runner"), key=int)
    cum = np.cumsum(w.select(cols).to_numpy().astype(float), axis=1)
    return bnd, dict(zip(w["runner"].to_list(), cum))


SUBSPLIT_HALF = 200.0  # m: a sub-split point sits mid-way through a 400 m section


def official_subsplits(info) -> tuple[list, dict]:
    """Full official record of the 200 m sub-splits: (distance-to-go at each sub-split point,
    {runner: cumulative official times there}), points in race order (e.g. ``[600.0, 200.0]``).

    HKJC prints each horse's time over the first and second 200 m of the race's last two 400 m sections;
    the point is the section's midpoint and its time the cumulative time at the section start plus the
    first 200 m. NaN where a horse has no sub-split or no time at the section start (a non-finisher's
    partial cell gives no section row, hence no sub-split). ``([], {})`` when the sectionals carry no
    sub-split columns (parsed before they were read). Separate from :func:`official_boundaries`, whose
    indices ``Mask(boundary=j)`` refers to."""
    sec = info.sectionals
    if "sub1" not in sec.columns:
        return [], {}
    bnd, cum = official_boundaries(info)
    sub1 = {(int(r), int(s)): v for r, s, v in sec.select("runner", "section", "sub1").iter_rows() if v is not None}
    pts, secs = [], []
    for i in sorted({s for _r, s in sub1}):  # 1-based section; it starts at boundary i - 2 (the start for i = 1)
        start = bnd[i - 2] if i >= 2 else float(info.distance)
        if abs(start - bnd[i - 1] - 2 * SUBSPLIT_HALF) < 1e-6:
            pts.append(start - SUBSPLIT_HALF)
            secs.append(i)
    out = {}
    for h, c in cum.items():
        t0 = [0.0 if i == 1 else float(c[i - 2]) for i in secs]
        out[h] = np.array([a + sub1[(h, i)] if (h, i) in sub1 else np.nan for a, i in zip(t0, secs)], dtype=float)
    return pts, out


def official_finish(info) -> dict:
    """Full official record: {horse: finish time s} (results FinishTime), in result order. Horses
    without a time (non-finishers) are absent; a disqualified horse keeps the time it ran."""
    out = {}
    for h, ft in info.results.select(["HorseNo", "FinishTime"]).iter_rows():
        if h is not None and ft is not None and np.isfinite(ft):
            out[int(h)] = float(ft)
    return out


def official_placing(info) -> dict:
    """Full official record: {horse: placing} for the placed horses, in placing order (the results' row
    order); the horses of a dead heat share a placing; unplaced horses (non-finishers, DISQ) are absent."""
    return {
        int(h): int(p)
        for h, p in info.results.select(["HorseNo", "Pla"]).iter_rows()
        if h is not None and p is not None
    }


def official_order(info) -> list:
    """Full official record: placed horse numbers in placing order (the results' row order)."""
    return list(official_placing(info))


@dataclass(frozen=True)
class Mask:
    """Information withheld from the reconstruction: official outcome records and video evidence.

    Official outcome. ``boundary=j``: the cumulative official time at sectional boundary j (0 = the first
    section end; the last index is the finish, which also withholds the finish times and placings), and
    the 200 m sub-split points within 200 m of it. ``boundary=(j, k, ...)`` withholds several boundaries at
    once, each as above (an evaluation tool: withheld points far from every visible boundary); a single
    index is kept as an int, so its spec stays ``boundary=j``.
    ``outcome=True``: every official outcome record (results, finish times, placings, sectionals,
    running positions); pre-race facts (distance, course, draw, runners, silks) stay visible.

    Video evidence (read from ``observe`` on). ``dets``: render detections, as ``dets.npz`` row indices,
    removed as if the detector had missed them. ``discs=P``: disc-number reads (circle render) on the
    race clock's blocks ``[kP, (k+1)P)`` with ``k % 2 == discs_phase`` (default 1, the odd blocks) count
    as unread. ``legend=P``: the silk render's legend reads (its 8 running-order slots) on the blocks
    ``[kP, (k+1)P)`` with ``k % 2 == legend_phase`` count as unread, in the same way.

    ``Mask()`` withholds nothing. Spec strings (``parse``, CLI ``--mask``, env ``HKJC_MASK``): ``none``,
    or ``;``-joined parts ``boundary=J`` | ``boundary=J+K+...`` | ``outcome``, ``dets=I,J,...``, ``discs=P`` | ``discs=P@PHASE``,
    ``legend=P`` | ``legend=P@PHASE``."""

    boundary: int | tuple | None = None
    outcome: bool = False
    dets: frozenset = frozenset()
    discs: float | None = None
    discs_phase: int = 1
    legend: float | None = None
    legend_phase: int = 1

    def __post_init__(self):
        if isinstance(self.boundary, (tuple, list, set, frozenset)):
            js = tuple(sorted({int(b) for b in self.boundary}))
            object.__setattr__(self, "boundary", js[0] if len(js) == 1 else js)
        js = self.boundaries()
        if self.boundary is not None and (self.outcome or not js or any(isinstance(b, bool) or b < 0 for b in js)
                                          or not all(isinstance(b, int) for b in js)):
            raise ValueError(f"invalid mask {self!r}: boundary is a non-negative index or indices, exclusive of outcome")
        object.__setattr__(self, "dets", frozenset(int(i) for i in self.dets))
        if any(i < 0 for i in self.dets):
            raise ValueError(f"invalid mask {self!r}: dets are dets.npz row indices")
        if self.discs is not None and not (self.discs > 0 and self.discs_phase in (0, 1)):
            raise ValueError(f"invalid mask {self!r}: discs is a block length > 0 s, phase 0 or 1")
        if self.legend is not None and not (self.legend > 0 and self.legend_phase in (0, 1)):
            raise ValueError(f"invalid mask {self!r}: legend is a block length > 0 s, phase 0 or 1")

    def __bool__(self) -> bool:
        return bool(self.outcome_part() or self.evidence_part())

    def boundaries(self) -> tuple:
        """The withheld boundary indices (empty when none)."""
        j = self.boundary
        return () if j is None else j if isinstance(j, tuple) else (j,)

    def outcome_part(self) -> str:
        """The official-outcome component of the spec ('' when nothing is withheld)."""
        if self.outcome:
            return "outcome"
        return "" if self.boundary is None else "boundary=" + "+".join(str(b) for b in self.boundaries())

    def evidence_part(self) -> str:
        """The video-evidence component of the spec ('' when nothing is withheld)."""
        parts = []
        if self.dets:
            parts.append("dets=" + ",".join(str(i) for i in sorted(self.dets)))
        if self.discs is not None:
            parts.append(f"discs={self.discs:g}" + ("" if self.discs_phase == 1 else f"@{self.discs_phase}"))
        if self.legend is not None:
            parts.append(f"legend={self.legend:g}" + ("" if self.legend_phase == 1 else f"@{self.legend_phase}"))
        return ";".join(parts)

    def __str__(self) -> str:
        return ";".join(p for p in (self.outcome_part(), self.evidence_part()) if p) or "none"

    @staticmethod
    def _blocks(t, period, phase) -> np.ndarray:
        t = np.asarray(t, float)
        if period is None:
            return np.zeros(t.shape, bool)
        return np.floor(t / period).astype(np.int64) % 2 == phase

    def discs_withheld(self, t) -> np.ndarray:
        """Per race-clock time: is a disc read at that time withheld?"""
        return self._blocks(t, self.discs, self.discs_phase)

    def legend_withheld(self, t) -> np.ndarray:
        """Per race-clock time: is a legend read at that time withheld?"""
        return self._blocks(t, self.legend, self.legend_phase)

    @classmethod
    def parse(cls, spec: str | None) -> Mask:
        s = (spec or "").strip().lower()
        if s in ("", "none"):
            return cls()
        kw = {}
        for part in (p.strip() for p in s.split(";")):
            key, _eq, val = part.partition("=")
            if key in kw or (key in ("outcome", "boundary") and ({"outcome", "boundary"} & kw.keys())):
                raise ValueError(f"mask spec {spec!r}: {key} given twice")
            if part == "outcome":
                kw["outcome"] = True
            elif key == "boundary" and val:
                js = tuple(int(b) for b in val.split("+"))
                kw["boundary"] = js[0] if len(js) == 1 else js
            elif key == "dets" and val:
                kw["dets"] = frozenset(int(i) for i in val.split(","))
            elif key in ("discs", "legend") and val:
                p, _at, ph = val.partition("@")
                kw[key] = float(p)
                if ph:
                    kw[f"{key}_phase"] = int(ph)
            else:
                raise ValueError(
                    f"mask spec {spec!r}: expected none | ';'-joined boundary=J[+K...] | outcome, dets=I,J,..., discs=P[@PHASE],"
                    " legend=P[@PHASE]"
                )
        return cls(**kw)

    @classmethod
    def from_env(cls) -> Mask:
        """``HKJC_MASK`` (a spec string); ``HKJC_FINAL_HOLDOUT=j`` is an alias for ``boundary=j``.
        Setting both is an error."""
        spec, hold = os.environ.get("HKJC_MASK"), os.environ.get("HKJC_FINAL_HOLDOUT")
        if spec is not None and hold is not None:
            raise ValueError("set HKJC_MASK or HKJC_FINAL_HOLDOUT, not both")
        return cls(boundary=int(hold)) if hold is not None else cls.parse(spec)


class MaskedRace:
    """The race as the reconstruction sees it: every pre-race fact of ``info`` (``PRE_RACE``) and the
    official outcome only through ``official_boundaries`` / ``official_subsplits`` / ``official_finish`` /
    ``official_order`` / ``official_placing``, with what ``mask`` withholds removed. It has no ``results`` / ``sectionals``."""

    PRE_RACE = (
        "date", "venue", "race_no", "distance", "course", "surface", "going", "race_class",
        "runners", "key", "geometry", "width", "n",
    )

    def __init__(self, info, mask: Mask = Mask()):
        self._info = info
        self.mask = mask

    def __getattr__(self, name):
        if name in MaskedRace.PRE_RACE:
            return getattr(self._info, name)
        raise AttributeError(f"MaskedRace has no {name!r}: official outcome is read through its accessors")

    def _finish_withheld(self) -> bool:
        """The outcome, or the last boundary (the finish), is withheld."""
        return self.mask.outcome or (self._info.sectionals["section"].n_unique() - 1) in self.mask.boundaries()

    def official_boundaries(self) -> tuple[list, dict]:
        """(distance-to-go at each sectional end, {runner: cumulative official times}); a withheld
        boundary is NaN for every runner; ([], {}) when the outcome is withheld."""
        if self.mask.outcome:
            return [], {}
        bnd, cum = official_boundaries(self._info)
        js = [j for j in self.mask.boundaries() if j < len(bnd)]
        if js:
            hidden = {}
            for h, c in cum.items():
                c = np.array(c, dtype=float)
                c[js] = np.nan
                hidden[h] = c
            cum = hidden
        return bnd, cum

    def official_subsplits(self) -> tuple[list, dict]:
        """(distance-to-go at each 200 m sub-split point, {runner: cumulative official times}); a point
        within 200 m of a withheld boundary is NaN for every runner (its time is the previous boundary's
        time plus a 200 m split, so the points on either side of a boundary give it away); ([], {}) when
        the outcome is withheld."""
        if self.mask.outcome:
            return [], {}
        pts, cum = official_subsplits(self._info)
        bnd = official_boundaries(self._info)[0] if self.mask.boundaries() and pts else []
        js = [j for j in self.mask.boundaries() if j < len(bnd)]
        if js:
            hide = np.array([any(abs(p - bnd[j]) <= SUBSPLIT_HALF + 1e-6 for j in js) for p in pts])
            cum = {h: np.where(hide, np.nan, c) for h, c in cum.items()}
        return pts, cum

    def official_finish(self) -> dict:
        """{horse: official finish time s}; empty when the outcome or the finish boundary is withheld."""
        if self._finish_withheld():
            return {}
        return official_finish(self._info)

    def official_order(self) -> list:
        """Placed horse numbers in official placing order; empty when the outcome or the finish is withheld."""
        if self._finish_withheld():
            return []
        return official_order(self._info)

    def official_placing(self) -> dict:
        """{horse: official placing} (dead heats share one); empty when the outcome or the finish is withheld."""
        if self._finish_withheld():
            return {}
        return official_placing(self._info)


def _cached(path: Path, fetch):
    if path.exists():
        return pl.read_parquet(path)
    df = fetch()
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path)
    return df


def race_info(date: str, venue: str, race_no: int) -> RaceInfo:
    """Fetch (cached under data_cache/race_official) the results + sectionals of one race."""
    from .historical import get_race_data
    from .sectional import get_sectional_times

    d = str(date).replace("-", "").replace("/", "")
    slash = f"{d[:4]}/{d[4:6]}/{d[6:]}"
    res = _cached(
        CACHE / f"{d}{venue}{race_no}_results.parquet", lambda: get_race_data(slash, venue, race_no)
    )
    sec = _cached(
        CACHE / f"{d}{venue}{race_no}_sectional.parquet",
        lambda: get_sectional_times(slash, race_no),
    )
    m = res.row(0, named=True)
    label = m["Course"]
    surface = "awt" if "ALL WEATHER" in label.upper() else "turf"
    course = "AWT" if surface == "awt" else (label.split('"')[1] if '"' in label else "A")
    runners = tuple(
        sorted(
            (
                Runner(int(r["HorseNo"]), r["Horse"].split(" (")[0], r["HorseID"], int(r["Dr"]))
                for r in res.iter_rows(named=True)
            ),
            key=lambda r: r.no,
        )
    )
    return RaceInfo(
        d,
        venue,
        int(race_no),
        float(m["RaceDistance"]),
        course,
        surface,
        m["Going"],
        m["RaceClass"],
        runners,
        res,
        sec,
    )
