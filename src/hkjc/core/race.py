"""The canonical :class:`Race` domain object — the input to agent-based modelling.

A race is *runners on a geometry under conditions*. This module bundles those into
one immutable structure so the rest of the stack stops passing loose parallel
arrays (``v_c``, ``w_prime``, ``draw``, ``distance``, ``course`` …) around. A
:class:`Race` resolves its own :class:`~hkjc.track.TrackGeometry` from
``(venue, distance, surface)`` and builds any :class:`~hkjc.model.RaceModel` from
itself, so an agent-based simulation is one call:

    race = Race.from_results(results_df)         # or .from_racecard(...)
    sim  = RaceSimulator(race.traffic_model(), n_samples=50_000)
    sim.prob(sim.wins(0))

Indices follow the rest of the library: position ``i`` in the per-runner arrays is
HKJC runner number ``i + 1``. Runners are held in starting (runner-number) order.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .track import TrackGeometry, track_for


@dataclass(frozen=True)
class Runner:
    """One starter. ``no`` is the HKJC runner number (1-based)."""
    no: int
    draw: int
    horse_id: str | None = None
    horse: str | None = None
    win_odds: float | None = None
    v_c: float | None = None          # critical velocity (m/s), from sectional energetics
    w_prime: float | None = None      # anaerobic distance reserve (m)
    finish_pos: int | None = None     # realised finishing position (results only)


# defaults for equal-ability draw-geometry studies: a typical HK gallop pace + reserve
DEFAULT_V_C = 16.0
DEFAULT_W_PRIME = 50.0


@dataclass(frozen=True)
class Race:
    date: str
    venue: str                        # "HV" | "ST"
    race_no: int
    distance: float
    runners: tuple[Runner, ...]
    surface: str = "turf"             # "turf" | "awt"
    course: str = "A"                 # movable-rail position
    going: str | None = None
    race_class: str | None = None
    extra: dict = field(default_factory=dict)

    # ----------------------------------------------------------------- geometry
    @property
    def n(self) -> int:
        return len(self.runners)

    @property
    def geometry(self) -> TrackGeometry:
        """The track geometry this race is run on (venue × surface × distance)."""
        return track_for(self.venue, self.distance, self.surface)

    @property
    def width(self) -> float:
        return self.geometry.width(self.course)

    @property
    def bend_fraction(self) -> float:
        """Share of the trip run on turns — the draw-bias exposure for this race."""
        return self.geometry.bend_fraction(self.distance)

    # ------------------------------------------------------------- runner arrays
    @property
    def draws(self) -> np.ndarray:
        return np.array([r.draw for r in self.runners], dtype=float)

    @property
    def win_odds(self) -> np.ndarray:
        return np.array([r.win_odds if r.win_odds else np.nan for r in self.runners], dtype=float)

    def _ability(self, attr: str, override, default: float) -> np.ndarray:
        if override is not None:
            arr = np.asarray(override, dtype=float)
            if arr.shape != (self.n,):
                raise ValueError(f"{attr} override must have shape ({self.n},)")
            return arr
        vals = [getattr(r, attr) for r in self.runners]
        return np.array([v if v is not None else default for v in vals], dtype=float)

    # --------------------------------------------------------------- model builders
    # NOTE: these convenience builders reach UP into ``model`` / ``calibration``. The
    # imports are function-local on purpose, so importing ``core`` never pulls in
    # ``model`` (the layering holds at module-load time; only calling a builder creates
    # the runtime dependency). If this ever feels wrong, move the builders to a
    # ``model.builders`` module of free functions taking a Race.
    def traffic_model(self, v_c=None, w_prime=None, **kwargs):
        """Build the Tier-A interacting :class:`~hkjc.model.TrafficRaceModel`.

        Resolves geometry/course/draw from the race; per-horse ``(v_c, w_prime)`` come
        from the runners (sectional energetics) unless overridden, falling back to
        equal-ability defaults for a pure draw-geometry study.
        """
        from ..model.calibration import CALIBRATED_BEND_CURVATURE
        from ..model.traffic import TrafficRaceModel
        if "bend_curvature" not in kwargs and self.venue in CALIBRATED_BEND_CURVATURE:
            kwargs["bend_curvature"] = CALIBRATED_BEND_CURVATURE[self.venue]
        return TrafficRaceModel(
            self._ability("v_c", v_c, DEFAULT_V_C),
            self._ability("w_prime", w_prime, DEFAULT_W_PRIME),
            self.draws, self.distance,
            track=self.geometry, course=self.course, **kwargs,
        )

    def energetics_model(self, v_c=None, w_prime=None, **kwargs):
        """Build the non-interacting :class:`~hkjc.model.EnergeticsRaceModel`."""
        from ..model.energetics import EnergeticsRaceModel
        return EnergeticsRaceModel(
            self._ability("v_c", v_c, DEFAULT_V_C),
            self._ability("w_prime", w_prime, DEFAULT_W_PRIME),
            self.distance, **kwargs,
        )

    def plackett_luce(self, **kwargs):
        """Build the market-odds Harville/Plackett-Luce baseline from win odds."""
        from ..model.plackett_luce import PlackettLuceModel
        return PlackettLuceModel.from_odds(self.win_odds, **kwargs)

    # ------------------------------------------------------------- constructors
    @classmethod
    def from_runners(cls, runners, *, date, venue, race_no, distance, **kwargs) -> "Race":
        rs = tuple(sorted(runners, key=lambda r: r.no))
        return cls(date=str(date), venue=str(venue), race_no=int(race_no),
                   distance=float(distance), runners=rs, **kwargs)

    @classmethod
    def from_racecard(cls, df, *, distance, course="A", surface="turf",
                      going=None, race_class=None) -> "Race":
        """Build from a :func:`hkjc.data.racecard.get_racecard` frame (one race).

        The racecard's per-runner table has no header metadata, so ``distance`` is
        required; ``course``/``surface``/``going`` come from the meeting context.
        """
        rows = df.to_dicts()
        meta = rows[0] if rows else {}
        runners = tuple(
            Runner(no=int(r["runner"]), draw=int(r["draw"]) if r.get("draw") is not None else 0,
                   horse_id=r.get("horse_id"), horse=r.get("horse"))
            for r in rows if r.get("runner") is not None and r.get("draw") is not None
        )
        return cls.from_runners(
            runners, date=meta.get("date"), venue=meta.get("venue"), race_no=meta.get("race_no"),
            distance=distance, course=course, surface=surface, going=going, race_class=race_class)

    @classmethod
    def from_results(cls, df) -> "Race":
        """Build from a :func:`hkjc.data.historical.get_race_data` frame (one race).

        Maps the results schema (PascalCase) to runners with draw, closing win odds
        and realised finishing position, and infers surface from the going code.
        """
        rows = df.to_dicts()
        if not rows:
            raise ValueError("empty results frame")
        m = rows[0]
        going = (m.get("Going") or "").strip() or None
        surface = "awt" if _is_awt(going, m.get("Course"), m.get("Track")) else "turf"
        runners = tuple(
            Runner(
                no=int(r["HorseNo"]), draw=int(r["Dr"]) if r.get("Dr") is not None else 0,
                horse_id=r.get("HorseID"), horse=(r.get("Horse") or None),
                win_odds=_as_float(r.get("WinOdds")), finish_pos=_as_int(r.get("Pla")),
            )
            for r in rows if r.get("HorseNo") is not None and r.get("Dr") is not None
        )
        return cls.from_runners(
            runners, date=m.get("Date"), venue=m.get("Venue"),
            race_no=m.get("RaceNumber") or m.get("RaceIndex"),
            distance=float(str(m.get("RaceDistance")).rstrip("M") or 0),
            course=(m.get("Course") or "A"), surface=surface, going=going,
            race_class=m.get("RaceClass"))


# The reliable surface signal is the Course/Track label ("ALL WEATHER TRACK"/"AWT").
# Going is only a weak fallback, restricted to AWT-*exclusive* codes — "GOOD"/"GD"
# appear in both scales, so they must NOT trigger AWT.
_AWT_GOINGS = {"WET FAST", "FAST", "SLOW", "WET SLOW", "RAIN AFFECTED",
               "NORMAL WATERING", "WF", "FT", "SL", "WS", "RA", "NW"}


def _is_awt(going, *labels) -> bool:
    for lab in labels:
        u = str(lab).upper().strip() if lab else ""
        if "ALL WEATHER" in u or u == "AWT":
            return True
    return bool(going) and going.strip().upper() in _AWT_GOINGS


def _as_float(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _as_int(x):
    try:
        return int(x)
    except (TypeError, ValueError):
        return None
