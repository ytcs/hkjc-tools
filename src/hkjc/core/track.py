"""Race-course geometry — the physical substrate for agent-based race models.

The draw bias is *geometric*: on a turn an inner lane is a shorter path, so a low
draw that holds the rail runs less ground. HKJC turf tracks also use a **movable
running rail** — the race's ``Course`` (A, A+2, B, B+2, B+3, C, C+3) shifts the
rail out to spread wear, changing the usable **width** (and slightly the
home-straight length). So geometry is parameterised by *venue × surface × course*,
not fixed per venue, and it is resolved from the race itself (:func:`track_for`).

Three structural regimes, all encoded here as :class:`TrackGeometry`:

- **Happy Valley** (turf, 1,450 m, right-handed) — a tight, bend-dominated circuit;
  the strongest draw bias in HK. Bend map from the sectional sampling points.
- **Sha Tin turf** (1,900 m, right-handed) — a galloping oval with long straights,
  so a *weaker* geometric draw bias than HV. Its 1,000 m sprint is run on a
  separate **straight chute** with *no* bends — :data:`SHA_TIN_STRAIGHT` — so there
  is no bend-geometry draw bias there at all (only going/watering effects).
- **Sha Tin all-weather** (1,560 m, right-handed, fixed single rail) — dirt; one
  configuration, uniform 22.8 m width.

Coordinates: ``d`` = metres-from-finish (a horse runs from d=distance down to d=0).
Lateral offset is metres outward from the (current) inner rail; on a bend, being
``off`` metres wider adds ``off · turn_angle`` of arc — the ground-lost mechanism
the traffic model integrates over the trip.

Bend segments for Happy Valley come from the HKJC sectional sampling points and are
firm; the Sha Tin oval / AWT bend partitions are reconstructed from the published
circumference and home-straight (the segment *boundaries* are approximate, the
straight-vs-turn split and total are not). Per-course widths and home straights are
the official HKJC figures.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TrackGeometry:
    name: str
    venue: str                    # "HV" | "ST"
    surface: str                  # "turf" | "awt"
    circumference: float          # metres (one lap of this configuration)
    handedness: str               # "right" => clockwise; inner rail favours low draws
    courses: dict                 # course -> (home_straight_m, usable_width_m)
    bend_segments: tuple          # ((start_d, end_d), ...) metres-from-finish, within one lap, that are TURNS

    def width(self, course: str = "A") -> float:
        """Usable track width (m) for a course setting; falls back to the base course."""
        c = (self.courses.get(course)
             or self.courses.get(course.split("+")[0])
             or next(iter(self.courses.values())))
        return c[1]

    def home_straight(self, course: str = "A") -> float:
        """Home-straight length (m) for a course setting."""
        c = (self.courses.get(course)
             or self.courses.get(course.split("+")[0])
             or next(iter(self.courses.values())))
        return c[0]

    def is_bend(self, d_from_finish: float) -> bool:
        """Is position ``d`` (metres from finish) on a turn (tiled over laps)?"""
        dm = d_from_finish % self.circumference
        return any(a <= dm < b for a, b in self.bend_segments)

    def bend_fraction(self, distance: float, step: float = 25.0) -> float:
        """Fraction of a race of ``distance`` run on turns (the draw-bias exposure)."""
        pts = max(1, int(distance / step))
        ds = [distance * (i + 0.5) / pts for i in range(pts)]
        return sum(self.is_bend(d) for d in ds) / pts


# --------------------------------------------------------------------------- venues
# Happy Valley — turf, 1,450 m, right-handed, bend-dominated. Bend segments from the
# sectional sampling points: home straight ~[0,300]; Sanatorium/Crescent/The Rock
# bends ~[300,770]; back straight ~[770,920]; HKFC/Stable bends + top turn ~[920,1450].
HAPPY_VALLEY = TrackGeometry(
    name="Happy Valley",
    venue="HV",
    surface="turf",
    circumference=1450.0,
    handedness="right",
    courses={
        "A": (312.0, 30.5), "A+2": (310.0, 28.5),
        "B": (338.0, 26.5), "B+2": (338.0, 24.5), "B+3": (338.0, 23.5),
        "C": (334.0, 22.5), "C+3": (335.0, 19.5),
    },
    bend_segments=((300.0, 770.0), (920.0, 1450.0)),
)

# Sha Tin turf — 1,900 m galloping oval, right-handed, 430 m home straight. Long
# straights => a far weaker geometric draw bias than HV. Bend boundaries are a
# two-turn reconstruction from circumference + home straight (straights ~910 m total,
# two ~495 m bends); widths/home-straight are official.
SHA_TIN_TURF = TrackGeometry(
    name="Sha Tin (turf)",
    venue="ST",
    surface="turf",
    circumference=1900.0,
    handedness="right",
    courses={
        "A": (430.0, 30.5), "A+2": (430.0, 28.5),
        "B": (430.0, 26.0), "B+2": (430.0, 24.0),
        "C": (430.0, 21.3), "C+3": (430.0, 18.3),
    },
    bend_segments=((430.0, 925.0), (1405.0, 1900.0)),
)

# Sha Tin straight 1,000 m sprint chute — a separate course with NO bends, so no
# bend-geometry draw bias (only the watering/going gradient across the track, which
# this geometry deliberately does not model). circumference set to 1,000 so the
# (empty) bend map tiles trivially.
SHA_TIN_STRAIGHT = TrackGeometry(
    name="Sha Tin (1000m straight)",
    venue="ST",
    surface="turf",
    circumference=1000.0,
    handedness="straight",
    courses={"A": (1000.0, 30.5)},
    bend_segments=(),
)

# Sha Tin all-weather (dirt) — 1,560 m, right-handed, 365 m home straight, fixed
# single rail (uniform 22.8 m). Two-turn reconstruction as for the turf oval.
SHA_TIN_AWT = TrackGeometry(
    name="Sha Tin (all-weather)",
    venue="ST",
    surface="awt",
    circumference=1560.0,
    handedness="right",
    courses={"AWT": (365.0, 22.8)},
    bend_segments=((365.0, 795.0), (1130.0, 1560.0)),
)

TRACKS = {
    "HV": HAPPY_VALLEY,
    "ST": SHA_TIN_TURF,
    "ST_STRAIGHT": SHA_TIN_STRAIGHT,
    "ST_AWT": SHA_TIN_AWT,
}

_AWT_ALIASES = {"awt", "aw", "dirt", "all weather", "all-weather", "all weather track"}


def track_for(venue: str, distance: float | None = None, surface: str = "turf") -> TrackGeometry:
    """Resolve the geometry for a race from its ``venue``, ``distance`` and ``surface``.

    - ``HV`` → Happy Valley (turf only).
    - ``ST`` + all-weather surface → Sha Tin AWT.
    - ``ST`` + turf + 1000 m → the straight sprint chute (no bends).
    - ``ST`` + turf otherwise → the Sha Tin turf oval.

    ``surface`` accepts HKJC strings (``"Turf"`` / ``"AWT"`` / ``"ALL WEATHER TRACK"``).
    """
    v = (venue or "").strip().upper()
    s = (surface or "turf").strip().lower()
    if v == "HV":
        return HAPPY_VALLEY
    if v == "ST":
        if s in _AWT_ALIASES:
            return SHA_TIN_AWT
        if distance is not None and int(distance) == 1000:
            return SHA_TIN_STRAIGHT
        return SHA_TIN_TURF
    raise KeyError(f"unknown venue {venue!r} (expected 'HV' or 'ST')")
