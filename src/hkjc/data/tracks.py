"""Race-course geometry per venue x surface x course setting (:func:`track_for`).

HKJC turf tracks use a movable running rail: the race's ``Course`` (A, A+2, B, B+2, B+3, C, C+3)
shifts the rail out, changing the usable width (and slightly the home straight).

Coordinates: ``d`` = metres to the finish (a horse runs from d=distance down to 0); lateral = metres
outward from the current inner rail.

Happy Valley bend segments come from the HKJC sectional sampling points; the Sha Tin oval/AWT bend
boundaries are reconstructed from circumference + home straight (boundaries approximate, total
straight-vs-turn split not). Per-course widths and home straights are official HKJC figures.
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
        """Fraction of a race of ``distance`` run on turns."""
        pts = max(1, int(distance / step))
        ds = [distance * (i + 0.5) / pts for i in range(pts)]
        return sum(self.is_bend(d) for d in ds) / pts


# ----------------------------------------------------------- physical scale constants
THOROUGHBRED_LENGTH_M = 2.4    # nose-to-tail; the HKJC margin unit ("a length", ~8 ft)
HORSE_LANE_WIDTH_M = 1.10      # effective lateral lane a galloping horse+jockey occupies
                              # (barrel ~0.5-0.6 m + leg spread/safety) = min centre-to-centre

# Uniformly spaced stalls across the full track width; draw d sits in stall d (scratchings leave
# gaps, so this is the venue standard, not the field size). Maps draw -> start lateral.
BARRIER_STALLS = {"HV": 14, "ST": 14}


# --------------------------------------------------------------------------- venues
# Happy Valley turf. From the sectional sampling points: home straight ~[0,300]; Sanatorium/Crescent/
# The Rock bends ~[300,770]; back straight ~[770,920]; HKFC/Stable bends + top turn ~[920,1450].
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

# Sha Tin turf oval. Widths from the official course diagram (racing.hkjc.com/en-us/local/page/racing-course).
# Bends: two-turn reconstruction from circumference + home straight (straights ~910 m, two ~495 m bends).
SHA_TIN_TURF = TrackGeometry(
    name="Sha Tin (turf)",
    venue="ST",
    surface="turf",
    circumference=1900.0,
    handedness="right",
    courses={
        "A": (430.0, 30.5), "A+2": (430.0, 28.5), "A+3": (430.0, 27.5),
        "B": (430.0, 26.0), "B+2": (430.0, 24.0),
        "C": (430.0, 21.3), "C+3": (430.0, 18.3),
    },
    bend_segments=((430.0, 925.0), (1405.0, 1900.0)),
)

# Sha Tin 1,000 m straight chute: a separate course with no bends; circumference 1,000 so the empty
# bend map tiles trivially.
SHA_TIN_STRAIGHT = TrackGeometry(
    name="Sha Tin (1000m straight)",
    venue="ST",
    surface="turf",
    circumference=1000.0,
    handedness="straight",
    courses={"A": (1000.0, 30.5)},
    bend_segments=(),
)

# Sha Tin all-weather: fixed single rail (uniform 22.8 m); bends reconstructed as for the turf oval.
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
