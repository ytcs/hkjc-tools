"""Race-course geometry per venue x surface x course setting (:func:`track_for`).

HKJC turf tracks use a movable running rail: the race's ``Course`` (A, A+2, B, B+2, B+3, C, C+3)
shifts the rail out, changing the usable width (and slightly the home straight).

Coordinates: ``d`` = metres to the finish (a horse runs from d=distance down to 0); lateral = metres
outward from the current inner rail.

Official (HKJC course page, racing.hkjc.com racing_course.asp): the per-course home straights and
widths, and the AWT's 365 m straight and 22.8 m width.

Measured: the turf bend segments and the Happy Valley circumference. A turn is where the rail's
curvature exceeds 1 mrad/m (radius under 1 km). Curvature is read from the rendered rail that
``hkjc.recon`` extracts (``rail.npz``) against the race's own distance-to-go metric (the render's
odometer; at the official 600 m point it agrees with the official time to 0.4 m, median). The sample is
193 reconstructed races from 2023-26: 91 at Happy Valley (courses A/B/C/C+3) and 102 on the Sha Tin
oval (A/A+3/B/B+2/C/C+3). Each boundary is the median over course settings of the per-course medians,
rounded to 5 m. Moving the threshold between 0.5 and 2 mrad/m moves the boundaries by at most 10 m.
The home bends rest on 78-91 races each. The far turns rest on fewer, because only long races render
them: the HV far-turn entry on 75 races and its end on 24. The Sha Tin far bend is run only from the
2,000 m+ starts. Its entry, 1,385 m, is the median over the 6 races of 2,000 m (the course-A median is
noise), and its end rests on 2 races.

Distances are on the odometer's line, about 4 m outside the rail. On the rail itself a boundary sits
4 m x (radians turned since the finish) closer to the finish: 11 m at the end of the home bend, 25 m
after a full lap. The rail moves out with the course setting, and the boundaries after the home bend
move with it (up to about 35 m between A and C+3). These per-venue segments do not model that.

Sha Tin turf starts of 1,200-1,800 m are on the back straight's extension and never run the far turn
(``SHA_TIN_TURF_BACK``). The AWT segments are a reconstruction from circumference + home straight; no
AWT race has been measured.
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


# --------------------------------------------------------------------------- venues
# Happy Valley turf, measured (see the module docstring):
# - home straight [0,315];
# - home bend [315,590], 2.8 rad (HKJC's Sanatorium Bend 450 and Crescent Bend 600 sampling points);
# - back straight [590,920] (The Rock 750 and Back Straight 900 lie on it);
# - HKFC + Stable bends [920,1480], 3.5 rad, one turn at 1 mrad/m running to the winning post. It eases to
#   about 2 mrad/m near 1,080-1,100 m, where the two bends meet.
# Circumference: the far turn ends at 1,456-1,501 m by course (median 1,478), and races over a lap meet the
# home bend again 1,480-1,505 m further out. So one lap on the odometer's line is about 1,480 m. The quoted
# ~1,450 m (not on the HKJC course page) plus 2*pi x 4 m for a line 4 m outside the rail gives 1,475 m.
# Not measured: the 1,000 and 1,200 m start areas (their rendered rails end before the start), so whether those
# starts sit on chutes is open; they are given the lap's segments.
HAPPY_VALLEY = TrackGeometry(
    name="Happy Valley",
    venue="HV",
    surface="turf",
    circumference=1480.0,
    handedness="right",
    courses={
        "A": (312.0, 30.5), "A+2": (310.0, 28.5),
        "B": (338.0, 26.5), "B+2": (338.0, 24.5), "B+3": (338.0, 23.5),
        "C": (334.0, 22.5), "C+3": (335.0, 19.5),
    },
    bend_segments=((315.0, 590.0), (920.0, 1480.0)),
)

# Sha Tin turf oval as run from the 2,000 m+ starts in front of the stands, measured (module docstring):
# home straight [0,430]; home bend [430,900], 3.1 rad; back straight [900,1385]; far (Stable) bend [1385,1900],
# 3.1 rad, ending at the winning post (measured end 1,896 m). Widths from the official course page
# (racing.hkjc.com/en-us/local/page/racing-course).
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
    bend_segments=((430.0, 900.0), (1385.0, 1900.0)),
)

# Sha Tin turf, 1,200-1,800 m. These starts are on the back straight's extension beyond the far bend; HKJC's start
# diagram puts the 1200-1800 m starts on the top straight. Measured on the 1,600 and 1,800 m races: the rail is
# straight (under 0.3 mrad/m) and supported from 900 m out to the end of the rendered rail (1,460-1,715 m). At those
# distances the 2,000 m races are already on the far bend, which starts at 1,385 m. Only the home bend is run.
SHA_TIN_TURF_BACK = TrackGeometry(
    name="Sha Tin (turf, back-straight start)",
    venue="ST",
    surface="turf",
    circumference=1900.0,
    handedness="right",
    courses=SHA_TIN_TURF.courses,
    bend_segments=((430.0, 900.0),),
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

# Sha Tin all-weather: fixed single rail (uniform 22.8 m). The bends are a reconstruction from circumference + home
# straight, not a measurement (no AWT race has an extracted rail).
SHA_TIN_AWT = TrackGeometry(
    name="Sha Tin (all-weather)",
    venue="ST",
    surface="awt",
    circumference=1560.0,
    handedness="right",
    courses={"AWT": (365.0, 22.8)},
    bend_segments=((365.0, 795.0), (1130.0, 1560.0)),
)

_AWT_ALIASES = {"awt", "aw", "dirt", "all weather", "all-weather", "all weather track"}


def track_for(venue: str, distance: float | None = None, surface: str = "turf") -> TrackGeometry:
    """Resolve the geometry for a race from its ``venue``, ``distance`` and ``surface``.

    - ``HV`` → Happy Valley (turf only).
    - ``ST`` + all-weather surface → Sha Tin AWT.
    - ``ST`` + turf + 1000 m → the straight sprint chute (no bends).
    - ``ST`` + turf, 1001-1800 m → the oval from a back-straight start (home bend only).
    - ``ST`` + turf otherwise (2000 m and over, or no distance) → the Sha Tin turf oval.

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
        if distance is not None and int(distance) <= 1800:
            return SHA_TIN_TURF_BACK
        return SHA_TIN_TURF
    raise KeyError(f"unknown venue {venue!r} (expected 'HV' or 'ST')")
