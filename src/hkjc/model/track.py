"""Track geometry for the interacting (traffic-flow) race model — Happy Valley.

The draw bias is *geometric*: on a turn, an inner lane is a shorter path, so a low
draw that holds the rail runs less ground. HKJC also uses a **movable running rail**
— the race's ``Course`` (A, A+2, B, B+2, B+3, C, C+3) sets how far the rail is out,
changing the usable **width** (and slightly the home-straight length). So geometry
is parameterised by the course of the day, not fixed per venue.

Specs below are the official HKJC figures (racing_course page). Happy Valley is a
tight right-handed (clockwise) 1,450m circuit; the bend map comes from the
sectional sampling points — it is bend-dominated, which is why its draw bias is the
strongest in HK and makes it the right first testbed.

Coordinates: ``d`` = metres-from-finish (a horse runs from d=distance down to d=0).
Lateral offset is metres outward from the (current) inner rail; on a bend, being
``off`` metres wider adds ``off · turn_angle`` of arc — the ground-lost mechanism.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TrackGeometry:
    name: str
    venue: str
    circumference: float          # metres
    handedness: str               # "right" => clockwise; inner rail favours low draws
    courses: dict                 # course -> (home_straight_m, usable_width_m)
    bend_segments: tuple          # ((start_d, end_d), ...) metres-from-finish, within one lap, that are TURNS

    def width(self, course: str) -> float:
        """Usable track width (m) for a course setting; falls back to the A course."""
        c = self.courses.get(course) or self.courses.get(course.split("+")[0]) or self.courses["A"]
        return c[1]

    def is_bend(self, d_from_finish: float) -> bool:
        """Is position ``d`` (metres from finish) on a turn (tiled over laps)?"""
        dm = d_from_finish % self.circumference
        return any(a <= dm < b for a, b in self.bend_segments)

    def bend_fraction(self, distance: float, step: float = 25.0) -> float:
        """Fraction of a race of ``distance`` run on turns (the draw-bias exposure)."""
        pts = max(1, int(distance / step))
        ds = [distance * (i + 0.5) / pts for i in range(pts)]
        return sum(self.is_bend(d) for d in ds) / pts


# Official HKJC figures. Bend segments derived from the HV sampling points:
# home straight ~[0,300]; Sanatorium/Crescent/The Rock bends ~[300,770]; back
# straight ~[770,920]; HKFC/Stable bends + top turn ~[920,1450].
HAPPY_VALLEY = TrackGeometry(
    name="Happy Valley",
    venue="HV",
    circumference=1450.0,
    handedness="right",
    courses={
        "A": (312.0, 30.5), "A+2": (310.0, 28.5),
        "B": (338.0, 26.5), "B+2": (338.0, 24.5), "B+3": (338.0, 23.5),
        "C": (334.0, 22.5), "C+3": (335.0, 19.5),
    },
    bend_segments=((300.0, 770.0), (920.0, 1450.0)),
)

TRACKS = {"HV": HAPPY_VALLEY}
