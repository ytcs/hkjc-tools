"""Geometry + draw bridge for race reconstruction.

Sources all track geometry (width, bend map, distance) and the barrier draw from the
canonical :class:`hkjc.core.Race`, so the CV reconstruction never re-encodes track
facts. The draw -> start-lateral map is the pre-race INITIAL CONDITION that anchors the
fusion at ``t=0`` (philosophy-clean: same class as distance/weight, not an outcome).

Coordinate note: the reconstruction works in ``arclen`` = metres run FROM THE START
(0 at the gate, up to ``distance`` at the line); ``core`` geometry is in ``d`` = metres
FROM THE FINISH. They are related by ``d = distance - arclen``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core.race import Race
from ..core.track import BARRIER_STALLS


def draw_to_start_lateral(draw: dict[int, int], width_m: float, n_stalls: int) -> dict[int, float]:
    """Map each runner's barrier DRAW -> its start lateral ``y_lat(t=0)`` in metres from the rail.

    The barrier stalls are uniformly spaced across the full width; the field sits one horse
    per stall, so ``stall_width = width / n_stalls`` and
    ``y_lat(draw d) = 0.5*stall_width + (d-1)*stall_width`` (stall 1's centre is half a stall
    off the rail; each higher draw steps out one stall).
    """
    sw = width_m / n_stalls
    return {n: 0.5 * sw + (d - 1) * sw for n, d in draw.items()}


@dataclass(frozen=True)
class RaceGeom:
    """Geometry the reconstruction needs, all resolved from a :class:`hkjc.core.Race`."""

    race: Race

    @property
    def distance(self) -> float:
        return self.race.distance

    @property
    def width(self) -> float:
        """Usable track width (m) for this race's venue x course."""
        return self.race.width

    @property
    def draw(self) -> dict[int, int]:
        return {r.no: r.draw for r in self.race.runners}

    def is_bend(self, arclen: float) -> bool:
        """Is the given arclen (m from start) on a turn?"""
        return self.race.geometry.is_bend(self.distance - arclen)

    def bend_weight(self, arclen, *, straight: float = 1.0, bend: float = 0.0):
        """Per-arclen reliability weight for the aerial lateral METRIC: full on straights,
        down-weighted on bends where render foreshortening degrades it. Scalar or array in."""
        a = np.atleast_1d(np.asarray(arclen, dtype=float))
        mask = np.array([self.race.geometry.is_bend(self.distance - x) for x in a])
        out = np.where(mask, bend, straight)
        return out if out.size > 1 else float(out[0])

    def draw_to_metres(self, n_stalls: int | None = None) -> dict[int, float]:
        """Start lateral ``y_lat(0)`` per runner from the draw + this venue's barrier-stall count."""
        n = n_stalls or BARRIER_STALLS.get(self.race.venue, self.race.n)
        return draw_to_start_lateral(self.draw, self.width, n)
