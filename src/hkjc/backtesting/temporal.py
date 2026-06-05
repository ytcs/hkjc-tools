"""``AsOfView`` — the enforced no-lookahead data accessor.

Leak-freedom by construction rather than by convention: wrap a frame with a cutoff time
and the only data you can reach is what existed strictly before it. Feature builders take
an :class:`AsOfView` (cutoff = the race's own date, or "now" live) so a future row is
*unreachable*, not merely un-selected. The strictly-prior expanding-mean features in
:mod:`hkjc.model.features` are the canonical thing this gates.
"""
from __future__ import annotations

from dataclasses import dataclass

import polars as pl


@dataclass(frozen=True)
class AsOfView:
    """A frame visible only up to (and excluding) ``cutoff`` on ``time_col``.

    Parameters
    ----------
    frame:    the full data frame (may contain rows from any time).
    cutoff:   the exclusive upper bound — only rows with ``time_col < cutoff`` are visible.
    time_col: the temporal ordering column (default ``"date"``).
    """

    frame: pl.DataFrame
    cutoff: object
    time_col: str = "date"

    def visible(self) -> pl.DataFrame:
        """Rows strictly before the cutoff — everything a model is allowed to use."""
        return self.frame.filter(pl.col(self.time_col) < self.cutoff)

    def at(self, cutoff) -> "AsOfView":
        """A new view at a different cutoff over the same frame/column."""
        return AsOfView(self.frame, cutoff, self.time_col)

    def __len__(self) -> int:
        return self.visible().height
