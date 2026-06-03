"""Walk-forward splitting and the sealed lockbox.

With <1000 races/year, the split discipline is what separates research from
self-deception (``docs/AGENT_ARCHITECTURE.md`` §3.4):

- **Walk-forward, never random.** Horses/jockeys/trainers/going persist across
  meetings, so a random split leaks the future. Train on the past, test on the
  strictly-later future, with an **embargo** gap between them.
- **Embargo measured in meeting-days**, not calendar days — the natural unit for
  racing and free of date-arithmetic ambiguity.
- **A sealed lockbox**: the final chronological slice the agent must not touch
  during search. :class:`Lockbox` enforces "read exactly once" with a persisted
  access log, so an accidental second peek is an error, not a silent leak.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np


@dataclass(frozen=True)
class Fold:
    """One walk-forward fold: training meeting-dates and the later test block."""

    index: int
    train: List
    test: List

    @property
    def train_span(self) -> Tuple:
        return (self.train[0], self.train[-1]) if self.train else (None, None)

    @property
    def test_span(self) -> Tuple:
        return (self.test[0], self.test[-1]) if self.test else (None, None)


def walk_forward_folds(
    dates: Sequence,
    n_folds: int = 5,
    embargo: int = 1,
    min_train: int = 1,
    window: Optional[int] = None,
) -> List[Fold]:
    """Build walk-forward folds over sorted unique meeting-dates.

    Parameters
    ----------
    dates:
        Meeting dates (any sortable type — ``date``, ``datetime`` or ISO strings).
    n_folds:
        Number of contiguous test blocks carved from the tail of the timeline.
    embargo:
        Meeting-days to drop *between* the end of train and the start of test
        (the gap that prevents leakage). ``embargo=1`` skips one meeting.
    min_train:
        Minimum number of training meeting-days before the first test block.
    window:
        If set, use a **rolling** train window of this many meeting-days instead of
        an **expanding** (anchored) one.
    """
    uniq = sorted(set(dates))
    d = len(uniq)
    start = min_train + embargo
    if start >= d:
        raise ValueError(
            f"not enough meeting-dates ({d}) for min_train={min_train} + embargo={embargo}")
    test_positions = list(range(start, d))
    blocks = [b for b in np.array_split(np.array(test_positions), n_folds) if len(b)]
    folds: List[Fold] = []
    for i, block in enumerate(blocks):
        test_start = int(block[0])
        train_end = test_start - embargo  # exclusive
        lo = 0 if window is None else max(0, train_end - window)
        train = uniq[lo:train_end]
        test = [uniq[j] for j in block]
        if train:
            folds.append(Fold(index=i, train=train, test=test))
    return folds


def split_lockbox(
    dates: Sequence, fraction: float = 0.2, min_lockbox: int = 1
) -> Tuple[List, List]:
    """Chronologically split dates into ``(research, lockbox)``.

    The most recent ``fraction`` of meeting-dates becomes the sealed lockbox; the
    earlier remainder is the research set used for walk-forward search.
    """
    uniq = sorted(set(dates))
    d = len(uniq)
    if d < 2:
        raise ValueError("need at least 2 meeting-dates to split a lockbox")
    k = max(min_lockbox, int(round(d * fraction)))
    k = min(k, d - 1)
    return uniq[: d - k], uniq[d - k:]


class LockboxBreach(RuntimeError):
    """Raised when sealed lockbox data is opened more than once without ``force``."""


class Lockbox:
    """A sealed hold-out set of meeting-dates that may be opened **once**.

    The point is the *discipline*: you fit and tune entirely on the research set,
    then open the lockbox a single time to confirm a promoted candidate. Every open
    is logged to a JSON state file, so a second open is a loud error rather than a
    silent leak that quietly invalidates the result.

    Construct fresh with :meth:`seal`, or reload an existing seal by passing the
    same ``path``.
    """

    def __init__(self, path: Union[str, Path]):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(
                f"no lockbox at {self.path}; create one with Lockbox.seal(dates, path)")
        self._state = json.loads(self.path.read_text(encoding="utf-8"))

    # ------------------------------------------------------------------ create
    @classmethod
    def seal(cls, dates: Sequence, path: Union[str, Path], now: Optional[datetime] = None) -> "Lockbox":
        """Seal ``dates`` into a new lockbox file. Refuses to overwrite an existing one."""
        path = Path(path)
        if path.exists():
            raise FileExistsError(f"lockbox already exists at {path}; refusing to reseal")
        path.parent.mkdir(parents=True, exist_ok=True)
        ts = (now or datetime.now(timezone.utc)).isoformat(timespec="seconds")
        state = {
            "dates": [str(x) for x in sorted(set(dates))],
            "sealed_at": ts,
            "opens": [],
        }
        path.write_text(json.dumps(state, indent=2), encoding="utf-8")
        return cls(path)

    # -------------------------------------------------------------------- read
    @property
    def opens(self) -> List[dict]:
        return list(self._state["opens"])

    @property
    def is_opened(self) -> bool:
        return len(self._state["opens"]) > 0

    @property
    def size(self) -> int:
        return len(self._state["dates"])

    def peek_status(self) -> dict:
        """Metadata only (sealed time, open count) — does NOT count as an open."""
        return {
            "path": str(self.path),
            "size": self.size,
            "sealed_at": self._state["sealed_at"],
            "opens": len(self._state["opens"]),
        }

    def open(self, reason: str, force: bool = False, now: Optional[datetime] = None) -> List[str]:
        """Reveal the held-out dates, logging the access. Raises on a second open.

        Pass ``force=True`` only with a deliberate, recorded reason (the breach is
        still logged) — e.g. a sanctioned re-evaluation after a harness fix.
        """
        if self.is_opened and not force:
            raise LockboxBreach(
                f"lockbox {self.path} already opened {len(self._state['opens'])}x "
                f"(first: {self._state['opens'][0]['reason']!r}); "
                "pass force=True to override and log a breach")
        ts = (now or datetime.now(timezone.utc)).isoformat(timespec="seconds")
        self._state["opens"].append({"at": ts, "reason": reason, "forced": bool(force)})
        self.path.write_text(json.dumps(self._state, indent=2), encoding="utf-8")
        return list(self._state["dates"])
