"""Tests for walk-forward folds and the sealed lockbox."""
import pytest

from hkjc.backtesting import walkforward as wf
from hkjc.backtesting.walkforward import Lockbox, LockboxBreachError


DATES = [f"2026-01-{d:02d}" for d in range(1, 13)]   # 12 meeting-dates


def test_folds_respect_embargo_and_order():
    folds = wf.walk_forward_folds(DATES, n_folds=3, embargo=1, min_train=2)
    assert len(folds) == 3
    for f in folds:
        # train strictly precedes test, with an embargo gap of >= 1 meeting
        last_train = max(f.train)
        first_test = min(f.test)
        assert last_train < first_test
        gap = DATES.index(first_test) - DATES.index(last_train)
        assert gap >= 2  # embargo=1 means at least one skipped date between them
        # no leakage: no overlap
        assert not (set(f.train) & set(f.test))


def test_expanding_train_grows():
    folds = wf.walk_forward_folds(DATES, n_folds=3, embargo=1, min_train=2)
    sizes = [len(f.train) for f in folds]
    assert sizes == sorted(sizes) and sizes[0] < sizes[-1]


def test_rolling_window_caps_train():
    folds = wf.walk_forward_folds(DATES, n_folds=3, embargo=1, min_train=2, window=3)
    assert all(len(f.train) <= 3 for f in folds)


def test_insufficient_dates_raises():
    with pytest.raises(ValueError):
        wf.walk_forward_folds(["2026-01-01", "2026-01-02"], n_folds=2, embargo=1, min_train=2)


def test_split_lockbox_is_chronological():
    research, lock = wf.split_lockbox(DATES, fraction=0.25)
    assert len(lock) == 3 and len(research) == 9
    assert max(research) < min(lock)


def test_lockbox_opens_once(tmp_path):
    path = tmp_path / "lb.json"
    Lockbox.seal(["2026-02-01", "2026-02-02"], path)

    lb = Lockbox(path)
    assert not lb.is_opened
    assert lb.peek_status()["opens"] == 0          # peeking is free
    dates = lb.open(reason="promote model v3")
    assert dates == ["2026-02-01", "2026-02-02"]

    # a second open is a breach...
    with pytest.raises(LockboxBreachError):
        Lockbox(path).open(reason="oops")
    # ...unless explicitly forced (and the breach is logged)
    Lockbox(path).open(reason="harness fixed", force=True)
    assert len(Lockbox(path).opens) == 2


def test_lockbox_seal_no_overwrite(tmp_path):
    path = tmp_path / "lb.json"
    Lockbox.seal(["2026-02-01"], path)
    with pytest.raises(FileExistsError):
        Lockbox.seal(["2026-02-02"], path)
