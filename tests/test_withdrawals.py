import numpy as np
import polars as pl

from hkjc.data.historical import _clean_race_data
from hkjc.recon import core


def _facts(numbers, row):
    """The list's facts on one row, as (ahead number, behind number, unlisted)."""
    L = core.list_rows(np.array([[0, *row]]), lambda f: np.asarray(f, float), numbers)
    St = core.statements(L, len(numbers))
    num = np.asarray(numbers)
    return L, [(int(num[a]), int(num[b]), bool(u)) for a, b, u in zip(St.a, St.b, St.unl)]


def test_list_facts_with_withdrawn_numbers_match_the_renumbered_field():
    """A field with withdrawn numbers (gaps) states the same facts as the same field numbered 1..N: columns are
    runners, not saddlecloth numbers, and no fact names a withdrawn number."""
    gapped = [1, 3, 5, 6, 7, 8, 9, 10, 11, 12]  # #2 and #4 withdrawn
    dense = list(range(1, 11))
    La, fa = _facts(gapped, gapped[:8])
    Lb, fb = _facts(dense, dense[:8])
    assert La.full.all() and Lb.full.all()
    ren = dict(zip(gapped, dense))
    assert [(ren[a], ren[b], u) for a, b, u in fa] == fb
    assert sum(u for *_x, u in fa) == 2  # the two unlisted runners, behind the 8th
    assert not {2, 4} & {x for a, b, _u in fa for x in (a, b)}


def test_a_withdrawn_number_in_the_list_states_nothing_about_it_and_no_unlisted_fact():
    numbers = [1, 3, 5, 6, 7, 8, 9, 10, 11, 12]
    L, f = _facts(numbers, [1, 3, 5, 6, 7, 8, 9, 2])  # the list shows a withdrawn number in slot 8
    assert not L.full.any()  # the row is not fully read as runners: it says nothing about the unlisted
    assert all(not u for *_x, u in f) and len(f) == 6  # the 6 adjacent pairs among slots 1-7
    assert all(2 not in (a, b) for a, b, _u in f)


def test_clean_race_data_drops_withdrawn_rows_and_keeps_the_numbers():
    """Withdrawn runners (WV-A, HV 2026-04-15 R6) are dropped; the remaining saddlecloth numbers keep their gaps."""
    raw = pl.DataFrame(
        {
            "Pla": ["1", "2", "3", "WV-A", "WV-A"],
            "HorseNo": ["3", "9", "1", "2", "4"],
            "ActWt": ["120", "126", "135", "130", "128"],
            "DeclarHorseWt": ["1100", "1050", "1150", "1080", "1090"],
            "Dr": ["11", "9", "6", "---", "---"],
            "WinOdds": ["4.1", "6.2", "3.3", "---", "---"],
            "FinishTime": ["1:09.76", "1:09.92", "1:10.00", "---", "---"],
        }
    )
    out = _clean_race_data(raw)
    assert out["HorseNo"].to_list() == [3, 9, 1] and out["Dr"].to_list() == [11, 9, 6]
