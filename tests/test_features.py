"""The extracted leak-free ability features compute strictly-prior means (no leakage)."""
import polars as pl

from hkjc.model.features import (early_position_table, form_table, jockey_table,
                                 prior_starts_table, recency_form_table, style_table,
                                 trainer_table)


def test_form_is_strictly_prior_expanding_mean():
    # one horse, three dated runs: form on run k must use only runs < k.
    e = pl.DataFrame({
        "date": ["2025-01-01", "2025-01-08", "2025-01-15"],
        "venue": ["ST"] * 3, "race_no": [1, 1, 1], "runner": [1, 1, 1],
        "horse_id": ["H1"] * 3,
    })
    w = pl.DataFrame({
        "date": ["2025-01-01", "2025-01-08", "2025-01-15"],
        "venue": ["ST"] * 3, "race_no": [1, 1, 1], "runner": [1, 1, 1],
        "finish_pos": [1, 5, 1], "field_size": [5, 5, 5],
    })
    out = form_table(e, w).sort("date")
    forms = out["form"].to_list()
    # rf = 1 - (pos-1)/(field-1): run1 rf=1.0, run2 rf=0.0, run3 rf=1.0
    assert forms[0] is None                 # debut: no prior
    assert abs(forms[1] - 1.0) < 1e-9       # prior = {run1 rf=1.0}
    assert abs(forms[2] - 0.5) < 1e-9       # prior = mean(1.0, 0.0)


def test_prior_starts_is_strictly_prior_count():
    # one horse, three dated runs: prior_starts on run k must count only runs < k.
    # Leak-prone (a count must never include the current row) -> regression-guarded.
    e = pl.DataFrame({
        "date": ["2025-01-01", "2025-01-08", "2025-01-15"],
        "venue": ["ST"] * 3, "race_no": [1, 1, 1], "runner": [1, 1, 1],
        "horse_id": ["H1"] * 3,
    })
    w = pl.DataFrame({
        "date": ["2025-01-01", "2025-01-08", "2025-01-15"],
        "venue": ["ST"] * 3, "race_no": [1, 1, 1], "runner": [1, 1, 1],
        "finish_pos": [1, 5, 1], "field_size": [5, 5, 5],
    })
    out = prior_starts_table(e, w).sort("date")
    ps = out["prior_starts"].to_list()
    assert ps[0] == 0                       # debut: zero prior starts
    assert ps[1] == 1                       # one prior run
    assert ps[2] == 2                       # two prior runs


def test_jockey_strike_rate_is_prior_only():
    rr = pl.DataFrame({
        "date": ["2025-01-01", "2025-01-08", "2025-01-15"],
        "Jockey": ["J1"] * 3, "HorseID": ["H1", "H2", "H3"], "Pla": [1, 2, 1],
    })
    out = jockey_table(rr).sort("date")
    sr = out["jky_sr"].to_list()
    assert sr[0] is None                    # first ride: no prior
    assert abs(sr[1] - 1.0) < 1e-9          # prior {win} -> 1.0
    assert abs(sr[2] - 0.5) < 1e-9          # prior {win, loss} -> 0.5


def test_recency_form_is_strictly_prior():
    # the production form (exp-weighted). First run must be null; later runs use prior runs only,
    # weighting recent more. Leak-prone feature -> regression-guarded.
    e = pl.DataFrame({
        "date": ["2025-01-01", "2025-01-08", "2025-01-15"],
        "venue": ["ST"] * 3, "race_no": [1, 1, 1], "runner": [1, 1, 1], "horse_id": ["H1"] * 3,
    })
    w = pl.DataFrame({
        "date": ["2025-01-01", "2025-01-08", "2025-01-15"],
        "venue": ["ST"] * 3, "race_no": [1, 1, 1], "runner": [1, 1, 1],
        "finish_pos": [1, 5, 1], "field_size": [5, 5, 5],   # rf = 1.0, 0.0, 1.0
    })
    out = recency_form_table(e, w, half_life=2.0).sort("date")
    fr = out["form_rec"].to_list()
    assert fr[0] is None                        # debut: no prior
    assert abs(fr[1] - 1.0) < 1e-9              # only prior is run1 (rf=1.0)
    assert 0.0 < fr[2] < 1.0                    # exp-weighted blend of runs 1,2 (prior only)


def test_trainer_strike_rate_is_prior_only():
    rr = pl.DataFrame({
        "date": ["2025-01-01", "2025-01-08", "2025-01-15"],
        "Trainer": ["T1"] * 3, "HorseID": ["H1", "H2", "H3"], "Pla": [1, 2, 1],
    })
    out = trainer_table(rr).sort("date")
    sr = out["trn_sr"].to_list()
    assert sr[0] is None
    assert abs(sr[1] - 1.0) < 1e-9
    assert abs(sr[2] - 0.5) < 1e-9


def test_early_position_is_strictly_prior():
    # the positional-fade feature: trailing mean early position must use only prior runs.
    rp = pl.DataFrame({
        "date": ["2025-01-01", "2025-01-08", "2025-01-15"],
        "venue": ["ST"] * 3, "race_no": [1, 1, 1], "runner": [1, 1, 1],
        "horse_id": ["H1"] * 3, "early_pos_norm": [0.2, 0.8, 0.5],
    })
    out = early_position_table(rp).sort("date")
    ep = out["early_pos"].to_list()
    assert ep[0] is None                        # debut: no prior
    assert abs(ep[1] - 0.2) < 1e-9              # prior = {run1 0.2}
    assert abs(ep[2] - 0.5) < 1e-9              # mean(0.2, 0.8)


def test_style_trailing_mean_prior_only():
    s = pl.DataFrame({
        "date": ["01/01/2025", "08/01/2025", "15/01/2025"],
        "venue": ["ST"] * 3, "race_no": [1, 1, 1], "runner": [1, 1, 1],
        "horse_id": ["H1"] * 3, "rel_early": [0.2, 0.4, 0.9],
    })
    out = style_table(s).sort("date")
    st = out["style"].to_list()
    assert st[0] is None
    assert abs(st[1] - 0.2) < 1e-9
    assert abs(st[2] - 0.3) < 1e-9          # mean(0.2, 0.4)
