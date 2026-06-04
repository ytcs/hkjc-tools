"""Tests for the research journal / memory."""
from hkjc.ledger import Ledger


def test_record_get_search_recent(tmp_path):
    ledger = Ledger(tmp_path)
    f = ledger.record(
        "CLV is the north-star metric",
        "Beating the closing line is the strongest leading indicator of edge.",
        kind="decision", tags=["clv", "evaluation"])
    assert f.id.endswith("clv-is-the-north-star-metric")
    assert ledger.get(f.id).title == f.title
    assert any(r.id == f.id for r in ledger.search("closing line"))
    assert any(r.id == f.id for r in ledger.by_tag("clv"))
    assert ledger.recent(1)[0].id == f.id


def test_search_ranks_title_over_body(tmp_path):
    ledger = Ledger(tmp_path)
    body_hit = ledger.record("Draw bias study", "An aside mentioning takeout once.", tags=[])
    title_hit = ledger.record("Takeout varies by pool", "Exotic pools skim more.", tags=["takeout"])
    res = ledger.search("takeout")
    assert res[0].id == title_hit.id
    assert {body_hit.id, title_hit.id} <= {r.id for r in res}


def test_supersede_marks_old(tmp_path):
    ledger = Ledger(tmp_path)
    a = ledger.record("Old belief", "x", kind="hypothesis")
    b = ledger.record("New belief", "y", kind="finding", supersedes=[a.id])
    assert ledger.get(a.id).status == "superseded"
    assert a.id in ledger.get(b.id).related


def test_status_update_and_filter(tmp_path):
    ledger = Ledger(tmp_path)
    f = ledger.record("Front-runner edge at HV", "needs testing", kind="hypothesis")
    ledger.set_status(f.id, "confirmed")
    assert ledger.get(f.id).status == "confirmed"
    assert ledger.search("", status="confirmed")[0].id == f.id


def test_reindex_roundtrip(tmp_path):
    ledger = Ledger(tmp_path)
    ledger.record("t1", "b1", tags=["x"])
    ledger.record("t2", "b2", tags=["y"])
    assert ledger.reindex() == 2
    assert (tmp_path / "index.jsonl").exists()
    assert (tmp_path / "INDEX.md").exists()
    # entries survive a fresh handle (durability)
    assert len(Ledger(tmp_path).all()) == 2
