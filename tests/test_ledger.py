"""Tests for the research journal / memory."""
from hkjc.ledger import Ledger


def test_record_get_search_recent(tmp_path):
    L = Ledger(tmp_path)
    f = L.record(
        "CLV is the north-star metric",
        "Beating the closing line is the strongest leading indicator of edge.",
        kind="decision", tags=["clv", "evaluation"])
    assert f.id.endswith("clv-is-the-north-star-metric")
    assert L.get(f.id).title == f.title
    assert any(r.id == f.id for r in L.search("closing line"))
    assert any(r.id == f.id for r in L.by_tag("clv"))
    assert L.recent(1)[0].id == f.id


def test_search_ranks_title_over_body(tmp_path):
    L = Ledger(tmp_path)
    body_hit = L.record("Draw bias study", "An aside mentioning takeout once.", tags=[])
    title_hit = L.record("Takeout varies by pool", "Exotic pools skim more.", tags=["takeout"])
    res = L.search("takeout")
    assert res[0].id == title_hit.id
    assert {body_hit.id, title_hit.id} <= {r.id for r in res}


def test_supersede_marks_old(tmp_path):
    L = Ledger(tmp_path)
    a = L.record("Old belief", "x", kind="hypothesis")
    b = L.record("New belief", "y", kind="finding", supersedes=[a.id])
    assert L.get(a.id).status == "superseded"
    assert a.id in L.get(b.id).related


def test_status_update_and_filter(tmp_path):
    L = Ledger(tmp_path)
    f = L.record("Front-runner edge at HV", "needs testing", kind="hypothesis")
    L.set_status(f.id, "confirmed")
    assert L.get(f.id).status == "confirmed"
    assert L.search("", status="confirmed")[0].id == f.id


def test_reindex_roundtrip(tmp_path):
    L = Ledger(tmp_path)
    L.record("t1", "b1", tags=["x"])
    L.record("t2", "b2", tags=["y"])
    assert L.reindex() == 2
    assert (tmp_path / "index.jsonl").exists()
    assert (tmp_path / "INDEX.md").exists()
    # entries survive a fresh handle (durability)
    assert len(Ledger(tmp_path).all()) == 2
