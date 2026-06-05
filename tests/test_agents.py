"""The multi-agent journal routes inboxes + threads; prompts carry the philosophy."""
from hkjc.agents import prompts, state
from hkjc.agents.journal import Journal, OpsLog


def test_inbox_routes_by_audience_and_excludes_self(tmp_path):
    j = Journal(root=tmp_path)
    j.record("draw still over-weighted", "t=-17", author="trading",
             channel="handoff", audience=["modelling"], kind="observation")
    j.record("kelly cap too tight", "frictions", author="trading",
             channel="handoff", audience=["strategist"])
    j.record("note to everyone", "broadcast", author="strategist", audience=["all"])

    mbox = j.inbox("modelling")
    titles = {e.title for e in mbox}
    assert "draw still over-weighted" in titles      # addressed to modelling
    assert "kelly cap too tight" not in titles        # addressed to strategist only
    assert "note to everyone" in titles               # all-fan-out reaches modelling
    # an agent never sees its own posts in its inbox
    assert all(e.author != "modelling" for e in mbox)


def test_thread_links_replies(tmp_path):
    j = Journal(root=tmp_path)
    parent = j.record("mass mechanism lead", "bodywt t=+9", author="modelling")
    j.reply(parent.id, "confirmed in sim", "mass slows", author="modelling", kind="result")
    chain = j.thread(parent.id)
    assert len(chain) == 2 and chain[0].id == parent.id


def test_status_and_supersede(tmp_path):
    j = Journal(root=tmp_path)
    old = j.record("FORM_GAIN should be 2", "stale", author="modelling")
    j.record("FORM_GAIN should be 4", "recalibrated", author="modelling", supersedes=[old.id])
    assert j.get(old.id).status == "superseded"


def test_ops_log_roundtrip(tmp_path):
    log = OpsLog(root=tmp_path / "ops", stream="paper_fills")
    log.append([{"race_id": "r1", "runner": 3, "stake": 10.0, "odds": 4.5}])
    out = log.read()
    assert out.height == 1 and out["stake"][0] == 10.0
    assert "logged_at" in out.columns


def test_prompts_carry_the_philosophy():
    text = prompts.load("modelling")
    for must in ["fit the market first", "goodness-of-fit", "private-info",
                 "mechanism", "diagnose"]:
        assert must.lower() in text.lower(), f"modelling prompt missing: {must}"
    assert "consume" in prompts.load("strategist").lower()
    assert "paper" in prompts.load("trading").lower()


def test_state_read_stub(tmp_path):
    assert "Strategy state" in state.read("strategist", root=tmp_path)
    state.write("trading", "# Trading state\n\nactive: kelly\n", root=tmp_path)
    assert "active: kelly" in state.read("trading", root=tmp_path)
