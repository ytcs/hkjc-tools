"""Smoke test: the package and its re-exports import cleanly after the reorg."""
import hkjc


def test_top_level_api():
    for name in [
        "data", "live", "historical", "speedpro", "features",
        "harville_model", "model", "simulation", "eval", "ledger",
        "RaceModel", "PlackettLuceModel", "RaceSimulator", "OddsTickStore",
        "Ledger", "pareto_filter", "speedpro_energy", "speedmap",
        "generate_historical_data",
    ]:
        assert hasattr(hkjc, name), f"hkjc.{name} missing"


def test_canonical_data_paths():
    import hkjc.data.live
    import hkjc.data.historical
    import hkjc.data.speedpro
    import hkjc.data.odds_ticks
    import hkjc.data.processing
    assert hkjc.live is hkjc.data.live


def test_eval_and_ledger_modules():
    import hkjc.eval.clv  # noqa: F401
    import hkjc.ledger.store  # noqa: F401
