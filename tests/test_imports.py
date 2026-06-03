"""Smoke test: the package and its re-exports import cleanly after the reorg."""
import hkjc


def test_top_level_api():
    for name in [
        "data", "live", "historical", "speedpro", "features",
        "harville_model", "model", "simulation", "eval", "ledger",
        "RaceModel", "PlackettLuceModel", "ConditionalLogit", "RaceSimulator",
        "OddsTickStore", "FeatureStore", "DataSource", "Ledger", "pareto_filter",
        "speedpro_energy", "speedmap", "generate_historical_data",
    ]:
        assert hasattr(hkjc, name), f"hkjc.{name} missing"


def test_canonical_data_paths():
    import hkjc.data.live
    import hkjc.data.historical
    import hkjc.data.speedpro
    import hkjc.data.odds_ticks
    import hkjc.data.processing
    import hkjc.data.source
    import hkjc.data.feature_store
    assert hkjc.live is hkjc.data.live


def test_eval_and_ledger_modules():
    import hkjc.eval.clv  # noqa: F401
    import hkjc.ledger.store  # noqa: F401


def test_model_package_paths():
    import hkjc.model.base
    import hkjc.model.plackett_luce
    import hkjc.model.conditional_logit
    import hkjc.model.baselines
    # RaceModel re-exported from the package root (back-compat after the split)
    assert hkjc.model.RaceModel is hkjc.model.base.RaceModel
