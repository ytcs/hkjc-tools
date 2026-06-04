"""Smoke test: the package imports cleanly and exposes its public API."""
import hkjc


def test_top_level_api():
    for name in [
        "data", "harville_model", "model", "simulation", "bets",
        "eval", "ledger", "RaceModel", "PlackettLuceModel", "ConditionalLogit",
        "RaceSimulator", "OddsTickStore", "FeatureStore", "DataSource", "Ledger",
        "pareto_filter", "generate_historical_data",
    ]:
        assert hasattr(hkjc, name), f"hkjc.{name} missing"


def test_canonical_data_paths():
    import hkjc.data.live
    import hkjc.data.historical
    import hkjc.data.odds_ticks
    import hkjc.data.processing
    import hkjc.data.source
    import hkjc.data.feature_store
    import hkjc.data.racecard
    import hkjc.data.sectional
    import hkjc.data.jackpot_history
    assert hkjc.data.live.live_odds is not None


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
