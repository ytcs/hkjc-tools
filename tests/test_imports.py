"""Smoke test: the package imports cleanly and exposes its public API."""
import hkjc


def test_top_level_api():
    for name in [
        "core", "data", "model", "bets",
        "eval", "ledger", "RaceModel", "PlackettLuceModel", "ConditionalLogit",
        "RaceSimulator", "OddsTickStore", "FeatureStore", "DataSource", "Ledger",
        "pareto_filter", "generate_historical_data",
    ]:
        assert hasattr(hkjc, name), f"hkjc.{name} missing"


def test_canonical_core_paths():
    import hkjc.core.contracts
    import hkjc.core.race
    import hkjc.core.track
    import hkjc.core.simulation
    import hkjc.core.utils
    import hkjc.core.hkjc_facts
    assert hkjc.core.RaceModel is hkjc.core.contracts.RaceModel


def test_canonical_data_paths():
    import hkjc.data.live
    import hkjc.data.historical
    import hkjc.data.ticks
    import hkjc.data.processing
    import hkjc.data.feature_store
    import hkjc.data.racecard
    import hkjc.data.sectional
    import hkjc.data.jackpot_history
    assert hkjc.data.live.live_odds is not None


def test_eval_and_ledger_modules():
    import hkjc.eval.clv  # noqa: F401
    import hkjc.ledger.store  # noqa: F401


def test_model_package_paths():
    import hkjc.model.plackett_luce
    import hkjc.model.conditional_logit
    import hkjc.model.baselines
    import hkjc.model.calibration
    import hkjc.model.harville_model
    import hkjc.model.eom
    # RaceModel contract now lives in core; the model package re-exports it.
    assert hkjc.model.RaceModel is hkjc.core.contracts.RaceModel
