"""The vision package imports with NO heavy CV deps installed (lazy-load contract).

Perception submodules (silk/identity/feeds/...) are covered by
``test_vision_recon_imports.py``; this file pins the eagerly-exported output schema.
"""


def test_vision_schema_imports_without_cv_deps():
    import hkjc.vision as v
    # schema dataclasses are pure stdlib and exported eagerly
    from hkjc.vision import Detection, RaceTrajectory, RunnerTrajectory, TrajPoint
    assert len(v.__all__) == 9
    # a schema dataclass is constructible
    p = TrajPoint(t=0.0, d=1200.0, off=0.5, speed=16.0)
    assert p.d == 1200.0 and Detection and RaceTrajectory and RunnerTrajectory
