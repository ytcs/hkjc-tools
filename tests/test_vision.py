"""The vision package imports with NO heavy CV deps installed (lazy-load contract)."""


def test_vision_imports_without_cv_deps():
    import hkjc.vision as v
    # schema dataclasses are pure stdlib and exported eagerly
    from hkjc.vision import Detection, RaceTrajectory, RunnerTrajectory, TrajPoint
    assert len(v.__all__) == 9
    # the stage modules import clean (heavy deps are lazy, inside functions)
    import hkjc.vision.detect
    import hkjc.vision.homography
    import hkjc.vision.pipeline  # noqa: F401
    # a schema dataclass is constructible
    p = TrajPoint(t=0.0, d=1200.0, off=0.5, speed=16.0)
    assert p.d == 1200.0 and Detection and RaceTrajectory and RunnerTrajectory
