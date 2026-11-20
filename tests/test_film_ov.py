import pytest


def test_fp32_compile_injects_precision_and_restores(monkeypatch):
    ov = pytest.importorskip("openvino")
    from hkjc.recon import film

    seen = []

    class Core:
        def compile_model(self, model, device_name=None, config=None):
            seen.append((device_name, dict(config or {})))

    monkeypatch.setattr(ov, "Core", Core)
    with film.fp32_compile():
        Core().compile_model("m", device_name="GPU", config={"PERFORMANCE_HINT": "LATENCY"})
    Core().compile_model("m", device_name="GPU")
    assert seen[0] == ("GPU", {"PERFORMANCE_HINT": "LATENCY", "INFERENCE_PRECISION_HINT": ov.Type.f32})
    assert seen[1] == ("GPU", {})


def test_film_device_missing_raises(monkeypatch):
    ov = pytest.importorskip("openvino")
    from hkjc.recon import film

    class Core:
        available_devices = ["CPU"]

    monkeypatch.setattr(ov, "Core", Core)
    monkeypatch.setenv("HKJC_FILM_DEVICE", "intel:gpu")
    with pytest.raises(RuntimeError, match="GPU not available"):
        film._detector()
