import pytest

ort = pytest.importorskip("onnxruntime")


def test_no_spin_wraps_sessions_once(monkeypatch):
    from hkjc.recon.ortsession import no_spin

    seen = {}

    class Fake:
        def __init__(self, path, so=None, *a, **k):
            seen["spin"] = so.get_session_config_entry("session.intra_op.allow_spinning")

    monkeypatch.setattr(ort, "InferenceSession", Fake)
    try:
        from rapidocr_onnxruntime.utils import infer_engine

        monkeypatch.setattr(infer_engine, "InferenceSession", Fake)
    except ImportError:
        pass
    no_spin()
    no_spin()
    assert ort.InferenceSession.__bases__ == (Fake,)
    ort.InferenceSession("model.onnx", providers=["CPUExecutionProvider"])
    assert seen["spin"] == "0"
