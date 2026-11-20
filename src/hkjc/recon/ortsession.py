"""ONNX Runtime sessions with intra-op spin-waiting off. Same thread count and work split, so outputs
are bit-identical; idle pool threads sleep instead of spinning (the spin was ~2-3x the useful CPU in
the avatar detector and the OCR recogniser)."""


def no_spin():
    """Make every later ``onnxruntime.InferenceSession`` (ultralytics' ONNX backend, RapidOCR) spin-free."""
    import onnxruntime as ort

    if getattr(ort.InferenceSession, "no_spin", False):
        return
    base = ort.InferenceSession

    class Session(base):
        no_spin = True

        def __init__(self, path_or_bytes, sess_options=None, *args, **kw):
            so = sess_options if sess_options is not None else ort.SessionOptions()
            so.add_session_config_entry("session.intra_op.allow_spinning", "0")
            super().__init__(path_or_bytes, so, *args, **kw)

    ort.InferenceSession = Session
    try:  # RapidOCR binds the class at import
        from rapidocr_onnxruntime.utils import infer_engine

        infer_engine.InferenceSession = Session
    except ImportError:
        pass
