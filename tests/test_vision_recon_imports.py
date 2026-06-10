"""The promoted 4-feed reconstruction modules import with NO heavy CV deps (lazy-load contract).

Mirrors ``tests/test_vision.py``: importing these modules must not pull cv2 / av / rapidocr /
scipy (those are lazy, inside functions). Pure numpy is permitted.
"""


def test_recon_modules_import_without_cv_deps():
    # shared primitives
    import hkjc.vision.geometry_cv as geom
    import hkjc.vision.ground_contact as gc
    import hkjc.vision.ocr as ocr
    # identity + per-feed observation models
    import hkjc.vision.identity as ident
    from hkjc.vision.feeds import FeedContext
    import hkjc.vision.feeds.aerial  # noqa: F401
    import hkjc.vision.feeds.aerial_detect as adet
    import hkjc.vision.feeds.aerial_rail as arail
    import hkjc.vision.aerial_layout as alayout
    import hkjc.vision.feeds.patrol  # noqa: F401
    import hkjc.vision.feeds.drone  # noqa: F401
    import hkjc.vision.feeds.rf  # noqa: F401

    # public names are present
    assert geom.FixedRail and geom.RailSpline and geom.robust_polyfit
    # promoted aerial primitives: dialed-in detector + field-selected rigid rail curve
    assert geom.candidate_rail_curves and geom.fit_rail_curve and geom.project_to_polyline
    assert adet.detect_edgefill and adet.AerialBlob
    # rail primitives + the promoted reconstruction layer (clean -> temporal fit -> global kappa)
    assert arail.line_candidates and arail.enumerate_lines and arail.select_running_rail
    assert arail.clean_pixels and arail.track_sequence and arail.fit_rails
    assert arail.build_kappa and arail.backproject_fixed_r and arail.radius_at
    assert arail.extract_rails           # top-level one-video pipeline (clean -> temporal fit)
    assert alayout.find_layout and alayout.Layout and alayout.render_active and alayout.content_box
    assert gc.shadow_base and gc.plumb_to_hoof
    assert ocr.ocr_numbers and ocr.best_per_number
    assert ident.Identity and ident.build_rf_anchor and ident.augment_silk_lut_from_patrol
    assert FeedContext
    # the fixed consensus weights are the validated values
    assert (ident.W_OWN_SILK, ident.W_RF_SILK, ident.W_ORDER, ident.W_DRAW) == (1.00, 0.45, 0.60, 0.70)


def test_feed_context_is_frozen_and_pure():
    from hkjc.vision.feeds import FeedContext

    ctx = FeedContext(video="x.mp4", distance=1650.0, track_width_m=22.5, field_size=12)
    assert ctx.distance == 1650.0 and ctx.field_size == 12
    try:
        ctx.distance = 1.0  # frozen -> must raise
        raised = False
    except Exception:
        raised = True
    assert raised


def test_identity_consensus_runs_on_a_tiny_synthetic_case():
    """A pure-numpy smoke test of consensus_assign (scipy is lazy but the inputs are numpy)."""
    import numpy as np

    from hkjc.vision.identity import Identity

    # two well-separated silks -> a correct one-to-one assignment
    sig_a = np.zeros(18)
    sig_a[2] = 1.0
    sig_b = np.zeros(18)
    sig_b[10] = 1.0
    ident = Identity(silk_lut={1: sig_a, 2: sig_b}, draw={1: 1, 2: 2}, field_size=2)
    out = ident.consensus_assign(
        track_sig={10: sig_a, 20: sig_b},
        track_order={10: 1.0, 20: 0.0},
        track_lat={10: 1.0, 20: 5.0},
        track_race_t={10: 5.0, 20: 5.0},
    )
    assert out[10][0] == 1 and out[20][0] == 2
