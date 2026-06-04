"""Tests for race-course geometry and the venue/surface/distance resolver."""
import pytest

from hkjc.track import (HAPPY_VALLEY as HV, SHA_TIN_AWT, SHA_TIN_STRAIGHT,
                        SHA_TIN_TURF, track_for)


def test_hv_specs_and_movable_rail():
    assert HV.circumference == 1450.0 and HV.handedness == "right"
    assert HV.width("A") == 30.5 and HV.width("C+3") == 19.5   # rail out -> narrower
    assert HV.width("C") == 22.5


def test_hv_bend_map_is_turn_dominated():
    assert HV.is_bend(150) is False    # home straight
    assert HV.is_bend(500) is True     # Crescent bend
    assert HV.is_bend(850) is False    # back straight
    assert HV.is_bend(1000) is True    # HKFC / stable bend
    # a 1200m race is run mostly on turns -> strong, persistent draw exposure
    assert HV.bend_fraction(1200) > 0.55


def test_sha_tin_turf_specs():
    assert SHA_TIN_TURF.circumference == 1900.0 and SHA_TIN_TURF.handedness == "right"
    assert SHA_TIN_TURF.home_straight("A") == 430.0
    assert SHA_TIN_TURF.width("A") == 30.5 and SHA_TIN_TURF.width("C+3") == 18.3


def test_sha_tin_has_weaker_draw_exposure_than_hv():
    # long galloping straights => less of the trip on turns than tight HV
    assert SHA_TIN_TURF.bend_fraction(1200) < HV.bend_fraction(1200)


def test_sha_tin_straight_chute_has_no_bends():
    assert SHA_TIN_STRAIGHT.bend_segments == ()
    assert SHA_TIN_STRAIGHT.bend_fraction(1000) == 0.0   # no geometric draw bias


def test_awt_is_fixed_single_rail():
    assert SHA_TIN_AWT.surface == "awt"
    assert SHA_TIN_AWT.width("AWT") == 22.8
    assert SHA_TIN_AWT.width("anything") == 22.8         # single configuration


def test_track_for_resolves_venue_surface_distance():
    assert track_for("HV", 1200) is HV
    assert track_for("ST", 1400) is SHA_TIN_TURF
    assert track_for("ST", 1000) is SHA_TIN_STRAIGHT     # 1000m turf is the chute
    assert track_for("ST", 1650, surface="AWT") is SHA_TIN_AWT
    assert track_for("ST", 1200, surface="ALL WEATHER TRACK") is SHA_TIN_AWT
    with pytest.raises(KeyError):
        track_for("XX", 1200)
