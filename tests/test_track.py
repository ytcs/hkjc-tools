"""Tests for the Happy Valley track geometry."""
from hkjc.model.track import HAPPY_VALLEY as HV


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
