import math

import pytest

from task5_shot_maps.coordinates import (
    attacks_right,
    normalize_coordinates,
    rink_to_image_pixels,
)


@pytest.mark.parametrize(
    "is_home,side,expected",
    [(True, "left", True), (True, "right", False), (False, "left", False), (False, "right", True)],
)
def test_attack_direction_from_home_defending_side(is_home, side, expected):
    assert attacks_right(is_home, side) is expected


def test_rotation_reflects_both_axes_not_absolute_x_only():
    assert normalize_coordinates(-70, 10, is_home=True, home_defending_side="right") == (70, -10)
    assert normalize_coordinates(70, 10, is_home=True, home_defending_side="left") == (70, 10)


def test_missing_coordinates_and_orientation_are_rejected():
    with pytest.raises(ValueError):
        normalize_coordinates(None, 2, is_home=True, home_defending_side="left")
    with pytest.raises(ValueError):
        normalize_coordinates(60, 2, is_home=True, home_defending_side=None)


def test_explicit_inference_is_supported_and_auditable_by_caller():
    assert normalize_coordinates(-50, -4, is_home=False, home_defending_side=None,
                                 inferred_attacks_right=False) == (50, 4)


def test_rink_to_provided_image_pixel_coordinates():
    assert rink_to_image_pixels(-100, 42.5) == (0, 0)
    assert rink_to_image_pixels(100, -42.5) == (732, 309)
    center = rink_to_image_pixels(0, 0)
    assert math.isclose(center[0], 366)
    assert math.isclose(center[1], 154.5)
