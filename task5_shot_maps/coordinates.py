"""Coordinate orientation and validation for NHL play-by-play events."""

from __future__ import annotations

import math
from typing import Optional


RINK_X_LIMIT = 100.0
RINK_Y_LIMIT = 42.5
OFFENSIVE_BLUE_LINE = 25.0


def attacks_right(is_home: bool, home_defending_side: str) -> bool:
    """Return whether the event owner's team attacks toward positive x.

    ``homeTeamDefendingSide`` describes the home team's own end. Therefore,
    the home team attacks right when it defends left, while the away team
    attacks right when the home team defends right.
    """

    side = str(home_defending_side).strip().lower()
    if side not in {"left", "right"}:
        raise ValueError(f"Unknown home defending side: {home_defending_side!r}")
    return (is_home and side == "left") or (not is_home and side == "right")


def normalize_coordinates(
    x: object,
    y: object,
    *,
    is_home: bool,
    home_defending_side: Optional[str],
    inferred_attacks_right: Optional[bool] = None,
) -> tuple[float, float]:
    """Orient a coordinate so the shooting team attacks toward positive x.

    The y-axis is also reflected when play is rotated by 180 degrees. This
    preserves left/right relative to the attacking team rather than merely
    applying ``abs(x)``. Missing orientation may use a separately inferred
    direction supplied by the caller. Invalid or out-of-rink coordinates
    raise ``ValueError`` so exclusions can be counted explicitly.
    """

    try:
        x_value = float(x)
        y_value = float(y)
    except (TypeError, ValueError) as exc:
        raise ValueError("Coordinates must be numeric") from exc

    if not (math.isfinite(x_value) and math.isfinite(y_value)):
        raise ValueError("Coordinates must be finite")
    if abs(x_value) > RINK_X_LIMIT or abs(y_value) > RINK_Y_LIMIT:
        raise ValueError(f"Coordinate outside NHL rink: ({x_value}, {y_value})")

    if home_defending_side is not None and str(home_defending_side).strip():
        direction = attacks_right(is_home, str(home_defending_side))
    elif inferred_attacks_right is not None:
        direction = bool(inferred_attacks_right)
    else:
        raise ValueError("Attack direction is unavailable")

    if direction:
        return x_value, y_value
    return -x_value, -y_value


def rink_to_image_pixels(
    x: float,
    y: float,
    *,
    width: int = 733,
    height: int = 310,
) -> tuple[float, float]:
    """Convert NHL feet to pixels for the provided full-rink image.

    The image covers x=[-100, 100] and y=[-42.5, 42.5]. Image y grows
    downward, hence the vertical inversion. Plotly uses physical coordinates
    directly; this helper makes the supplied-image convention testable.
    """

    if width < 2 or height < 2:
        raise ValueError("Image dimensions must be at least 2 by 2 pixels")
    x_value = float(x)
    y_value = float(y)
    if abs(x_value) > RINK_X_LIMIT or abs(y_value) > RINK_Y_LIMIT:
        raise ValueError("Coordinate outside NHL rink")
    pixel_x = (x_value + RINK_X_LIMIT) / (2 * RINK_X_LIMIT) * (width - 1)
    pixel_y = (RINK_Y_LIMIT - y_value) / (2 * RINK_Y_LIMIT) * (height - 1)
    return pixel_x, pixel_y
