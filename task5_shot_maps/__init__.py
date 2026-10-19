"""Task 5: reproducible NHL offensive-zone shot maps."""

from .coordinates import normalize_coordinates
from .rates import RateResult, compute_shot_rates

__all__ = ["RateResult", "compute_shot_rates", "normalize_coordinates"]
