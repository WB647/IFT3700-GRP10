import numpy as np
import pandas as pd
import pytest

from task5_shot_maps.rates import compute_shot_rates, gaussian_smooth_preserve_sum


def _games():
    return pd.DataFrame(
        [
            {"game_id": 1, "home_team": "AAA", "away_team": "BBB"},
            {"game_id": 2, "home_team": "AAA", "away_team": "CCC"},
        ]
    )


def _shots():
    return pd.DataFrame(
        [
            {"team": "AAA", "x_attack": 30.0, "y_attack": 0.0},
            {"team": "AAA", "x_attack": 30.0, "y_attack": 0.0},
            {"team": "BBB", "x_attack": 30.0, "y_attack": 0.0},
            {"team": "CCC", "x_attack": np.nan, "y_attack": 0.0},
            {"team": "CCC", "x_attack": -70.0, "y_attack": 0.0},
        ]
    )


def test_spatial_counting_league_formula_and_team_excess():
    result = compute_shot_rates(
        _shots(), _games(), x_edges=np.array([25, 35, 45]), y_edges=np.array([-5, 5])
    )
    assert result.league_games == 2
    assert result.league_counts[0, 0] == 3
    assert result.league_rate[0, 0] == pytest.approx(3 / (2 * 2))
    assert result.team_counts["AAA"][0, 0] == 2
    assert result.team_rates["AAA"][0, 0] == pytest.approx(2 / 2)
    assert result.excess_rates["AAA"][0, 0] == pytest.approx(1 - 0.75)
    assert result.excess_rates["CCC"][0, 0] == pytest.approx(-0.75)


def test_game_denominators_come_from_participation_not_shot_rows():
    result = compute_shot_rates(
        _shots(), _games(), x_edges=np.array([25, 35]), y_edges=np.array([-5, 5])
    )
    assert result.team_games == {"AAA": 2, "BBB": 1, "CCC": 1}
    assert "CCC" in result.team_rates


def test_missing_and_outside_coordinates_are_audited_separately():
    result = compute_shot_rates(
        _shots(), _games(), x_edges=np.array([25, 35]), y_edges=np.array([-5, 5])
    )
    assert result.outside_display_zone == 1
    assert result.league_counts.sum() == 3


def test_zero_game_input_cannot_divide_by_zero():
    with pytest.raises(ValueError, match="At least one unique game"):
        compute_shot_rates(_shots(), _games().iloc[0:0])


def test_computation_is_reproducible():
    first = compute_shot_rates(_shots(), _games())
    second = compute_shot_rates(_shots(), _games())
    np.testing.assert_array_equal(first.league_rate, second.league_rate)
    for team in first.excess_rates:
        np.testing.assert_array_equal(first.excess_rates[team], second.excess_rates[team])


def test_gaussian_smoothing_spreads_values_and_preserves_signed_sum():
    values = np.zeros((7, 7))
    values[0, 0] = 2.0
    values[3, 3] = -0.5
    smoothed = gaussian_smooth_preserve_sum(values, sigma_bins=1.0, radius_bins=3)
    assert smoothed.sum() == pytest.approx(values.sum(), abs=1e-12)
    assert 0 < smoothed[0, 0] < 2.0
    assert np.count_nonzero(smoothed) > 2


@pytest.mark.parametrize("sigma,radius", [(0, 3), (-1, 3), (1, 0)])
def test_gaussian_smoothing_rejects_invalid_parameters(sigma, radius):
    with pytest.raises(ValueError):
        gaussian_smooth_preserve_sum(np.ones((2, 2)), sigma_bins=sigma, radius_bins=radius)
