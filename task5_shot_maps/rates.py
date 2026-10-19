"""Spatial shot counts and per-team-hour excess-rate calculations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


DEFAULT_X_EDGES = np.arange(25.0, 100.0 + 5.0, 5.0)
DEFAULT_Y_EDGES = np.arange(-42.5, 42.5 + 5.0, 5.0)


def gaussian_smooth_preserve_sum(
    values: np.ndarray,
    *,
    sigma_bins: float = 1.0,
    radius_bins: int = 3,
) -> np.ndarray:
    """Spread each bin over its neighbours with a truncated Gaussian kernel.

    Weights are renormalized separately for every source bin after clipping
    the kernel at rink boundaries. Consequently no rate leaks outside the
    offensive-zone grid and the array sum is preserved (up to floating-point
    precision). Units remain shots per team-hour distributed across 5x5-foot
    bins. This operation is linear, so smoothing an excess is equivalent to
    smoothing team and league rates separately and subtracting them.
    """

    array = np.asarray(values, dtype=float)
    if array.ndim != 2 or not np.isfinite(array).all():
        raise ValueError("Smoothing requires a finite two-dimensional array")
    if sigma_bins <= 0:
        raise ValueError("sigma_bins must be positive")
    if radius_bins < 1:
        raise ValueError("radius_bins must be at least 1")

    offsets = np.arange(-radius_bins, radius_bins + 1)
    one_dimensional = np.exp(-0.5 * (offsets / sigma_bins) ** 2)
    kernel = np.outer(one_dimensional, one_dimensional)
    smoothed = np.zeros_like(array, dtype=float)
    rows, columns = array.shape

    for source_x in range(rows):
        target_x0 = max(0, source_x - radius_bins)
        target_x1 = min(rows, source_x + radius_bins + 1)
        kernel_x0 = target_x0 - (source_x - radius_bins)
        kernel_x1 = kernel_x0 + (target_x1 - target_x0)
        for source_y in range(columns):
            target_y0 = max(0, source_y - radius_bins)
            target_y1 = min(columns, source_y + radius_bins + 1)
            kernel_y0 = target_y0 - (source_y - radius_bins)
            kernel_y1 = kernel_y0 + (target_y1 - target_y0)
            local_kernel = kernel[kernel_x0:kernel_x1, kernel_y0:kernel_y1]
            smoothed[target_x0:target_x1, target_y0:target_y1] += (
                array[source_x, source_y] * local_kernel / local_kernel.sum()
            )
    return smoothed


@dataclass(frozen=True)
class RateResult:
    """All quantities needed to audit and display one season."""

    league_counts: np.ndarray
    league_rate: np.ndarray
    team_counts: dict[str, np.ndarray]
    team_rates: dict[str, np.ndarray]
    excess_rates: dict[str, np.ndarray]
    team_games: dict[str, int]
    league_games: int
    x_edges: np.ndarray
    y_edges: np.ndarray
    outside_display_zone: int


def _histogram(frame: pd.DataFrame, x_edges: np.ndarray, y_edges: np.ndarray) -> np.ndarray:
    valid = frame[["x_attack", "y_attack"]].dropna()
    counts, _, _ = np.histogram2d(
        valid["x_attack"].to_numpy(dtype=float),
        valid["y_attack"].to_numpy(dtype=float),
        bins=(x_edges, y_edges),
    )
    return counts


def _team_game_counts(games: pd.DataFrame) -> dict[str, int]:
    appearances = pd.concat(
        [
            games[["game_id", "home_team"]].rename(columns={"home_team": "team"}),
            games[["game_id", "away_team"]].rename(columns={"away_team": "team"}),
        ],
        ignore_index=True,
    ).drop_duplicates(["game_id", "team"])
    return appearances.groupby("team")["game_id"].nunique().astype(int).to_dict()


def compute_shot_rates(
    shots: pd.DataFrame,
    games: pd.DataFrame,
    *,
    x_edges: np.ndarray = DEFAULT_X_EDGES,
    y_edges: np.ndarray = DEFAULT_Y_EDGES,
) -> RateResult:
    """Compute league, team, and raw excess shot rates by spatial bin.

    ``league_rate = league_shots / (2 * league_games)`` and
    ``excess = team_shots / team_games - league_rate``. Each game is counted
    once for the league and once per participating team, even if a team has no
    retained shot. Coordinates outside the displayed offensive zone are
    audited separately and are not silently folded with ``abs(x)``.
    """

    required_shots = {"team", "x_attack", "y_attack"}
    required_games = {"game_id", "home_team", "away_team"}
    if missing := required_shots.difference(shots.columns):
        raise ValueError(f"Missing shot columns: {sorted(missing)}")
    if missing := required_games.difference(games.columns):
        raise ValueError(f"Missing game columns: {sorted(missing)}")

    x_edges = np.asarray(x_edges, dtype=float)
    y_edges = np.asarray(y_edges, dtype=float)
    if len(x_edges) < 2 or len(y_edges) < 2 or not (
        np.all(np.diff(x_edges) > 0) and np.all(np.diff(y_edges) > 0)
    ):
        raise ValueError("Bin edges must be strictly increasing")

    league_games = int(games["game_id"].nunique())
    if league_games <= 0:
        raise ValueError("At least one unique game is required")
    team_games = _team_game_counts(games)
    if not team_games or any(value <= 0 for value in team_games.values()):
        raise ValueError("Every team denominator must be positive")

    finite = shots[["x_attack", "y_attack"]].notna().all(axis=1)
    inside = (
        finite
        & shots["x_attack"].between(x_edges[0], x_edges[-1], inclusive="both")
        & shots["y_attack"].between(y_edges[0], y_edges[-1], inclusive="both")
    )
    display_shots = shots.loc[inside]
    outside = int((finite & ~inside).sum())
    league_counts = _histogram(display_shots, x_edges, y_edges)
    league_rate = league_counts / (2.0 * league_games)

    team_counts: dict[str, np.ndarray] = {}
    team_rates: dict[str, np.ndarray] = {}
    excess_rates: dict[str, np.ndarray] = {}
    for team in sorted(team_games):
        counts = _histogram(display_shots.loc[display_shots["team"] == team], x_edges, y_edges)
        rate = counts / float(team_games[team])
        team_counts[team] = counts
        team_rates[team] = rate
        excess_rates[team] = rate - league_rate

    return RateResult(
        league_counts=league_counts,
        league_rate=league_rate,
        team_counts=team_counts,
        team_rates=team_rates,
        excess_rates=excess_rates,
        team_games=team_games,
        league_games=league_games,
        x_edges=x_edges,
        y_edges=y_edges,
        outside_display_zone=outside,
    )
