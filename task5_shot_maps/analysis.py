"""Compact, data-backed metrics used to write the Task 5 discussion."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .rates import RateResult


def summarize_teams(result: RateResult, season: str) -> pd.DataFrame:
    """Return per-team spatial metrics without interpreting shot quality."""

    x = (result.x_edges[:-1] + result.x_edges[1:]) / 2.0
    y = (result.y_edges[:-1] + result.y_edges[1:]) / 2.0
    xx, yy = np.meshgrid(x, y, indexing="ij")
    zones = {
        "net_front": (xx >= 75) & (np.abs(yy) <= 12.5),
        "inner_slot": (xx >= 55) & (xx < 75) & (np.abs(yy) <= 17.5),
    }
    rows: list[dict[str, object]] = []
    league_total = float(result.league_rate.sum())
    for team in sorted(result.team_rates):
        rate = result.team_rates[team]
        excess = result.excess_rates[team]
        total = float(rate.sum())
        row: dict[str, object] = {
            "season": season,
            "team": team,
            "games": result.team_games[team],
            "displayed_shots_per_game": total,
            "league_displayed_shots_per_team_game": league_total,
            "displayed_excess_shots_per_game": float(excess.sum()),
        }
        for name, mask in zones.items():
            zone_rate = float(rate[mask].sum())
            row[f"{name}_shots_per_game"] = zone_rate
            row[f"{name}_share"] = zone_rate / total if total else 0.0
            row[f"{name}_excess_per_game"] = float(excess[mask].sum())
        rows.append(row)
    return pd.DataFrame(rows)
