"""Command-line generation of all five required interactive shot maps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .analysis import summarize_teams
from .data import SeasonData, load_season
from .figure import build_figure, write_figure_html
from .rates import RateResult, compute_shot_rates


DEFAULT_SEASONS = ("20162017", "20172018", "20182019", "20192020", "20202021")


def season_label(season: str) -> str:
    return f"{season[:4]}-{season[6:]}"


def _labels(data: SeasonData) -> dict[str, str]:
    pairs = pd.concat(
        [
            data.games[["home_team", "home_label"]].set_axis(["team", "label"], axis=1),
            data.games[["away_team", "away_label"]].set_axis(["team", "label"], axis=1),
        ],
        ignore_index=True,
    ).drop_duplicates("team")
    return dict(zip(pairs["team"], pairs["label"]))


def generate_all(
    data_dir: Path,
    output_dir: Path,
    seasons: tuple[str, ...] = DEFAULT_SEASONS,
    game_types: tuple[int, ...] = (2,),
) -> list[Path]:
    """Generate HTML, audit JSON, and analysis CSV for all requested seasons."""

    loaded: dict[str, SeasonData] = {}
    rates: dict[str, RateResult] = {}
    for season in seasons:
        loaded[season] = load_season(season, data_dir, game_types=game_types)
        rates[season] = compute_shot_rates(loaded[season].shots, loaded[season].games)

    values = np.concatenate(
        [matrix.ravel() for result in rates.values() for matrix in result.excess_rates.values()]
    )
    color_limit = float(np.quantile(np.abs(values), 0.98))
    if not np.isfinite(color_limit) or color_limit <= 0:
        color_limit = 1.0

    output_dir.mkdir(parents=True, exist_ok=True)
    html_paths: list[Path] = []
    audit: dict[str, object] = {
        "game_types": list(game_types),
        "bin_size_feet": [5.0, 5.0],
        "smoothing": {
            "available_as_toggle": True,
            "method": "mass-preserving truncated Gaussian",
            "sigma_bins": 0.6,
            "sigma_feet": 3.0,
            "radius_bins": 2,
            "radius_feet": 10.0,
            "bin_size_feet": 5.0,
            "boundary_handling": "per-source kernel renormalization inside display grid",
            "rendering": "smoothed heatmap with visual interpolation and contour guides",
        },
        "color_scale_quantile": 0.98,
        "shared_color_limit": color_limit,
        "seasons": {},
    }
    metrics = []
    for season in seasons:
        result = rates[season]
        figure = build_figure(
            result,
            season_label(season),
            team_labels=_labels(loaded[season]),
            color_limit=color_limit,
        )
        destination = output_dir / f"shot_map_{season}.html"
        html_paths.append(write_figure_html(figure, destination))
        metrics.append(summarize_teams(result, season))
        audit["seasons"][season] = {
            "league_games": result.league_games,
            "team_games": result.team_games,
            "unusable_coordinates": loaded[season].exclusions["unusable_coordinates"],
            "orientation_sources": loaded[season].exclusions["orientation_sources"],
            "orientation_metadata_conflicts": loaded[season].exclusions[
                "orientation_metadata_conflicts"
            ],
            "valid_coordinates_outside_offensive_display": result.outside_display_zone,
            "teams": sorted(result.team_games),
        }

    (output_dir / "generation_audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    pd.concat(metrics, ignore_index=True).to_csv(output_dir / "analysis_metrics.csv", index=False)
    return html_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("task5_outputs/html"))
    parser.add_argument("--seasons", nargs="+", default=list(DEFAULT_SEASONS))
    parser.add_argument(
        "--game-types", nargs="+", type=int, default=[2],
        help="NHL game types (2=regular season, 3=playoffs).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = generate_all(
        args.data_dir,
        args.output_dir,
        tuple(args.seasons),
        tuple(args.game_types),
    )
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
