"""Validate generated Task 5 artifacts against the complete real dataset."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

from .data import load_season
from .figure import SMOOTH_RADIUS_BINS, SMOOTH_SIGMA_BINS, build_figure
from .generate import DEFAULT_SEASONS
from .rates import compute_shot_rates, gaussian_smooth_preserve_sum


def validate_all(data_dir: Path, output_dir: Path) -> dict[str, object]:
    """Run deterministic integrity checks and write a machine-readable report."""

    report: dict[str, object] = {"seasons": {}, "all_checks_passed": True}
    for season in DEFAULT_SEASONS:
        data = load_season(season, data_dir)
        result = compute_shot_rates(data.shots, data.games)
        valid = data.shots["x_attack"].notna() & data.shots["y_attack"].notna()
        zone = data.shots["zone_code"].astype(str).str.upper()
        offensive_wrong_side = int((valid & zone.eq("O") & data.shots["x_attack"].lt(0)).sum())
        defensive_wrong_side = int((valid & zone.eq("D") & data.shots["x_attack"].gt(0)).sum())
        duplicate_events = int(data.shots.duplicated(["game_id", "event_id"]).sum())

        league_error = float(
            np.max(np.abs(result.league_rate - result.league_counts / (2 * result.league_games)))
        )
        team_error = 0.0
        excess_error = 0.0
        for team in result.team_rates:
            team_error = max(
                team_error,
                float(np.max(np.abs(
                    result.team_rates[team]
                    - result.team_counts[team] / result.team_games[team]
                ))),
            )
            excess_error = max(
                excess_error,
                float(np.max(np.abs(
                    result.excess_rates[team]
                    - (result.team_rates[team] - result.league_rate)
                ))),
            )

        smoothing_sum_error = max(
            abs(float(
                gaussian_smooth_preserve_sum(
                    result.league_rate,
                    sigma_bins=SMOOTH_SIGMA_BINS,
                    radius_bins=SMOOTH_RADIUS_BINS,
                ).sum()
                - result.league_rate.sum()
            )),
            *(abs(float(gaussian_smooth_preserve_sum(
                values,
                sigma_bins=SMOOTH_SIGMA_BINS,
                radius_bins=SMOOTH_RADIUS_BINS,
            ).sum() - values.sum()))
              for values in result.excess_rates.values()),
        )

        html_path = output_dir / f"shot_map_{season}.html"
        html = html_path.read_text(encoding="utf-8")
        dropdown_buttons = len(re.findall(r'"method":"update"', html))
        expected_teams = len(result.team_games)
        figure = build_figure(result, season)
        view_labels = [button.label for button in figure.layout.updatemenus[1].buttons]
        view_toggle_valid = (
            len(figure.data) == 3
            and figure.data[0].type == "heatmap"
            and figure.data[1].type == "heatmap"
            and figure.data[2].type == "contour"
            and view_labels == ["Valeurs brutes 5×5 pi", "Carte lissée"]
            and list(figure.layout.updatemenus[1].buttons[1].args[0]["visible"])
            == [False, True, True]
        )
        checks = {
            "unique_game_event_keys": duplicate_events == 0,
            "offensive_zone_oriented_positive_x": offensive_wrong_side == 0,
            "defensive_zone_oriented_negative_x": defensive_wrong_side == 0,
            "league_formula_exact": league_error == 0.0,
            "team_formula_exact": team_error == 0.0,
            "excess_formula_exact": excess_error == 0.0,
            "smoothing_preserves_total": smoothing_sum_error <= 1e-12,
            "dropdown_contains_every_team": dropdown_buttons == expected_teams,
            "raw_smoothed_toggle_available": view_toggle_valid,
            "standalone_plotly_embedded": "plotly.js v" in html,
            "html_render_centered": ".plotly-graph-div{margin-left:auto!important" in html,
        }
        report["seasons"][season] = {
            "games": result.league_games,
            "teams": expected_teams,
            "shots_and_goals": len(data.shots),
            "duplicate_game_event_keys": duplicate_events,
            "offensive_zone_wrong_side": offensive_wrong_side,
            "defensive_zone_wrong_side": defensive_wrong_side,
            "league_formula_max_error": league_error,
            "team_formula_max_error": team_error,
            "excess_formula_max_error": excess_error,
            "smoothing_sum_max_error": smoothing_sum_error,
            "dropdown_buttons": dropdown_buttons,
            "view_buttons": view_labels,
            "checks": checks,
        }
        report["all_checks_passed"] = bool(report["all_checks_passed"] and all(checks.values()))

    destination = output_dir / "validation_report.json"
    destination.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("task5_outputs/html"))
    args = parser.parse_args()
    report = validate_all(args.data_dir, args.output_dir)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["all_checks_passed"]:
        raise SystemExit("La validation de la tâche 5 a échoué.")


if __name__ == "__main__":
    main()
