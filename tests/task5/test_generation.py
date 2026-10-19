import json
from pathlib import Path

import pandas as pd

from task5_shot_maps.figure import build_figure
from task5_shot_maps.generate import DEFAULT_SEASONS, generate_all
from task5_shot_maps.rates import compute_shot_rates
from task5_shot_maps.data import load_game


def _synthetic_game(season: str) -> dict:
    return {
        "id": int(f"{season[:4]}020001"),
        "season": int(season),
        "gameType": 2,
        "homeTeam": {"id": 1, "abbrev": "AAA", "commonName": {"default": "Alphas"}},
        "awayTeam": {"id": 2, "abbrev": "BBB", "commonName": {"default": "Betas"}},
        "rosterSpots": [],
        "plays": [
            {
                "eventId": 10,
                "typeDescKey": "shot-on-goal",
                "periodDescriptor": {"number": 1, "periodType": "REG"},
                "timeInPeriod": "01:00",
                "situationCode": "1551",
                "homeTeamDefendingSide": "left",
                "details": {"eventOwnerTeamId": 1, "xCoord": 70, "yCoord": 5,
                            "shootingPlayerId": 11, "goalieInNetId": 22, "shotType": "wrist"},
            },
            {
                "eventId": 11,
                "typeDescKey": "goal",
                "periodDescriptor": {"number": 1, "periodType": "REG"},
                "timeInPeriod": "02:00",
                "situationCode": "1551",
                "homeTeamDefendingSide": "left",
                "details": {"eventOwnerTeamId": 2, "xCoord": -75, "yCoord": -3,
                            "scoringPlayerId": 33, "goalieInNetId": 44, "shotType": "snap"},
            },
        ],
    }


def test_dropdown_selects_every_team():
    shots = pd.DataFrame([
        {"team": "AAA", "x_attack": 70, "y_attack": 5},
        {"team": "BBB", "x_attack": 75, "y_attack": 3},
    ])
    games = pd.DataFrame([{"game_id": 1, "home_team": "AAA", "away_team": "BBB"}])
    figure = build_figure(compute_shot_rates(shots, games), "test")
    team_buttons = figure.layout.updatemenus[0].buttons
    view_buttons = figure.layout.updatemenus[1].buttons
    assert len(figure.data) == 3
    assert [trace.type for trace in figure.data] == ["heatmap", "heatmap", "contour"]
    assert [button.label for button in team_buttons] == ["AAA", "BBB"]
    assert len(team_buttons[1].args[0]["z"]) == 3
    assert [button.label for button in view_buttons] == [
        "Valeurs brutes 5×5 pi", "Carte lissée"
    ]
    assert list(view_buttons[0].args[0]["visible"]) == [True, False, False]
    assert list(view_buttons[1].args[0]["visible"]) == [False, True, True]


def test_zone_code_orients_archival_event_without_defending_side(tmp_path: Path):
    game = _synthetic_game("20162017")
    del game["plays"][1]["homeTeamDefendingSide"]
    game["plays"][1]["details"]["zoneCode"] = "O"
    path = tmp_path / "game.json"
    path.write_text(json.dumps(game), encoding="utf-8")
    shots, _ = load_game(path)
    away = shots.loc[shots["team"] == "BBB"].iloc[0]
    assert (away["x_attack"], away["y_attack"]) == (75.0, 3.0)
    assert away["orientation_source"] == "zone_code"


def test_event_zone_wins_when_game_side_metadata_is_inconsistent(tmp_path: Path):
    game = _synthetic_game("20192020")
    game["plays"][0]["details"]["zoneCode"] = "O"
    game["plays"][0]["details"]["xCoord"] = -70
    game["plays"][0]["details"]["yCoord"] = 5
    game["plays"][0]["homeTeamDefendingSide"] = "left"
    path = tmp_path / "conflict.json"
    path.write_text(json.dumps(game), encoding="utf-8")
    shots, _ = load_game(path)
    home = shots.loc[shots["team"] == "AAA"].iloc[0]
    assert (home["x_attack"], home["y_attack"]) == (70.0, -5.0)
    assert home["orientation_source"] == "zone_code"
    assert bool(home["orientation_metadata_conflict"])


def test_generate_all_creates_the_five_required_html_files(tmp_path: Path):
    data_dir = tmp_path / "data"
    for season in DEFAULT_SEASONS:
        directory = data_dir / season
        directory.mkdir(parents=True)
        game = _synthetic_game(season)
        (directory / f"{game['id']}.json").write_text(json.dumps(game), encoding="utf-8")

    output = tmp_path / "html"
    paths = generate_all(data_dir, output)
    assert len(paths) == 5
    assert all(path.exists() and path.stat().st_size > 100_000 for path in paths)
    assert all(
        ".plotly-graph-div{margin-left:auto!important" in path.read_text(encoding="utf-8")
        for path in paths
    )
    assert {path.name for path in paths} == {
        f"shot_map_{season}.html" for season in DEFAULT_SEASONS
    }
    assert (output / "generation_audit.json").exists()
    assert (output / "analysis_metrics.csv").exists()
