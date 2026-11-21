import json
from pathlib import Path

import numpy as np

from milestone2.features import FEATURE_COLUMNS, build_dataset, build_game_features


def synthetic_game(game_id: int = 2017021065, season: int = 20172018) -> dict:
    common = {"periodDescriptor": {"number": 1, "periodType": "REG"}}
    return {
        "id": game_id,
        "season": season,
        "gameType": 2,
        "homeTeam": {"id": 1, "abbrev": "WPG", "commonName": {"default": "Jets"}},
        "awayTeam": {"id": 2, "abbrev": "WSH", "commonName": {"default": "Capitals"}},
        "rosterSpots": [],
        "plays": [
            {**common, "eventId": 1, "typeDescKey": "faceoff", "timeInPeriod": "01:00",
             "details": {"xCoord": 0, "yCoord": 0}},
            {**common, "eventId": 2, "typeDescKey": "shot-on-goal", "timeInPeriod": "01:05",
             "homeTeamDefendingSide": "left", "situationCode": "1551",
             "details": {"eventOwnerTeamId": 1, "xCoord": 70, "yCoord": 10,
                         "zoneCode": "O", "shootingPlayerId": 11, "shotType": "wrist"}},
            {**common, "eventId": 3, "typeDescKey": "shot-on-goal", "timeInPeriod": "01:07",
             "homeTeamDefendingSide": "left", "situationCode": "1551",
             "details": {"eventOwnerTeamId": 1, "xCoord": 78, "yCoord": -4,
                         "zoneCode": "O", "shootingPlayerId": 11, "shotType": "snap"}},
            {**common, "eventId": 4, "typeDescKey": "hit", "timeInPeriod": "01:10",
             "details": {"xCoord": 65, "yCoord": -8}},
            {**common, "eventId": 5, "typeDescKey": "goal", "timeInPeriod": "01:13",
             "homeTeamDefendingSide": "left", "situationCode": "1551",
             "details": {"eventOwnerTeamId": 1, "xCoord": 82, "yCoord": 2,
                         "zoneCode": "O", "scoringPlayerId": 11, "shotType": "tip-in"}},
        ],
    }


def write_game(path: Path, game: dict | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(game or synthetic_game()), encoding="utf-8")
    return path


def test_build_game_features_has_every_mandatory_column(tmp_path: Path):
    frame = build_game_features(write_game(tmp_path / "game.json"))
    assert len(frame) == 3
    assert set(FEATURE_COLUMNS).issubset(frame.columns)
    assert {"empty_net", "is_goal", "game_id", "event_id"}.issubset(frame.columns)
    assert frame["game_id"].eq(2017021065).all()


def test_previous_event_and_rebound_features(tmp_path: Path):
    frame = build_game_features(write_game(tmp_path / "game.json")).set_index("event_id")
    first, rebound, goal = frame.loc[2], frame.loc[3], frame.loc[5]
    assert first["last_event_type"] == "faceoff"
    assert first["time_since_last_event"] == 5
    assert bool(rebound["rebound"])
    assert rebound["time_since_last_event"] == 2
    assert rebound["change_in_shot_angle"] > 0
    assert not bool(goal["rebound"])
    assert goal["change_in_shot_angle"] == 0
    assert np.isclose(goal["distance_from_last_event"], np.hypot(17, 10))
    assert np.isclose(goal["speed"], np.hypot(17, 10) / 3)


def test_build_dataset_filters_non_regular_games(tmp_path: Path):
    season_dir = tmp_path / "20172018"
    write_game(season_dir / "regular.json")
    playoff = synthetic_game(2017030001)
    playoff["gameType"] = 3
    write_game(season_dir / "playoff.json", playoff)
    frame = build_dataset(tmp_path, ["20172018"], strict=False)
    assert frame["game_type"].eq(2).all()
    assert frame["game_id"].nunique() == 1


def test_build_dataset_rejects_missing_seasons(tmp_path: Path):
    try:
        build_dataset(tmp_path, ["20212022"])
    except FileNotFoundError as error:
        assert "20212022" in str(error)
    else:
        raise AssertionError("Missing seasons must not be silently ignored")


def test_build_dataset_rejects_partial_regular_season(tmp_path: Path):
    write_game(tmp_path / "20172018" / "one_game.json")
    try:
        build_dataset(tmp_path, ["20172018"])
    except ValueError as error:
        assert "1 games, expected 1271" in str(error)
    else:
        raise AssertionError("A partial season must not be accepted for modeling")
