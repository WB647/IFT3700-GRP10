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


def test_overtime_length_depends_on_game_type():
    from milestone2.features import _game_seconds

    # Regular season: a single 5-minute overtime after 60 minutes.
    assert _game_seconds(4, "02:00", "OT", 2) == 3600 + 120
    # Playoffs: every overtime lasts 20 minutes, so the 2nd OT starts at 80:00.
    assert _game_seconds(4, "02:00", "OT", 3) == 3600 + 120
    assert _game_seconds(5, "05:00", "OT", 3) == 4800 + 300
    assert _game_seconds(6, "00:30", "OT", 3) == 6000 + 30


def test_playoff_double_overtime_features(tmp_path: Path):
    game = synthetic_game(2023030111, 20232024)
    game["gameType"] = 3
    for play in game["plays"]:
        play["periodDescriptor"] = {"number": 5, "periodType": "OT"}
    frame = build_game_features(write_game(tmp_path / "playoff.json", game)).set_index("event_id")
    assert frame.loc[2, "game_seconds"] == 4800 + 65
    assert frame.loc[2, "time_since_last_event"] == 5
    assert frame["game_type"].eq(3).all()


def test_build_dataset_can_select_playoffs(tmp_path: Path):
    season_dir = tmp_path / "20232024"
    write_game(season_dir / "regular.json", synthetic_game(2023020001, 20232024))
    playoff = synthetic_game(2023030111, 20232024)
    playoff["gameType"] = 3
    write_game(season_dir / "playoff.json", playoff)

    playoffs = build_dataset(tmp_path, ["20232024"], game_types=(3,), strict=False)
    assert playoffs["game_type"].eq(3).all()
    assert set(playoffs["game_id"]) == {2023030111}

    both = build_dataset(tmp_path, ["20232024"], game_types=(2, 3), strict=False)
    assert set(both["game_id"]) == {2023020001, 2023030111}


def test_playoffs_are_not_rejected_by_regular_season_count(tmp_path: Path):
    playoff = synthetic_game(2023030111, 20232024)
    playoff["gameType"] = 3
    write_game(tmp_path / "20232024" / "playoff.json", playoff)
    frame = build_dataset(tmp_path, ["20232024"], game_types=(3,))
    assert len(frame) == 3


def context_game() -> dict:
    """Match synthétique : but local, tir visiteur en avantage numérique, tir local dans un filet désert."""

    common = {"periodDescriptor": {"number": 3, "periodType": "REG"}, "homeTeamDefendingSide": "left"}
    return {
        "id": 2022020500,
        "season": 20222023,
        "gameType": 2,
        "homeTeam": {"id": 1, "abbrev": "MTL"},
        "awayTeam": {"id": 2, "abbrev": "TOR"},
        "rosterSpots": [],
        "plays": [
            {**common, "eventId": 1, "typeDescKey": "goal", "timeInPeriod": "10:00", "situationCode": "1551",
             "details": {"eventOwnerTeamId": 1, "xCoord": 80, "yCoord": 5, "zoneCode": "O",
                         "scoringPlayerId": 11, "shotType": "wrist"}},
            {**common, "eventId": 2, "typeDescKey": "faceoff", "timeInPeriod": "10:00",
             "details": {"eventOwnerTeamId": 2, "xCoord": 0, "yCoord": 0}},
            {**common, "eventId": 3, "typeDescKey": "shot-on-goal", "timeInPeriod": "10:20", "situationCode": "1541",
             "details": {"eventOwnerTeamId": 2, "xCoord": -70, "yCoord": 10, "zoneCode": "O",
                         "shootingPlayerId": 21, "shotType": "slap"}},
            {**common, "eventId": 4, "typeDescKey": "stoppage", "timeInPeriod": "18:00", "details": {}},
            {**common, "eventId": 5, "typeDescKey": "shot-on-goal", "timeInPeriod": "19:00", "situationCode": "0651",
             "details": {"eventOwnerTeamId": 1, "xCoord": 40, "yCoord": 0, "zoneCode": "O",
                         "shootingPlayerId": 12, "shotType": "wrist"}},
        ],
    }


def test_feature_lists_keep_task5_columns_unchanged():
    from milestone2.features import FEATURE_COLUMNS_V2, NEW_FEATURE_COLUMNS

    assert len(FEATURE_COLUMNS) == 15
    assert FEATURE_COLUMNS_V2 == FEATURE_COLUMNS + NEW_FEATURE_COLUMNS
    assert not set(FEATURE_COLUMNS) & set(NEW_FEATURE_COLUMNS)


def test_context_features(tmp_path: Path):
    from milestone2.features import NEW_FEATURE_COLUMNS

    frame = build_game_features(write_game(tmp_path / "game.json", context_game())).set_index("event_id")
    assert set(NEW_FEATURE_COLUMNS).issubset(frame.columns)
    goal, powerplay, empty = frame.loc[1], frame.loc[3], frame.loc[5]

    # Le but ne compte pas dans son propre écart au score.
    assert goal["score_diff"] == 0
    assert goal["is_home"] == 1.0 and goal["empty_net"] == 0.0
    assert np.isnan(goal["last_event_same_team"])  # aucun événement avant

    # Tir visiteur à 5 contre 4, en retard d'un but, après une mise au jeu gagnée.
    assert (powerplay["shooter_skaters"], powerplay["opponent_skaters"], powerplay["skater_diff"]) == (5, 4, 1)
    assert powerplay["score_diff"] == -1
    assert powerplay["is_home"] == 0.0
    assert powerplay["last_event_same_team"] == 1.0
    assert powerplay["empty_net"] == 0.0

    # Tir local vers le filet désert (gardien visiteur retiré, 6 patineurs adverses).
    assert empty["empty_net"] == 1.0
    assert (empty["shooter_skaters"], empty["opponent_skaters"], empty["skater_diff"]) == (5, 6, -1)
    assert empty["score_diff"] == 1
    assert np.isnan(empty["last_event_same_team"])  # arrêt de jeu : aucune équipe


def test_unknown_situation_code_gives_missing_values(tmp_path: Path):
    game = context_game()
    del game["plays"][2]["situationCode"]
    frame = build_game_features(write_game(tmp_path / "game.json", game)).set_index("event_id")
    row = frame.loc[3]
    assert np.isnan(row["shooter_skaters"]) and np.isnan(row["skater_diff"]) and np.isnan(row["empty_net"])