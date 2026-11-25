"""Build the mandatory milestone-2 Task 4 features from NHL play-by-play."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from task5_shot_maps.data import load_game


NET_X = 89.0
TRAIN_SEASONS = tuple(f"{year}{year + 1}" for year in range(2016, 2023))
TEST_SEASON = "20232024"
REGULAR_SEASON, PLAYOFFS = 2, 3
EXPECTED_REGULAR_GAMES = {
    "20162017": 1230,
    "20172018": 1271,
    "20182019": 1271,
    "20192020": 1082,
    "20202021": 868,
    "20212022": 1312,
    "20222023": 1312,
    "20232024": 1312,
}
FEATURE_COLUMNS = [
    "game_seconds",
    "period",
    "x",
    "y",
    "shot_distance",
    "shot_angle",
    "shot_type",
    "last_event_type",
    "last_event_x",
    "last_event_y",
    "time_since_last_event",
    "distance_from_last_event",
    "rebound",
    "change_in_shot_angle",
    "speed",
]

#Caractéristiques ajoutées pour la tentative A de la tâche 6. FEATURE_COLUMNS ne change pas 
NEW_FEATURE_COLUMNS = [
    "empty_net",
    "shooter_skaters",
    "opponent_skaters",
    "skater_diff",
    "is_home",
    "last_event_same_team",
    "score_diff",
]

FEATURE_COLUMNS_V2 = FEATURE_COLUMNS + NEW_FEATURE_COLUMNS
IDENTIFIER_COLUMNS = ["season", "game_type", "game_id", "event_id", "team"]


def _clock_seconds(value: object) -> float:
    """Convert an NHL ``MM:SS`` period clock to elapsed seconds."""

    if not isinstance(value, str) or ":" not in value:
        return np.nan
    try:
        minutes, seconds = (int(part) for part in value.split(":", 1))
    except (TypeError, ValueError):
        return np.nan
    if minutes < 0 or seconds < 0 or seconds >= 60:
        return np.nan
    return float(60 * minutes + seconds)


def _game_seconds(
    period: object, time_in_period: object, period_type: object, game_type: object = REGULAR_SEASON
) -> float:
    """Retourne le nombre de secondes écoulées depuis le début du match."""

    elapsed = _clock_seconds(time_in_period)
    try:
        number = int(period)
    except (TypeError, ValueError):
        return np.nan
    if not np.isfinite(elapsed) or number < 1:
        return np.nan
    try:
        is_playoff = int(game_type) == PLAYOFFS
    except (TypeError, ValueError):
        is_playoff = False
    if number <= 3 or is_playoff:
        return (number - 1) * 1200.0 + elapsed
    if str(period_type).upper() == "OT":
        return 3600.0 + (number - 4) * 300.0 + elapsed
    return (number - 1) * 1200.0 + elapsed


def _rotation(raw_x: object, raw_y: object, attack_x: object, attack_y: object) -> float:
    """Recover the ±1 rotation already established by the Task-5 adapter."""

    for raw, oriented in ((raw_x, attack_x), (raw_y, attack_y)):
        try:
            raw_value, oriented_value = float(raw), float(oriented)
        except (TypeError, ValueError):
            continue
        if np.isfinite(raw_value) and np.isfinite(oriented_value) and raw_value != 0:
            return 1.0 if raw_value * oriented_value > 0 else -1.0
    return 1.0


def _signed_angle(x: float, y: float) -> float:
    return float(np.degrees(np.arctan2(y, NET_X - x)))


def _flag(value: object) -> float:
    """Convertit un booléen en 1.0/0.0 et garde les valeurs inconnues comme NaN."""

    if value is None or (isinstance(value, float) and np.isnan(value)) or value is pd.NA:
        return np.nan
    return float(bool(value))


def _skaters(strength_state: object) -> tuple[float, float]:
    """Lit « 5v4 » (équipe qui tire contre adversaire) et retourne (5.0, 4.0), ou (NaN, NaN)."""

    try:
        own, opponent = str(strength_state).split("v")
        return float(own), float(opponent)
    except (TypeError, ValueError):
        return np.nan, np.nan


def _angle_difference(first: float, second: float) -> float:
    """Smallest absolute difference between two angles, in degrees."""

    return abs((first - second + 180.0) % 360.0 - 180.0)


def build_game_features(path: Path | str) -> pd.DataFrame:
    """Create one row per non-shootout shot/goal with Task-4 features.

    Shot coordinates and every preceding-event coordinate are rotated with the
    same direction inferred by the existing Task-3/Task-5 data pipeline. The
    previous event is the immediately preceding item in the raw ``plays`` list,
    not merely the previous shot.

    Tâche 6 : ajoute aussi les colonnes de "NEW_FEATURE_COLUMNS". Le score
    utilisé par "score_diff" est celui d'avant le tir."""

    path = Path(path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    shots, _ = load_game(path)
    shot_by_event = {int(row.event_id): row for row in shots.itertuples(index=False)}
    rows: list[dict[str, object]] = []
    plays = raw.get("plays", [])
    game_type = raw.get("gameType")
    home_id = raw.get("homeTeam", {}).get("id")
    away_id = raw.get("awayTeam", {}).get("id")
    goals = {home_id: 0, away_id: 0}

    for index, play in enumerate(plays):
        descriptor = play.get("periodDescriptor", {})

        # Score avant cet événement, puis mise à jour si c'est un but (hors tirs de barrage).

        score_before = dict(goals)
        if play.get("typeDescKey") == "goal" and descriptor.get("periodType") != "SO":
            scorer = play.get("details", {}).get("eventOwnerTeamId")
            if scorer in goals:
                goals[scorer] += 1
        if play.get("typeDescKey") not in {"shot-on-goal", "goal"}:
            continue
        if descriptor.get("periodType") == "SO":
            continue
        event_id = play.get("eventId")
        try:
            shot = shot_by_event[int(event_id)]
        except (KeyError, TypeError, ValueError):
            continue

        current_seconds = _game_seconds(
            descriptor.get("number"), play.get("timeInPeriod"), descriptor.get("periodType"), game_type
        )
        rotation = _rotation(shot.x, shot.y, shot.x_attack, shot.y_attack)
        x = float(shot.x_attack) if pd.notna(shot.x_attack) else np.nan
        y = float(shot.y_attack) if pd.notna(shot.y_attack) else np.nan
        distance = float(np.hypot(NET_X - x, y)) if np.isfinite(x) and np.isfinite(y) else np.nan
        signed_angle = _signed_angle(x, y) if np.isfinite(x) and np.isfinite(y) else np.nan

        previous = plays[index - 1] if index else None
        previous_type: object = pd.NA
        previous_x = previous_y = previous_seconds = np.nan
        if previous is not None:
            previous_type = previous.get("typeDescKey", pd.NA)
            previous_details = previous.get("details", {})
            try:
                previous_x = rotation * float(previous_details.get("xCoord"))
                previous_y = rotation * float(previous_details.get("yCoord"))
            except (TypeError, ValueError):
                previous_x = previous_y = np.nan
            previous_descriptor = previous.get("periodDescriptor", {})
            previous_seconds = _game_seconds(
                previous_descriptor.get("number"),
                previous.get("timeInPeriod"),
                previous_descriptor.get("periodType"),
                game_type,
            )

        elapsed = (
            current_seconds - previous_seconds
            if np.isfinite(current_seconds) and np.isfinite(previous_seconds)
            else np.nan
        )
        if np.isfinite(elapsed) and elapsed < 0:
            elapsed = np.nan
        previous_distance = (
            float(np.hypot(x - previous_x, y - previous_y))
            if all(np.isfinite(value) for value in (x, y, previous_x, previous_y))
            else np.nan
        )
        rebound = bool(previous_type == "shot-on-goal") if pd.notna(previous_type) else False
        change = 0.0
        if rebound and all(np.isfinite(value) for value in (signed_angle, previous_x, previous_y)):
            change = _angle_difference(signed_angle, _signed_angle(previous_x, previous_y))
        speed = previous_distance / elapsed if np.isfinite(elapsed) and elapsed > 0 else np.nan

        #Tache 6 : contexte du tir (avantage numérique, score, équipe de l'événement précédent).

        shooter_id = play.get("details", {}).get("eventOwnerTeamId")
        if shooter_id not in goals:
            shooter_id = home_id if bool(shot.is_home) else away_id
        opponent_id = away_id if shooter_id == home_id else home_id
        shooter_skaters, opponent_skaters = _skaters(shot.strength_state)
        previous_owner = previous.get("details", {}).get("eventOwnerTeamId") if previous is not None else None
        same_team = np.nan if previous_owner is None else float(previous_owner == shooter_id)

        rows.append(
            {
                "season": int(raw["season"]),
                "game_type": int(raw["gameType"]),
                "game_id": int(raw["id"]),
                "event_id": int(event_id),
                "team": str(shot.team),
                "game_seconds": current_seconds,
                "period": int(descriptor.get("number")),
                "x": x,
                "y": y,
                "shot_distance": distance,
                "shot_angle": abs(signed_angle) if np.isfinite(signed_angle) else np.nan,
                "shot_type": shot.shot_type,
                "last_event_type": previous_type,
                "last_event_x": previous_x,
                "last_event_y": previous_y,
                "time_since_last_event": elapsed,
                "distance_from_last_event": previous_distance,
                "rebound": bool(rebound),
                "change_in_shot_angle": change,
                "speed": speed,
                "empty_net": _flag(shot.empty_net),
                "shooter_skaters": shooter_skaters,
                "opponent_skaters": opponent_skaters,
                "skater_diff": shooter_skaters - opponent_skaters,
                "is_home": _flag(shot.is_home),
                "last_event_same_team": same_team,
                "score_diff": float(score_before[shooter_id] - score_before[opponent_id]),
                "is_goal": int(bool(shot.is_goal)),
            }
        )
    return pd.DataFrame(rows)


def build_dataset(
    data_dir: Path | str,
    seasons: Iterable[str] = TRAIN_SEASONS,
    *,
    game_types: Iterable[int] = (REGULAR_SEASON,),
    strict: bool = True,
) -> pd.DataFrame:
    """Construit le jeu de données pour les saisons et types de matchs demandés.

    "game_types=(2,)" (par défaut) garde la saison régulière, "(3,)" les
    séries. Les saisons manquantes et les saisons régulières incomplètes sont refusées.
    """

    data_dir = Path(data_dir)
    requested = tuple(str(season) for season in seasons)
    allowed = {int(value) for value in game_types}
    missing = [season for season in requested if not any((data_dir / season).glob("*.json"))]
    if missing and strict:
        raise FileNotFoundError(f"Missing raw seasons: {', '.join(missing)}")

    frames: list[pd.DataFrame] = []
    for season in requested:
        paths_by_type: dict[int, list[Path]] = {}
        for path in sorted((data_dir / season).glob("*.json")):
            try:
                game_type = int(json.loads(path.read_text(encoding="utf-8"))["gameType"])
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                continue
            paths_by_type.setdefault(game_type, []).append(path)
        expected = EXPECTED_REGULAR_GAMES.get(season)
        regular_count = len(paths_by_type.get(REGULAR_SEASON, []))
        if strict and REGULAR_SEASON in allowed and expected is not None and regular_count != expected:
            raise ValueError(
                f"Incomplete regular season {season}: {regular_count} games, expected {expected}"
            )
        for game_type in sorted(allowed):
            frames.extend(build_game_features(path) for path in paths_by_type.get(game_type, []))
    if not frames:
        raise ValueError(f"No games of type {sorted(allowed)} were available")
    return pd.concat(frames, ignore_index=True)