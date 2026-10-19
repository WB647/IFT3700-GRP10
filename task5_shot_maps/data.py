"""Task-5 adapter around the existing tidy-data pipeline."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from tache3_tidydata import game_to_df

from .coordinates import attacks_right, normalize_coordinates


@dataclass(frozen=True)
class SeasonData:
    """Shots, game participation, and auditable exclusion counts."""

    shots: pd.DataFrame
    games: pd.DataFrame
    exclusions: dict[str, object]


def _team_label(team: dict) -> str:
    """Choose a stable display label while retaining the API abbreviation."""

    abbrev = str(team.get("abbrev", team.get("id", "?")))
    for key in ("name", "commonName", "placeName"):
        value = team.get(key)
        if isinstance(value, dict) and value.get("default"):
            return f"{value['default']} ({abbrev})"
        if isinstance(value, str) and value:
            return f"{value} ({abbrev})"
    return abbrev


def _metadata_by_event(game: dict) -> dict[object, tuple[object, object]]:
    return {
        play.get("eventId"): (
            play.get("homeTeamDefendingSide"),
            play.get("details", {}).get("zoneCode"),
        )
        for play in game.get("plays", [])
        if play.get("typeDescKey") in {"shot-on-goal", "goal"}
    }


def _infer_period_directions(rows: pd.DataFrame) -> dict[tuple[int, str], bool]:
    """Infer direction from a period/team median only when metadata is absent.

    NHL shots overwhelmingly occur toward the opponent's end. A non-zero
    median therefore gives a transparent fallback. This is never used when
    ``homeTeamDefendingSide`` is available and its use is counted.
    """

    result: dict[tuple[int, str], bool] = {}
    for (period, team), group in rows.groupby(["period", "team"], dropna=False):
        votes: list[bool] = []
        for row in group.itertuples(index=False):
            try:
                x = float(row.x)
            except (TypeError, ValueError):
                continue
            zone = str(row.zone_code).upper()
            if x == 0:
                continue
            if zone == "O":
                votes.append(x > 0)
            elif zone == "D":
                votes.append(x < 0)
        if votes:
            result[(int(period), str(team))] = sum(votes) >= len(votes) / 2
    return result


def load_game(path: Path) -> tuple[pd.DataFrame, dict[str, object]]:
    """Load one game through task 3 and add Task-5-only orientation fields."""

    path = Path(path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    shots = game_to_df(path).copy()
    home = raw["homeTeam"]
    away = raw["awayTeam"]
    home_abbrev = str(home["abbrev"])
    event_metadata = _metadata_by_event(raw)
    shots["home_defending_side"] = shots["event_id"].map(
        lambda event_id: event_metadata.get(event_id, (None, None))[0]
    )
    shots["zone_code"] = shots["event_id"].map(
        lambda event_id: event_metadata.get(event_id, (None, None))[1]
    )
    inferred = _infer_period_directions(shots)

    shots["is_home"] = shots["team"].astype(str).eq(home_abbrev)
    normalized: list[tuple[float, float] | None] = []
    sources: list[str] = []
    metadata_conflicts: list[bool] = []
    for row in shots.itertuples(index=False):
        side = row.home_defending_side
        zone = str(row.zone_code).upper()
        try:
            raw_x = float(row.x)
        except (TypeError, ValueError):
            raw_x = 0.0
        zone_direction = None
        if raw_x != 0 and zone == "O":
            zone_direction = raw_x > 0
        elif raw_x != 0 and zone == "D":
            zone_direction = raw_x < 0
        period_direction = inferred.get((int(row.period), str(row.team)))
        inferred_direction = zone_direction if zone_direction is not None else period_direction
        side_available = not pd.isna(side) and bool(str(side).strip())
        conflict = False
        if side_available and zone_direction is not None:
            try:
                conflict = attacks_right(bool(row.is_home), str(side)) != zone_direction
            except ValueError:
                conflict = False
        metadata_conflicts.append(conflict)
        # Event-level zone metadata is more reliable than the small number of
        # archival games whose defending-side field contradicts every shot.
        side_for_normalization = str(side) if side_available and inferred_direction is None else None
        try:
            normalized.append(
                normalize_coordinates(
                    row.x,
                    row.y,
                    is_home=bool(row.is_home),
                    home_defending_side=side_for_normalization,
                    inferred_attacks_right=inferred_direction,
                )
            )
            if zone_direction is not None:
                sources.append("zone_code")
            elif period_direction is not None:
                sources.append("inferred_period_zone_majority")
            else:
                sources.append("home_defending_side")
        except ValueError:
            normalized.append(None)
            sources.append("unusable")

    shots["x_attack"] = [value[0] if value else np.nan for value in normalized]
    shots["y_attack"] = [value[1] if value else np.nan for value in normalized]
    shots["orientation_source"] = sources
    shots["orientation_metadata_conflict"] = metadata_conflicts
    shots["team_label"] = np.where(
        shots["is_home"], _team_label(home), _team_label(away)
    )

    game = {
        "season": str(raw["season"]),
        "game_type": int(raw["gameType"]),
        "game_id": int(raw["id"]),
        "home_team": home_abbrev,
        "away_team": str(away["abbrev"]),
        "home_label": _team_label(home),
        "away_label": _team_label(away),
    }
    return shots, game


def load_season(
    season: str,
    data_dir: Path | str,
    *,
    game_types: Iterable[int] = (2,),
) -> SeasonData:
    """Load a season and report every coordinate exclusion.

    Regular season (game type 2) is the default so comparisons align with
    official regular-season standings. Playoffs can be included explicitly.
    """

    season = str(season)
    directory = Path(data_dir) / season
    files = sorted(directory.glob("*.json"))
    if not files:
        raise FileNotFoundError(
            f"No JSON games found in {directory}. Run download_nhl_data.py first."
        )

    allowed = {int(value) for value in game_types}
    shot_frames: list[pd.DataFrame] = []
    games: list[dict[str, object]] = []
    for path in files:
        frame, game = load_game(path)
        if int(game["game_type"]) in allowed:
            shot_frames.append(frame)
            games.append(game)

    if not games:
        raise ValueError(f"No games of type {sorted(allowed)} found for {season}")
    game_frame = pd.DataFrame(games).drop_duplicates("game_id").reset_index(drop=True)
    shots = pd.concat(shot_frames, ignore_index=True) if shot_frames else pd.DataFrame()
    invalid = int(shots["x_attack"].isna().sum())
    source_counts = {
        str(key): int(value)
        for key, value in shots["orientation_source"].value_counts().to_dict().items()
    }
    return SeasonData(
        shots=shots,
        games=game_frame,
        exclusions={
            "unusable_coordinates": invalid,
            "orientation_sources": source_counts,
            "orientation_metadata_conflicts": int(shots["orientation_metadata_conflict"].sum()),
        },
    )
