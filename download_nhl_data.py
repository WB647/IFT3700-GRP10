import json
import time
from pathlib import Path

import requests

BASE_URL = "https://api-web.nhle.com/v1"
DATA_DIR = Path("data/raw")

# Includes teams that played during the required seasons.
# Vegas and Seattle did not play in every season.
TEAMS = [
    "ANA", "ARI", "BOS", "BUF", "CGY", "CAR", "CHI", "COL",
    "CBJ", "DAL", "DET", "EDM", "FLA", "LAK", "MIN", "MTL",
    "NSH", "NJD", "NYI", "NYR", "OTT", "PHI", "PIT", "SJS",
    "STL", "TBL", "TOR", "VAN", "VGK", "WSH", "WPG", "SEA",
]


def get_json(url):
    """Send a GET request and return the response as JSON."""
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return response.json()


def get_game_ids(season):
    """Collect regular-season and playoff game IDs for one season."""
    game_ids = set()

    for team in TEAMS:
        url = f"{BASE_URL}/club-schedule-season/{team}/{season}"

        try:
            schedule = get_json(url)
        except requests.RequestException as error:
            print(f"Could not load the {team} schedule: {error}")
            continue

        for game in schedule.get("games", []):
            # Game type 2 is the regular season; game type 3 is the playoffs.
            if game.get("gameType") in (2, 3) and game.get("id"):
                game_ids.add(game["id"])

        time.sleep(0.1)

    return sorted(game_ids)


def download_season(season):
    """Download and cache play-by-play JSON files for one season."""
    season_dir = DATA_DIR / season
    season_dir.mkdir(parents=True, exist_ok=True)

    game_ids = get_game_ids(season)
    print(f"{season}: found {len(game_ids)} regular-season and playoff games")

    failed_ids = []

    for index, game_id in enumerate(game_ids, start=1):
        output_file = season_dir / f"{game_id}.json"

        # Skip files that have already been downloaded.
        if output_file.exists():
            continue

        url = f"{BASE_URL}/gamecenter/{game_id}/play-by-play"

        try:
            data = get_json(url)
            output_file.write_text(
                json.dumps(data, ensure_ascii=False),
                encoding="utf-8",
            )
        except (requests.RequestException, json.JSONDecodeError) as error:
            print(f"Could not download game {game_id}: {error}")
            failed_ids.append(str(game_id))

        if index % 100 == 0:
            print(f"Processed {index}/{len(game_ids)} games")

        time.sleep(0.01)

    if failed_ids:
        failed_file = season_dir / "failed_ids.txt"
        failed_file.write_text("\n".join(failed_ids), encoding="utf-8")
        print(f"Failed game IDs were saved to {failed_file}")

    print(f"Finished processing season {season}.")


if __name__ == "__main__":
    # Test the bulk download with one season first.    
    for year in range(2016, 2024):
      season = f"{year}{year + 1}"
      download_season(season)

