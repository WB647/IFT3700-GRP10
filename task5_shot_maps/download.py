"""Cached, moderately concurrent downloader used only by Task 5.

This adapter reuses Task 1's API constants and request function without
altering that implementation. It downloads regular-season games only because
Task 5 is compared with regular-season standings.
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

from download_nhl_data import BASE_URL, TEAMS, get_json

from .generate import DEFAULT_SEASONS


def _with_retries(url: str, attempts: int = 5) -> dict:
    error: Exception | None = None
    for attempt in range(attempts):
        try:
            return get_json(url)
        except (requests.RequestException, json.JSONDecodeError) as exc:
            error = exc
            if attempt + 1 < attempts:
                retry_after = None
                if isinstance(exc, requests.HTTPError) and exc.response is not None:
                    retry_after = exc.response.headers.get("Retry-After")
                try:
                    delay = float(retry_after) if retry_after is not None else 2.0 * (2 ** attempt)
                except ValueError:
                    delay = 2.0 * (2 ** attempt)
                time.sleep(min(delay, 60.0))
    assert error is not None
    raise error


def regular_game_ids(season: str, workers: int = 2) -> list[int]:
    """Discover unique regular-season IDs through cached-compatible schedules."""

    ids: set[int] = set()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_with_retries, f"{BASE_URL}/club-schedule-season/{team}/{season}"): team
            for team in TEAMS
        }
        for future in as_completed(futures):
            team = futures[future]
            try:
                schedule = future.result()
            except (requests.RequestException, json.JSONDecodeError) as exc:
                print(f"Horaire indisponible pour {team} {season}: {exc}", flush=True)
                continue
            ids.update(
                int(game["id"])
                for game in schedule.get("games", [])
                if game.get("gameType") == 2 and game.get("id")
            )
    if not ids:
        raise RuntimeError(f"Aucun match de saison régulière trouvé pour {season}")
    return sorted(ids)


def download_regular_season(
    season: str,
    data_dir: Path | str = Path("data/raw"),
    *,
    workers: int = 2,
) -> tuple[int, list[int]]:
    """Download missing regular-season games, preserving Task 1's cache layout."""

    destination = Path(data_dir) / str(season)
    destination.mkdir(parents=True, exist_ok=True)
    ids = regular_game_ids(str(season), workers)
    missing = [game_id for game_id in ids if not (destination / f"{game_id}.json").exists()]
    print(f"{season}: {len(ids)} matchs, {len(missing)} à télécharger", flush=True)

    failures: list[int] = []

    def fetch(game_id: int) -> int:
        data = _with_retries(f"{BASE_URL}/gamecenter/{game_id}/play-by-play")
        target = destination / f"{game_id}.json"
        temporary = destination / f".{game_id}.json.tmp"
        temporary.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        temporary.replace(target)
        time.sleep(0.15)
        return game_id

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fetch, game_id): game_id for game_id in missing}
        completed = 0
        for future in as_completed(futures):
            game_id = futures[future]
            try:
                future.result()
            except (requests.RequestException, json.JSONDecodeError, OSError) as exc:
                failures.append(game_id)
                print(f"Échec {game_id}: {exc}", flush=True)
            completed += 1
            if completed % 100 == 0 or completed == len(missing):
                print(f"{season}: {completed}/{len(missing)} fichiers manquants traités", flush=True)

    failure_path = destination / "task5_failed_ids.txt"
    if failures:
        failure_path.write_text("\n".join(map(str, sorted(failures))), encoding="utf-8")
    elif failure_path.exists():
        failure_path.unlink()
    return len(ids), sorted(failures)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--seasons", nargs="+", default=list(DEFAULT_SEASONS))
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if not 1 <= args.workers <= 16:
        parser.error("--workers doit être compris entre 1 et 16")
    failed = False
    for season in args.seasons:
        _, failures = download_regular_season(season, args.data_dir, workers=args.workers)
        failed = failed or bool(failures)
    if failed:
        raise SystemExit("Certains téléchargements ont échoué; relancez la commande pour reprendre le cache.")


if __name__ == "__main__":
    main()
