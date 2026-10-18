import json
from pathlib import Path
 
import matplotlib.pyplot as plt
from ipywidgets import IntSlider, interact
from matplotlib.patches import Circle
 
# Paths are relative to the repository root, whatever folder the notebook is run from.
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "raw"
RINK_IMAGE = ROOT / "figures" / "nhl_rink.png"  # rink image (200 ft x 85 ft)
 
 
def get_seasons():
    """Return the season folders found in the data directory (e.g. 20162017)."""
    if not DATA_DIR.exists():
        return []
    return sorted(p.name for p in DATA_DIR.iterdir() if p.is_dir() and p.name.isdigit())
 
 
def get_games(season, game_type):
    """Return the JSON files of one season. game_type: '02' regular season, '03' playoffs."""
    return sorted((DATA_DIR / season).glob(f"{season[:4]}{game_type}*.json"))
 
 
def draw_rink(ax):
    """Draw the rink (x from -100 to 100 ft, y from -42.5 to 42.5 ft)."""
    if RINK_IMAGE.exists():
        ax.imshow(plt.imread(RINK_IMAGE), extent=[-100, 100, -42.5, 42.5])
    else:
        ax.plot([-100, 100, 100, -100, -100], [-42.5, -42.5, 42.5, 42.5, -42.5], "k")
        ax.axvline(0, color="red")  # center line
        ax.axvline(-25, color="blue")  # blue lines
        ax.axvline(25, color="blue")
        ax.axvline(-89, color="red", linewidth=1)  # goal lines
        ax.axvline(89, color="red", linewidth=1)
        ax.add_patch(Circle((0, 0), 15, fill=False, color="blue"))  # center circle
 
    ax.set_xlim(-100, 100)
    ax.set_ylim(-42.5, 42.5)
    ax.set_xticks(range(-100, 101, 25))
    ax.set_yticks([-42.5, -21.25, 0, 21.25, 42.5])
    ax.set_xlabel("feet")
    ax.set_ylabel("feet")
 
 
def show_event(season, game_type, game_index, event_index):
    """Print the details of one event and draw its coordinates on the rink."""
    games = get_games(season, game_type)
    if not games:
        print("No game found for this selection.")
        return
 
    # The sliders have a fixed maximum, so stay inside the real number of games/events.
    game_index = min(game_index, len(games) - 1)
    game = json.loads(games[game_index].read_text(encoding="utf-8"))
 
    plays = game.get("plays", [])
    if not plays:
        print("No event in this game.")
        return
    event_index = min(event_index, len(plays) - 1)
    play = plays[event_index]
    details = play.get("details", {})
 
    home, away = game["homeTeam"]["abbrev"], game["awayTeam"]["abbrev"]
    period = play["periodDescriptor"]["number"]
    print(game["gameDate"])
    print(f"Game ID: {game['id']}; {home} (home) vs {away} (away)  [{game_index + 1}/{len(games)}]")
    print(f"Event {event_index + 1}/{len(plays)}: {play['typeDescKey']}, "
          f"period {period}, {play['timeInPeriod']}")
    print(details)
 
    fig, ax = plt.subplots(figsize=(8, 3.5))
    draw_rink(ax)
 
    # Team names above the rink, on the side they defend (left/right of the rink).
    left, right = (home, away) if play.get("homeTeamDefendingSide") == "left" else (away, home)
    ax.text(-50, 44, left, ha="center", va="bottom")
    ax.text(50, 44, right, ha="center", va="bottom")
 
    title = f"{play['typeDescKey']}\n{play['timeInPeriod']} P-{period}"
    if "xCoord" in details and "yCoord" in details:
        ax.scatter(details["xCoord"], details["yCoord"], s=120, color="blue")
    else:
        title += " (no coordinates)"
    ax.set_title(title, y=1.1)
    plt.show()
 
 
def start():
    """Display the interactive tool (call it from a notebook cell)."""
    seasons = get_seasons()
    if not seasons:
        print(f"No season found in {DATA_DIR}. Run download_nhl_data.py first.")
        return
 
    interact(
        show_event,
        season=seasons,
        game_type=[("Saison régulière", "02"), ("Séries", "03")],
        game_index=IntSlider(min=0, max=1400),
        event_index=IntSlider(min=0, max=600),
    )
 