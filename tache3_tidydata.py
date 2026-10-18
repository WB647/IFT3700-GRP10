import json
from pathlib import Path
 
import pandas as pd
 
DATA_DIR = Path(__file__).resolve().parent / "data" / "raw"
 
 
def game_to_df(path):
    """Convert the shots and goals of one game JSON file into a DataFrame (one row per event)."""
    game = json.loads(Path(path).read_text(encoding="utf-8"))
    home, away = game["homeTeam"], game["awayTeam"]
 
    # Player names are not in the events, only in rosterSpots.
    names = {p["playerId"]: f"{p['firstName']['default']} {p['lastName']['default']}"
             for p in game.get("rosterSpots", [])}
 
    rows = []
    for play in game["plays"]:
        if play["typeDescKey"] not in ("shot-on-goal", "goal"):
            continue
        if play["periodDescriptor"]["periodType"] == "SO":  # skip the shootout
            continue
 
        d = play.get("details", {})
        is_home = d.get("eventOwnerTeamId") == home["id"]
        shooter_id = d.get("scoringPlayerId") if play["typeDescKey"] == "goal" else d.get("shootingPlayerId")
        goalie_id = d.get("goalieInNetId")
 
        # situationCode = [away goalie][away skaters][home skaters][home goalie]
        own = opp = empty_net = strength = None
        code = play.get("situationCode")
        if code and len(code) == 4:
            away_g, away_s, home_s, home_g = (int(c) for c in code)
            own, opp = (home_s, away_s) if is_home else (away_s, home_s)
            empty_net = (away_g if is_home else home_g) == 0  # goalie of the defending team
            strength = "even" if own == opp else ("powerplay" if own > opp else "shorthanded")
 
        rows.append({
            "season": game["season"],
            "game_type": game["gameType"],
            "game_id": game["id"],
            "event_id": play["eventId"],
            "period": play["periodDescriptor"]["number"],
            "time_in_period": play["timeInPeriod"],
            "team": home["abbrev"] if is_home else away["abbrev"],
            "is_goal": play["typeDescKey"] == "goal",
            "x": d.get("xCoord"),
            "y": d.get("yCoord"),
            "shooter": names.get(shooter_id),
            "goalie": names.get(goalie_id),
            "shot_type": d.get("shotType"),
            "empty_net": empty_net,
            "strength": strength,
            "strength_state": f"{own}v{opp}" if own is not None else None,
        })
    return pd.DataFrame(rows)
 
 
def season_to_df(season): 
    """Convert all the games of one season (e.g. '20162017') into one DataFrame. Used for the blog, only the first 10 lines are printed."""
    files = sorted((DATA_DIR / str(season)).glob("*.json"))
    return pd.concat([game_to_df(f) for f in files], ignore_index=True)

"""for task 4-5"""
def all_seasons_to_df(): 
    """Convert all the downloaded seasons into one DataFrame."""
    seasons = sorted(p.name for p in DATA_DIR.iterdir() if p.is_dir())
    return pd.concat([season_to_df(s) for s in seasons], ignore_index=True)
 
 
if __name__ == "__main__":
    df = season_to_df("20162017")
    print(df.head(10))
    print(len(df), "events")
    print(df.isna().sum())  # missing values per column
    print(df["game_type"].value_counts())
    print("buts :", df["is_goal"].sum())
    print("filets déserts :", df["empty_net"].sum())
    print("gardien manquant sans filet désert :", (df["goalie"].isna() & ~df["empty_net"]).sum())
 