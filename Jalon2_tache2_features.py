from pathlib import Path
 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm
 
from tache3_tidydata import DATA_DIR, season_to_df
 
ROOT = Path(__file__).resolve().parent
NET_X = 89  # the goal line is 89 feet from the center of the rink
 
 
def load_train_data():
    """Regular season 2016-17 to 2022-23 (2023-24 is the test set, it is never read)."""
    seasons = sorted(p.name for p in DATA_DIR.iterdir() if p.is_dir() and p.name < "20232024")
    df = pd.concat([season_to_df(s) for s in seasons], ignore_index=True)
    return df[df["game_type"] == 2]
 
 
def add_features(df):
    """Add distance and angle to the net, and keep goal (0/1) and empty net (0/1)."""
    # Direction of attack of each team in each period: the side where it takes most of its shots.
    # (homeTeamDefendingSide is missing in some seasons and looks wrong in some games.)
    direction = np.sign(df.groupby(["game_id", "period", "team"])["x"].transform("median")).replace(0, np.nan)
 
    # Check against the official field, where it exists (+1 = the team attacks the right net).
    field = df["home_defending_side"].map({"left": 1, "right": -1})
    field = field.where(df["is_home"], -field)
    known = field.notna() & direction.notna()
    print(f"Events without homeTeamDefendingSide: {field.isna().sum()}")
    print(f"Agreement with homeTeamDefendingSide: {(field[known] == direction[known]).mean():.1%}")
 
    ahead = NET_X - direction * df["x"]  # feet in front of the goal line (negative = behind the net)
    return df.assign(
        distance=np.hypot(ahead, df["y"]),
        angle=np.degrees(np.arctan2(df["y"].abs(), ahead)),  # 0 = straight in front of the net
        is_goal=df["is_goal"].astype(int),
        empty_net=df["empty_net"].astype("boolean").astype("Int64"),  # keeps missing values
    )[["season", "game_id", "event_id", "period", "time_in_period", "x", "y",
       "distance", "angle", "is_goal", "empty_net"]]
 
 
def save(name):
    (ROOT / "figures").mkdir(exist_ok=True)
    plt.savefig(ROOT / "figures" / name, dpi=150, bbox_inches="tight")
    plt.close()
 
 
def make_figures(data):
    data = data.dropna(subset=["distance", "angle"])
    goal = data["is_goal"] == 1
 
    for col, bins, label in [("distance", np.arange(0, 205, 5), "Distance from the net (feet)"),
                             ("angle", np.arange(0, 185, 5), "Angle from the net (degrees)")]:
        # Q1: number of shots (goals and non-goals apart); log scale because goals are rare
        plt.hist([data[col][~goal], data[col][goal]], bins=bins, label=["No goal", "Goal"])
        plt.yscale("log")
        plt.xlabel(label)
        plt.ylabel("Number of shots (log scale)")
        plt.legend()
        save(f"etape2_t2_q1_{col}_hist.png")
 
        # Q2: goal rate = goals / (goals + non-goals), ignoring bins with fewer than 50 shots
        rate = data["is_goal"].groupby(pd.cut(data[col], bins), observed=True).agg(["mean", "count"])
        rate = rate[rate["count"] >= 50]
        plt.plot([interval.mid for interval in rate.index], rate["mean"], marker="o")
        plt.xlabel(label)
        plt.ylabel("Goal rate (goals / shots)")
        save(f"etape2_t2_q2_goal_rate_{col}.png")
 
    # Q1: 2D histogram, distance vs angle
    plt.hist2d(data["distance"], data["angle"], bins=[np.arange(0, 205, 5), np.arange(0, 185, 5)], norm=LogNorm())
    plt.colorbar(label="Number of shots (log scale)")
    plt.xlabel("Distance from the net (feet)")
    plt.ylabel("Angle from the net (degrees)")
    save("etape2_t2_q1_distance_angle_2d.png")
 
    # Q3: goals only, empty net vs not empty
    goals = data[goal & data["empty_net"].notna()]
    plt.hist([goals["distance"][goals["empty_net"] == 0], goals["distance"][goals["empty_net"] == 1]],
             bins=np.arange(0, 205, 5), label=["Net not empty", "Empty net"])
    plt.yscale("log")
    plt.xlabel("Distance from the net (feet)")
    plt.ylabel("Number of goals (log scale)")
    plt.legend()
    save("etape2_t2_q3_goals_empty_net.png")
 
 
if __name__ == "__main__":
    data = add_features(load_train_data())
    print(len(data), "events")
    print("Missing values:", data[["distance", "angle", "empty_net"]].isna().sum().to_dict())
    data.to_csv(ROOT / "data" / "etape2_shots_train.csv", index=False)  # data/ is not tracked by git
    make_figures(data)
 
    # Goals from far away on a net that was not empty are very rare: check them by hand.
    far = data[(data["is_goal"] == 1) & (data["empty_net"] == 0) & (data["distance"] > 100)]
    print(far.sort_values("distance", ascending=False).head(15).to_string())

    print(len(far), "goals from more than 100 feet with a non-empty net")