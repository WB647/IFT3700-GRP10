from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from tache3_tidydata import season_to_df

from pathlib import Path

FIGURES_DIR = Path(__file__).parent / "Figures"
FIGURES_DIR.mkdir(exist_ok=True)


df = season_to_df("20182019")

summary = (
    df.dropna(subset=["shot_type"])
    .groupby("shot_type", as_index=False)
    .agg(
        shots=("is_goal", "size"),
        goals=("is_goal", "sum"),
    )
)
summary["goal_percentage"] = 100 * summary["goals"] / summary["shots"]
summary = summary.sort_values("shots", ascending=False)

x = np.arange(len(summary))
width = 0.38
fig, ax1 = plt.subplots(figsize=(10, 6))

ax1.bar(x - width / 2, summary["shots"], width, label="Shots")
ax1.bar(x + width / 2, summary["goals"], width, label="Goals")
ax1.set_xlabel("Shot type")
ax1.set_ylabel("Number of events")
ax1.set_xticks(x)
ax1.set_xticklabels(summary["shot_type"], rotation=30, ha="right")
ax1.legend(loc="upper left")

ax2 = ax1.twinx()
ax2.plot(
    x,
    summary["goal_percentage"],
    color="red",
    marker="o",
    label="Goal percentage",
)
ax2.set_ylabel("Goals / shots (%)")
ax2.legend(loc="upper right")

plt.title("Shots and goals by shot type, 2018–19")
plt.tight_layout()

Path("Figures").mkdir(exist_ok=True)
plt.savefig("Figures/task4_q1_20182019.png", dpi=200)
plt.show()

print(summary.sort_values("goal_percentage", ascending=False))

def plot_goal_probability_by_distance():
    seasons = ["20182019", "20192020", "20202021"]
    bin_width = 5

    fig, ax = plt.subplots(figsize=(10, 6))

    for season in seasons:
        df = season_to_df(season)

        # Keep events with usable coordinates and goal labels
        df = df.dropna(subset=["x", "y", "is_goal"]).copy()

        # Estimate distance to the nearer goal at x = -89 or x = 89
        distance_to_left_goal = ((df["x"] + 89) ** 2 + df["y"] ** 2) ** 0.5
        distance_to_right_goal = ((df["x"] - 89) ** 2 + df["y"] ** 2) ** 0.5
        df["distance"] = distance_to_left_goal.combine(
            distance_to_right_goal, min
        )

        # Group shots into 5-foot distance bins
        df["distance_bin"] = (df["distance"] // bin_width) * bin_width
        summary = df.groupby("distance_bin")["is_goal"].mean() * 100

        ax.plot(
            summary.index,
            summary.values,
            marker="o",
            label=f"{season[:4]}–{season[4:]}"
        )

    ax.set_title("Goal probability by shot distance")
    ax.set_xlabel("Distance to the nearer goal (feet)")
    ax.set_ylabel("Goals / recorded shots (%)")
    ax.grid(True, alpha=0.3)
    ax.legend(title="Season")
    fig.tight_layout()

    output_path = FIGURES_DIR / "task4_q2_goal_probability_by_distance.png"
    fig.savefig(output_path, dpi=300)
    print(f"Saved figure to {output_path}")
    plt.show()


plot_goal_probability_by_distance()

def plot_goal_percentage_by_distance_and_type():
    season = "20182019"
    df = season_to_df(season)

    df = df.dropna(subset=["x", "y", "is_goal", "shot_type"]).copy()

    distance_to_left_goal = ((df["x"] + 89) ** 2 + df["y"] ** 2) ** 0.5
    distance_to_right_goal = ((df["x"] - 89) ** 2 + df["y"] ** 2) ** 0.5
    df["distance"] = distance_to_left_goal.combine(distance_to_right_goal, min)

    bin_width = 10
    df["distance_bin"] = (df["distance"] // bin_width) * bin_width

    grouped = (
        df.groupby(["distance_bin", "shot_type"], observed=True)["is_goal"]
        .agg(["mean", "count"])
        .reset_index()
    )

    # Hide cells with fewer than 20 shots to reduce noise
    grouped.loc[grouped["count"] < 20, "mean"] = float("nan")

    summary = grouped.pivot(
        index="distance_bin",
        columns="shot_type",
        values="mean"
    ) * 100

    cmap = plt.cm.YlOrRd.copy()
    cmap.set_bad("lightgray")

    values = np.ma.masked_invalid(summary.to_numpy())

    fig, ax = plt.subplots(figsize=(10, 7))
    image = ax.imshow(
        values,
        aspect="auto",
        origin="lower",
        cmap=cmap,
        vmin=0,
        vmax=25
    )

    ax.set_xticks(range(len(summary.columns)))
    ax.set_xticklabels(summary.columns, rotation=35, ha="right")
    ax.set_yticks(range(len(summary.index)))
    ax.set_yticklabels([f"{int(distance)}–{int(distance) + bin_width}" for distance in summary.index])

    ax.set_title("Goal percentage by distance and shot type, 2018–19")
    ax.set_xlabel("Shot type")
    ax.set_ylabel("Distance to the nearer goal (feet)")

    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("Goals / recorded shots (%)")

    fig.tight_layout()

    output_path = FIGURES_DIR / "task4_q3_goal_percentage_distance_and_type.png"
    fig.savefig(output_path, dpi=300)
    print(f"Saved figure to {output_path}")
    plt.show()


plot_goal_percentage_by_distance_and_type()