"""CLI for Task 4 feature generation and the required W&B dataset artifact."""

from __future__ import annotations

import argparse
from pathlib import Path

from .features import TRAIN_SEASONS, build_dataset, build_game_features
from .wandb_tracking import log_dataset_artifact


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("milestone2_outputs"))
    parser.add_argument("--dataset", action="store_true", help="Also build all 2016/17-2022/23 data.")
    parser.add_argument("--wandb-mode", choices=["disabled", "offline", "online"], default="disabled")
    parser.add_argument("--wandb-project", default="ift3700-milestone-2")
    parser.add_argument("--wandb-entity")
    args = parser.parse_args()

    artifact_dir = args.output_dir / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    game_path = args.data_dir / "20172018" / "2017021065.json"
    game = build_game_features(game_path)
    csv_path = artifact_dir / "wpg_v_wsh_2017021065.csv"
    game.to_csv(csv_path, index=False)
    url = log_dataset_artifact(
        game,
        csv_path,
        project=args.wandb_project,
        entity=args.wandb_entity,
        mode=args.wandb_mode,
    )
    print(f"Artifact CSV: {csv_path} ({len(game)} rows)")
    if url:
        print(f"W&B run: {url}")

    if args.dataset:
        dataset = build_dataset(args.data_dir, TRAIN_SEASONS)
        dataset_path = args.output_dir / "task4_features_2016_2022.csv.gz"
        dataset.to_csv(dataset_path, index=False, compression="gzip")
        print(f"Training dataset: {dataset_path} ({len(dataset)} rows)")


if __name__ == "__main__":
    main()
