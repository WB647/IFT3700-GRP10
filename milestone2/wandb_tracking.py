"""Optional Weights & Biases integration without hard-coded credentials."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd


def _ensure_writable_wandb_dirs() -> None:
    """Keep W&B cache/staging files in the project instead of user AppData."""

    root = Path.cwd() / "wandb"
    locations = {
        "WANDB_DIR": root,
        "WANDB_DATA_DIR": root / "data",
        "WANDB_CACHE_DIR": root / "cache",
        "WANDB_CONFIG_DIR": root / "config",
    }
    for variable, directory in locations.items():
        directory.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault(variable, str(directory))


def _init_run(name: str, project: str, entity: str | None, mode: str, **kwargs: Any):
    _ensure_writable_wandb_dirs()
    import wandb

    init_mode = "online" if mode == "online" else mode
    return wandb.init(
        project=project,
        entity=entity or os.environ.get("WANDB_ENTITY"),
        name=name,
        mode=init_mode,
        **kwargs,
    )


def log_dataset_artifact(
    frame: pd.DataFrame,
    csv_path: Path,
    *,
    project: str,
    entity: str | None = None,
    mode: str = "disabled",
) -> str | None:
    """Log the required game table under its exact assignment name."""

    if mode == "disabled":
        return None
    _ensure_writable_wandb_dirs()
    import wandb

    run = _init_run(
        "task4-wpg-v-wsh-2017021065",
        project,
        entity,
        mode,
        tags=["task-4", "dataset", "2017021065"],
        config={"game_id": 2017021065, "rows": len(frame), "columns": list(frame.columns)},
    )
    artifact = wandb.Artifact("wpg_v_wsh_2017021065", type="dataset")
    artifact.add(wandb.Table(dataframe=frame), "wpg_v_wsh_2017021065")
    artifact.add_file(str(csv_path), name=csv_path.name)
    run.log_artifact(artifact)
    url = run.url
    run.finish()
    return url


def log_model_run(
    *,
    name: str,
    model_path: Path,
    metrics: dict[str, float],
    config: dict[str, Any],
    figure_paths: list[Path],
    project: str,
    entity: str | None,
    mode: str,
    register: bool,
) -> str | None:
    """Track one experiment and optionally link its model to the registry."""

    if mode == "disabled":
        return None
    _ensure_writable_wandb_dirs()
    import wandb

    run = _init_run(
        name,
        project,
        entity,
        mode,
        tags=["task-5", "xgboost", name],
        config=config,
    )
    run.log(metrics)
    run.log({path.stem: wandb.Image(str(path)) for path in figure_paths})
    artifact = wandb.Artifact(name, type="model", metadata={"metrics": metrics, **config})
    artifact.add_file(str(model_path))
    logged = run.log_artifact(artifact)
    if register and mode == "online":
        run.link_artifact(logged, target_path=f"model-registry/{name}")
    url = run.url
    run.finish()
    return url
