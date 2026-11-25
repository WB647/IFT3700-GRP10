"""Tâche 6 : lancer, comparer et suivre les expériences « Faites de votre mieux ».

Commandes (depuis la racine du projet) ::

    python -m milestone2.train_task6 list
    python -m milestone2.train_task6 build-dataset
    python -m milestone2.train_task6 run --experiment xgb_task5
    python -m milestone2.train_task6 run --experiment <name> --wandb-mode online --register
    python -m milestone2.train_task6 run --experiment xgb_tuned_v2 --param n_iter=40
    python -m milestone2.train_task6 run --experiment xgb_isotonic_v2 --param base=xgb_weighted_v2 --name xgb_weighted_isotonic
    python -m milestone2.train_task6 figures --curves xgb_task5 <name> ... --final <name>

Chaque ``run`` utilise la séparation de la tâche 5 (entraînement 2016/17-2021/22,
validation 2022/23), enregistre les probabilités de validation, les métriques
et le modèle dans ``milestone2_outputs/task6/``, et peut envoyer l'expérience
à W&B. ``figures`` construit ensuite les quatre figures du blogue à partir des
probabilités enregistrées, sans rien réentraîner. La saison de test 2023/24
n'est jamais lue.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter
from typing import Any

import pandas as pd

from .evaluation import (
    load_aligned_predictions,
    probability_metrics,
    save_predictions,
    summary_table,
    to_markdown,
)
from .features import TRAIN_SEASONS, build_dataset
from .models import VALIDATION_SEASON, make_evaluation_figures, save_model, split_train_validation
from .task6_experiments import EXPERIMENTS, Context
from .wandb_tracking import log_model_run


DEFAULT_DATASET = Path("milestone2_outputs/task6_features_2016_2022.csv.gz")
DEFAULT_OUTPUT_DIR = Path("milestone2_outputs")
DEFAULT_PROJECT = "ift3700-milestone-2"
REFERENCE = "xgb_task5"
REQUIRED_COLUMNS = ["season", "game_id", "event_id", "is_goal"]


def task6_dir(output_dir: Path) -> Path:
    return Path(output_dir) / "task6"


def prediction_path(output_dir: Path, name: str) -> Path:
    return task6_dir(output_dir) / "predictions" / f"{name}.csv.gz"


def _jsonable(value: Any) -> Any:
    """Convertit les valeurs numpy (et le reste) pour pouvoir les enregistrer en JSON."""

    return json.loads(json.dumps(value, default=lambda item: item.item() if hasattr(item, "item") else str(item)))


def build_task6_dataset(data_dir: Path, output: Path) -> Path:
    """Reconstruit le jeu de données 2016/17-2022/23 (saison régulière) avec les caractéristiques actuelles."""

    frame = build_dataset(data_dir, TRAIN_SEASONS)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False, compression="gzip")
    print(f"Jeu de données : {output} ({len(frame)} tirs, {len(frame.columns)} colonnes)")
    return output


def load_dataset(path: Path, columns: list[str]) -> pd.DataFrame:
    """Lit le jeu de données et vérifie que toutes les colonnes nécessaires sont présentes."""

    if not Path(path).exists():
        raise FileNotFoundError(
            f"{path} est introuvable. Lance d'abord : python -m milestone2.train_task6 build-dataset"
        )
    frame = pd.read_csv(path)
    missing = sorted((set(columns) | set(REQUIRED_COLUMNS)) - set(frame.columns))
    if missing:
        raise ValueError(
            f"Colonnes absentes de {path} : {missing}. "
            "Reconstruis le jeu de données avec : python -m milestone2.train_task6 build-dataset"
        )
    frame = frame.dropna(subset=["is_goal"]).copy()
    frame["is_goal"] = frame["is_goal"].astype(int)
    return frame


def run_experiment(
    name: str,
    *,
    dataset: Path = DEFAULT_DATASET,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    wandb_mode: str = "disabled",
    wandb_project: str = DEFAULT_PROJECT,
    wandb_entity: str | None = None,
    register: bool = False,
    params: dict[str, Any] | None = None,
    run_name: str | None = None,
) -> dict[str, Any]:
    """Entraîne une expérience, l'évalue sur 2022/23, enregistre tout et l'envoie à W&B.

    ``params`` contient les options propres à l'expérience (``--param clé=valeur``).
    ``run_name`` (``--name``) enregistre l'exécution sous un autre nom que celui
    de l'expérience, pour garder plusieurs variantes côte à côte (fichiers,
    figures et expérience W&B ``task6_<nom>``).
    """

    if name not in EXPERIMENTS:
        raise KeyError(f"Expérience inconnue : {name}. Disponibles : {sorted(EXPERIMENTS)}")
    spec = EXPERIMENTS[name]
    experiment_name, name = name, (run_name or name)
    output_dir = Path(output_dir)
    frame = load_dataset(dataset, spec.features + spec.extra_columns)
    train, validation = split_train_validation(frame)
    print(f"[{name}] {spec.description}")
    print(f"[{name}] entraînement : {len(train)} tirs, validation : {len(validation)} tirs")

    folder = task6_dir(output_dir)
    context = Context(
        output_dir=output_dir,
        features=spec.features,
        figures_dir=folder / "figures" / "runs",
        params=dict(params or {}),
        name=name,
    )
    start = perf_counter()
    model, settings = spec.train(train, context)
    train_seconds = perf_counter() - start
    # Figures supplémentaires produites par l'expérience (p. ex. importance des caractéristiques).
    extra_figures = [Path(path) for path in settings.pop("figures", [])]
    start = perf_counter()
    probability = model.predict_proba(validation[spec.features])[:, 1]
    predict_seconds = perf_counter() - start

    metrics = probability_metrics(validation["is_goal"], probability)
    metrics.update(train_seconds=round(train_seconds, 2), predict_seconds=round(predict_seconds, 2))
    save_predictions(validation, probability, prediction_path(output_dir, name))
    model_path = save_model(model, folder / "models" / f"{name}.joblib")
    figure_paths = _run_figures(name, output_dir) + extra_figures

    config = _jsonable(
        {
            "experiment": experiment_name,
            "run_name": name,
            "description": spec.description,
            "features": spec.features,
            "training_seasons": sorted(train["season"].astype(int).unique().tolist()),
            "validation_season": VALIDATION_SEASON,
            "params": dict(params or {}),
            "settings": settings,
        }
    )
    url = None
    if wandb_mode != "disabled" and spec.log_to_wandb:
        url = log_model_run(
            name=f"task6_{name}",
            model_path=model_path,
            metrics=metrics,
            config=config,
            figure_paths=figure_paths,
            project=wandb_project,
            entity=wandb_entity,
            mode=wandb_mode,
            register=register,
            tags=["task-6", *spec.tags],
        )
    elif wandb_mode != "disabled":
        print(f"[{name}] non envoyé à W&B (expérience de référence, déjà suivie à la tâche 5)")

    record = {**config, "metrics": metrics, "wandb_url": url, "registered": bool(url and register)}
    metrics_path = folder / "metrics" / f"{name}.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(_jsonable(record), indent=2, ensure_ascii=False), encoding="utf-8")

    print(
        f"[{name}] AUC {metrics['auc']:.4f} | log loss {metrics['log_loss']:.4f} | "
        f"Brier {metrics['brier']:.4f} | ECE {metrics['ece']:.4f} | xG/buts {metrics['xg_goal_ratio']:.3f} | "
        f"entraînement {train_seconds:.1f} s"
    )
    if url:
        print(f"[{name}] W&B : {url}")
    return record


def parse_params(values: list[str] | None) -> dict[str, Any]:
    """Transforme ``["n_iter=40", "game_fraction=0.5"]`` en ``{"n_iter": 40, "game_fraction": 0.5}``."""

    params: dict[str, Any] = {}
    for item in values or []:
        if "=" not in item:
            raise ValueError(f"--param attend clé=valeur, reçu : {item!r}")
        key, raw = item.split("=", 1)
        try:
            params[key.strip()] = json.loads(raw)
        except json.JSONDecodeError:
            params[key.strip()] = raw
    return params


def _run_figures(name: str, output_dir: Path) -> list[Path]:
    """Quatre figures pour une expérience, à côté de la référence de la tâche 5 si elle existe."""

    paths = {name: prediction_path(output_dir, name)}
    reference = prediction_path(output_dir, REFERENCE)
    if name != REFERENCE and reference.exists():
        paths = {f"{REFERENCE} (référence)": reference, **paths}
    y, probabilities = load_aligned_predictions(paths)
    return make_evaluation_figures(y, probabilities, task6_dir(output_dir) / "figures" / "runs", prefix=name)


def make_task6_figures(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    curves: list[str] | None = None,
    final: str | None = None,
) -> dict[str, Any]:
    """Construit les quatre figures du blogue et le tableau récapitulatif à partir des prédictions enregistrées."""

    folder = task6_dir(output_dir) / "predictions"
    available = sorted(path.name.removesuffix(".csv.gz") for path in folder.glob("*.csv.gz"))
    curves = list(curves or available)
    if final and final not in curves:
        curves.append(final)
    missing = [name for name in curves if name not in available]
    if missing:
        raise FileNotFoundError(f"Aucune prédiction pour {missing}. Lance d'abord : run --experiment {missing[0]}")

    def label(name: str) -> str:
        if name == final:
            return f"{name} (final)"
        if name == REFERENCE:
            return f"{name} (référence)"
        return name

    y, probabilities = load_aligned_predictions({label(name): prediction_path(output_dir, name) for name in curves})
    figures = make_evaluation_figures(y, probabilities, Path(output_dir) / "figures", prefix="task6")
    table = summary_table(y, probabilities)
    table.to_csv(task6_dir(output_dir) / "summary.csv")
    markdown = to_markdown(table)
    (task6_dir(output_dir) / "summary.md").write_text(markdown, encoding="utf-8")
    print(markdown)
    for path in figures:
        print(f"Figure : {path}")
    return {"figures": figures, "summary": table}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("list", help="Lister les expériences disponibles.")

    build = commands.add_parser("build-dataset", help="Reconstruire le jeu de données 2016/17-2022/23.")
    build.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    build.add_argument("--output", type=Path, default=DEFAULT_DATASET)

    run = commands.add_parser("run", help="Entraîner et évaluer une ou plusieurs expériences.")
    run.add_argument("--experiment", nargs="+", required=True, choices=sorted(EXPERIMENTS))
    run.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    run.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    run.add_argument("--wandb-mode", choices=["disabled", "offline", "online"], default="disabled")
    run.add_argument("--wandb-project", default=DEFAULT_PROJECT)
    run.add_argument("--wandb-entity")
    run.add_argument("--register", action="store_true", help="Lier le modèle au registre W&B (mode online).")
    run.add_argument(
        "--name",
        help="Enregistrer l'exécution sous ce nom (une seule expérience), p. ex. pour une variante.",
    )
    run.add_argument(
        "--param",
        action="append",
        metavar="CLÉ=VALEUR",
        help="Option propre à l'expérience, p. ex. --param n_iter=40 (répétable).",
    )

    figures = commands.add_parser("figures", help="Construire les 4 figures et le tableau du blogue.")
    figures.add_argument("--curves", nargs="+", help="Expériences à afficher (par défaut : toutes).")
    figures.add_argument("--final", help="Expérience à mettre en évidence comme modèle final.")
    figures.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)

    args = parser.parse_args()
    if args.command == "list":
        for name, spec in sorted(EXPERIMENTS.items()):
            wandb_note = "" if spec.log_to_wandb else " [pas envoyée à W&B]"
            print(f"{name:<28} {len(spec.features):>2} caractéristiques  {spec.description}{wandb_note}")
    elif args.command == "build-dataset":
        build_task6_dataset(args.data_dir, args.output)
    elif args.command == "run":
        if args.name and len(args.experiment) > 1:
            parser.error("--name ne s'utilise qu'avec une seule expérience")
        for name in args.experiment:
            run_experiment(
                name,
                dataset=args.dataset,
                output_dir=args.output_dir,
                wandb_mode=args.wandb_mode,
                wandb_project=args.wandb_project,
                wandb_entity=args.wandb_entity,
                register=args.register,
                params=parse_params(args.param),
                run_name=args.name,
            )
    else:
        make_task6_figures(args.output_dir, args.curves, args.final)


if __name__ == "__main__":
    main()