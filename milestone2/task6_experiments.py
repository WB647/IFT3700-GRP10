"""Catalogue des expériences de la tâche 6.

Chaque tentative est une fonction décorée par "@experiment". La fonction
reçoit le DataFrame d'entraînement (2016/17-2021/22) et retourne le modèle
entraîné ainsi qu'un dictionnaire de paramètres à enregistrer dans W&B. Le
modèle doit accepter "frame[features]" dans "predict_proba". Le plus simple
est un Pipeline scikit-learn qui contient sa propre préparation des données :
la tâche 7 pourra alors l'utiliser directement sur les données 2023/24.

Pour ajouter une tentative ::

    @experiment(
        "xgb_new_features",
        features=FEATURE_COLUMNS + ["empty_net"],
        description="XGBoost, hyperparamètres de la tâche 5, nouvelles caractéristiques",
        tags=("xgboost", "features-v2"),
    )
    def xgb_new_features(train, context):
        model = make_pipeline(context.features, task5_best_params(context.output_dir))
        model.fit(train[context.features], train["is_goal"])
        return model, {"hyperparameters": task5_best_params(context.output_dir)}
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import joblib
import pandas as pd

from .features import FEATURE_COLUMNS
from .models import make_pipeline


# Meilleure configuration trouvée par la recherche aléatoire de la tâche 5 (task5_results.json).
TASK5_BEST_PARAMS: dict[str, Any] = {
    "n_estimators": 250,
    "max_depth": 5,
    "learning_rate": 0.07,
    "min_child_weight": 1,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
}


@dataclass(frozen=True)
class Context:
    """Informations transmises à chaque fonction d'expérience."""

    output_dir: Path
    features: list[str]


@dataclass(frozen=True)
class Experiment:
    name: str
    description: str
    features: list[str]
    train: Callable[[pd.DataFrame, Context], tuple[Any, dict[str, Any]]]
    tags: tuple[str, ...] = ()
    log_to_wandb: bool = True
    extra_columns: list[str] = field(default_factory=list)


EXPERIMENTS: dict[str, Experiment] = {}


def experiment(
    name: str,
    *,
    features: list[str],
    description: str,
    tags: tuple[str, ...] = (),
    log_to_wandb: bool = True,
    extra_columns: list[str] | None = None,
) -> Callable:
    """Enregistre une fonction d'expérience sous le nom ``name``.

    ``extra_columns`` liste les colonnes dont la fonction a besoin en plus des
    caractéristiques (par exemple ``game_id`` pour une validation croisée
    groupée par match).
    """

    def decorator(function: Callable[[pd.DataFrame, Context], tuple[Any, dict[str, Any]]]):
        if name in EXPERIMENTS:
            raise ValueError(f"Experiment {name!r} is defined twice")
        EXPERIMENTS[name] = Experiment(
            name=name,
            description=description,
            features=list(features),
            train=function,
            tags=tuple(tags),
            log_to_wandb=log_to_wandb,
            extra_columns=list(extra_columns or []),
        )
        return function

    return decorator


def task5_best_params(output_dir: Path) -> dict[str, Any]:
    """Hyperparamètres choisis à la tâche 5, lus dans task5_results.json s'il existe."""

    path = Path(output_dir) / "task5_results.json"
    try:
        saved = json.loads(path.read_text(encoding="utf-8"))["best_parameters"]
    except (FileNotFoundError, KeyError, json.JSONDecodeError):
        return dict(TASK5_BEST_PARAMS)
    return {key.removeprefix("model__"): value for key, value in saved.items()}


# ---------------------------------------------------------------------------
# Référence : le XGBoost réglé de la tâche 5
# ---------------------------------------------------------------------------

@experiment(
    "xgb_task5",
    features=FEATURE_COLUMNS,
    description="Référence : XGBoost complet réglé de la tâche 5 (déjà dans le registre)",
    tags=("xgboost", "reference"),
    log_to_wandb=False,
)
def xgb_task5(train: pd.DataFrame, context: Context) -> tuple[Any, dict[str, Any]]:
    """Réutilise le modèle sauvegardé de la tâche 5; le réentraîne avec les mêmes paramètres s'il manque."""

    path = context.output_dir / "models" / "xgb_tuned_full.joblib"
    if path.exists():
        return joblib.load(path), {"source": str(path), "refit": False}
    params = task5_best_params(context.output_dir)
    model = make_pipeline(context.features, params)
    model.fit(train[context.features], train["is_goal"])
    return model, {"source": "refit with task-5 hyperparameters", "refit": True, "hyperparameters": params}


# ---------------------------------------------------------------------------
# Les tentatives de la tâche 6 (A, B, C, D)
# ---------------------------------------------------------------------------
