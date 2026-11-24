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
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.inspection import permutation_importance

from .features import FEATURE_COLUMNS, FEATURE_COLUMNS_V2, NEW_FEATURE_COLUMNS
from .models import make_pipeline


#Meilleure configuration trouvée par la recherche aléatoire de la tâche 5 (task5_results.json).
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
    """Informations transmises à chaque fonction d'expérience.

    "figures_dir" est le dossier où une expérience peut enregistrer des
    figures supplémentaires; elle retourne leurs chemins dans
    "settings["figures"]" pour qu'elles soient aussi envoyées à W&B.
    """

    output_dir: Path
    features: list[str]
    figures_dir: Path | None = None


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

    "extra_columns" liste les colonnes dont la fonction a besoin en plus des
    caractéristiques (par exemple "game_id" pour une validation croisée
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
# Référence : le XGBoost réglé de la tâche 5, auquel chaque tentative est comparée.
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
# Figures communes aux tentatives
# ---------------------------------------------------------------------------

def importance_figure(
    model: Any,
    train: pd.DataFrame,
    context: Context,
    name: str,
    *,
    sample_size: int = 50_000,
) -> Path:
    """Importance par permutation, calculée sur un échantillon de l'entraînement.

    Comme à la tâche 5, la mesure est la baisse d'AUC quand on mélange une
    colonne, et la validation 2022/23 n'est pas utilisée. Les nouvelles
    caractéristiques de la tâche 6 sont en orange, celles de la tâche 4 en bleu.
    """

    sample = train.sample(min(sample_size, len(train)), random_state=42)
    result = permutation_importance(
        model,
        sample[context.features],
        sample["is_goal"],
        scoring="roc_auc",
        n_repeats=3,
        random_state=42,
        n_jobs=1,
    )
    importance = pd.Series(result.importances_mean, index=context.features).sort_values()
    colors = ["tab:orange" if feature in NEW_FEATURE_COLUMNS else "tab:blue" for feature in importance.index]
    figure, axis = plt.subplots(figsize=(8, 7))
    axis.barh(importance.index, importance.to_numpy(), color=colors)
    axis.set_xlabel("Diminution moyenne de l'AUC après permutation")
    axis.set_title("Importance des caractéristiques (orange : nouvelles, tâche 6)")
    directory = context.figures_dir or (context.output_dir / "task6" / "figures" / "runs")
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}_feature_importance.png"
    figure.tight_layout()
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)
    importance.sort_values(ascending=False).to_csv(directory / f"{name}_feature_importance.csv", header=["auc_drop"])
    return path


# ---------------------------------------------------------------------------
# Tentative A : nouvelles caractéristiques, mêmes hyperparamètres que la tâche 5.
# Seules les caractéristiques changent : tout écart avec xgb_task5 vient d'elles.
# ---------------------------------------------------------------------------

@experiment(
    "xgb_features_v2",
    features=FEATURE_COLUMNS_V2,
    description="Tentative A : XGBoost (hyperparamètres de la tâche 5) avec les nouvelles caractéristiques",
    tags=("xgboost", "features-v2", "attempt-a"),
)
def xgb_features_v2(train: pd.DataFrame, context: Context) -> tuple[Any, dict[str, Any]]:
    params = task5_best_params(context.output_dir)
    model = make_pipeline(context.features, params)
    model.fit(train[context.features], train["is_goal"])
    figure = importance_figure(model, train, context, "xgb_features_v2")
    return model, {"hyperparameters": params, "new_features": NEW_FEATURE_COLUMNS, "figures": [figure]}


# ---------------------------------------------------------------------------
# Les tentatives B, C et D seront ajoutées ci-dessous.
# ---------------------------------------------------------------------------