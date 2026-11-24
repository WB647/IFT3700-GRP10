"""Catalogue des expériences de la tâche 6.

Chaque tentative est une fonction décorée par ``@experiment``. La fonction
reçoit le DataFrame d'entraînement (2016/17-2021/22) et retourne le modèle
entraîné ainsi qu'un dictionnaire de paramètres à enregistrer dans W&B. Le
modèle doit accepter ``frame[features]`` dans ``predict_proba``. Le plus simple
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
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder
from xgboost import XGBClassifier

from .features import FEATURE_COLUMNS, FEATURE_COLUMNS_V2, NEW_FEATURE_COLUMNS
from .models import CATEGORICAL_FEATURES, make_pipeline


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
    """Informations transmises à chaque fonction d'expérience.

    ``figures_dir`` est le dossier où une expérience peut enregistrer des
    figures supplémentaires; elle retourne leurs chemins dans
    ``settings["figures"]`` pour qu'elles soient aussi envoyées à W&B.
    ``params`` contient les options passées en ligne de commande avec
    ``--param clé=valeur`` (p. ex. le nombre de configurations à essayer).
    """

    output_dir: Path
    features: list[str]
    figures_dir: Path | None = None
    params: dict[str, Any] = field(default_factory=dict)


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
# Tentative B : XGBoost mieux réglé, avec les caractéristiques de la tentative A.
# ---------------------------------------------------------------------------

def make_native_pipeline(features: list[str], model_params: dict[str, Any] | None = None) -> Pipeline:
    """Pipeline XGBoost qui laisse XGBoost gérer lui-même les valeurs manquantes.

    Contrairement à ``make_pipeline`` (tâche 5), les colonnes numériques ne sont
    pas remplacées par la médiane : elles sont seulement converties en nombres
    réels, et XGBoost apprend de quel côté de chaque séparation envoyer les
    valeurs manquantes. Le one-hot est dense pour que les zéros restent des
    zéros (dans une matrice creuse, XGBoost les traiterait comme manquants).
    """

    categorical = [name for name in features if name in CATEGORICAL_FEATURES]
    numeric = [name for name in features if name not in categorical]
    transformers: list[tuple[str, Any, list[str]]] = []
    if numeric:
        transformers.append(("numeric", FunctionTransformer(np.asarray, kw_args={"dtype": float}), numeric))
    if categorical:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical,
            )
        )
    params: dict[str, Any] = {"random_state": 42, "n_jobs": -1, "eval_metric": "logloss", "tree_method": "hist"}
    params.update(model_params or {})
    return Pipeline(
        [
            ("preprocessor", ColumnTransformer(transformers, remainder="drop", sparse_threshold=0.0)),
            ("model", XGBClassifier(**params)),
        ]
    )


# Espace de recherche : celui de la tâche 5, élargi, avec trois paramètres de régularisation
# (reg_lambda : pénalité L2, reg_alpha : pénalité L1, gamma : gain minimal pour séparer un nœud).
SEARCH_SPACE: dict[str, list[Any]] = {
    "model__n_estimators": [200, 300, 400, 600, 800],
    "model__max_depth": [3, 4, 5, 6, 7, 8],
    "model__learning_rate": [0.02, 0.03, 0.05, 0.07, 0.1],
    "model__min_child_weight": [1, 3, 5, 10, 20],
    "model__subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
    "model__colsample_bytree": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    "model__reg_lambda": [0.5, 1.0, 2.0, 5.0, 10.0],
    "model__reg_alpha": [0.0, 0.1, 0.5, 1.0],
    "model__gamma": [0.0, 0.1, 0.5, 1.0],
}


def tuning_figure(results: pd.DataFrame, context: Context, name: str) -> Path:
    """Log loss en validation croisée de chaque configuration, selon la profondeur et le taux d'apprentissage."""

    loss = -results["mean_test_score"]
    best = loss.idxmin()
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    panels = [
        ("param_model__max_depth", "Profondeur maximale (max_depth)"),
        ("param_model__learning_rate", "Taux d'apprentissage (learning_rate)"),
    ]
    for axis, (column, label) in zip(axes, panels):
        values = results[column].astype(float)
        axis.scatter(values, loss, alpha=0.7, label="configuration essayée")
        axis.scatter(values[best], loss[best], color="tab:red", s=90, label="meilleure configuration")
        axis.set_xlabel(label)
    axes[0].set_ylabel("Log loss moyenne en validation croisée (plus bas = mieux)")
    axes[1].legend()
    figure.suptitle(f"Recherche d'hyperparamètres : {len(results)} configurations, validation croisée par match")
    directory = context.figures_dir or (context.output_dir / "task6" / "figures" / "runs")
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}_tuning.png"
    figure.tight_layout()
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)
    return path


def _sample_games(train: pd.DataFrame, fraction: float, seed: int = 42) -> pd.DataFrame:
    """Garde une fraction des matchs (tous leurs tirs), pour accélérer la recherche."""

    if fraction >= 1:
        return train
    games = train["game_id"].unique()
    rng = np.random.default_rng(seed)
    chosen = rng.choice(games, size=max(2, int(len(games) * fraction)), replace=False)
    return train[train["game_id"].isin(chosen)]


@experiment(
    "xgb_tuned_v2",
    features=FEATURE_COLUMNS_V2,
    description="Tentative B : XGBoost réglé (log loss, validation croisée par match, valeurs manquantes natives)",
    tags=("xgboost", "features-v2", "attempt-b", "tuned"),
    extra_columns=["game_id"],
)
def xgb_tuned_v2(train: pd.DataFrame, context: Context) -> tuple[Any, dict[str, Any]]:
    """Recherche aléatoire sur une partie des matchs, puis réentraînement sur tout l'entraînement.

    Options (``--param``) : ``n_iter`` (configurations essayées, 30 par défaut),
    ``folds`` (plis de validation croisée, 3), ``game_fraction`` (part des matchs
    d'entraînement utilisée pour la recherche, 0.3).
    """

    n_iter = int(context.params.get("n_iter", 30))
    folds = int(context.params.get("folds", 3))
    fraction = float(context.params.get("game_fraction", 0.3))
    sample = _sample_games(train, fraction)
    print(
        f"[xgb_tuned_v2] recherche : {n_iter} configurations × {folds} plis sur "
        f"{sample['game_id'].nunique()} matchs ({len(sample)} tirs)"
    )

    # Chaque configuration est entraînée sur un seul fil; les configurations tournent en parallèle.
    search = RandomizedSearchCV(
        make_native_pipeline(context.features, {"n_jobs": 1}),
        SEARCH_SPACE,
        n_iter=n_iter,
        scoring="neg_log_loss",
        cv=GroupKFold(n_splits=folds),
        random_state=42,
        n_jobs=-1,
        refit=False,
        verbose=1,
    )
    search.fit(sample[context.features], sample["is_goal"], groups=sample["game_id"])

    results = pd.DataFrame(search.cv_results_)
    best_params = {key.removeprefix("model__"): value for key, value in search.best_params_.items()}
    model = make_native_pipeline(context.features, best_params)
    model.fit(train[context.features], train["is_goal"])

    figure = tuning_figure(results, context, "xgb_tuned_v2")
    table = results[[column for column in results.columns if column.startswith("param_")]].copy()
    table.columns = [column.removeprefix("param_model__") for column in table.columns]
    table["cv_log_loss"] = -results["mean_test_score"]
    table["cv_log_loss_std"] = results["std_test_score"]
    table.sort_values("cv_log_loss").to_csv(figure.with_suffix(".csv"), index=False)

    best = int(np.argmax(results["mean_test_score"]))
    return model, {
        "hyperparameters": best_params,
        "cv_log_loss": float(-results["mean_test_score"][best]),
        "cv_log_loss_std": float(results["std_test_score"][best]),
        "n_iter": n_iter,
        "folds": folds,
        "game_fraction": fraction,
        "search_games": int(sample["game_id"].nunique()),
        "search_shots": int(len(sample)),
        "figures": [figure],
    }


# ---------------------------------------------------------------------------
# Les tentatives C et D seront ajoutées ci-dessous.
# ---------------------------------------------------------------------------