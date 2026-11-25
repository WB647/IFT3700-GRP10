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

import copy
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
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, OrdinalEncoder, QuantileTransformer
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
    ``name`` est le nom sous lequel l'exécution est enregistrée : celui de
    l'expérience, ou celui donné avec ``--name`` pour une variante.
    """

    output_dir: Path
    features: list[str]
    figures_dir: Path | None = None
    params: dict[str, Any] = field(default_factory=dict)
    name: str = ""


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
    figure = importance_figure(model, train, context, context.name or "xgb_features_v2")
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

    figure = tuning_figure(results, context, context.name or "xgb_tuned_v2")
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
# Tentative C : autres familles de modèles, avec les caractéristiques de la tentative A.
# Les paramètres par défaut peuvent être changés avec --param (p. ex. --param alpha=0.01).
# ---------------------------------------------------------------------------

def _split_features(features: list[str]) -> tuple[list[str], list[str]]:
    """Sépare les caractéristiques numériques et catégorielles (type de tir, type du dernier événement)."""

    categorical = [name for name in features if name in CATEGORICAL_FEATURES]
    return [name for name in features if name not in categorical], categorical


def _override(defaults: dict[str, Any], context: Context) -> dict[str, Any]:
    """Remplace les valeurs par défaut par celles passées avec --param (clés connues seulement)."""

    params = dict(defaults)
    params.update({key: value for key, value in context.params.items() if key in defaults})
    return params


HGB_DEFAULTS: dict[str, Any] = {
    "learning_rate": 0.05,
    "max_iter": 500,
    "max_leaf_nodes": 31,
    "min_samples_leaf": 50,
    "l2_regularization": 1.0,
}


def make_hgb_pipeline(features: list[str], params: dict[str, Any] | None = None) -> Pipeline:
    """HistGradientBoosting de scikit-learn : valeurs manquantes et catégories gérées nativement.

    Les catégories sont numérotées (OrdinalEncoder) puis déclarées comme
    catégorielles au modèle, qui les sépare en groupes au lieu de les traiter
    comme des nombres. Une catégorie jamais vue devient une valeur manquante.
    L'arrêt précoce utilise 10 % de l'entraînement, jamais la validation 2022/23.
    """

    numeric, categorical = _split_features(features)
    transformers: list[tuple[str, Any, list[str]]] = [
        ("numeric", FunctionTransformer(np.asarray, kw_args={"dtype": float}), numeric)
    ]
    if categorical:
        encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan, dtype=float)
        transformers.append(("categorical", encoder, categorical))
    is_categorical = [False] * len(numeric) + [True] * len(categorical)
    model = HistGradientBoostingClassifier(
        **{**HGB_DEFAULTS, **(params or {})},
        categorical_features=is_categorical,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        scoring="loss",
        random_state=42,
    )
    return Pipeline(
        [("preprocessor", ColumnTransformer(transformers, remainder="drop", sparse_threshold=0.0)), ("model", model)]
    )


def make_scaled_preprocessor(features: list[str]) -> ColumnTransformer:
    """Préparation pour les modèles sensibles à l'échelle (réseau de neurones, régression logistique).

    Numérique : médiane + indicateur « valeur manquante », puis normalisation
    par quantiles (certaines variables, comme la vitesse, ont des valeurs
    extrêmes qui gêneraient l'apprentissage). Catégoriel : one-hot.
    """

    numeric, categorical = _split_features(features)
    transformers: list[tuple[str, Any, list[str]]] = [
        (
            "numeric",
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                    ("scaler", QuantileTransformer(output_distribution="normal", n_quantiles=1000, random_state=42)),
                ]
            ),
            numeric,
        )
    ]
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
    return ColumnTransformer(transformers, remainder="drop", sparse_threshold=0.0)


MLP_DEFAULTS: dict[str, Any] = {
    "hidden_layer_sizes": [64, 32],
    "alpha": 1e-3,
    "learning_rate_init": 1e-3,
    "batch_size": 512,
    "max_iter": 100,
    "patience": 10,
}


def fit_mlp_on_log_loss(
    train: pd.DataFrame, features: list[str], params: dict[str, Any], held_out_fraction: float = 0.1
) -> tuple[Pipeline, dict[str, Any]]:
    """Entraîne le réseau époque par époque et garde la meilleure époque selon la log loss.

    L'arrêt précoce intégré à scikit-learn surveille l'exactitude, qui bouge à
    peine quand environ 90 % des tirs ne sont pas des buts. Ici, 10 % des matchs
    d'entraînement sont mis de côté; après chaque époque, on calcule la log
    loss sur ces matchs et on garde les poids de la meilleure époque. On arrête
    après ``patience`` époques sans amélioration. La validation 2022/23 n'est
    jamais utilisée.
    """

    held_out = _sample_games(train, held_out_fraction, seed=7)
    fit_part = train[~train.index.isin(held_out.index)]
    preprocessor = make_scaled_preprocessor(features).fit(fit_part[features])
    x_fit, y_fit = preprocessor.transform(fit_part[features]), fit_part["is_goal"].to_numpy()
    x_held, y_held = preprocessor.transform(held_out[features]), held_out["is_goal"].to_numpy()
    monitor = np.random.default_rng(42).choice(len(x_fit), size=min(50_000, len(x_fit)), replace=False)

    network = MLPClassifier(
        hidden_layer_sizes=tuple(params["hidden_layer_sizes"]),
        alpha=params["alpha"],
        learning_rate_init=params["learning_rate_init"],
        batch_size=params["batch_size"],
        random_state=42,
    )
    history: list[dict[str, float]] = []
    best_loss, best_network, best_epoch = np.inf, None, 0
    for epoch in range(1, int(params["max_iter"]) + 1):
        network.partial_fit(x_fit, y_fit, classes=np.array([0, 1]))  # une époque complète
        held_loss = log_loss(y_held, network.predict_proba(x_held)[:, 1], labels=[0, 1])
        train_loss = log_loss(y_fit[monitor], network.predict_proba(x_fit[monitor])[:, 1], labels=[0, 1])
        history.append({"epoch": epoch, "train_log_loss": train_loss, "held_out_log_loss": held_loss})
        if held_loss < best_loss - 1e-5:
            best_loss, best_network, best_epoch = held_loss, copy.deepcopy(network), epoch
        elif epoch - best_epoch >= int(params["patience"]):
            break

    model = Pipeline([("preprocessor", preprocessor), ("model", best_network)])
    return model, {
        "epochs_run": len(history),
        "best_epoch": best_epoch,
        "held_out_log_loss": float(best_loss),
        "held_out_games": int(held_out["game_id"].nunique()),
        "history": history,
    }


def mlp_curve_figure(history: list[dict[str, float]], best_epoch: int, context: Context, name: str) -> Path:
    """Courbe d'apprentissage du réseau : log loss d'entraînement et des matchs mis de côté, par époque."""

    table = pd.DataFrame(history)
    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.plot(table["epoch"], table["train_log_loss"], label="entraînement (échantillon de 50 000 tirs)")
    axis.plot(table["epoch"], table["held_out_log_loss"], label="10 % des matchs mis de côté")
    axis.axvline(best_epoch, color="tab:red", linestyle="--", label=f"meilleure époque ({best_epoch}), poids conservés")
    axis.set_xlabel("Époque")
    axis.set_ylabel("Log loss (plus bas = mieux)")
    axis.set_title("Réseau de neurones : arrêt précoce sur la log loss")
    axis.legend()
    directory = context.figures_dir or (context.output_dir / "task6" / "figures" / "runs")
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}_learning_curve.png"
    figure.tight_layout()
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)
    return path


@experiment(
    "hgb_v2",
    features=FEATURE_COLUMNS_V2,
    description="Tentative C : HistGradientBoosting (scikit-learn), catégories natives, arrêt précoce",
    tags=("hist-gradient-boosting", "features-v2", "attempt-c"),
)
def hgb_v2(train: pd.DataFrame, context: Context) -> tuple[Any, dict[str, Any]]:
    params = _override(HGB_DEFAULTS, context)
    model = make_hgb_pipeline(context.features, params)
    model.fit(train[context.features], train["is_goal"])
    return model, {"hyperparameters": params, "iterations_used": int(model[-1].n_iter_)}


@experiment(
    "mlp_v2",
    features=FEATURE_COLUMNS_V2,
    description="Tentative C : réseau de neurones (MLP), normalisation par quantiles, arrêt précoce sur la log loss",
    tags=("neural-network", "mlp", "features-v2", "attempt-c"),
    extra_columns=["game_id"],
)
def mlp_v2(train: pd.DataFrame, context: Context) -> tuple[Any, dict[str, Any]]:
    params = _override(MLP_DEFAULTS, context)
    layers = params["hidden_layer_sizes"]
    # Accepte --param hidden_layer_sizes=64 (une couche) ou =[128,64] (plusieurs couches).
    params["hidden_layer_sizes"] = [int(layers)] if isinstance(layers, (int, float)) else [int(size) for size in layers]
    model, details = fit_mlp_on_log_loss(train, context.features, params)
    figure = mlp_curve_figure(details.pop("history"), details["best_epoch"], context, context.name or "mlp_v2")
    return model, {"hyperparameters": params, **details, "figures": [figure]}


@experiment(
    "logreg_v2",
    features=FEATURE_COLUMNS_V2,
    description="Tentative C : régression logistique sur toutes les caractéristiques (référence linéaire)",
    tags=("logistic-regression", "features-v2", "attempt-c"),
)
def logreg_v2(train: pd.DataFrame, context: Context) -> tuple[Any, dict[str, Any]]:
    params = _override({"C": 1.0, "max_iter": 2000}, context)
    model = Pipeline(
        [
            ("preprocessor", make_scaled_preprocessor(context.features)),
            ("model", LogisticRegression(C=params["C"], max_iter=params["max_iter"])),
        ]
    )
    model.fit(train[context.features], train["is_goal"])
    return model, {"hyperparameters": params}


# ---------------------------------------------------------------------------
# Tentative D : déséquilibre des classes et calibration.
# ---------------------------------------------------------------------------

def _fit_base(name: str, frame: pd.DataFrame, context: Context) -> tuple[Any, dict[str, Any]]:
    """Entraîne un des modèles XGBoost de base de la tentative D sur ``frame``.

    - ``xgb_features_v2`` : configuration de la tentative A;
    - ``xgb_weighted_v2`` : la même, avec les buts pondérés (``scale_pos_weight``);
    - ``xgb_tuned_v2`` : hyperparamètres trouvés par la tentative B (lancer B d'abord).
    """

    params = task5_best_params(context.output_dir)
    if name == "xgb_features_v2":
        model = make_pipeline(context.features, params)
        info: dict[str, Any] = {"hyperparameters": params}
    elif name == "xgb_weighted_v2":
        goals = int(frame["is_goal"].sum())
        weight = float(context.params.get("scale_pos_weight", (len(frame) - goals) / max(goals, 1)))
        model = make_pipeline(context.features, {**params, "scale_pos_weight": weight})
        info = {"hyperparameters": params, "scale_pos_weight": weight}
    elif name == "xgb_tuned_v2":
        path = context.output_dir / "task6" / "metrics" / "xgb_tuned_v2.json"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} est introuvable. Lance d'abord : python -m milestone2.train_task6 run --experiment xgb_tuned_v2"
            )
        params = json.loads(path.read_text(encoding="utf-8"))["settings"]["hyperparameters"]
        model = make_native_pipeline(context.features, params)
        info = {"hyperparameters": params}
    else:
        raise ValueError(f"Modèle de base inconnu : {name} (xgb_features_v2, xgb_weighted_v2 ou xgb_tuned_v2)")
    model.fit(frame[context.features], frame["is_goal"])
    return model, info


@experiment(
    "xgb_weighted_v2",
    features=FEATURE_COLUMNS_V2,
    description="Tentative D : XGBoost avec buts pondérés (scale_pos_weight = non-buts / buts), sinon comme A",
    tags=("xgboost", "features-v2", "attempt-d", "class-weight"),
)
def xgb_weighted_v2(train: pd.DataFrame, context: Context) -> tuple[Any, dict[str, Any]]:
    """Seule différence avec la tentative A : chaque but compte autant que ~9 non-buts à l'entraînement.

    Le classement des tirs (AUC) change peu, mais les probabilités sont
    gonflées : le modèle apprend sur une population où les buts semblent
    beaucoup plus fréquents qu'en réalité. À comparer avec xgb_features_v2.
    """

    return _fit_base("xgb_weighted_v2", train, context)


class CalibratedModel:
    """Modèle de base suivi d'une fonction de calibration apprise sur une saison mise de côté.

    ``isotonic`` : fonction croissante en escalier, très flexible (régression
    isotonique). ``sigmoid`` : méthode de Platt, une régression logistique sur
    le logit de la probabilité brute, plus rigide mais plus stable. Les
    probabilités sont bornées dans [1e-4, 1 - 1e-4] pour qu'un but ne reçoive
    jamais une probabilité nulle (log loss infinie).
    """

    EPSILON = 1e-4

    def __init__(self, base: Any, method: str) -> None:
        if method not in {"isotonic", "sigmoid"}:
            raise ValueError(f"Méthode de calibration inconnue : {method}")
        self.base = base
        self.method = method
        self.calibrator: Any = None
        self.classes_ = np.array([0, 1])

    @staticmethod
    def _logit(probability: np.ndarray) -> np.ndarray:
        clipped = np.clip(probability, 1e-6, 1 - 1e-6)
        return np.log(clipped / (1 - clipped)).reshape(-1, 1)

    def fit_calibrator(self, raw: np.ndarray, y: np.ndarray) -> "CalibratedModel":
        if self.method == "isotonic":
            self.calibrator = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(raw, y)
        else:
            self.calibrator = LogisticRegression(C=1e6, max_iter=1000).fit(self._logit(raw), y)
        return self

    def calibrate(self, raw: np.ndarray) -> np.ndarray:
        if self.method == "isotonic":
            calibrated = self.calibrator.predict(raw)
        else:
            calibrated = self.calibrator.predict_proba(self._logit(raw))[:, 1]
        return np.clip(calibrated, self.EPSILON, 1 - self.EPSILON)

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        probability = self.calibrate(self.base.predict_proba(frame)[:, 1])
        return np.column_stack([1 - probability, probability])


def calibration_map_figure(
    model: CalibratedModel, raw: np.ndarray, y: np.ndarray, context: Context, name: str, *, n_bins: int = 15
) -> Path:
    """Ce que la calibration a appris : probabilité brute (x) → probabilité calibrée (y).

    Les points sont les taux de buts observés par groupe de tirs de la saison
    de calibration; la courbe est la fonction apprise; la diagonale correspond
    à un modèle déjà parfaitement calibré.
    """

    edges = np.quantile(raw, np.linspace(0, 1, n_bins + 1))
    bins = np.clip(np.searchsorted(edges[1:-1], raw, side="right"), 0, n_bins - 1)
    observed = pd.DataFrame({"raw": raw, "goal": y, "bin": bins}).groupby("bin").mean()
    grid = np.linspace(raw.min(), raw.max(), 300)
    upper = float(max(raw.max(), observed["goal"].max())) * 1.05
    figure, axis = plt.subplots(figsize=(6.5, 6))
    axis.plot([0, upper], [0, upper], linestyle="--", color="gray", label="aucune correction (diagonale)")
    axis.scatter(observed["raw"], observed["goal"], color="tab:blue", label="taux de buts observé (groupes de tirs)")
    axis.plot(grid, model.calibrate(grid), color="tab:red", label=f"fonction apprise ({model.method})")
    axis.set_xlim(0, upper)
    axis.set_ylim(0, upper)
    axis.set_xlabel("Probabilité brute du modèle de base")
    axis.set_ylabel("Probabilité après calibration")
    axis.set_title("Calibration apprise sur la saison mise de côté")
    axis.legend()
    directory = context.figures_dir or (context.output_dir / "task6" / "figures" / "runs")
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}_calibration_map.png"
    figure.tight_layout()
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)
    return path


def _calibrated_experiment(method: str) -> Callable[[pd.DataFrame, Context], tuple[Any, dict[str, Any]]]:
    """Crée la fonction d'une expérience de calibration (isotonique ou sigmoïde)."""

    name = f"xgb_{method}_v2"

    def train_calibrated(train: pd.DataFrame, context: Context) -> tuple[Any, dict[str, Any]]:
        """Base entraînée sur toutes les saisons d'entraînement sauf la dernière, calibrée sur la dernière.

        La dernière saison d'entraînement (2021/22) sert à apprendre la
        calibration : le modèle de base ne l'a pas vue. La validation 2022/23
        reste réservée à l'évaluation. Option : ``--param base=xgb_tuned_v2``
        (ou ``xgb_weighted_v2``) pour calibrer un autre modèle que celui de A.
        """

        base_name = str(context.params.get("base", "xgb_features_v2"))
        seasons = sorted(int(season) for season in train["season"].unique())
        if len(seasons) < 2:
            raise ValueError("La calibration demande au moins deux saisons d'entraînement")
        calibration_season = seasons[-1]
        fit_part = train[train["season"] != calibration_season]
        calibration_part = train[train["season"] == calibration_season]
        base, base_info = _fit_base(base_name, fit_part, context)
        raw = base.predict_proba(calibration_part[context.features])[:, 1]
        model = CalibratedModel(base, method).fit_calibrator(raw, calibration_part["is_goal"].to_numpy())
        y_calibration = calibration_part["is_goal"].to_numpy()
        figure = calibration_map_figure(model, raw, y_calibration, context, context.name or name)
        return model, {
            "base": base_name,
            "base_settings": base_info,
            "method": method,
            "base_training_seasons": seasons[:-1],
            "calibration_season": calibration_season,
            "calibration_shots": int(len(calibration_part)),
            "figures": [figure],
        }

    return train_calibrated


for _method, _label in (("isotonic", "isotonique"), ("sigmoid", "sigmoïde (Platt)")):
    experiment(
        f"xgb_{_method}_v2",
        features=FEATURE_COLUMNS_V2,
        description=f"Tentative D : XGBoost (A par défaut, --param base=...) + calibration {_label} apprise sur 2021/22",
        tags=("xgboost", "features-v2", "attempt-d", "calibration", _method),
    )(_calibrated_experiment(_method))