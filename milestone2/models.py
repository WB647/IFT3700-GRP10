"""XGBoost training, tuning, feature selection, and evaluation for Task 5."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score, roc_curve
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

from .features import FEATURE_COLUMNS


BASELINE_FEATURES = ["shot_distance", "shot_angle"]
CATEGORICAL_FEATURES = ["shot_type", "last_event_type"]
VALIDATION_SEASON = 20222023


def make_logistic_reference() -> Pipeline:
    """Reproduce the Task-3 distance/angle logistic reference for comparison."""

    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("model", LogisticRegression(max_iter=1000)),
        ]
    )


def split_train_validation(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Use 2022/23 only for validation and every earlier requested season for training."""

    seasons = pd.to_numeric(frame["season"], errors="coerce")
    train = frame.loc[seasons < VALIDATION_SEASON].copy()
    validation = frame.loc[seasons == VALIDATION_SEASON].copy()
    if train.empty or validation.empty:
        raise ValueError("Training seasons and validation season 2022/23 must both be present")
    return train, validation


def make_pipeline(features: list[str], model_params: dict[str, Any] | None = None) -> Pipeline:
    """Build a leakage-free preprocessing/XGBoost pipeline."""

    categorical = [name for name in features if name in CATEGORICAL_FEATURES]
    numeric = [name for name in features if name not in categorical]
    transformers: list[tuple[str, Pipeline, list[str]]] = []
    if numeric:
        transformers.append(
            ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric)
        )
    if categorical:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical,
            )
        )
    params: dict[str, Any] = {
        "random_state": 42,
        "n_jobs": -1,
        "eval_metric": "logloss",
        "tree_method": "hist",
    }
    params.update(model_params or {})
    return Pipeline(
        [
            ("preprocessor", ColumnTransformer(transformers, remainder="drop")),
            ("model", XGBClassifier(**params)),
        ]
    )


def model_metrics(y_true: pd.Series, probabilities: np.ndarray) -> dict[str, float]:
    predictions = (probabilities >= 0.5).astype(int)
    return {
        "auc": float(roc_auc_score(y_true, probabilities)),
        "log_loss": float(log_loss(y_true, probabilities)),
        "accuracy": float(accuracy_score(y_true, predictions)),
    }


def tune_full_model(
    train: pd.DataFrame,
    *,
    n_iter: int = 12,
    cv: int = 3,
    random_state: int = 42,
) -> RandomizedSearchCV:
    """Tune the full Task-4 feature set using ROC AUC cross-validation."""

    # RandomizedSearchCV parallelizes candidates; each candidate stays single-threaded
    # to avoid multiplying the number of worker threads.
    pipeline = make_pipeline(FEATURE_COLUMNS, {"n_jobs": 1})
    parameters = {
        "model__n_estimators": [150, 250, 400],
        "model__max_depth": [3, 5, 7],
        "model__learning_rate": [0.03, 0.07, 0.12],
        "model__min_child_weight": [1, 5, 10],
        "model__subsample": [0.7, 0.9, 1.0],
        "model__colsample_bytree": [0.7, 0.9, 1.0],
    }
    search = RandomizedSearchCV(
        pipeline,
        parameters,
        n_iter=n_iter,
        scoring="roc_auc",
        cv=cv,
        random_state=random_state,
        n_jobs=-1,
        verbose=1,
        return_train_score=False,
    )
    search.fit(train[FEATURE_COLUMNS], train["is_goal"])
    return search


def select_features(
    model: Pipeline,
    evaluation_frame: pd.DataFrame,
    *,
    sample_size: int = 50_000,
) -> tuple[list[str], pd.Series]:
    """Select original columns using permutation importance on training data."""

    sample = evaluation_frame.sample(min(sample_size, len(evaluation_frame)), random_state=42)
    result = permutation_importance(
        model,
        sample[FEATURE_COLUMNS],
        sample["is_goal"],
        scoring="roc_auc",
        n_repeats=3,
        random_state=42,
        n_jobs=1,
    )
    importance = pd.Series(result.importances_mean, index=FEATURE_COLUMNS).sort_values(ascending=False)
    positive = importance[importance > 0]
    threshold = float(positive.median()) if not positive.empty else 0.0
    selected = importance[importance >= threshold].index.tolist()
    for required in BASELINE_FEATURES:
        if required not in selected:
            selected.append(required)
    return selected, importance


def _percentile_curve(y: np.ndarray, probability: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    percentiles = pd.Series(probability).rank(method="average", pct=True).to_numpy() * 100.0
    bins = np.arange(0, 105, 5)
    grouped = pd.DataFrame({"y": y, "p": percentiles}).groupby(
        pd.cut(percentiles, bins, include_lowest=True), observed=True
    )["y"].mean()
    return np.array([interval.mid for interval in grouped.index]), grouped.to_numpy()


def make_evaluation_figures(
    y_true: pd.Series,
    probabilities: dict[str, np.ndarray],
    output_dir: Path,
) -> list[Path]:
    """Create the four figures required by Tasks 3 and 5."""

    output_dir.mkdir(parents=True, exist_ok=True)
    y = y_true.to_numpy(dtype=int)
    paths: list[Path] = []

    for name, probability in probabilities.items():
        fpr, tpr, _ = roc_curve(y, probability)
        plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y, probability):.3f})")
    plt.plot([0, 1], [0, 1], "k--", label="Aléatoire")
    plt.xlabel("Taux de faux positifs")
    plt.ylabel("Taux de vrais positifs")
    plt.legend()
    paths.append(_save_plot(output_dir / "task5_roc_auc.png"))

    for name, probability in probabilities.items():
        x, rate = _percentile_curve(y, probability)
        plt.plot(x, rate, marker="o", label=name)
    plt.xlabel("Centile de probabilité du modèle")
    plt.ylabel("Taux de buts")
    plt.legend()
    paths.append(_save_plot(output_dir / "task5_goal_rate_percentile.png"))

    for name, probability in probabilities.items():
        order = np.argsort(-probability)
        ordered_goals = y[order]
        fraction_shots = np.arange(1, len(y) + 1) / len(y) * 100.0
        cumulative = np.cumsum(ordered_goals) / max(ordered_goals.sum(), 1)
        plt.plot(fraction_shots, cumulative, label=name)
    plt.xlabel("Pourcentage des tirs, du xG le plus élevé au plus faible")
    plt.ylabel("Proportion cumulative des buts")
    plt.legend()
    paths.append(_save_plot(output_dir / "task5_cumulative_goals.png"))

    plt.plot([0, 1], [0, 1], "k--", label="Calibration parfaite")
    for name, probability in probabilities.items():
        fraction, mean = calibration_curve(y, probability, n_bins=15, strategy="quantile")
        plt.plot(mean, fraction, marker="o", label=name)
    plt.xlabel("Probabilité moyenne prédite")
    plt.ylabel("Fréquence observée des buts")
    plt.legend()
    paths.append(_save_plot(output_dir / "task5_reliability.png"))
    return paths


def save_tuning_figure(search: RandomizedSearchCV, destination: Path) -> Path:
    results = pd.DataFrame(search.cv_results_).sort_values("rank_test_score").head(10).iloc[::-1]
    labels = [f"essai {index}" for index in results["rank_test_score"].astype(int)]
    plt.barh(labels, results["mean_test_score"])
    plt.xlabel("AUC moyenne en validation croisée")
    plt.title("Dix meilleures configurations XGBoost")
    return _save_plot(destination)


def save_importance_figure(importance: pd.Series, destination: Path) -> Path:
    importance.sort_values().plot.barh(figsize=(8, 6))
    plt.xlabel("Diminution moyenne de l'AUC après permutation")
    plt.title("Importance des caractéristiques originales")
    return _save_plot(destination)


def _save_plot(destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(destination, dpi=160, bbox_inches="tight")
    plt.close()
    return destination


def save_model(model: Any, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, destination)
    return destination


def write_results(payload: dict[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
