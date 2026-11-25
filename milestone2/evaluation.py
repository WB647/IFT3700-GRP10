"""Métriques de probabilité et prédictions communes aux tâches 6 et 7."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score


KEY_COLUMNS = ["game_id", "event_id"]


def expected_calibration_error(y_true: Iterable[int], probability: Iterable[float], n_bins: int = 15) -> float:
    """Écart moyen entre la probabilité prédite et le taux de buts observé (ECE).

    Les tirs sont répartis en "n_bins" groupes de même taille,
    comme dans le diagramme de fiabilité. Chaque groupe compte en proportion
    de son nombre de tirs.
    """

    y = np.asarray(y_true, dtype=float)
    p = np.asarray(probability, dtype=float)
    edges = np.quantile(p, np.linspace(0.0, 1.0, n_bins + 1))
    bins = np.clip(np.searchsorted(edges[1:-1], p, side="right"), 0, n_bins - 1)
    error = 0.0
    for index in range(n_bins):
        mask = bins == index
        if mask.any():
            error += mask.mean() * abs(y[mask].mean() - p[mask].mean())
    return float(error)


def probability_metrics(y_true: Iterable[int], probability: Iterable[float]) -> dict[str, float]:
    """Métriques d'un modèle de probabilité de but.

    "log_loss", "brier" et "ece" jugent les probabilités elles-mêmes
    (plus bas = mieux); "auc" juge le classement des tirs (plus haut = mieux);
    "xg_goal_ratio" est la somme des probabilités prédites divisée par le
    nombre de buts (1 = autant de buts espérés que de buts réels).
    """

    y = np.asarray(y_true, dtype=int)
    p = np.asarray(probability, dtype=float)
    return {
        "auc": float(roc_auc_score(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "brier": float(brier_score_loss(y, p)),
        "ece": expected_calibration_error(y, p),
        "xg_goal_ratio": float(p.sum() / max(y.sum(), 1)),
        "accuracy": float(accuracy_score(y, (p >= 0.5).astype(int))),
        "shots": int(len(y)),
        "goals": int(y.sum()),
    }


def save_predictions(frame: pd.DataFrame, probability: np.ndarray, destination: Path) -> Path:
    """Enregistre une probabilité par tir, identifiée par (game_id, event_id)."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    table = frame[KEY_COLUMNS + ["is_goal"]].copy()
    table["probability"] = np.asarray(probability, dtype=float)
    table.to_csv(destination, index=False)
    return destination


def load_aligned_predictions(paths: dict[str, Path]) -> tuple[pd.Series, dict[str, np.ndarray]]:
    """Charge plusieurs fichiers de prédictions et garde seulement les tirs communs.

    Retourne les vraies étiquettes et un tableau de probabilités par nom, tous
    dans le même ordre : chaque courbe d'une figure porte ainsi sur les mêmes tirs.
    """

    if not paths:
        raise ValueError("No prediction file to load")
    merged: pd.DataFrame | None = None
    for name, path in paths.items():
        table = pd.read_csv(path).set_index(KEY_COLUMNS)
        table = table.rename(columns={"probability": name, "is_goal": f"is_goal__{name}"})
        merged = table if merged is None else merged.join(table, how="inner")
    assert merged is not None
    label_columns = [column for column in merged.columns if column.startswith("is_goal__")]
    labels = merged[label_columns[0]].astype(int)
    for column in label_columns[1:]:
        if not merged[column].astype(int).equals(labels):
            raise ValueError(f"The goal labels differ between prediction files ({column})")
    largest = max(len(pd.read_csv(path, usecols=KEY_COLUMNS)) for path in paths.values())
    if len(merged) < largest:
        print(f"Attention : {largest - len(merged)} tirs absents d'au moins un fichier ont été ignorés.")
    return labels.rename("is_goal"), {name: merged[name].to_numpy() for name in paths}


def summary_table(y_true: pd.Series, probabilities: dict[str, np.ndarray]) -> pd.DataFrame:
    """Une ligne de métriques par modèle, triée par log loss (le meilleur en premier)."""

    rows = {name: probability_metrics(y_true, probability) for name, probability in probabilities.items()}
    table = pd.DataFrame(rows).T
    columns = ["auc", "log_loss", "brier", "ece", "xg_goal_ratio", "accuracy", "shots", "goals"]
    return table[columns].sort_values("log_loss")


def to_markdown(table: pd.DataFrame) -> str:
    """Tableau Markdown prêt à coller dans le blogue (sans dépendance supplémentaire)."""

    header = "| Modèle | AUC ↑ | Log loss ↓ | Brier ↓ | ECE ↓ | xG / buts |"
    lines = [header, "|---|---:|---:|---:|---:|---:|"]
    for name, row in table.iterrows():
        lines.append(
            f"| {name} | {row['auc']:.4f} | {row['log_loss']:.4f} | {row['brier']:.4f} "
            f"| {row['ece']:.4f} | {row['xg_goal_ratio']:.3f} |"
        )
    return "\n".join(lines) + "\n"
