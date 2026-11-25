"""Tâche 7 : évaluer les 5 modèles retenus sur l'ensemble de test 2023/24.

Ce que fait le script : 

1. télécharge les 5 modèles depuis le registre W&B (3 régressions logistiques
   de la tâche 3, le meilleur XGBoost de la tâche 5, le modèle final de la tâche 6);
2. construit les données de test 2023/24 (saison régulière et séries) avec le
   code qui a servi à entraîner chaque modèle (tâche 2 pour les régressions
   logistiques, tâche 4/6 pour les autres), et garde les mêmes tirs pour tous;
3. calcule les métriques (avec intervalles de confiance par bootstrap sur les
   matchs), les mêmes sur la validation 2022/23 pour comparer, et les 8 figures;
4. enregistre tout dans "milestone2_outputs/task7/" et dans une expérience
   W&B "task7_test_evaluation".
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

from Jalon2_tache2_features import add_features
from tache3_tidydata import game_to_df

from .evaluation import KEY_COLUMNS, probability_metrics, save_predictions
from .features import PLAYOFFS, REGULAR_SEASON, TEST_SEASON, build_dataset
from .models import VALIDATION_SEASON, make_evaluation_figures
from .wandb_tracking import _init_run


DEFAULT_PROJECT = "ift3700-milestone-2"
SET_LABELS = {
    "validation": "Validation 2022-23",
    "regular": "Test 2023-24, saison régulière",
    "playoffs": "Test 2023-24, séries",
}


@dataclass(frozen=True)
class ModelSpec:
    """Un modèle à évaluer : son nom, sa collection dans le registre et son fichier local."""

    name: str
    collection: str
    local_path: Path


def model_specs(final: str, output_dir: Path, logreg_dir: Path) -> list[ModelSpec]:
    """Les 5 modèles demandés par l'énoncé, dans l'ordre des légendes."""

    specs = [
        ModelSpec(f"logreg_{name}", f"logreg_{name}", logreg_dir / f"logreg_{name}.joblib")
        for name in ("distance", "angle", "distance_angle")
    ]
    specs.append(ModelSpec("xgb_tuned_full", "xgb_tuned_full", output_dir / "models" / "xgb_tuned_full.joblib"))
    specs.append(ModelSpec(final, f"task6_{final}", output_dir / "task6" / "models" / f"{final}.joblib"))
    return specs


def model_features(model: Any) -> list[str]:
    """Colonnes attendues par un modèle, lues dans le modèle lui-même (``feature_names_in_``)."""

    for candidate in (model, getattr(model, "base", None)):
        names = getattr(candidate, "feature_names_in_", None)
        if names is not None:
            return [str(name) for name in names]
    raise ValueError(f"Impossible de trouver les colonnes attendues par {type(model).__name__}")


def load_models(
    specs: list[ModelSpec], *, source: str, run: Any, registry: str, alias: str, download_dir: Path
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Charge les modèles depuis le registre W&B (``use_artifact``) ou depuis le disque."""

    models: dict[str, Any] = {}
    origins: dict[str, dict[str, Any]] = {}
    for spec in specs:
        if source == "registry":
            artifact = run.use_artifact(f"{registry}/{spec.collection}:{alias}")
            folder = Path(artifact.download(root=str(download_dir / spec.collection)))
            files = sorted(folder.glob("*.joblib"))
            if len(files) != 1:
                raise FileNotFoundError(f"{spec.collection} : un seul fichier .joblib attendu, trouvé {files}")
            models[spec.name] = joblib.load(files[0])
            origins[spec.name] = {"source": "registry", "artifact": artifact.name, "version": artifact.version}
        else:
            if not spec.local_path.exists():
                raise FileNotFoundError(f"Modèle introuvable : {spec.local_path}")
            models[spec.name] = joblib.load(spec.local_path)
            origins[spec.name] = {"source": "local", "path": str(spec.local_path)}
        columns = model_features(models[spec.name])
        shown = ", ".join(columns) if len(columns) <= 3 else f"{len(columns)} caractéristiques"
        print(f"Modèle {spec.name} : {origins[spec.name]} ({shown})")
    return models, origins


def _through_csv(frame: pd.DataFrame, path: Path) -> pd.DataFrame:
    """Enregistre puis relit en CSV : les types (valeurs manquantes, booléens) sont
    ainsi exactement ceux que les modèles ont vus à l'entraînement, lui aussi lu en CSV."""

    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, compression="gzip")
    return pd.read_csv(path)


def merge_pipelines(task2: pd.DataFrame, task4: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Réunit les deux versions des caractéristiques sur (game_id, event_id).

    On ne garde que les tirs présents des deux côtés et ayant une distance et un
    angle (sans coordonnées, une régression logistique ne peut pas prédire) :
    les 5 modèles sont ainsi évalués exactement sur les mêmes tirs.
    """

    left = task2[KEY_COLUMNS + ["distance", "angle", "is_goal"]].rename(columns={"is_goal": "is_goal_task2"})
    merged = task4.merge(left, on=KEY_COLUMNS, how="inner", validate="one_to_one")
    mismatched = int((merged["is_goal"].astype(int) != merged["is_goal_task2"].astype(int)).sum())
    if mismatched:
        raise ValueError(f"{mismatched} tirs n'ont pas la même étiquette dans les deux pipelines")
    complete = merged.dropna(subset=["distance", "angle"]).drop(columns="is_goal_task2")
    complete["is_goal"] = complete["is_goal"].astype(int)
    info = {
        "task4_shots": int(len(task4)),
        "task2_shots": int(len(task2)),
        "unmatched": int(len(task4) + len(task2) - 2 * len(merged)),
        "excluded_missing_coordinates": int(len(merged) - len(complete)),
        "evaluated_shots": int(len(complete)),
    }
    return complete.reset_index(drop=True), info


def build_test_sets(data_dir: Path, output_dir: Path, *, strict: bool = True) -> dict[str, tuple[pd.DataFrame, dict]]:
    """Construit les deux ensembles de test 2023/24 (saison régulière et séries)."""

    season_dir = Path(data_dir) / TEST_SEASON
    print(f"Lecture des matchs {TEST_SEASON} (pipeline de la tâche 2)...", flush=True)
    raw = pd.concat([game_to_df(path) for path in sorted(season_dir.glob("*.json"))], ignore_index=True)
    print(f"Lecture des matchs {TEST_SEASON} (pipeline des tâches 4 et 6)...", flush=True)
    task4_all = build_dataset(data_dir, [TEST_SEASON], game_types=(REGULAR_SEASON, PLAYOFFS), strict=strict)

    sets: dict[str, tuple[pd.DataFrame, dict]] = {}
    for name, game_type in (("regular", REGULAR_SEASON), ("playoffs", PLAYOFFS)):
        task2 = add_features(raw[raw["game_type"] == game_type])
        task4 = _through_csv(
            task4_all[task4_all["game_type"] == game_type],
            output_dir / "task7" / f"test_features_{name}.csv.gz",
        )
        sets[name] = merge_pipelines(task2, task4)
    return sets


def build_validation_set(task2_csv: Path, task6_dataset: Path) -> tuple[pd.DataFrame, dict] | None:
    """Les tirs de validation 2022/23, pour comparer les mêmes métriques avant et après le test."""

    missing = [str(path) for path in (task2_csv, task6_dataset) if not Path(path).exists()]
    if missing:
        print(f"Attention : validation 2022-23 ignorée, fichiers introuvables : {missing}")
        return None
    task2 = pd.read_csv(task2_csv)
    task4 = pd.read_csv(task6_dataset)
    return merge_pipelines(
        task2[task2["season"] == VALIDATION_SEASON], task4[task4["season"] == VALIDATION_SEASON]
    )


def describe_set(frame: pd.DataFrame, info: dict[str, int]) -> dict[str, Any]:
    """Quelques statistiques sur un ensemble, utiles pour la discussion."""

    description = {
        "games": int(frame["game_id"].nunique()),
        "shots": int(len(frame)),
        "goals": int(frame["is_goal"].sum()),
        "goal_rate": float(frame["is_goal"].mean()),
        "empty_net_share": float(frame["empty_net"].mean()) if "empty_net" in frame else np.nan,
        "mean_shot_distance": float(frame["shot_distance"].mean()) if "shot_distance" in frame else np.nan,
    }
    description.update(info)
    return description


def bootstrap_intervals(
    frame: pd.DataFrame,
    probabilities: dict[str, np.ndarray],
    final: str,
    n_boot: int,
    seed: int = 42,
) -> dict[str, dict[str, float]]:
    """Intervalles de confiance à 95 % par bootstrap sur les matchs.

    On tire des matchs avec remise (et tous leurs tirs, car les tirs d'un même
    match ne sont pas indépendants), on recalcule les métriques, et on garde les
    percentiles 2,5 et 97,5. ``delta_*`` est l'écart avec le modèle final, calculé
    sur les mêmes tirages : un intervalle qui contient 0 indique un écart non significatif.
    """

    if n_boot <= 0:
        return {}
    y = frame["is_goal"].to_numpy()
    _, inverse = np.unique(frame["game_id"].to_numpy(), return_inverse=True)
    order = np.argsort(inverse, kind="stable")
    games = np.split(order, np.cumsum(np.bincount(inverse))[:-1])
    rng = np.random.default_rng(seed)
    samples: dict[str, dict[str, list[float]]] = {
        name: {"auc": [], "log_loss": [], "delta_auc": [], "delta_log_loss": []} for name in probabilities
    }
    for _ in range(n_boot):
        index = np.concatenate([games[g] for g in rng.integers(0, len(games), len(games))])
        y_boot = y[index]
        if y_boot.min() == y_boot.max():
            continue  # un tirage sans but (ou sans non-but) : AUC indéfinie
        scores = {
            name: (roc_auc_score(y_boot, p[index]), log_loss(y_boot, p[index], labels=[0, 1]))
            for name, p in probabilities.items()
        }
        for name, (auc, loss) in scores.items():
            samples[name]["auc"].append(auc)
            samples[name]["log_loss"].append(loss)
            samples[name]["delta_auc"].append(auc - scores[final][0])
            samples[name]["delta_log_loss"].append(loss - scores[final][1])
    intervals: dict[str, dict[str, float]] = {}
    for name, values in samples.items():
        intervals[name] = {}
        for metric, series in values.items():
            low, high = np.percentile(series, [2.5, 97.5]) if series else (np.nan, np.nan)
            intervals[name][f"{metric}_low"] = float(low)
            intervals[name][f"{metric}_high"] = float(high)
    return intervals


def _format_interval(value: float, low: float | None, high: float | None) -> str:
    if low is None or high is None or np.isnan(low):
        return f"{value:.4f}"
    return f"{value:.4f} [{low:.4f}, {high:.4f}]"


def summary_markdown(metrics: pd.DataFrame, descriptions: dict[str, dict[str, Any]], final: str) -> str:
    """Tableaux Markdown prêts à coller dans le blogue."""

    lines = ["## Données", "", "| Ensemble | Matchs | Tirs | Buts | Taux de buts | Filet désert | Distance moyenne | Tirs exclus |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name, item in descriptions.items():
        lines.append(
            f"| {SET_LABELS[name]} | {item['games']} | {item['shots']} | {item['goals']} | {item['goal_rate']:.2%} "
            f"| {item['empty_net_share']:.2%} | {item['mean_shot_distance']:.1f} pi | {item['excluded_missing_coordinates']} |"
        )
    for name in descriptions:
        with_delta = name != "validation" and "delta_log_loss_low" in metrics
        lines += ["", f"## {SET_LABELS[name]}", "",
                  "| Modèle | AUC [IC 95 %] | Log loss [IC 95 %] | Brier | ECE | xG / buts |"
                  + (" Écart de log loss avec le final [IC 95 %] |" if with_delta else ""),
                  "|---|---:|---:|---:|---:|---:|" + ("---:|" if with_delta else "")]
        rows = metrics[metrics["set"] == name]
        final_loss = float(rows.loc[rows["model"] == final, "log_loss"].iloc[0])
        for _, row in rows.iterrows():
            label = f"**{row['model']} (final)**" if row["model"] == final else row["model"]
            line = (
                f"| {label} | {_format_interval(row['auc'], row.get('auc_low'), row.get('auc_high'))} "
                f"| {_format_interval(row['log_loss'], row.get('log_loss_low'), row.get('log_loss_high'))} "
                f"| {row['brier']:.4f} | {row['ece']:.4f} | {row['xg_goal_ratio']:.3f} |"
            )
            if with_delta:
                delta = row["log_loss"] - final_loss
                # Un intervalle qui contient 0 : écart non significatif avec le modèle final.
                line += " — |" if row["model"] == final else (
                    f" {delta:+.4f} [{row['delta_log_loss_low']:+.4f}, {row['delta_log_loss_high']:+.4f}] |"
                )
            lines.append(line)
    pivot = metrics.pivot(index="model", columns="set", values=["auc", "log_loss"])
    order = [name for name in ("validation", "regular", "playoffs") if name in descriptions]
    lines += ["", "## Comparaison validation / test", "",
              "| Modèle | " + " | ".join(f"AUC {SET_LABELS[s]}" for s in order) + " | "
              + " | ".join(f"Log loss {SET_LABELS[s]}" for s in order) + " |",
              "|---|" + "---:|" * (2 * len(order))]
    for model in metrics["model"].unique():
        values = [f"{pivot.loc[model, ('auc', s)]:.4f}" for s in order]
        values += [f"{pivot.loc[model, ('log_loss', s)]:.4f}" for s in order]
        lines.append(f"| {model} | " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def evaluate(
    *,
    data_dir: Path = Path("data/raw"),
    output_dir: Path = Path("milestone2_outputs"),
    logreg_dir: Path = Path("data/models"),
    task2_csv: Path = Path("data/etape2_shots_train.csv"),
    task6_dataset: Path = Path("milestone2_outputs/task6_features_2016_2022.csv.gz"),
    final: str = "hgb_v2",
    source: str = "registry",
    alias: str = "latest",
    registry: str = "wandb-registry-model",
    n_boot: int = 500,
    wandb_mode: str = "online",
    wandb_project: str = DEFAULT_PROJECT,
    wandb_entity: str | None = None,
    strict: bool = True,
) -> dict[str, Any]:
    """Évalue les 5 modèles sur la validation 2022/23 et sur les deux ensembles de test 2023/24."""

    output_dir = Path(output_dir)
    folder = output_dir / "task7"
    if source == "registry" and wandb_mode != "online":
        raise ValueError("Le registre W&B demande --wandb-mode online (ou --source local pour un essai).")
    specs = model_specs(final, output_dir, Path(logreg_dir))

    run = None
    if wandb_mode != "disabled":
        run = _init_run(
            "task7_test_evaluation",
            wandb_project,
            wandb_entity,
            wandb_mode,
            job_type="evaluation",
            tags=["task-7", "test", "evaluation"],
            config={"test_season": TEST_SEASON, "validation_season": VALIDATION_SEASON, "final_model": final,
                    "source": source, "alias": alias, "bootstrap": n_boot,
                    "models": [spec.collection for spec in specs]},
        )
    models, origins = load_models(
        specs, source=source, run=run, registry=registry, alias=alias, download_dir=folder / "downloaded_models"
    )

    sets: dict[str, tuple[pd.DataFrame, dict]] = {}
    validation = build_validation_set(Path(task2_csv), Path(task6_dataset))
    if validation is not None:
        sets["validation"] = validation
    sets.update(build_test_sets(Path(data_dir), output_dir, strict=strict))

    rows: list[dict[str, Any]] = []
    descriptions: dict[str, dict[str, Any]] = {}
    figures: list[Path] = []
    for set_name, (frame, info) in sets.items():
        descriptions[set_name] = describe_set(frame, info)
        print(f"{SET_LABELS[set_name]} : {descriptions[set_name]}")
        probabilities = {
            name: model.predict_proba(frame[model_features(model)])[:, 1] for name, model in models.items()
        }
        intervals = bootstrap_intervals(frame, probabilities, final, n_boot if set_name != "validation" else 0)
        for name, probability in probabilities.items():
            save_predictions(frame, probability, folder / "predictions" / set_name / f"{name}.csv.gz")
            rows.append({"set": set_name, "model": name, **probability_metrics(frame["is_goal"], probability),
                         **intervals.get(name, {})})
        if set_name != "validation":
            labels = {name: (f"{name} (final)" if name == final else name) for name in probabilities}
            figures += make_evaluation_figures(
                frame["is_goal"],
                {labels[name]: probability for name, probability in probabilities.items()},
                output_dir / "figures",
                prefix=f"task7_{set_name}",
            )

    metrics = pd.DataFrame(rows)
    folder.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(folder / "metrics.csv", index=False)
    markdown = summary_markdown(metrics, descriptions, final)
    (folder / "summary.md").write_text(markdown, encoding="utf-8")
    print(markdown)

    url = None
    if run is not None:
        import wandb

        run.log({path.stem: wandb.Image(str(path)) for path in figures})
        run.log({"task7_metrics": wandb.Table(dataframe=metrics)})
        run.summary.update(
            {f"{row['set']}/{row['model']}/{metric}": row[metric]
             for row in rows for metric in ("auc", "log_loss", "brier", "ece", "xg_goal_ratio")}
        )
        url = run.url
        run.finish()

    record = {"models": origins, "sets": descriptions, "figures": [str(path) for path in figures], "wandb_url": url}
    (folder / "run.json").write_text(json.dumps(record, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    for path in figures:
        print(f"Figure : {path}")
    if url:
        print(f"W&B : {url}")
    return {**record, "metrics": metrics}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("milestone2_outputs"))
    parser.add_argument("--logreg-dir", type=Path, default=Path("data/models"),
                        help="Dossier des régressions logistiques (seulement avec --source local).")
    parser.add_argument("--task2-csv", type=Path, default=Path("data/etape2_shots_train.csv"))
    parser.add_argument("--task6-dataset", type=Path,
                        default=Path("milestone2_outputs/task6_features_2016_2022.csv.gz"))
    parser.add_argument("--final", default="hgb_v2", help="Modèle final de la tâche 6 (collection task6_<nom>).")
    parser.add_argument("--source", choices=["registry", "local"], default="registry")
    parser.add_argument("--alias", default="latest", help="Version à prendre dans le registre (latest, final, v0...).")
    parser.add_argument("--registry", default="wandb-registry-model")
    parser.add_argument("--bootstrap", type=int, default=500, help="Tirages du bootstrap (0 pour désactiver).")
    parser.add_argument("--wandb-mode", choices=["disabled", "offline", "online"], default="online")
    parser.add_argument("--wandb-project", default=DEFAULT_PROJECT)
    parser.add_argument("--wandb-entity")
    parser.add_argument("--no-strict", action="store_true",
                        help="Ne pas vérifier que la saison régulière 2023-24 est complète (1312 matchs).")
    args = parser.parse_args()
    evaluate(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        logreg_dir=args.logreg_dir,
        task2_csv=args.task2_csv,
        task6_dataset=args.task6_dataset,
        final=args.final,
        source=args.source,
        alias=args.alias,
        registry=args.registry,
        n_boot=args.bootstrap,
        wandb_mode=args.wandb_mode,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        strict=not args.no_strict,
    )


if __name__ == "__main__":
    main()