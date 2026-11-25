import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from Jalon2_tache2_features import add_features
from milestone2.evaluate_test import (
    build_test_sets,
    evaluate,
    merge_pipelines,
    model_features,
)
from milestone2.features import FEATURE_COLUMNS, FEATURE_COLUMNS_V2, build_dataset
from milestone2.models import make_pipeline
from milestone2.task6_experiments import CalibratedModel, make_hgb_pipeline
from tache3_tidydata import game_to_df


def random_game(game_id: int, season: int, game_type: int, rng: np.random.Generator, shots: int = 24) -> dict:
    """Match synthétique : 24 tirs alternés entre les deux équipes, sur 3 périodes.

    L'équipe locale attaque le filet de droite (x > 0), l'équipe visiteuse celui
    de gauche; la probabilité de but diminue avec la distance. Le dernier tir du
    match n'a pas de coordonnées, pour vérifier qu'il est exclu de l'évaluation.
    """

    plays = [{"eventId": 1, "typeDescKey": "faceoff", "timeInPeriod": "00:00",
              "periodDescriptor": {"number": 1, "periodType": "REG"},
              "details": {"eventOwnerTeamId": 1, "xCoord": 0, "yCoord": 0}}]
    per_period = shots // 3
    for k in range(shots):
        home = k % 2 == 0
        distance = rng.uniform(5, 60)
        goal = rng.random() < 0.6 * np.exp(-distance / 15)
        seconds = 30 + 60 * (k % per_period)
        details = {
            "eventOwnerTeamId": 1 if home else 2,
            "xCoord": int((89 - 0.8 * distance) * (1 if home else -1)),
            "yCoord": int(rng.uniform(-25, 25)),
            "zoneCode": "O",
            "shotType": str(rng.choice(["wrist", "snap", "slap"])),
            ("scoringPlayerId" if goal else "shootingPlayerId"): 11 if home else 21,
        }
        if k == shots - 1:
            del details["xCoord"], details["yCoord"]
        plays.append({
            "eventId": k + 2,
            "typeDescKey": "goal" if goal else "shot-on-goal",
            "timeInPeriod": f"{seconds // 60:02d}:{seconds % 60:02d}",
            "periodDescriptor": {"number": 1 + k // per_period, "periodType": "REG"},
            "homeTeamDefendingSide": "left",
            "situationCode": "1551",
            "details": details,
        })
    return {"id": game_id, "season": season, "gameType": game_type, "rosterSpots": [],
            "homeTeam": {"id": 1, "abbrev": "MTL"}, "awayTeam": {"id": 2, "abbrev": "TOR"}, "plays": plays}


def write_season(data_dir: Path, season: int, games: dict[int, int], rng: np.random.Generator) -> None:
    """Écrit ``games[type]`` matchs synthétiques de chaque type pour une saison."""

    folder = data_dir / str(season)
    folder.mkdir(parents=True, exist_ok=True)
    for game_type, count in games.items():
        for number in range(1, count + 1):
            game_id = int(f"{str(season)[:4]}0{game_type}{number:04d}")
            game = random_game(game_id, season, game_type, rng)
            (folder / f"{game_id}.json").write_text(json.dumps(game), encoding="utf-8")


@pytest.fixture()
def project(tmp_path: Path) -> dict[str, Path]:
    """Un petit projet complet : données brutes, données d'entraînement/validation et 5 modèles locaux."""

    rng = np.random.default_rng(0)
    data_dir = tmp_path / "data" / "raw"
    write_season(data_dir, 20212022, {2: 40}, rng)  # entraînement
    write_season(data_dir, 20222023, {2: 15}, rng)  # validation
    write_season(data_dir, 20232024, {2: 20, 3: 6}, rng)  # test : saison régulière et séries
    output_dir = tmp_path / "milestone2_outputs"

    # Données « tâche 2 » (distance, angle) pour 2021-22 et 2022-23, comme data/etape2_shots_train.csv.
    raw = pd.concat([game_to_df(p) for s in ("20212022", "20222023") for p in sorted((data_dir / s).glob("*.json"))])
    task2 = add_features(raw)
    task2_csv = tmp_path / "data" / "etape2_shots_train.csv"
    task2.to_csv(task2_csv, index=False)
    # Données « tâche 6 » (22 caractéristiques), comme task6_features_2016_2022.csv.gz.
    task6 = build_dataset(data_dir, ["20212022", "20222023"], strict=False)
    task6_csv = output_dir / "task6_features_2016_2022.csv.gz"
    task6_csv.parent.mkdir(parents=True)
    task6.to_csv(task6_csv, index=False, compression="gzip")
    task2_train = task2[task2["season"] == 20212022].dropna(subset=["distance", "angle"])
    task6_train = pd.read_csv(task6_csv).query("season == 20212022")

    logreg_dir = tmp_path / "data" / "models"
    logreg_dir.mkdir(parents=True)
    for name, columns in {"distance": ["distance"], "angle": ["angle"], "distance_angle": ["distance", "angle"]}.items():
        model = LogisticRegression().fit(task2_train[columns], task2_train["is_goal"])
        joblib.dump(model, logreg_dir / f"logreg_{name}.joblib")
    xgb = make_pipeline(FEATURE_COLUMNS, {"n_estimators": 20, "max_depth": 2})
    xgb.fit(task6_train[FEATURE_COLUMNS], task6_train["is_goal"])
    (output_dir / "models").mkdir()
    joblib.dump(xgb, output_dir / "models" / "xgb_tuned_full.joblib")
    hgb = make_hgb_pipeline(FEATURE_COLUMNS_V2, {"max_iter": 20})
    hgb.fit(task6_train[FEATURE_COLUMNS_V2], task6_train["is_goal"])
    (output_dir / "task6" / "models").mkdir(parents=True)
    joblib.dump(hgb, output_dir / "task6" / "models" / "hgb_v2.joblib")
    return {"data_dir": data_dir, "output_dir": output_dir, "logreg_dir": logreg_dir,
            "task2_csv": task2_csv, "task6_dataset": task6_csv}


def run_local(project: dict[str, Path], **options) -> dict:
    return evaluate(
        data_dir=project["data_dir"],
        output_dir=project["output_dir"],
        logreg_dir=project["logreg_dir"],
        task2_csv=project["task2_csv"],
        task6_dataset=project["task6_dataset"],
        source="local",
        wandb_mode="disabled",
        strict=False,
        **{"n_boot": 30, **options},
    )


def test_end_to_end_evaluation(project):
    result = run_local(project)
    metrics, sets = result["metrics"], result["sets"]
    output_dir = project["output_dir"]

    # Trois ensembles × cinq modèles, avec le bon nombre de matchs.
    assert set(metrics["set"]) == {"validation", "regular", "playoffs"}
    assert metrics.groupby("set")["model"].nunique().eq(5).all()
    assert sets["regular"]["games"] == 20 and sets["playoffs"]["games"] == 6 and sets["validation"]["games"] == 15
    # Le tir sans coordonnées de chaque match est exclu, pour tous les modèles.
    assert sets["regular"]["excluded_missing_coordinates"] == 20
    assert sets["playoffs"]["shots"] == 6 * 23

    # Huit figures (4 par ensemble de test), aucune pour la validation.
    figures = sorted(path.name for path in (output_dir / "figures").glob("task7_*.png"))
    assert len(figures) == 8
    assert all(name.startswith(("task7_regular_", "task7_playoffs_")) for name in figures)

    # Intervalles de confiance sur le test seulement; l'écart du modèle final avec lui-même est nul.
    test_rows = metrics[metrics["set"] != "validation"]
    assert test_rows["auc_low"].notna().all()
    assert (test_rows["auc_low"] <= test_rows["auc"] + 1e-9).all() and (test_rows["auc"] <= test_rows["auc_high"] + 1e-9).all()
    final = test_rows[test_rows["model"] == "hgb_v2"]
    assert np.allclose(final[["delta_log_loss_low", "delta_log_loss_high"]].to_numpy(), 0)
    assert metrics.loc[metrics["set"] == "validation", "auc_low"].isna().all()

    # Prédictions sauvegardées pour chaque modèle et chaque ensemble (utile pour la partie 8).
    for set_name in ("validation", "regular", "playoffs"):
        assert len(list((output_dir / "task7" / "predictions" / set_name).glob("*.csv.gz"))) == 5
    summary = (output_dir / "task7" / "summary.md").read_text(encoding="utf-8")
    assert "**hgb_v2 (final)**" in summary and "Comparaison validation / test" in summary
    assert "Écart de log loss avec le final" in summary
    record = json.loads((output_dir / "task7" / "run.json").read_text(encoding="utf-8"))
    assert record["models"]["hgb_v2"]["source"] == "local"


def test_validation_metrics_match_saved_task6_predictions(project):
    """Les métriques de validation recalculées doivent être celles du modèle tel qu'enregistré."""

    result = run_local(project, n_boot=0)
    validation = pd.read_csv(project["output_dir"] / "task7" / "predictions" / "validation" / "hgb_v2.csv.gz")
    frame = pd.read_csv(project["task6_dataset"]).merge(validation[["game_id", "event_id"]], on=["game_id", "event_id"])
    model = joblib.load(project["output_dir"] / "task6" / "models" / "hgb_v2.joblib")
    expected = model.predict_proba(frame[FEATURE_COLUMNS_V2])[:, 1]
    assert np.allclose(np.sort(expected), np.sort(validation["probability"]))
    assert "auc_low" not in result["metrics"].columns


def test_test_sets_use_both_pipelines_on_the_same_shots(project):
    sets = build_test_sets(project["data_dir"], project["output_dir"], strict=False)
    frame, info = sets["regular"]
    assert {"distance", "angle", *FEATURE_COLUMNS_V2}.issubset(frame.columns)
    assert info["unmatched"] == 0
    assert frame[["game_id", "event_id"]].duplicated().sum() == 0
    assert frame["game_type"].eq(2).all() and sets["playoffs"][0]["game_type"].eq(3).all()


def test_merge_rejects_inconsistent_labels():
    task2 = pd.DataFrame({"game_id": [1, 1], "event_id": [1, 2], "distance": [10.0, 20.0], "angle": [5.0, 6.0],
                          "is_goal": [1, 0]})
    task4 = pd.DataFrame({"game_id": [1, 1], "event_id": [1, 2], "is_goal": [0, 0]})
    with pytest.raises(ValueError, match="étiquette"):
        merge_pipelines(task2, task4)


def test_model_features_reads_fitted_columns():
    frame = pd.DataFrame({"distance": [5.0, 30.0, 10.0, 50.0], "is_goal": [1, 0, 1, 0]})
    model = LogisticRegression().fit(frame[["distance"]], frame["is_goal"])
    assert model_features(model) == ["distance"]
    assert model_features(CalibratedModel(model, "sigmoid")) == ["distance"]


def test_registry_requires_online_wandb(project):
    with pytest.raises(ValueError, match="online"):
        evaluate(data_dir=project["data_dir"], output_dir=project["output_dir"], source="registry",
                 wandb_mode="offline")