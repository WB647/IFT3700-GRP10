from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from milestone2.evaluation import (
    expected_calibration_error,
    load_aligned_predictions,
    probability_metrics,
    save_predictions,
)
from milestone2.features import FEATURE_COLUMNS
from milestone2.models import make_pipeline
from milestone2.task6_experiments import EXPERIMENTS, Experiment
from milestone2.train_task6 import make_task6_figures, prediction_path, run_experiment


def synthetic_dataset(path: Path, rows: int = 1500) -> Path:
    """Tirs dont la probabilité de but diminue avec la distance, sur trois saisons."""

    rng = np.random.default_rng(0)
    distance = rng.uniform(5, 80, rows)
    frame = pd.DataFrame(
        {
            "season": rng.choice([20202021, 20212022, 20222023], rows, p=[0.4, 0.4, 0.2]),
            "game_type": 2,
            "game_id": rng.integers(2020020001, 2020020200, rows),
            "event_id": np.arange(rows),
            "team": "MTL",
            "game_seconds": rng.uniform(0, 3600, rows),
            "period": rng.integers(1, 4, rows),
            "x": 89 - distance * 0.8,
            "y": rng.uniform(-30, 30, rows),
            "shot_distance": distance,
            "shot_angle": rng.uniform(0, 80, rows),
            "shot_type": rng.choice(["wrist", "snap", "slap"], rows),
            "last_event_type": rng.choice(["faceoff", "hit", "shot-on-goal"], rows),
            "last_event_x": rng.uniform(-90, 90, rows),
            "last_event_y": rng.uniform(-40, 40, rows),
            "time_since_last_event": rng.uniform(0, 60, rows),
            "distance_from_last_event": rng.uniform(0, 150, rows),
            "rebound": rng.random(rows) < 0.1,
            "change_in_shot_angle": rng.uniform(0, 90, rows),
            "speed": rng.uniform(0, 50, rows),
            "empty_net": (rng.random(rows) < 0.02).astype(float),
            "shooter_skaters": rng.choice([4.0, 5.0, 6.0], rows, p=[0.1, 0.8, 0.1]),
            "opponent_skaters": rng.choice([4.0, 5.0], rows, p=[0.2, 0.8]),
            "is_home": rng.integers(0, 2, rows).astype(float),
            "last_event_same_team": rng.choice([0.0, 1.0, np.nan], rows),
            "score_diff": rng.integers(-3, 4, rows).astype(float),
        }
    )
    frame["skater_diff"] = frame["shooter_skaters"] - frame["opponent_skaters"]
    base = 0.35 * np.exp(-distance / 20)
    frame["is_goal"] = (rng.random(rows) < np.where(frame["empty_net"] == 1, 0.8, base)).astype(int)
    frame.to_csv(path, index=False, compression="gzip")
    return path


def test_expected_calibration_error_known_values():
    y = np.array([0, 0, 1, 1])
    assert expected_calibration_error(y, np.full(4, 0.5), n_bins=2) == pytest.approx(0.0)
    assert expected_calibration_error(y, np.full(4, 0.9), n_bins=2) == pytest.approx(0.4)


def test_probability_metrics_reports_probability_quality():
    y = np.array([0, 0, 0, 1])
    metrics = probability_metrics(y, np.array([0.1, 0.2, 0.3, 0.4]))
    assert metrics["auc"] == pytest.approx(1.0)
    assert metrics["xg_goal_ratio"] == pytest.approx(1.0)
    assert metrics["brier"] == pytest.approx(np.mean([0.01, 0.04, 0.09, 0.36]))
    assert metrics["shots"] == 4 and metrics["goals"] == 1


def test_predictions_are_aligned_on_common_shots(tmp_path: Path):
    frame = pd.DataFrame({"game_id": [1, 1, 2], "event_id": [10, 11, 10], "is_goal": [0, 1, 0]})
    save_predictions(frame, np.array([0.1, 0.8, 0.2]), tmp_path / "a.csv.gz")
    save_predictions(frame.iloc[[2, 0]], np.array([0.3, 0.4]), tmp_path / "b.csv.gz")
    y, probabilities = load_aligned_predictions({"a": tmp_path / "a.csv.gz", "b": tmp_path / "b.csv.gz"})
    assert len(y) == 2
    for position, (game_id, event_id) in enumerate(y.index):
        expected_b = {(1, 10): 0.4, (2, 10): 0.3}[(game_id, event_id)]
        assert probabilities["b"][position] == pytest.approx(expected_b)


def test_reference_is_refit_when_task5_model_is_missing(tmp_path: Path):
    dataset = synthetic_dataset(tmp_path / "data.csv.gz")
    record = run_experiment("xgb_task5", dataset=dataset, output_dir=tmp_path)
    assert record["settings"]["refit"] is True
    assert record["validation_season"] == 20222023
    assert 20222023 not in record["training_seasons"]
    assert prediction_path(tmp_path, "xgb_task5").exists()
    assert (tmp_path / "task6" / "metrics" / "xgb_task5.json").exists()
    assert (tmp_path / "task6" / "models" / "xgb_task5.joblib").exists()


def test_reference_reuses_saved_task5_model(tmp_path: Path):
    dataset = synthetic_dataset(tmp_path / "data.csv.gz")
    frame = pd.read_csv(dataset)
    model = make_pipeline(FEATURE_COLUMNS, {"n_estimators": 5})
    model.fit(frame[FEATURE_COLUMNS], frame["is_goal"])
    (tmp_path / "models").mkdir()
    joblib.dump(model, tmp_path / "models" / "xgb_tuned_full.joblib")
    record = run_experiment("xgb_task5", dataset=dataset, output_dir=tmp_path)
    assert record["settings"]["refit"] is False


def test_new_experiment_then_blog_figures(tmp_path: Path, monkeypatch):
    dataset = synthetic_dataset(tmp_path / "data.csv.gz")

    def small_model(train, context):
        model = make_pipeline(context.features, {"n_estimators": 20, "max_depth": 2})
        model.fit(train[context.features], train["is_goal"])
        return model, {"n_estimators": np.int64(20)}

    monkeypatch.setitem(
        EXPERIMENTS,
        "small_xgb",
        Experiment(name="small_xgb", description="test", features=["shot_distance", "shot_angle"],
                   train=small_model, tags=("test",)),
    )
    run_experiment("xgb_task5", dataset=dataset, output_dir=tmp_path)
    record = run_experiment("small_xgb", dataset=dataset, output_dir=tmp_path)
    assert record["settings"] == {"n_estimators": 20}
    run_figures = sorted((tmp_path / "task6" / "figures" / "runs").glob("small_xgb_*.png"))
    assert len(run_figures) == 4

    result = make_task6_figures(tmp_path, final="small_xgb")
    names = sorted(path.name for path in result["figures"])
    assert names == sorted(
        ["task6_roc_auc.png", "task6_goal_rate_percentile.png", "task6_cumulative_goals.png", "task6_reliability.png"]
    )
    assert set(result["summary"].index) == {"small_xgb (final)", "xgb_task5 (référence)"}
    assert "small_xgb (final)" in (tmp_path / "task6" / "summary.md").read_text(encoding="utf-8")


def test_missing_feature_column_explains_how_to_rebuild(tmp_path: Path, monkeypatch):
    dataset = synthetic_dataset(tmp_path / "data.csv.gz")
    monkeypatch.setitem(
        EXPERIMENTS,
        "needs_new_column",
        Experiment(name="needs_new_column", description="test", features=["column_not_built_yet"],
                   train=lambda train, context: (None, {})),
    )
    with pytest.raises(ValueError, match="build-dataset"):
        run_experiment("needs_new_column", dataset=dataset, output_dir=tmp_path)


def test_attempt_a_new_features(tmp_path: Path):
    from milestone2.features import NEW_FEATURE_COLUMNS

    dataset = synthetic_dataset(tmp_path / "data.csv.gz")
    record = run_experiment("xgb_features_v2", dataset=dataset, output_dir=tmp_path)
    assert record["settings"]["new_features"] == NEW_FEATURE_COLUMNS
    assert "figures" not in record["settings"]  # retirée de la configuration, envoyée comme image
    runs = tmp_path / "task6" / "figures" / "runs"
    assert (runs / "xgb_features_v2_feature_importance.png").exists()
    importance = pd.read_csv(runs / "xgb_features_v2_feature_importance.csv", index_col=0)
    assert len(importance) == 22

    # Dans les données synthétiques, le filet désert fait presque toujours un but.
    assert importance.loc["empty_net", "auc_drop"] > 0

def test_native_pipeline_keeps_missing_values_and_zeros(tmp_path: Path):
    from milestone2.task6_experiments import make_native_pipeline

    frame = pd.DataFrame(
        {
            "speed": [np.nan, 0.0, 12.5, 3.0],
            "rebound": [True, False, False, True],
            "shot_type": ["wrist", np.nan, "slap", "wrist"],
            "is_goal": [0, 1, 0, 1],
        }
    )
    features = ["speed", "rebound", "shot_type"]
    model = make_native_pipeline(features, {"n_estimators": 3})
    model.fit(frame[features], frame["is_goal"])
    matrix = model[:-1].transform(frame[features])
    assert isinstance(matrix, np.ndarray) and matrix.dtype == float
    assert np.isnan(matrix[0, 0])  # valeur manquante conservée, pas remplacée par la médiane
    assert matrix[1, 0] == 0.0 and matrix[1, 1] == 0.0  # les zéros restent des zéros
    joblib.dump(model, tmp_path / "model.joblib")
    reloaded = joblib.load(tmp_path / "model.joblib")
    assert np.allclose(reloaded.predict_proba(frame[features]), model.predict_proba(frame[features]))


def test_parse_params():
    from milestone2.train_task6 import parse_params

    assert parse_params(["n_iter=40", "game_fraction=0.5", "name=abc"]) == {
        "n_iter": 40,
        "game_fraction": 0.5,
        "name": "abc",
    }
    assert parse_params(None) == {}
    with pytest.raises(ValueError):
        parse_params(["n_iter"])


def test_attempt_b_small_search(tmp_path: Path):
    dataset = synthetic_dataset(tmp_path / "data.csv.gz")
    params = {"n_iter": 2, "folds": 2, "game_fraction": 0.5}
    record = run_experiment("xgb_tuned_v2", dataset=dataset, output_dir=tmp_path, params=params)
    settings = record["settings"]
    assert record["params"] == params
    assert settings["n_iter"] == 2 and settings["folds"] == 2
    assert settings["cv_log_loss"] > 0
    assert {"max_depth", "learning_rate", "reg_lambda"}.issubset(settings["hyperparameters"])
    runs = tmp_path / "task6" / "figures" / "runs"
    assert (runs / "xgb_tuned_v2_tuning.png").exists()
    assert len(pd.read_csv(runs / "xgb_tuned_v2_tuning.csv")) == 2


def test_hgb_pipeline_handles_missing_and_unknown_categories(tmp_path: Path):
    from milestone2.task6_experiments import make_hgb_pipeline

    frame = pd.DataFrame(
        {
            "shot_distance": [10.0, np.nan, 30.0, 45.0] * 10,
            "shot_type": ["wrist", "slap", np.nan, "snap"] * 10,
            "is_goal": [1, 0, 0, 0] * 10,
        }
    )
    features = ["shot_distance", "shot_type"]
    model = make_hgb_pipeline(features, {"max_iter": 10})
    model.fit(frame[features], frame["is_goal"])
    new_shots = pd.DataFrame({"shot_distance": [np.nan, 12.0], "shot_type": ["bat", "wrist"]})
    probability = model.predict_proba(new_shots)[:, 1]
    assert np.all((probability > 0) & (probability < 1))
    joblib.dump(model, tmp_path / "hgb.joblib")
    assert np.allclose(joblib.load(tmp_path / "hgb.joblib").predict_proba(new_shots)[:, 1], probability)


@pytest.mark.parametrize(
    "name, params",
    [
        ("hgb_v2", {"max_iter": 30}),
        # Petit jeu de données : petits lots, sinon le réseau ne fait que 2 mises à jour par époque.
        ("mlp_v2", {"max_iter": 30, "hidden_layer_sizes": 16, "batch_size": 32}),
        ("logreg_v2", {"C": 0.5}),
    ],
)
def test_attempt_c_models(tmp_path: Path, name: str, params: dict):
    dataset = synthetic_dataset(tmp_path / "data.csv.gz")
    record = run_experiment(name, dataset=dataset, output_dir=tmp_path, params=params)
    for key, value in params.items():
        expected = [value] if key == "hidden_layer_sizes" else value
        assert record["settings"]["hyperparameters"][key] == expected
    assert 0.5 < record["metrics"]["auc"] <= 1
    assert (tmp_path / "task6" / "models" / f"{name}.joblib").exists()
    if name == "mlp_v2":
        assert (tmp_path / "task6" / "figures" / "runs" / "mlp_v2_learning_curve.png").exists()
        settings = record["settings"]
        assert 1 <= settings["best_epoch"] <= settings["epochs_run"] <= 30
        assert settings["held_out_log_loss"] > 0
        assert "history" not in settings  # la courbe va dans la figure, pas dans la configuration
    if name == "hgb_v2":
        assert 1 <= record["settings"]["iterations_used"] <= 30