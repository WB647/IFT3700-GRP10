from pathlib import Path

import numpy as np
import pandas as pd

from milestone2.features import FEATURE_COLUMNS
from milestone2.models import (
    BASELINE_FEATURES,
    make_evaluation_figures,
    make_pipeline,
    model_metrics,
    split_train_validation,
)


def model_frame() -> pd.DataFrame:
    rows = []
    for index in range(80):
        rows.append(
            {
                "season": 20212022 if index < 60 else 20222023,
                "is_goal": int(index % 7 == 0),
                "game_seconds": index * 10,
                "period": 1 + index % 3,
                "x": 50 + index % 35,
                "y": -20 + index % 40,
                "shot_distance": 5 + index % 70,
                "shot_angle": index % 80,
                "shot_type": "wrist" if index % 2 else "snap",
                "last_event_type": "faceoff" if index % 3 else "shot-on-goal",
                "last_event_x": 40 + index % 30,
                "last_event_y": -10 + index % 20,
                "time_since_last_event": 1 + index % 10,
                "distance_from_last_event": 2 + index % 25,
                "rebound": index % 3 == 0,
                "change_in_shot_angle": index % 30,
                "speed": 0.5 + index % 8,
            }
        )
    return pd.DataFrame(rows)


def test_split_is_strictly_season_based():
    train, validation = split_train_validation(model_frame())
    assert set(train["season"]) == {20212022}
    assert set(validation["season"]) == {20222023}


def test_baseline_pipeline_trains_and_returns_probabilities():
    train, validation = split_train_validation(model_frame())
    model = make_pipeline(BASELINE_FEATURES, {"n_estimators": 5, "max_depth": 2})
    model.fit(train[BASELINE_FEATURES], train["is_goal"])
    probabilities = model.predict_proba(validation[BASELINE_FEATURES])[:, 1]
    assert len(probabilities) == len(validation)
    assert np.all((probabilities >= 0) & (probabilities <= 1))
    assert set(model_metrics(validation["is_goal"], probabilities)) == {"auc", "log_loss", "accuracy"}


def test_full_pipeline_accepts_all_features():
    train, validation = split_train_validation(model_frame())
    model = make_pipeline(FEATURE_COLUMNS, {"n_estimators": 5, "max_depth": 2})
    model.fit(train[FEATURE_COLUMNS], train["is_goal"])
    assert model.predict_proba(validation[FEATURE_COLUMNS]).shape == (20, 2)


def test_four_evaluation_figures_are_created(tmp_path: Path):
    validation = model_frame().query("season == 20222023")
    first = np.linspace(0.01, 0.8, len(validation))
    second = first[::-1]
    paths = make_evaluation_figures(
        validation["is_goal"], {"premier": first, "second": second}, tmp_path
    )
    assert len(paths) == 4
    assert all(path.exists() and path.stat().st_size > 0 for path in paths)
