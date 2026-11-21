"""Train the three required Task-5 XGBoost experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .features import FEATURE_COLUMNS
from .models import (
    BASELINE_FEATURES,
    make_evaluation_figures,
    make_logistic_reference,
    make_pipeline,
    model_metrics,
    save_importance_figure,
    save_model,
    save_tuning_figure,
    select_features,
    split_train_validation,
    tune_full_model,
    write_results,
)
from .wandb_tracking import log_model_run


def _clean(frame: pd.DataFrame) -> pd.DataFrame:
    required = set(FEATURE_COLUMNS) | {"season", "is_goal"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing dataset columns: {sorted(missing)}")
    clean = frame.dropna(subset=["is_goal"]).copy()
    clean["is_goal"] = clean["is_goal"].astype(int)
    return clean


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("milestone2_outputs/task4_features_2016_2022.csv.gz"))
    parser.add_argument("--output-dir", type=Path, default=Path("milestone2_outputs"))
    parser.add_argument("--tuning-iterations", type=int, default=12)
    parser.add_argument("--cv", type=int, default=3)
    parser.add_argument("--wandb-mode", choices=["disabled", "offline", "online"], default="disabled")
    parser.add_argument("--wandb-project", default="ift3700-milestone-2")
    parser.add_argument("--wandb-entity")
    args = parser.parse_args()

    frame = _clean(pd.read_csv(args.dataset))
    train, validation = split_train_validation(frame)
    models_dir = args.output_dir / "models"
    figures_dir = args.output_dir / "figures"

    logistic = make_logistic_reference()
    logistic.fit(train[BASELINE_FEATURES], train["is_goal"])
    logistic_probability = logistic.predict_proba(validation[BASELINE_FEATURES])[:, 1]

    baseline = make_pipeline(BASELINE_FEATURES)
    baseline.fit(train[BASELINE_FEATURES], train["is_goal"])
    baseline_probability = baseline.predict_proba(validation[BASELINE_FEATURES])[:, 1]

    search = tune_full_model(train, n_iter=args.tuning_iterations, cv=args.cv)
    full = search.best_estimator_
    full_probability = full.predict_proba(validation[FEATURE_COLUMNS])[:, 1]
    tuning_figure = save_tuning_figure(search, figures_dir / "task5_hyperparameter_tuning.png")

    # Feature selection uses training data only; 2022/23 remains an untouched
    # validation set until every modeling decision has been made.
    selected_features, importance = select_features(full, train)
    best_params = {
        key.removeprefix("model__"): value for key, value in search.best_params_.items()
    }
    selected = make_pipeline(selected_features, best_params)
    selected.fit(train[selected_features], train["is_goal"])
    selected_probability = selected.predict_proba(validation[selected_features])[:, 1]
    importance_figure = save_importance_figure(importance, figures_dir / "task5_feature_selection.png")

    probabilities = {
        "Régression logistique": logistic_probability,
        "XGBoost distance + angle": baseline_probability,
        "XGBoost réglé complet": full_probability,
        "XGBoost sélectionné": selected_probability,
    }
    evaluation_figures = make_evaluation_figures(validation["is_goal"], probabilities, figures_dir)
    experiment_models = {"xgb_baseline": baseline, "xgb_tuned_full": full, "xgb_selected": selected}
    experiment_features = {
        "xgb_baseline": BASELINE_FEATURES,
        "xgb_tuned_full": FEATURE_COLUMNS,
        "xgb_selected": selected_features,
    }
    experiment_probabilities = {
        "xgb_baseline": baseline_probability,
        "xgb_tuned_full": full_probability,
        "xgb_selected": selected_probability,
    }

    results: dict[str, object] = {
        "training_seasons": sorted(train["season"].astype(int).unique().tolist()),
        "validation_season": 20222023,
        "training_rows": len(train),
        "validation_rows": len(validation),
        "logistic_reference_metrics": model_metrics(validation["is_goal"], logistic_probability),
        "best_parameters": search.best_params_,
        "selected_features": selected_features,
        "experiments": {},
    }
    for name, model in experiment_models.items():
        metrics = model_metrics(validation["is_goal"], experiment_probabilities[name])
        path = save_model(model, models_dir / f"{name}.joblib")
        config = {
            "features": experiment_features[name],
            "validation_season": 20222023,
            "best_parameters": search.best_params_ if name != "xgb_baseline" else {},
        }
        url = log_model_run(
            name=name,
            model_path=path,
            metrics=metrics,
            config=config,
            figure_paths=evaluation_figures + [tuning_figure, importance_figure],
            project=args.wandb_project,
            entity=args.wandb_entity,
            mode=args.wandb_mode,
            register=name != "xgb_baseline",
        )
        results["experiments"][name] = {"metrics": metrics, "wandb_url": url}

    write_results(results, args.output_dir / "task5_results.json")
    print(pd.DataFrame({name: value["metrics"] for name, value in results["experiments"].items()}).T)
    print(f"Selected features: {selected_features}")


if __name__ == "__main__":
    main()
