from pathlib import Path
 
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wandb
from sklearn.calibration import CalibrationDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, auc, roc_auc_score, roc_curve
 
ROOT = Path(__file__).resolve().parent
PROJECT = "milestone-2"  # the Wandb workspace comes from the WANDB_ENTITY environment variable
MODELS = {"distance": ["distance"], "angle": ["angle"], "distance_angle": ["distance", "angle"]}
 
 
def load_data():
    """Shots from task 2, without missing distance or angle. Validation = season 2022-23, training = the others."""
    df = pd.read_csv(ROOT / "data" / "etape2_shots_train.csv").dropna(subset=["distance", "angle"])
    is_val = df["season"] == 20222023
    return df[~is_val], df[is_val]
 
 
def train_model(name, features, train, val):
    """Train one logistic regression, log it to Wandb, and return the validation probabilities of a goal."""
    run = wandb.init(project=PROJECT, name=f"logreg_{name}", tags=["baseline", "logistic-regression", name],
                     config={"features": features, "validation_season": 20222023})
    clf = LogisticRegression()  # default parameters
    clf.fit(train[features], train["is_goal"])
    proba = clf.predict_proba(val[features])[:, 1]  # probability of the class "goal"
    pred = clf.predict(val[features])
 
    accuracy = accuracy_score(val["is_goal"], pred)
    print(f"{name}: accuracy {accuracy:.4f}, shots predicted as goals: {pred.mean():.4%}")
    run.log({"accuracy": accuracy, "auc": roc_auc_score(val["is_goal"], proba)})
 
    path = ROOT / "data" / "models" / f"logreg_{name}.joblib"
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, path)
    run.log_artifact(str(path), name=f"logreg_{name}", type="model")
    run.finish()
    return proba
 
 
def save(name, xlabel, ylabel):
    (ROOT / "figures").mkdir(exist_ok=True)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.savefig(ROOT / "figures" / f"etape2_t3_{name}.png", dpi=150, bbox_inches="tight")
    plt.close()
 
 
def make_figures(y, probas):
    """Four figures with one curve per model. probas = {model name: predicted probabilities of a goal}."""
    bins = np.arange(0, 101, 5)
 
    for name, p in probas.items():  # 1. ROC curves
        fpr, tpr, _ = roc_curve(y, p)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc(fpr, tpr):.3f})")
    save("roc", "False positive rate", "True positive rate")
 
    for name, p in probas.items():  # 2. goal rate per percentile of the predicted probability
        percentile = pd.Series(p).rank(pct=True).to_numpy() * 100
        rate = pd.Series(y).groupby(pd.cut(percentile, bins, include_lowest=True), observed=True).mean()
        plt.plot([interval.mid for interval in rate.index], rate, label=name)
    save("goal_rate", "Shot probability model percentile", "Goal rate (goals / shots)")
 
    for name, p in probas.items():  # 3. cumulative proportion of goals, from the highest probability down
        goals = y[np.argsort(-p)]
        plt.plot(100 - np.arange(1, len(y) + 1) / len(y) * 100, np.cumsum(goals) / goals.sum(), label=name)
    plt.xlim(100, 0)
    save("cumulative_goals", "Shot probability model percentile", "Cumulative proportion of goals")
 
    plt.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")  # 4. reliability diagram
    for name, p in probas.items():
        CalibrationDisplay.from_predictions(y, p, n_bins=20, strategy="quantile", ax=plt.gca(), name=name, ref_line=False)
    save("reliability", "Mean predicted probability", "Fraction of goals")
 
 
if __name__ == "__main__":
    train, val = load_data()
    print(f"{len(train)} training shots, {len(val)} validation shots, goal rate {val['is_goal'].mean():.4f}")
    probas = {name: train_model(name, features, train, val) for name, features in MODELS.items()}
    probas["random"] = np.random.default_rng(0).uniform(size=len(val))  # baseline: U(0, 1) probabilities
    make_figures(val["is_goal"].to_numpy(), probas)
 