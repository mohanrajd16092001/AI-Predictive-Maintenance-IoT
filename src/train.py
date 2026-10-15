"""Train on separate machines, evaluate once on held-out machines."""
import json
import platform

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    ConfusionMatrixDisplay, accuracy_score, average_precision_score,
    confusion_matrix, precision_score, recall_score, f1_score,
    precision_recall_curve,
)
from sklearn.pipeline import Pipeline

from src.config import (
    HORIZON, METRICS_FILE, MODEL_FILE, ROOT, SEED, SENSORS, SPLITS_FILE,
    TARGET, prepare_directories,
)
from src.data import load_data
from src.features import make_features


def split_by_machine(frame):
    machines = np.sort(frame.machine_id.unique())
    if len(machines) < 10:
        raise ValueError("At least 10 machines are required.")
    machines = np.random.default_rng(SEED).permutation(machines)
    train_end = int(len(machines) * 0.70)
    validation_end = int(len(machines) * 0.85)
    return {
        "train": machines[:train_end].tolist(),
        "validation": machines[train_end:validation_end].tolist(),
        "test": machines[validation_end:].tolist(),
    }


def score(y, predicted, probability):
    return {
        "rows": int(len(y)),
        "positive_rows": int(y.sum()),
        "accuracy": float(accuracy_score(y, predicted)),
        "precision": float(precision_score(y, predicted, zero_division=0)),
        "recall": float(recall_score(y, predicted, zero_division=0)),
        "f1": float(f1_score(y, predicted, zero_division=0)),
        "average_precision": float(average_precision_score(y, probability)),
        "confusion_matrix": confusion_matrix(y, predicted, labels=[0, 1]).tolist(),
    }


def train_model():
    prepare_directories()
    frame = load_data()
    # A current failure is already known, so exclude terminal event readings.
    examples = frame.loc[frame.failure == 0].copy()
    splits = split_by_machine(examples)
    parts = {name: examples.loc[examples.machine_id.isin(ids)] for name, ids in splits.items()}
    features = {name: make_features(part) for name, part in parts.items()}
    labels = {name: part[TARGET].astype(int) for name, part in parts.items()}
    for name, target in labels.items():
        if target.nunique() != 2:
            raise ValueError(f"{name} must contain both target classes.")

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", RandomForestClassifier(
            n_estimators=120, max_depth=10, min_samples_leaf=3,
            class_weight="balanced", random_state=SEED, n_jobs=-1,
        )),
    ])
    model.fit(features["train"], labels["train"])
    # The threshold is fixed in advance. No tuning on validation or test.
    threshold = 0.5
    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(features["train"], labels["train"])
    metrics = {
        "data_source": "synthetic; invented sensor degradation formulas",
        "target": TARGET,
        "horizon_readings": HORIZON,
        "threshold": threshold,
        "split_policy": "70/15/15 by machine; no machine shared across splits",
        "seed": SEED,
        "python_version": platform.python_version(),
        "sklearn_version": sklearn.__version__,
        "machine_counts": {name: len(ids) for name, ids in splits.items()},
    }
    results = {}
    for name in ["validation", "test"]:
        probability = model.predict_proba(features[name])[:, 1]
        predicted = (probability >= threshold).astype(int)
        metrics[name] = score(labels[name], predicted, probability)
        base_probability = baseline.predict_proba(features[name])[:, 1]
        metrics[f"{name}_majority_baseline"] = score(
            labels[name], baseline.predict(features[name]), base_probability,
        )
        results[name] = (predicted, probability)
    joblib.dump({
        "pipeline": model, "threshold": threshold, "sensors": SENSORS,
        "feature_names": list(features["train"].columns), "target": TARGET,
        "horizon": HORIZON, "sklearn_version": sklearn.__version__,
    }, MODEL_FILE)
    SPLITS_FILE.write_text(json.dumps(splits, indent=2), encoding="utf-8")
    METRICS_FILE.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    test_output = parts["test"].copy()
    test_output["predicted_failure"] = results["test"][0]
    test_output["failure_risk_score"] = results["test"][1]
    test_output.to_csv(ROOT / "outputs" / "test_predictions.csv", index=False)
    save_plots(parts, labels, model, features, results)
    print("Random Forest trained. Model and fitted imputer saved together.")
    for name in ["validation", "test"]:
        print(name, json.dumps(metrics[name], indent=2))
    print("These metrics describe this synthetic generator only.")
    return metrics


def save_plots(parts, labels, model, features, results):
    plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(7, 5))
    ConfusionMatrixDisplay.from_predictions(
        labels["test"], results["test"][0], labels=[0, 1],
        display_labels=["No failure soon", "Failure within 10"], cmap="Blues", ax=ax, colorbar=False,
    )
    ax.set_title("Synthetic held-out machines | Confusion matrix")
    fig.tight_layout()
    fig.savefig(ROOT / "images" / "confusion_matrix.png")
    plt.close(fig)

    precision, recall, _ = precision_recall_curve(labels["test"], results["test"][1])
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(recall, precision, color="#176b87", label="Random Forest")
    ax.axhline(labels["test"].mean(), color="#b05a23", linestyle="--", label="Positive prevalence")
    ax.set(xlabel="Recall", ylabel="Precision", title="Synthetic test data | Precision and recall", xlim=(0, 1.02), ylim=(0, 1.02))
    ax.legend()
    fig.tight_layout()
    fig.savefig(ROOT / "images" / "precision_recall.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    importance = model.named_steps["classifier"].feature_importances_
    order = np.argsort(importance)
    ax.barh(np.array(features["train"].columns)[order], importance[order], color="#176b87")
    ax.set(xlabel="Impurity-based importance (descriptive)", title="Features used by the synthetic-data model")
    fig.tight_layout()
    fig.savefig(ROOT / "images" / "feature_importance.png")
    plt.close(fig)

    example_id = parts["test"].machine_id.iloc[0]
    example = parts["test"].loc[parts["test"].machine_id == example_id]
    fig, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
    for ax, sensor, unit in zip(axes, SENSORS, ["Celsius", "Arbitrary units", "Amps"]):
        ax.plot(np.arange(len(example)), example[sensor], color="#176b87")
        ax.set(ylabel=f"{sensor}\n({unit})")
    axes[0].set_title(f"Synthetic sensor history | {example_id}")
    axes[-1].set_xlabel("Simulated minute / reading number")
    fig.tight_layout()
    fig.savefig(ROOT / "images" / "sensor_history.png")
    plt.close(fig)
