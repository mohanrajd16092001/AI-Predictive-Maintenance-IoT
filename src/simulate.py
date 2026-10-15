"""Replay an unseen simulated machine, producing alerts and virtual actions."""
import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import ROOT, SPLITS_FILE, TARGET
from src.data import load_data
from src.features import make_features
from src.predict import format_prediction, load_model


def simulate(machine_id=None, delay=0):
    if delay < 0:
        raise ValueError("Delay must be nonnegative.")
    bundle = load_model()
    if not SPLITS_FILE.exists():
        raise FileNotFoundError("Machine splits missing. Run training again.")
    test_ids = json.loads(SPLITS_FILE.read_text(encoding="utf-8"))["test"]
    machine_id = machine_id or test_ids[0]
    if machine_id not in test_ids:
        raise ValueError(f"Choose an unseen test machine from {test_ids}")
    data = load_data()
    history = data.loc[(data.machine_id == machine_id) & (data.failure == 0)].sort_values("timestamp").copy()
    if history.empty:
        raise ValueError("No readings found for this machine. Regenerate and retrain together.")
    # Every feature row uses only that row's present readings. The future target
    # below is written for evaluation and never passed to the predictor.
    scores = bundle["pipeline"].predict_proba(make_features(history))[:, 1]
    records = []
    for index, (row, score) in enumerate(zip(history.to_dict("records"), scores)):
        result = format_prediction(score, bundle)
        row.update(result)
        records.append(row)
        if delay or index >= len(history) - 10:
            print(f"{row['timestamp']} | risk score={score:.3f} | {result['motor_action']}")
        if delay:
            time.sleep(delay)
    replay = pd.DataFrame(records)
    replay.to_csv(ROOT / "outputs" / "simulation_predictions.csv", index=False)
    alerts = replay.loc[replay.failure_predicted]
    actual_event = data.loc[(data.machine_id == machine_id) & (data.failure == 1), "timestamp"].iloc[0]
    first_alert = None if alerts.empty else alerts.timestamp.iloc[0]
    summary = {
        "machine_id": machine_id,
        "data_source": "synthetic; held-out test machine",
        "readings_replayed": len(replay),
        "alert_readings": len(alerts),
        "false_alert_readings": int(((replay.failure_predicted) & (replay[TARGET] == 0)).sum()),
        "missed_positive_readings": int(((~replay.failure_predicted) & (replay[TARGET] == 1)).sum()),
        "first_alert": None if first_alert is None else str(first_alert),
        "simulated_failure_time": str(actual_event),
        "first_alert_lead_minutes": None if first_alert is None else float((actual_event - first_alert).total_seconds() / 60),
        "action_mode": "virtual STOP/RUN only; replay does not alter the recorded machine history",
        "lead_note": "An alert earlier than the 10-reading target window counts as a false positive for this target.",
    }
    (ROOT / "outputs" / "simulation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.plot(np.arange(len(replay)), scores, label="Model risk score", color="#176b87")
    ax.axhline(bundle["threshold"], linestyle="--", color="#a04428", label="Alert threshold")
    ax.fill_between(np.arange(len(replay)), 0, 1, where=replay[TARGET].astype(bool), color="#ddad46", alpha=0.25, label="True future-failure window (evaluation)")
    ax.set(xlabel="Simulated minute / reading number", ylabel="Uncalibrated risk score", ylim=(-0.03, 1.03), title=f"Virtual replay | unseen machine {machine_id}")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(ROOT / "images" / "failure_prediction.png", dpi=140)
    plt.close(fig)
    print(json.dumps(summary, indent=2))
    return summary
