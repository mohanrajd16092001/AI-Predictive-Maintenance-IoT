"""Shared model loading, input validation and simulated actions."""
import math
import numbers

import joblib
import pandas as pd

from src.config import MODEL_FILE, SENSORS
from src.features import make_features


def load_model():
    if not MODEL_FILE.exists():
        raise FileNotFoundError("Model missing. Run: python main.py train")
    # Load only this project's locally generated artifact.
    return joblib.load(MODEL_FILE)


def validate_reading(payload):
    if not isinstance(payload, dict):
        raise ValueError("JSON input must be an object.")
    if set(payload) != set(SENSORS):
        raise ValueError("Provide exactly temperature, vibration, and current.")
    # Deliberately broad demo input bounds, not hardware or safety ratings.
    bounds = {"temperature": (-50, 200), "vibration": (0, 100), "current": (0, 100)}
    cleaned = {}
    for sensor in SENSORS:
        value = payload[sensor]
        if isinstance(value, bool) or not isinstance(value, numbers.Real) or not math.isfinite(value):
            raise ValueError(f"{sensor} must be a finite number.")
        low, high = bounds[sensor]
        if not low <= value <= high:
            raise ValueError(f"{sensor} must be within demo input bounds {low}..{high}.")
        cleaned[sensor] = float(value)
    return cleaned


def simulated_motor_action(failure_predicted):
    return "STOP (simulation only)" if failure_predicted else "RUN (simulation only)"


def format_prediction(score, bundle):
    predicted = bool(score >= bundle["threshold"])
    return {
        "Prediction": "Machine failure predicted!" if predicted else "Machine is running normally.",
        "failure_predicted": predicted,
        "failure_risk_score": round(float(score), 6),
        "threshold": bundle["threshold"],
        "horizon_readings": bundle["horizon"],
        "motor_action": simulated_motor_action(predicted),
        "data_source": "synthetic simulation",
        "score_note": "Uncalibrated model score; not a validated real-world failure probability.",
    }


def predict_reading(payload, bundle=None):
    reading = validate_reading(payload)
    bundle = load_model() if bundle is None else bundle
    features = make_features(pd.DataFrame([reading]))
    score = bundle["pipeline"].predict_proba(features)[0, 1]
    return format_prediction(score, bundle)
