"""Shared paths and explicit simulation assumptions."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "iot_sensor_data.csv"
MODEL_FILE = ROOT / "models" / "predictive_maintenance_model.pkl"
METRICS_FILE = ROOT / "outputs" / "metrics.json"
SPLITS_FILE = ROOT / "outputs" / "machine_splits.json"
SENSORS = ["temperature", "vibration", "current"]
TARGET = "failure_next_10_readings"
HORIZON = 10
SEED = 42


def prepare_directories():
    for folder in ["data", "models", "outputs", "images"]:
        (ROOT / folder).mkdir(parents=True, exist_ok=True)
