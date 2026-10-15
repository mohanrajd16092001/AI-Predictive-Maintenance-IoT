"""Generate fictional run-to-failure histories, without external datasets."""
import numpy as np
import pandas as pd

from src.config import DATA_FILE, HORIZON, SEED, TARGET, prepare_directories


def generate_data(machine_count=80, seed=SEED):
    """One reading per simulated minute; each machine ends at a failure event.

    Wear is an invented latent variable used ONLY by this generator. Sensor
    formulas and units are demonstration assumptions, not calibrated physics.
    The model never receives wear, lifetime, time to failure, or event labels.
    """
    if machine_count < 10:
        raise ValueError("Use at least 10 machines for separate machine splits.")
    rng = np.random.default_rng(seed)
    histories = []
    for number in range(machine_count):
        last_step = int(rng.integers(160, 261))
        steps = np.arange(last_step + 1)
        wear = steps / last_step
        temp_base = rng.normal(30, 2)
        current_base = rng.normal(0.5, 0.04)
        temperature = temp_base + 50 * wear**3 + rng.normal(0, 1.3, len(steps))
        vibration = np.maximum(0, 0.15 + 4.5 * wear**4 + rng.normal(0, 0.12, len(steps)))
        current = np.maximum(0, current_base + 1.2 * wear**2 + rng.normal(0, 0.04, len(steps)))
        event = (steps == last_step).astype(int)
        # Strictly FUTURE failures: offsets 1..10, excluding the current row.
        future_event = ((last_step - steps >= 1) & (last_step - steps <= HORIZON)).astype(int)
        frame = pd.DataFrame({
            "machine_id": f"SIM-{number + 1:03d}",
            "timestamp": pd.date_range("2026-01-01", periods=len(steps), freq="min"),
            "temperature": temperature.round(4),
            "vibration": vibration.round(4),
            "current": current.round(4),
            "failure": event,
            TARGET: future_event,
        })
        # A small reproducible missing-data challenge for the training imputer.
        for sensor in ["temperature", "vibration", "current"]:
            frame.loc[rng.random(len(frame)) < 0.01, sensor] = np.nan
        histories.append(frame)
    result = pd.concat(histories, ignore_index=True)
    prepare_directories()
    result.to_csv(DATA_FILE, index=False)
    print(f"Saved {len(result):,} synthetic readings from {machine_count} machines to {DATA_FILE}")
    print("Units: temperature = Celsius; current = A; vibration = arbitrary simulated units.")
    print(f"Target: failure in the next {HORIZON} readings (one simulated minute per reading).")
    return result


def load_data():
    if not DATA_FILE.exists():
        raise FileNotFoundError("Dataset missing. Run: python main.py generate")
    frame = pd.read_csv(DATA_FILE, parse_dates=["timestamp"])
    required = {"machine_id", "timestamp", "temperature", "vibration", "current", "failure", TARGET}
    if not required.issubset(frame.columns):
        raise ValueError(f"Dataset missing columns: {sorted(required - set(frame.columns))}")
    return frame
