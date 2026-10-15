"""One feature function shared by training, simulation and the API."""
import pandas as pd

from src.config import SENSORS


def make_features(readings):
    """Use only present sensor values; no future information or identifiers.

    Products provide simple interactions. Missing values are left for the
    pipeline imputer, which is fitted on training machines only.
    """
    frame = readings[SENSORS].apply(pd.to_numeric, errors="raise").copy()
    frame["temperature_x_vibration"] = frame.temperature * frame.vibration
    frame["vibration_x_current"] = frame.vibration * frame.current
    frame["temperature_x_current"] = frame.temperature * frame.current
    return frame
