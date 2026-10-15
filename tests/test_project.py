"""Checks for future labels, leakage prevention, and the prediction contract.

Run after `python main.py run`: python -m unittest discover -s tests -v
"""
import json
import unittest

import numpy as np
import pandas as pd

from src.api import create_app
from src.config import HORIZON, ROOT, SENSORS, SPLITS_FILE, TARGET
from src.data import load_data
from src.features import make_features
from src.predict import load_model, predict_reading


class ProjectChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = load_data()
        cls.bundle = load_model()
        cls.client = create_app(cls.bundle).test_client()
        cls.splits = json.loads(SPLITS_FILE.read_text(encoding="utf-8"))

    def test_labels_refer_to_strictly_future_events(self):
        for _, group in self.data.groupby("machine_id"):
            group = group.sort_values("timestamp")
            events = group.failure.to_numpy()
            expected = np.array([int(events[i+1:i+1+HORIZON].any()) for i in range(len(events))])
            np.testing.assert_array_equal(group[TARGET].to_numpy(), expected)
            self.assertEqual(int(events.sum()), 1)

    def test_machines_are_disjoint_and_cover_dataset(self):
        train, validation, test = [set(self.splits[name]) for name in ["train", "validation", "test"]]
        self.assertFalse(train & validation or train & test or validation & test)
        self.assertEqual(train | validation | test, set(self.data.machine_id))

    def test_imputer_uses_only_training_readings(self):
        training = self.data.loc[(self.data.failure == 0) & self.data.machine_id.isin(self.splits["train"])]
        expected = make_features(training).median().to_numpy()
        np.testing.assert_allclose(self.bundle["pipeline"].named_steps["imputer"].statistics_, expected)

    def test_features_ignore_identifiers_and_labels(self):
        sample = self.data.iloc[:5].copy()
        original = make_features(sample)
        sample["machine_id"] = "OTHER"
        sample["failure"] = 1 - sample["failure"]
        sample[TARGET] = 1 - sample[TARGET]
        pd.testing.assert_frame_equal(original, make_features(sample))
        self.assertEqual(list(original.columns[:3]), SENSORS)

    def test_api_matches_direct_prediction_and_action(self):
        for payload in [dict(temperature=35, vibration=0.3, current=0.6), dict(temperature=80, vibration=4.5, current=1.7)]:
            response = self.client.post("/predict", json=payload)
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.get_json(), predict_reading(payload, self.bundle))
            result = response.get_json()
            self.assertEqual(result["failure_predicted"], result["failure_risk_score"] >= result["threshold"])
            self.assertEqual(result["motor_action"], "STOP (simulation only)" if result["failure_predicted"] else "RUN (simulation only)")

    def test_api_rejects_invalid_inputs(self):
        valid = dict(temperature=35, vibration=0.3, current=0.6)
        payloads = [None, [], {}, {**valid, "failure": 1}, {**valid, "temperature": "hot"}, {**valid, "current": -1}, {**valid, "vibration": True}, {**valid, "temperature": float("nan")}, {**valid, "temperature": float("inf")}]
        for payload in payloads:
            with self.subTest(payload=payload):
                response = self.client.post("/predict", data=json.dumps(payload), content_type="application/json")
                self.assertEqual(response.status_code, 400)
                self.assertIn("error", response.get_json())
        self.assertEqual(self.client.post("/predict", data="bad", content_type="application/json").status_code, 400)

    def test_dashboard_and_all_charts_are_available(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        response.close()
        self.assertEqual(self.client.get("/metrics").status_code, 200)
        for filename in ["sensor_history.png", "failure_prediction.png", "confusion_matrix.png", "precision_recall.png", "feature_importance.png"]:
            with self.subTest(filename=filename):
                response = self.client.get("/images/" + filename)
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.mimetype, "image/png")
                response.close()


if __name__ == "__main__":
    unittest.main()
