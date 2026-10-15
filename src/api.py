"""Local Flask demo: browser dashboard and document-compatible /predict API."""
import json

from flask import Flask, jsonify, request, send_from_directory
from werkzeug.exceptions import BadRequest, UnsupportedMediaType

from src.config import METRICS_FILE, ROOT
from src.predict import load_model, predict_reading


def create_app(bundle=None):
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 4096
    model = load_model() if bundle is None else bundle

    @app.get("/")
    def dashboard():
        return send_from_directory(ROOT / "src", "dashboard.html")

    @app.get("/health")
    def health():
        return jsonify({"status": "ready", "mode": "synthetic simulation"})

    @app.post("/predict")
    def predict():
        try:
            payload = request.get_json()
            return jsonify(predict_reading(payload, model))
        except (ValueError, BadRequest, UnsupportedMediaType) as error:
            return jsonify({"error": str(error)}), 400

    @app.get("/metrics")
    def metrics():
        if not METRICS_FILE.exists():
            return jsonify({"error": "Run training first."}), 404
        return jsonify(json.loads(METRICS_FILE.read_text(encoding="utf-8")))

    @app.get("/images/<name>")
    def chart(name):
        allowed = {"sensor_history.png", "failure_prediction.png", "confusion_matrix.png", "precision_recall.png", "feature_importance.png"}
        if name not in allowed:
            return jsonify({"error": "Unknown chart."}), 404
        return send_from_directory(ROOT / "images", name)

    return app
