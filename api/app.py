"""
Model-serving REST API (Flask).
  GET  /health   liveness check and model version
  POST /predict  JSON order(s) -> profitability prediction and probability
  GET  /metrics  Prometheus metrics (request count, latency, errors, predictions)
"""
import json
import time

import joblib
import pandas as pd
from flask import Flask, Response, jsonify, request
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

from etl.config import MODEL_DIR
from etl.logging_config import get_logger

log = get_logger("api")
app = Flask(__name__)

MODEL = joblib.load(MODEL_DIR / "model.joblib")
CARD = json.loads((MODEL_DIR / "model_card.json").read_text())
FEATURES = CARD["features"]

REQUESTS = Counter("api_requests_total", "Total API requests", ["endpoint", "status"])
LATENCY = Histogram("api_request_latency_seconds", "Request latency in seconds", ["endpoint"])
PREDICTIONS = Counter("model_predictions_total", "Predictions served", ["predicted_class"])


@app.get("/health")
def health():
    REQUESTS.labels("/health", "200").inc()
    return jsonify(status="ok", model=CARD["model"], version=CARD["registered_version"])


@app.post("/predict")
def predict():
    start = time.perf_counter()
    try:
        payload = request.get_json(force=True)
        records = payload if isinstance(payload, list) else [payload]
        X = pd.DataFrame(records)
        missing = [f for f in FEATURES if f not in X.columns]
        if missing:
            REQUESTS.labels("/predict", "400").inc()
            return jsonify(error=f"missing features: {missing}"), 400
        proba = MODEL.predict_proba(X[FEATURES])[:, 1]
        out = [{"is_profitable": int(p >= 0.5), "probability": round(float(p), 4)} for p in proba]
        for o in out:
            PREDICTIONS.labels(str(o["is_profitable"])).inc()
        REQUESTS.labels("/predict", "200").inc()
        log.info("Served %d prediction(s)", len(out))
        return jsonify(predictions=out, model_version=CARD["registered_version"])
    except Exception as exc:  # logged and counted, never crashes the server
        REQUESTS.labels("/predict", "500").inc()
        log.exception("Prediction failed: %s", exc)
        return jsonify(error="prediction failed"), 500
    finally:
        LATENCY.labels("/predict").observe(time.perf_counter() - start)


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
