"""
Train and track: reads the curated warehouse table, tunes two scikit-learn
models, logs every run to MLflow (parameters, metrics, model artifact), registers
the best model, and saves a local copy for the API.
"""
import json

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sqlalchemy import create_engine

from etl.config import MLFLOW_EXPERIMENT, MLFLOW_TRACKING_URI, MODEL_DIR, MODEL_NAME, WAREHOUSE_URL
from etl.logging_config import get_logger

log = get_logger("train")

NUMERIC = ["discount", "quantity_z", "unit_price_usd_z", "log_sales_usd", "tenure_months_z",
           "loyalty_member", "review_sentiment", "has_review", "order_month"]
CATEGORICAL = ["category", "segment", "region", "currency"]
TARGET = "is_profitable"

CANDIDATES = [
    ("logistic_regression", LogisticRegression(max_iter=1000), [{"C": 0.1}, {"C": 1.0}]),
    ("random_forest", RandomForestClassifier(random_state=8530, n_jobs=-1),
     [{"n_estimators": 100, "max_depth": 8}, {"n_estimators": 200, "max_depth": 12}]),
]


def load_training_data() -> pd.DataFrame:
    return pd.read_sql("SELECT * FROM orders_features", create_engine(WAREHOUSE_URL))


def train() -> dict:
    df = load_training_data()
    X, y = df[NUMERIC + CATEGORICAL].astype({c: "float64" for c in NUMERIC}), df[TARGET]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=8530)
    prep = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL)],
                             remainder="passthrough")

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    best = {"f1": -1}
    for name, estimator, grid in CANDIDATES:
        for params in grid:
            with mlflow.start_run(run_name=f"{name}-{params}") as run:
                model = Pipeline([("prep", prep), ("clf", estimator.set_params(**params))]).fit(X_tr, y_tr)
                pred, proba = model.predict(X_te), model.predict_proba(X_te)[:, 1]
                metrics = {"accuracy": accuracy_score(y_te, pred), "f1": f1_score(y_te, pred),
                           "roc_auc": roc_auc_score(y_te, proba)}
                mlflow.log_params({"model": name, **params, "train_rows": len(X_tr), "test_rows": len(X_te)})
                mlflow.log_metrics(metrics)
                info = mlflow.sklearn.log_model(model, name="model", signature=infer_signature(X_te, pred),
                                         input_example=X_te.head(3),
                                         # model trained in this run, so its tree type is trusted
                                         skops_trusted_types=["sklearn.tree._tree.Tree"])
                log.info("%s %s -> %s", name, params, {k: round(v, 4) for k, v in metrics.items()})
                if metrics["f1"] > best["f1"]:
                    best = {**metrics, "run_id": run.info.run_id, "model": name, "params": params,
                            "estimator": model, "model_uri": info.model_uri}

    baseline = max(y_te.mean(), 1 - y_te.mean())
    mv = mlflow.register_model(best.pop("model_uri"), MODEL_NAME)
    joblib.dump(best.pop("estimator"), MODEL_DIR / "model.joblib")
    summary = {**best, "baseline_accuracy": baseline, "registered_version": mv.version,
               "features": NUMERIC + CATEGORICAL}
    (MODEL_DIR / "model_card.json").write_text(json.dumps(summary, indent=2, default=str))
    log.info("Best model %s registered as %s v%s", best["model"], MODEL_NAME, mv.version)
    return summary


if __name__ == "__main__":
    print(json.dumps(train(), indent=2, default=str))
