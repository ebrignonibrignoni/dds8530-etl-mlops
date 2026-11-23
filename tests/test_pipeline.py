"""Unit and integration tests run by GitHub Actions on every push."""
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from etl.transform import review_features, transform  # noqa: E402


def test_review_sentiment_scores_text():
    reviews = pd.DataFrame({"order_id": ["A", "B"], "text": ["Great quality, fast delivery.", "Arrived damaged, late delivery."]})
    out = review_features(reviews).set_index("order_id")
    assert out.loc["A", "review_sentiment"] > 0
    assert out.loc["B", "review_sentiment"] < 0


def test_transform_cleans_and_normalizes(tmp_path):
    raw = pd.DataFrame({
        "order_id": ["O1", "O1", "O2", "O3", "O4"],
        "order_date": pd.to_datetime(["2025-01-05"] * 5),
        "customer_id": ["C1"] * 5,
        "category": ["Technology", "Technology", None, "Furniture", "Furniture"],
        "quantity": [2, 2, 1, -1, 3],
        "discount": [0.0, 0.0, 0.1, 0.0, 0.2],
        "currency": ["USD", "USD", "EUR", "USD", "GBP"],
        "sales_local": [200.0, 200.0, 92.0, 50.0, None],
        "profit_usd": [20.0, 20.0, -5.0, 1.0, 3.0],
    })
    raw.to_parquet(tmp_path / "part_000.parquet", index=False)
    customers = pd.DataFrame({"customer_id": ["C1"], "segment": ["Consumer"], "region": ["West"],
                              "tenure_months": [12], "loyalty_member": [1]})
    reviews = pd.DataFrame({"order_id": ["O1"], "text": ["Excellent, love it."]})
    df, report = transform(str(tmp_path), customers, {"EUR": 0.92, "GBP": 0.79, "CAD": 1.38}, reviews)

    assert report["rows_in"] == 5
    assert report["rows_out"] == 2                      # duplicate, negative qty, missing sales removed
    assert set(df["order_id"]) == {"O1", "O2"}
    assert df.loc[df.order_id == "O2", "sales_usd"].iloc[0] == pytest.approx(100.0)   # 92 EUR -> 100 USD
    assert df.loc[df.order_id == "O2", "category"].iloc[0] == "Unknown"             # imputed
    assert {"sales_usd_z", "log_sales_usd", "review_sentiment"} <= set(df.columns)


def test_api_health_and_predict():
    model_dir = Path(__file__).resolve().parents[1] / "models"
    if not (model_dir / "model.joblib").exists():
        pytest.skip("model not trained yet")
    from api.app import app, FEATURES
    client = app.test_client()
    assert client.get("/health").get_json()["status"] == "ok"
    sample = {f: 0 for f in FEATURES}
    sample.update({"category": "Technology", "segment": "Consumer", "region": "West", "currency": "USD",
                   "discount": 0.0, "loyalty_member": 1, "order_month": 6})
    resp = client.post("/predict", data=json.dumps(sample), content_type="application/json")
    assert resp.status_code == 200
    assert 0.0 <= resp.get_json()["predictions"][0]["probability"] <= 1.0
    assert client.post("/predict", json={"discount": 0.1}).status_code == 400   # missing features rejected
    assert b"api_requests_total" in client.get("/metrics").data
