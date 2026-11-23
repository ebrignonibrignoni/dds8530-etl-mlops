"""
TRANSFORM: clean, normalize, and engineer features with Dask, which splits the
data into partitions and processes them in parallel.
"""
import dask.dataframe as dd
import numpy as np
import pandas as pd

from etl.config import DASK_PARTITIONS
from etl.logging_config import get_logger

log = get_logger("transform")

POS_WORDS = ["great", "fast", "excellent", "love", "perfect"]
NEG_WORDS = ["damaged", "late", "poor", "missing", "disappointed"]


def review_features(reviews: pd.DataFrame) -> pd.DataFrame:
    """Feature extraction from unstructured text: simple lexicon sentiment score."""
    t = reviews["text"].str.lower()
    pos = sum(t.str.count(w) for w in POS_WORDS)
    neg = sum(t.str.count(w) for w in NEG_WORDS)
    return pd.DataFrame({"order_id": reviews["order_id"], "review_sentiment": (pos - neg).astype(float),
                         "has_review": 1})


def transform(stage_path: str, customers: pd.DataFrame, fx: dict, reviews: pd.DataFrame):
    report = {}
    ddf = dd.read_parquet(stage_path).repartition(npartitions=DASK_PARTITIONS)
    report["rows_in"] = int(ddf.shape[0].compute())

    # 1. Cleaning
    ddf = ddf.drop_duplicates()
    report["after_dedup"] = int(ddf.shape[0].compute())
    ddf = ddf[ddf["quantity"] > 0]
    report["after_invalid_qty"] = int(ddf.shape[0].compute())
    ddf = ddf.dropna(subset=["sales_local"])
    ddf["category"] = ddf["category"].fillna("Unknown")

    # 2. Currency normalization using the API rates (local amount / units per USD)
    rate_map = {"USD": 1.0, **{k: float(v) for k, v in fx.items()}}
    ddf["fx_rate"] = ddf["currency"].astype(str).map(rate_map, meta=("fx_rate", "f8")).astype("float64")
    ddf["sales_usd"] = (ddf["sales_local"] / ddf["fx_rate"]).round(2)

    # 3. Feature engineering
    ddf["order_month"] = ddf["order_date"].dt.month
    ddf["unit_price_usd"] = (ddf["sales_usd"] / (ddf["quantity"] * (1 - ddf["discount"]))).round(2)
    ddf["is_profitable"] = (ddf["profit_usd"] > 0).astype(int)

    df = ddf.compute()
    df = df.merge(customers, on="customer_id", how="left")
    df = df.merge(review_features(reviews), on="order_id", how="left")
    df["review_sentiment"] = df["review_sentiment"].fillna(0.0)
    df["has_review"] = df["has_review"].fillna(0).astype(int)

    # 4. Normalization (z-score) of numeric model inputs; raw values kept for reporting
    for col in ["sales_usd", "unit_price_usd", "quantity", "tenure_months"]:
        mu, sd = df[col].mean(), df[col].std()
        df[f"{col}_z"] = ((df[col] - mu) / sd).round(4)
    df["log_sales_usd"] = np.log1p(df["sales_usd"]).round(4)

    report["rows_out"] = len(df)
    report["missing_sales_dropped"] = report["after_invalid_qty"] - len(df)
    report["category_imputed"] = int((df["category"] == "Unknown").sum())
    log.info("Transform report: %s", report)
    return df, report
