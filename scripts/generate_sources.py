"""
Creates the three simulated source systems used by the pipeline:
  1. data/raw/orders.csv        - 100,000 retail order lines (structured, large)
  2. data/raw/crm_source.db     - SQLite operational database with a customers table
  3. data/raw/reviews.json      - free-text customer reviews (unstructured)
A small share of records is deliberately defective (missing values, duplicates,
negative quantities) so the transform stage has real cleaning work to do.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import create_engine

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from etl.config import ORDERS_CSV, REVIEWS_JSON, SOURCE_DB_URL, FX_CACHE  # noqa: E402

RNG = np.random.default_rng(8530)
N_ORDERS = 100_000
N_CUSTOMERS = 5_000
CURRENCY_RATE = {"USD": 1.0, "EUR": 0.92, "GBP": 0.79, "CAD": 1.38}  # units per 1 USD


def customers() -> pd.DataFrame:
    seg = RNG.choice(["Consumer", "Corporate", "Home Office"], N_CUSTOMERS, p=[0.5, 0.3, 0.2])
    return pd.DataFrame({
        "customer_id": [f"C{i:05d}" for i in range(N_CUSTOMERS)],
        "segment": seg,
        "region": RNG.choice(["Central", "East", "South", "West"], N_CUSTOMERS),
        "tenure_months": RNG.integers(1, 120, N_CUSTOMERS),
        "loyalty_member": RNG.choice([0, 1], N_CUSTOMERS, p=[0.6, 0.4]),
    })


def orders(cust: pd.DataFrame) -> pd.DataFrame:
    cat = RNG.choice(["Furniture", "Office Supplies", "Technology"], N_ORDERS, p=[0.25, 0.45, 0.30])
    base = np.select([cat == "Furniture", cat == "Office Supplies"], [350.0, 40.0], 600.0)
    unit_price_usd = np.round(base * RNG.lognormal(0, 0.5, N_ORDERS), 2)
    qty = RNG.integers(1, 10, N_ORDERS)
    discount = RNG.choice([0, 0.1, 0.2, 0.3, 0.4, 0.5], N_ORDERS, p=[0.45, 0.2, 0.15, 0.1, 0.06, 0.04])
    currency = RNG.choice(list(CURRENCY_RATE), N_ORDERS, p=[0.7, 0.15, 0.1, 0.05])
    cust_idx = RNG.integers(0, N_CUSTOMERS, N_ORDERS)
    loyalty = cust.loyalty_member.values[cust_idx]
    margin = np.select([cat == "Furniture", cat == "Office Supplies"], [0.12, 0.25], 0.18)
    profit_rate = margin - 0.9 * discount + 0.04 * loyalty + RNG.normal(0, 0.08, N_ORDERS)
    sales_usd = unit_price_usd * qty * (1 - discount)
    df = pd.DataFrame({
        "order_id": [f"O{i:06d}" for i in range(N_ORDERS)],
        "order_date": pd.Timestamp("2025-01-01") + pd.to_timedelta(RNG.integers(0, 365, N_ORDERS), unit="D"),
        "customer_id": cust.customer_id.values[cust_idx],
        "category": cat,
        "quantity": qty,
        "discount": discount,
        "currency": currency,
        "sales_local": np.round(sales_usd * np.vectorize(CURRENCY_RATE.get)(currency), 2),
        "profit_usd": np.round(sales_usd * profit_rate, 2),
    })
    # defects: ~1% missing sales, ~0.5% missing category, 0.2% negative qty, 500 duplicate rows
    df.loc[RNG.choice(N_ORDERS, 1000, replace=False), "sales_local"] = np.nan
    df.loc[RNG.choice(N_ORDERS, 500, replace=False), "category"] = None
    df.loc[RNG.choice(N_ORDERS, 200, replace=False), "quantity"] = -1
    return pd.concat([df, df.sample(500, random_state=1)], ignore_index=True)


def reviews(ord_df: pd.DataFrame) -> list:
    pos = ["great quality", "fast delivery", "excellent", "love it", "works perfectly"]
    neg = ["arrived damaged", "late delivery", "poor quality", "missing parts", "disappointed"]
    sample = ord_df.drop_duplicates("order_id").sample(20_000, random_state=2)
    out = []
    for oid, p in zip(sample.order_id, sample.profit_usd.fillna(0)):
        words = RNG.choice(pos if (p > 0) == (RNG.random() < 0.8) else neg, 2, replace=False)
        out.append({"order_id": oid, "text": f"{words[0].capitalize()}, {words[1]}."})
    return out


def main():
    cust = customers()
    create_engine(SOURCE_DB_URL).connect().close()
    cust.to_sql("customers", create_engine(SOURCE_DB_URL), if_exists="replace", index=False)
    ord_df = orders(cust)
    ord_df.to_csv(ORDERS_CSV, index=False)
    REVIEWS_JSON.write_text(json.dumps(reviews(ord_df)))
    FX_CACHE.write_text(json.dumps({"base": "USD", "rates": {k: v for k, v in CURRENCY_RATE.items() if k != "USD"}}))
    print(f"orders.csv: {len(ord_df):,} rows | customers: {len(cust):,} | reviews: 20,000")


if __name__ == "__main__":
    main()
