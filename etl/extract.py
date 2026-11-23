"""
EXTRACT: pull data from three kinds of sources.
  - CSV file, read in batches (chunks) so memory stays flat as the file grows
  - Operational database, queried through SQLAlchemy
  - Web API (exchange rates) through requests, with retries and a cached fallback
  - Unstructured JSON reviews
"""
import json
import time

import pandas as pd
import requests
from sqlalchemy import create_engine, text

from etl.config import CHUNK_SIZE, FX_API_URL, FX_CACHE, ORDERS_CSV, PROCESSED_DIR, REVIEWS_JSON, SOURCE_DB_URL
from etl.logging_config import get_logger

log = get_logger("extract")


def extract_orders_csv() -> str:
    """Stream the CSV in batches and write each batch to a staging Parquet file."""
    stage = PROCESSED_DIR / "orders_stage"
    stage.mkdir(exist_ok=True)
    for old in stage.glob("*.parquet"):
        old.unlink()
    total = 0
    for i, chunk in enumerate(pd.read_csv(ORDERS_CSV, chunksize=CHUNK_SIZE, parse_dates=["order_date"])):
        chunk.to_parquet(stage / f"part_{i:03d}.parquet", index=False)
        total += len(chunk)
        log.info("CSV batch %d: %d rows", i, len(chunk))
    log.info("Extracted %d order rows from CSV in batches of %d", total, CHUNK_SIZE)
    return str(stage)


def extract_customers_db() -> pd.DataFrame:
    engine = create_engine(SOURCE_DB_URL)
    with engine.connect() as conn:
        df = pd.read_sql(text("SELECT customer_id, segment, region, tenure_months, loyalty_member FROM customers"), conn)
    log.info("Extracted %d customers from database via SQLAlchemy", len(df))
    return df


def extract_fx_rates(retries: int = 3) -> dict:
    """Fetch USD exchange rates from a public REST API; fall back to the last cached copy."""
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(FX_API_URL, params={"from": "USD", "to": "EUR,GBP,CAD"}, timeout=10)
            resp.raise_for_status()
            rates = resp.json()["rates"]
            FX_CACHE.write_text(json.dumps({"base": "USD", "rates": rates}))
            log.info("Fetched live FX rates from API: %s", rates)
            return rates
        except (requests.RequestException, KeyError, ValueError) as exc:
            log.warning("FX API attempt %d/%d failed: %s", attempt, retries, exc)
            time.sleep(attempt)
    rates = json.loads(FX_CACHE.read_text())["rates"]
    log.warning("Using cached FX rates: %s", rates)
    return rates


def extract_reviews() -> pd.DataFrame:
    df = pd.DataFrame(json.loads(REVIEWS_JSON.read_text()))
    log.info("Extracted %d unstructured reviews", len(df))
    return df
