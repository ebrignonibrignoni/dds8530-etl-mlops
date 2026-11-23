"""Central configuration for the ETL and MLOps pipeline."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
LOG_DIR = ROOT / "logs"

# Source systems
ORDERS_CSV = RAW_DIR / "orders.csv"                       # structured, large file
SOURCE_DB_URL = f"sqlite:///{RAW_DIR / 'crm_source.db'}"  # operational database (customers)
REVIEWS_JSON = RAW_DIR / "reviews.json"                   # unstructured text
FX_API_URL = os.getenv("FX_API_URL", "https://api.frankfurter.app/latest")
FX_CACHE = RAW_DIR / "fx_rates_cache.json"                # fallback when the API is unreachable

# Destination (data warehouse); swap for PostgreSQL by setting WAREHOUSE_URL,
# e.g. postgresql+psycopg2://user:pass@host:5432/warehouse
WAREHOUSE_URL = os.getenv("WAREHOUSE_URL", f"sqlite:///{DATA_DIR / 'warehouse.db'}")
S3_BUCKET = os.getenv("S3_BUCKET")                        # optional cloud copy

# Processing
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "25000"))        # batch size for extraction
DASK_PARTITIONS = int(os.getenv("DASK_PARTITIONS", "4"))

# MLOps
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{ROOT / 'mlflow.db'}")
MLFLOW_EXPERIMENT = "order-profitability"
MODEL_NAME = "order_profitability_classifier"
MODEL_DIR = ROOT / "models"

for d in (RAW_DIR, PROCESSED_DIR, LOG_DIR, MODEL_DIR):
    d.mkdir(parents=True, exist_ok=True)
