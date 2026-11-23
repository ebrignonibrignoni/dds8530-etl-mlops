"""
LOAD: write the curated table to a relational warehouse with SQLAlchemy
(SQLite by default, PostgreSQL via WAREHOUSE_URL) and optionally copy a
Parquet snapshot to Amazon S3 when S3_BUCKET is set.
"""
import pandas as pd
from sqlalchemy import create_engine, text

from etl.config import PROCESSED_DIR, S3_BUCKET, WAREHOUSE_URL
from etl.logging_config import get_logger

log = get_logger("load")


def load(df: pd.DataFrame, table: str = "orders_features") -> int:
    engine = create_engine(WAREHOUSE_URL)
    df.to_sql(table, engine, if_exists="replace", index=False, chunksize=10_000)
    with engine.begin() as conn:
        conn.execute(text(f"CREATE INDEX IF NOT EXISTS ix_{table}_order ON {table}(order_id)"))
        n = conn.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar()
    log.info("Loaded %d rows into %s.%s", n, WAREHOUSE_URL.split(':')[0], table)

    snapshot = PROCESSED_DIR / f"{table}.parquet"
    df.to_parquet(snapshot, index=False)
    if S3_BUCKET:
        import boto3
        boto3.client("s3").upload_file(str(snapshot), S3_BUCKET, f"processed/{snapshot.name}")
        log.info("Copied snapshot to s3://%s/processed/%s", S3_BUCKET, snapshot.name)
    return n
