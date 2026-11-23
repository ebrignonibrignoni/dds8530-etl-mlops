"""
Apache Airflow DAG: schedules the ETL and MLOps pipeline daily.
extract -> transform -> load -> train -> validate_model
Each task passes small results through XCom and large data through files.
"""
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

from airflow.decorators import dag, task

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

default_args = {"owner": "ebrignoni", "retries": 2, "retry_delay": timedelta(minutes=5)}


@dag(dag_id="etl_mlops_pipeline", schedule="@daily", start_date=datetime(2026, 11, 1),
     catchup=False, default_args=default_args, tags=["dds8530", "etl", "mlops"])
def etl_mlops_pipeline():

    @task
    def extract() -> dict:
        from etl.extract import extract_customers_db, extract_fx_rates, extract_orders_csv
        stage = extract_orders_csv()
        extract_customers_db()          # validates source connectivity
        return {"stage": stage, "fx": extract_fx_rates()}

    @task
    def transform_and_load(meta: dict) -> dict:
        from etl.extract import extract_customers_db, extract_reviews
        from etl.load import load
        from etl.transform import transform
        df, report = transform(meta["stage"], extract_customers_db(), meta["fx"], extract_reviews())
        report["rows_loaded"] = load(df)
        return report

    @task
    def train_model(report: dict) -> dict:
        from pipeline.train import train
        summary = train()
        return {k: summary[k] for k in ("model", "accuracy", "f1", "roc_auc", "baseline_accuracy", "registered_version")}

    @task
    def validate_model(summary: dict) -> None:
        """Quality gate: fail the run if the model does not beat the majority-class baseline."""
        if summary["accuracy"] <= summary["baseline_accuracy"]:
            raise ValueError(f"Model did not beat baseline: {json.dumps(summary)}")

    validate_model(train_model(transform_and_load(extract())))


etl_mlops_pipeline()
