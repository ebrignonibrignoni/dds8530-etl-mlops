"""
End-to-end runner: extract -> transform -> load -> train.
Used locally, by the Airflow DAG tasks, and by GitHub Actions.
"""
import json
import time

from etl.extract import extract_customers_db, extract_fx_rates, extract_orders_csv, extract_reviews
from etl.load import load
from etl.logging_config import get_logger
from etl.transform import transform
from pipeline.train import train

log = get_logger("pipeline")


def run() -> dict:
    timings, t0 = {}, time.perf_counter()

    t = time.perf_counter()
    stage = extract_orders_csv()
    customers, fx, reviews = extract_customers_db(), extract_fx_rates(), extract_reviews()
    timings["extract_s"] = round(time.perf_counter() - t, 2)

    t = time.perf_counter()
    df, report = transform(stage, customers, fx, reviews)
    timings["transform_s"] = round(time.perf_counter() - t, 2)

    t = time.perf_counter()
    report["rows_loaded"] = load(df)
    timings["load_s"] = round(time.perf_counter() - t, 2)

    t = time.perf_counter()
    model = train()
    timings["train_s"] = round(time.perf_counter() - t, 2)
    timings["total_s"] = round(time.perf_counter() - t0, 2)

    result = {"etl": report, "model": {k: model[k] for k in ("model", "params", "accuracy", "f1", "roc_auc",
                                                             "baseline_accuracy", "registered_version")},
              "timings": timings}
    log.info("Pipeline complete: %s", json.dumps(result, default=str))
    return result


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, default=str))
