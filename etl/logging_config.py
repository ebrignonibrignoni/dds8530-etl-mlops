"""Shared logging setup: console plus a rotating file in logs/pipeline.log."""
import logging
from logging.handlers import RotatingFileHandler

from etl.config import LOG_DIR


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    file_handler = RotatingFileHandler(LOG_DIR / "pipeline.log", maxBytes=1_000_000, backupCount=3)
    file_handler.setFormatter(fmt)
    logger.addHandler(console)
    logger.addHandler(file_handler)
    logger.propagate = False
    return logger
