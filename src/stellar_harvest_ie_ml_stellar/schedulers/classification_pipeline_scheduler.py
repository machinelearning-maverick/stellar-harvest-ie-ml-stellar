import logging
import os
import time

import schedule
from dotenv import load_dotenv

from stellar_harvest_ie_config.logging_config import setup_logging

setup_logging()

from stellar_harvest_ie_config.utils.log_decorators import log_io
from stellar_harvest_ie_ml_stellar.pipelines.classification_pipeline import (
    run_classification_pipeline,
)

logger = logging.getLogger(
    "stellar_harvest_ie_ml_stellar.schedulers.classification_pipeline_scheduler"
)


@log_io()
def job():
    logger.info("Running classification training pipeline...")
    try:
        metrics = run_classification_pipeline()
        logger.info(f"Training finished: metrics={metrics}")
    except Exception:
        logger.error("Training failed.", exc_info=True)


def main(env_path="/run/secrets/env"):
    load_dotenv(env_path)

    try:
        schedule_day = os.environ["SCHEDULE_CLASSIFY_DAY"]
        schedule_at = os.environ["SCHEDULE_CLASSIFY_AT"]
    except KeyError as e:
        logger.error(f"Missing required environment variable: {e}")
        return

    msg = f"Classify training scheduler starting; {schedule_day} at {schedule_at}"
    logger.info(msg)

    getattr(schedule.every(), schedule_day.lower()).at(schedule_at).do(job)

    while True:
        schedule.run_pending()
        time.sleep(1)


if __name__ == "__main__":
    main()
