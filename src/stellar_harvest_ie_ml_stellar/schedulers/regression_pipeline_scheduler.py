import asyncio
import logging
import os
import sys
import time

import schedule
from dotenv import load_dotenv

from stellar_harvest_ie_config.logging_config import setup_logging

setup_logging()

from stellar_harvest_ie_ml_stellar.pipelines.regression_pipeline import (
    run_regression_pipeline,
)

logger = logging.getLogger(
    "stellar_harvest_ie_ml_stellar.schedulers.regression_pipeline_scheduler"
)


def job():
    logger.info("Running regression training pipeline...")
    try:
        n_forecast_steps = int(os.environ["SWPC_REGRESSION_N_FORECAST_STEPS"])
        result = asyncio.run(run_regression_pipeline(n_forecast_steps=n_forecast_steps))
        logger.info(f"Pipeline finished: metrics={result['metrics']}")
    except Exception:
        logger.error("Pipeline failed.", exc_info=True)


def main(env_path="/run/secrets/env"):
    load_dotenv(env_path if os.path.exists(env_path) else ".env")

    if "--run-once" in sys.argv:
        logger.info("--run-once flag detected; running pipeline immediately.")
        job()
        return

    try:
        schedule_day = os.environ["SCHEDULE_REGRESSION_DAY"]
        schedule_at = os.environ["SCHEDULE_REGRESSION_AT"]
    except KeyError as e:
        logger.error(f"Missing required environment variable: {e}")
        return

    logger.info(f"Regression scheduler starting; {schedule_day} at {schedule_at}")
    getattr(schedule.every(), schedule_day.lower()).at(schedule_at).do(job)

    while True:
        schedule.run_pending()
        time.sleep(1)


if __name__ == "__main__":
    main()
