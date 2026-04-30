import logging
import os
import time

import schedule
from dotenv import load_dotenv

from stellar_harvest_ie_config.logging_config import setup_logging

setup_logging()

from stellar_harvest_ie_config.utils.log_decorators import log_io
from stellar_harvest_ie_ml_stellar.pipelines.regression_pipeline import (
    run_regression_pipeline,
)

logger = logging.getLogger(
    "stellar_harvest_ie_ml_stellar.schedulers.regression_pipeline_scheduler"
)


@log_io()
def job():
    logger.info("Running classification training pipeline...")
    try:
        n_forecast_steps = os.environ["SWPC_REGRESSION_N_FORECAST_STEPS"]
        metrics = run_regression_pipeline(n_forecast_steps=n_forecast_steps)
        logger.info(f"Training finished: metrics={metrics}")
    except Exception:
        logger.error("Training failed.", exc_info=True)


def main(env_path="/run/secrets/env"):
    load_dotenv(env_path if os.path.exists(env_path) else ".env")

    try:
        schedule_day = os.environ["SCHEDULE_REGRESSION_DAY"]
        schedule_at = os.environ["SCHEDULE_REGRESSION_AT"]
    except KeyError as e:
        logger.error(f"Missing required environment variable: {e}")
        return

    msg = f"Regression training scheduler starting; {schedule_day} at {schedule_at}"
    logger.info(msg)

    getattr(schedule.every(), schedule_day.lower()).at(schedule_at).do(job)

    while True:
        schedule.run_pending()
        time.sleep(1)


if __name__ == "__main__":
    main()
