from stellar_harvest_ie_config.logging_config import setup_logging

setup_logging()

import logging
from datetime import datetime, timedelta

import pandas as pd

from stellar_harvest_ie_ml_stellar.data.loader import kp_index_entities_to_df
from stellar_harvest_ie_ml_stellar.models.regression.features import (
    resample_1minute_3hours,
    extract,
)
from stellar_harvest_ie_ml_stellar.models.regression.config.core import config
from stellar_harvest_ie_models.stellar.swpc.entities import (
    KpIndexEntity,
    KpForecastEntity,
)

logger = logging.getLogger(__name__)

# 250 rows at the native 3h cadence - enough to survive max_lag=216 + horizon=1
_KP_ROWS_REGRESSION = [
    KpIndexEntity(
        time_tag=datetime(2024, 1, 1, 0, 0) + timedelta(hours=3 * i),
        kp_index=[1, 4, 7][i % 3],
        estimated_kp=float([1.0, 4.0, 7.0][i % 3]),
        kp=["1Z", "4P", "7M"][i % 3],
    )
    for i in range(250)
]


def test_resample_1minute_3hours():
    df = kp_index_entities_to_df(_KP_ROWS_REGRESSION)
    logger.info(f"pd.DataFrame - Kp index: {df}")
    pass


def test_extract():
    df = pd.DataFrame(
        {
            "time_tag": ["2025-06-15T12:00:00Z", "2025-06-15T13:00:00Z"],
            "kp_index": [2, 6],
            "estimated_kp": [1.5, 2.0],
            "kp": ["1M", "2Z"],
        }
    )

    X, y = extract(df)

    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert X.shape[0] == 2
    assert y.shape[0] == 2

    for column in config.model_cfg.features_raw:
        assert column in list(X.columns)
