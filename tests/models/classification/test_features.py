import pandas as pd

from stellar_harvest_ie_ml_stellar.models.classification.features import extract
from stellar_harvest_ie_ml_stellar.models.classification.config.core import config

from stellar_harvest_ie_config.logging_config import setup_logging

setup_logging()


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
