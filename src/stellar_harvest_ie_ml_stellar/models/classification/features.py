import pandas as pd
from typing import Tuple

from stellar_harvest_ie_config.utils.log_decorators import log_io
from stellar_harvest_ie_ml_stellar.models.classification.config.core import config


@log_io
def extract(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    df_copy = df.copy()
    tt: str = "time_tag"

    # 1) timestamps
    df_copy[tt] = pd.to_datetime(df_copy[tt], utc=True)
    df_copy["year"] = df_copy[tt].dt.year
    df_copy["month"] = df_copy[tt].dt.month
    df_copy["day"] = df_copy[tt].dt.day
    df_copy["hour"] = df_copy[tt].dt.hour
    df_copy["minute"] = df_copy[tt].dt.minute
    df_copy.drop(tt, axis=1, inplace=True)

    # 2) categorize target
    def categorize(k: int) -> int:
        return 0 if k <= 3 else (1 if k <= 5 else 2)

    df_copy[config.model_config.target] = df_copy["kp_index"].map(categorize)

    X = df_copy[config.model_config.features]
    y = df_copy[config.model_config.target]

    return X, y
