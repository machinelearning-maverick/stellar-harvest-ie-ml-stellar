import pandas as pd

from stellar_harvest_ie_config.utils.log_decorators import log_io
from stellar_harvest_ie_ml_stellar.models.classification.config.core import config


@log_io(skip_types={
    pd.DataFrame: lambda v: f"<DataFrame shape={v.shape} columns={list(v.columns)}>",
})
def validate(df: pd.DataFrame) -> None:
    if df.empty:
        raise ValueError("DataFrame is empty")

    missing = set(config.model_cfg.input_features) - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame is missing required columns: {missing}")

    nulls = df[config.model_cfg.input_features].isnull().any()
    if nulls.any():
        raise ValueError(f"Null values found in columns: {nulls[nulls].index.tolist()}")
