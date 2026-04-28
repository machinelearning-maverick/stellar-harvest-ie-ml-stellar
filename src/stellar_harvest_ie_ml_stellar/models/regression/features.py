import numpy as np
import pandas as pd
from typing import Tuple

from stellar_harvest_ie_config.utils.log_decorators import log_io
from stellar_harvest_ie_ml_stellar.models.regression.config.core import config


@log_io(
    skip_types_input={
        pd.DataFrame: lambda v: f"<DataFrame shape={v.shape} columns={list(v.columns)}>",
        pd.Series: lambda v: f"<Series name={v.name} len={len(v)}>",
    },
    skip_types_output={
        pd.DataFrame: lambda v: f"<DataFrame shape={v.shape} columns={list(v.columns)}>",
        pd.Series: lambda v: f"<Series name={v.name} len={len(v)}>",
    },
)
def extract(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    cfg = config.model_cfg
    df = df.copy()

    # 1) Resample 1-minute -> 3-hour blocks (last value per bucket)
    df["time_tag"] = pd.to_datetime(df["time_tag"], utc=True)
    df = (
        df.set_index("time_tag")
        .sort_index()
        .resample(cfg.resample_rule)
        .agg({"estimated_kp": "last", "kp_index": "last"})
        .dropna(subset=["estimated_kp"])
    )

    # 2) Lag features (past values only)
    for lag in cfg.lags:
        df[f"kp_lag{lag}"] = df["estimated_kp"].shift(lag)

    # 3) Rolling-window features (also past-only because shift(1) before rolling)
    df["kp_roll8_mean"] = df["estimated_kp"].shift(1).rolling(8).mean()  # last 24h mean
    df["kp_roll8_max"] = df["estimated_kp"].shift(1).rolling(8).max()  # last 24h peak

    # 4) Cyclical UT-hour features
    hour = df.index.hour
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)

    # 5) Future target — regression: the actual estimated_kp h blocks ahead
    df["target"] = df["estimated_kp"].shift(-cfg.horizon)

    # 6) Drop rows with NaN from lags (head) and horizon (tail)
    df = df.dropna()

    feature_cols = [f"kp_lag{l}" for l in cfg.lags] + [
        "kp_roll8_mean",
        "kp_roll8_max",
        "hour_sin",
        "hour_cos",
    ]
    X = df[feature_cols]
    y = df["target"].astype(float)
    return X, y
