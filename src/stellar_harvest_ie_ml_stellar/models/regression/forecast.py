from typing import List, Tuple

import numpy as np
import pandas as pd
from datetime import datetime, timezone

from stellar_harvest_ie_config.utils.log_decorators import log_io
from stellar_harvest_ie_ml_stellar.models.regression.config.core import (
    config,
    ModelConfig,
)
from stellar_harvest_ie_ml_stellar.models.regression import __version__ as model_version

from stellar_harvest_ie_models.stellar.swpc.entities import KpForecastEntity


def kp_to_g_level(kp: float) -> int:
    return 0 if kp < 5 else min(5, int(kp) - 4)


# @log_io()
def build_feature_row(
    series: pd.Series, next_time: pd.Timestamp, cfg: ModelConfig
) -> pd.DataFrame:
    feat = {f"kp_lag{l}": series.iloc[-l] for l in cfg.lags}
    feat["kp_roll8_mean"] = series.iloc[-8:].mean()
    feat["kp_roll8_max"] = series.iloc[-8:].max()
    feat["hour_sin"] = np.sin(2 * np.pi * next_time.hour / 24)
    feat["hour_cos"] = np.cos(2 * np.pi * next_time.hour / 24)
    return pd.DataFrame([feat])


# @log_io()
def forecast_step(
    model, series: pd.Series, step: int, cfg: ModelConfig
) -> Tuple[dict, pd.Series]:
    last_time = series.index[-1]
    next_time = last_time + pd.Timedelta(cfg.resample_rule)

    X_future = build_feature_row(series, next_time, cfg)
    y_hat = float(np.clip(model.predict(X_future)[0], 0.0, 9.0))

    prediction = {"time_tag": next_time, "predicted_kp": y_hat, "horizon": step}
    series = series.copy()
    series.loc[next_time] = y_hat

    return prediction, series


@log_io()
def predictions_to_entities(
    predictions: List[dict], issued_at: datetime, model_version: str
) -> List[KpForecastEntity]:
    return [
        KpForecastEntity(
            forecast_time=p["time_tag"],
            issued_at=issued_at,
            horizon=p["horizon"],
            predicted_kp=p["predicted_kp"],
            predicted_g=kp_to_g_level(p["predicted_kp"]),
            model_name="hgb_regressor",
            model_version=model_version,
        )
        for p in predictions
    ]


async def forecast(
    model, df: pd.DataFrame, n_steps: int = 8
) -> Tuple[pd.DataFrame, List[KpForecastEntity]]:
    cfg = config.model_cfg
    issued_at = datetime.now(timezone.utc)

    df["time_tag"] = pd.to_datetime(df["time_tag"], utc=True)
    df = (
        df.set_index("time_tag")
        .sort_index()
        .resample(cfg.resample_rule)
        .agg({"estimated_kp": "last"})
        .dropna()
    )

    predictions = []
    series = df["estimated_kp"].copy()

    for step in range(1, n_steps + 1):
        prediction, series = forecast_step(model, series, step, cfg)
        predictions.append(prediction)

    entities = predictions_to_entities(predictions, issued_at, model_version)
    return pd.DataFrame(predictions), entities
