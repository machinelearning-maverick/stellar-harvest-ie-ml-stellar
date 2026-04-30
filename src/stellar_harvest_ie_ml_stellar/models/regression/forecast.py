from typing import List, Tuple

import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone

from stellar_harvest_ie_models.stellar.swpc.entities import KpForecastEntity

from stellar_harvest_ie_ml_stellar.data.loader import load_planetary_kp_index
from stellar_harvest_ie_ml_stellar.models.regression.config.core import config
from stellar_harvest_ie_ml_stellar.models.regression import __version__ as model_version



def kp_to_g_level(kp: float) -> int:
    return 0 if kp < 5 else min(5, int(kp) - 4)


async def forecast(model, n_steps: int = 8) -> Tuple[pd.DataFrame, List[KpForecastEntity]]:
    """Produce the next n_steps Kp predictions, one per 3-hour block."""
    cfg = config.model_cfg
    issued_at = datetime.now(timezone.utc)  # one timestamp for the whole run

    # 1) Load history, resample, same as in extract()
    df = await load_planetary_kp_index()
    df["time_tag"] = pd.to_datetime(df["time_tag"], utc=True)
    df = (
        df.set_index("time_tag")
        .sort_index()
        .resample(cfg.resample_rule)
        .agg({"estimated_kp": "last"})
        .dropna()
    )

    predictions = []
    series = df["estimated_kp"].copy()  # we'll append predictions to this

    for step in range(1, n_steps + 1):
        last_time = series.index[-1]
        next_time = last_time + pd.Timedelta(cfg.resample_rule)

        # Build the feature row for predicting `next_time`
        feat = {f"kp_lag{l}": series.iloc[-l] for l in cfg.lags}
        feat["kp_roll8_mean"] = series.iloc[-8:].mean()
        feat["kp_roll8_max"] = series.iloc[-8:].max()
        feat["hour_sin"] = np.sin(2 * np.pi * next_time.hour / 24)
        feat["hour_cos"] = np.cos(2 * np.pi * next_time.hour / 24)

        X_future = pd.DataFrame([feat])
        y_hat = float(np.clip(model.predict(X_future)[0], 0.0, 9.0))

        predictions.append(
            {"time_tag": next_time, "predicted_kp": y_hat, "horizon": step}
        )
        # Append prediction so the next iteration's lags can use it (recursive forecasting)
        series.loc[next_time] = y_hat

        entities = [
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

    return pd.DataFrame(predictions), entities
