import numpy as np
import pandas as pd
from datetime import timedelta

from stellar_harvest_ie_ml_stellar.data.loader import load_planetary_kp_index
from stellar_harvest_ie_ml_stellar.models.regression.config.core import config


async def forecast(model, n_steps: int = 8) -> pd.DataFrame:
    """Produce the next n_steps Kp predictions, one per 3-hour block."""
    cfg = config.model_cfg

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

    return pd.DataFrame(predictions)
