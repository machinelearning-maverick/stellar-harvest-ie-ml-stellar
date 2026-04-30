import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from stellar_harvest_ie_config.utils.log_decorators import log_io


@log_io(
    skip_types_input={
        pd.Series: lambda v: f"<Series name={v.name} len={len(v)}>",
        np.ndarray: lambda v: f"<ndarray shape={v.shape} dtype={v.dtype}>",
    }
)
def evaluate(X_test: pd.DataFrame, y_test: pd.Series, y_preds: np.ndarray) -> dict:
    y_preds = np.clip(y_preds, 0.0, 9.0)  # Kp is bounded

    # Persistence baseline: predict that Kp doesn't change over the horizon.
    # Using lag1 from features as the naive forecast.
    y_naive = X_test["kp_lag1"].values

    return {
        "mae": float(mean_absolute_error(y_test, y_preds)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, y_preds))),
        "r2": float(r2_score(y_test, y_preds)),
        "mae_baseline": float(mean_absolute_error(y_test, y_naive)),
        "rmse_baseline": float(np.sqrt(mean_squared_error(y_test, y_naive))),
    }
