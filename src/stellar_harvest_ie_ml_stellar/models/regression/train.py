import pandas as pd
from typing import Tuple

from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingRegressor

from stellar_harvest_ie_config.utils.log_decorators import log_io
from stellar_harvest_ie_ml_stellar.models.regression.config.core import config


@log_io(
    skip_types_input={
        pd.DataFrame: lambda v: f"<DataFrame shape={v.shape} columns={list(v.columns)}>",
        pd.Series: lambda v: f"<Series name={v.name} len={len(v)}>",
    }
)
def train(
    X: pd.DataFrame, y: pd.Series
) -> Tuple[
    HistGradientBoostingRegressor, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series
]:
    cfg = config.model_cfg

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg.test_size, shuffle=False  # critical: chronological
    )

    model = HistGradientBoostingRegressor(
        max_iter=cfg.n_estimators,
        max_depth=cfg.max_depth,
        learning_rate=0.05,
        random_state=cfg.random_state,
    )
    model.fit(X_train, y_train)
    return model, X_train, X_test, y_train, y_test
