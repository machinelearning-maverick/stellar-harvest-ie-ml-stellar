from pytest import raises
from datetime import datetime, timedelta

import pandas as pd

from stellar_harvest_ie_models.stellar.swpc.entities import KpIndexEntity
from stellar_harvest_ie_ml_stellar.data.loader import kp_entities_to_df
from stellar_harvest_ie_ml_stellar.models.regression.validate import validate
from stellar_harvest_ie_ml_stellar.models.regression.features import extract
from stellar_harvest_ie_ml_stellar.models.regression.train import train
from stellar_harvest_ie_ml_stellar.models.regression.predict import predict
from stellar_harvest_ie_ml_stellar.models.regression.evaluate import evaluate
from sklearn.ensemble import HistGradientBoostingRegressor
from stellar_harvest_ie_ml_stellar.models.regression.config.core import config


_KP_ROWS = [
    KpIndexEntity(
        time_tag=datetime(2024, 1, 1, 0, 0),
        kp_index=2,
        estimated_kp=2.33,
        kp="2Z",
    ),
    KpIndexEntity(
        time_tag=datetime(2024, 1, 1, 3, 0),
        kp_index=4,
        estimated_kp=4.67,
        kp="1P",
    ),
    KpIndexEntity(
        time_tag=datetime(2024, 1, 1, 6, 0),
        kp_index=7,
        estimated_kp=7.0,
        kp="0Z",
    ),
]


# 250 rows at the native 3h cadence — enough to survive max_lag=216 + horizon=1
_KP_ROWS_REGRESSION = [
    KpIndexEntity(
        time_tag=datetime(2024, 1, 1, 0, 0) + timedelta(hours=3 * i),
        kp_index=[1, 4, 7][i % 3],
        estimated_kp=float([1.0, 4.0, 7.0][i % 3]),
        kp=["1Z", "4P", "7M"][i % 3],
    )
    for i in range(250)
]


def test_evaluate():
    df = kp_entities_to_df(_KP_ROWS_REGRESSION)
    X, y = extract(df=df)
    model, _, X_test, _, y_test = train(X=X, y=y)

    result = evaluate(model=model, X_test=X_test, y_test=y_test)

    assert isinstance(result, dict)
    assert set(result.keys()) == {"mae", "rmse", "r2", "mae_baseline", "rmse_baseline"}
    assert result["mae"] >= 0.0
    assert result["rmse"] >= 0.0
    assert result["mae_baseline"] >= 0.0
    assert result["rmse_baseline"] >= 0.0
    assert isinstance(result["r2"], float)


def test_predict():
    df = kp_entities_to_df(_KP_ROWS_REGRESSION)
    X, y = extract(df=df)
    model, _, X_test, _, _ = train(X=X, y=y)

    result = predict(model=model, X_test=X_test)

    assert isinstance(result, dict)
    assert set(result.keys()) == {"predictions", "version", "validation_errors"}
    assert len(result["predictions"]) == X_test.shape[0]
    assert result["predictions"].min() >= 0.0
    assert result["predictions"].max() <= 9.0
    assert isinstance(result["version"], str)
    assert result["validation_errors"] is None


def test_train_split():
    df = kp_entities_to_df(_KP_ROWS_REGRESSION)
    X, y = extract(df=df)

    _, _, _, y_train, y_test = train(X=X, y=y)

    assert len(y_train) + len(y_test) == len(y)
    # shuffle=False: train gets first n rows, test gets last m rows
    assert y_train.tolist() == y.iloc[: len(y_train)].tolist()
    assert y_test.tolist() == y.iloc[len(y_train) :].tolist()


def test_train_model():
    df = kp_entities_to_df(_KP_ROWS_REGRESSION)
    X, y = extract(df=df)

    model, _, _, _, _ = train(X=X, y=y)

    assert isinstance(model, HistGradientBoostingRegressor)
    assert model.max_iter == config.model_cfg.n_estimators
    assert model.random_state == config.model_cfg.random_state
    assert hasattr(model, "_is_fitted")  # fitted


def test_train_split_shapes():
    df = kp_entities_to_df(_KP_ROWS_REGRESSION)
    X, y = extract(df=df)

    _, X_train, X_test, y_train, y_test = train(X=X, y=y)

    assert isinstance(X_train, pd.DataFrame)
    assert isinstance(X_test, pd.DataFrame)
    assert X_train.shape[1] == X_test.shape[1] == X.shape[1]
    assert len(X_train) + len(X_test) == len(X)
    assert len(y_train) + len(y_test) == len(y)


def test_extract():
    df = kp_entities_to_df(_KP_ROWS_REGRESSION)

    X, y = extract(df=df)

    expected_feature_cols = [f"kp_lag{l}" for l in config.model_cfg.lags] + [
        "kp_roll8_mean",
        "kp_roll8_max",
        "hour_sin",
        "hour_cos",
    ]
    expected_valid_rows = len(df) - max(config.model_cfg.lags) - config.model_cfg.horizon

    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert list(X.columns) == expected_feature_cols
    assert y.name == "target"
    assert y.dtype == float
    assert len(X) == len(y) == expected_valid_rows
    assert len(X) < len(df)
    assert not X.isnull().any().any()
    assert not y.isnull().any()
    assert X["hour_sin"].between(-1, 1).all()
    assert X["hour_cos"].between(-1, 1).all()


async def test_validate():
    df = kp_entities_to_df(_KP_ROWS)
    assert isinstance(df, pd.DataFrame)

    validate(df=df)


async def test_validate_missing_columns():
    df = kp_entities_to_df(_KP_ROWS)
    df = df.drop(columns=["kp_index"])

    with raises(ValueError, match="missing required columns"):
        validate(df=df)
