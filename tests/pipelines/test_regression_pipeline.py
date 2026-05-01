from pytest import raises
from datetime import datetime, timedelta

import pandas as pd

from stellar_harvest_ie_ml_stellar.models.regression.config.core import config
from stellar_harvest_ie_models.stellar.swpc.entities import (
    KpIndexEntity,
    KpForecastEntity,
)
from stellar_harvest_ie_ml_stellar.data.loader import kp_index_entities_to_df

from stellar_harvest_ie_ml_stellar.models.regression.validate import validate
from stellar_harvest_ie_ml_stellar.models.regression.features import extract
from stellar_harvest_ie_ml_stellar.models.regression.train import train
from stellar_harvest_ie_ml_stellar.models.regression.predict import predict
from stellar_harvest_ie_ml_stellar.models.regression.evaluate import evaluate
from stellar_harvest_ie_ml_stellar.models.regression.forecast import (
    forecast,
    build_feature_row,
    forecast_step,
    predictions_to_entities,
)


from sklearn.ensemble import HistGradientBoostingRegressor

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


async def test_validate():
    df = kp_index_entities_to_df(_KP_ROWS)
    assert isinstance(df, pd.DataFrame)

    validate(df=df)


async def test_validate_missing_columns():
    df = kp_index_entities_to_df(_KP_ROWS)
    df = df.drop(columns=["kp_index"])

    with raises(ValueError, match="missing required columns"):
        validate(df=df)


def test_extract():
    df = kp_index_entities_to_df(_KP_ROWS_REGRESSION)

    X, y = extract(df=df)

    expected_feature_cols = [f"kp_lag{l}" for l in config.model_cfg.lags] + [
        "kp_roll8_mean",
        "kp_roll8_max",
        "hour_sin",
        "hour_cos",
    ]
    expected_valid_rows = (
        len(df) - max(config.model_cfg.lags) - config.model_cfg.horizon
    )

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


def test_train_split():
    df = kp_index_entities_to_df(_KP_ROWS_REGRESSION)
    X, y = extract(df=df)

    _, _, _, y_train, y_test = train(X=X, y=y)

    assert len(y_train) + len(y_test) == len(y)
    # shuffle=False: train gets first n rows, test gets last m rows
    assert y_train.tolist() == y.iloc[: len(y_train)].tolist()
    assert y_test.tolist() == y.iloc[len(y_train) :].tolist()


def test_train_model():
    df = kp_index_entities_to_df(_KP_ROWS_REGRESSION)
    X, y = extract(df=df)

    model, _, _, _, _ = train(X=X, y=y)

    assert isinstance(model, HistGradientBoostingRegressor)
    assert model.max_iter == config.model_cfg.n_estimators
    assert model.random_state == config.model_cfg.random_state
    assert hasattr(model, "_is_fitted")  # fitted


def test_train_split_shapes():
    df = kp_index_entities_to_df(_KP_ROWS_REGRESSION)
    X, y = extract(df=df)

    _, X_train, X_test, y_train, y_test = train(X=X, y=y)

    assert isinstance(X_train, pd.DataFrame)
    assert isinstance(X_test, pd.DataFrame)
    assert X_train.shape[1] == X_test.shape[1] == X.shape[1]
    assert len(X_train) + len(X_test) == len(X)
    assert len(y_train) + len(y_test) == len(y)


def test_predict():
    df = kp_index_entities_to_df(_KP_ROWS_REGRESSION)
    X, y = extract(df=df)
    model, _, X_test, _, _ = train(X=X, y=y)

    result = predict(model=model, X_test=X_test)

    assert isinstance(result, dict)
    assert set(result.keys()) == {"y_preds", "version", "validation_errors"}
    assert len(result["y_preds"]) == X_test.shape[0]
    assert result["y_preds"].min() >= 0.0
    assert result["y_preds"].max() <= 9.0
    assert isinstance(result["version"], str)
    assert result["validation_errors"] is None


def test_evaluate():
    df = kp_index_entities_to_df(_KP_ROWS_REGRESSION)
    X, y = extract(df=df)
    model, _, X_test, _, y_test = train(X=X, y=y)
    predict_result = predict(model, X_test=X_test)

    result = evaluate(X_test=X_test, y_test=y_test, y_preds=predict_result["y_preds"])

    assert isinstance(result, dict)
    assert set(result.keys()) == {"mae", "rmse", "r2", "mae_baseline", "rmse_baseline"}
    assert result["mae"] >= 0.0
    assert result["rmse"] >= 0.0
    assert result["mae_baseline"] >= 0.0
    assert result["rmse_baseline"] >= 0.0
    assert isinstance(result["r2"], float)


def _build_series() -> pd.Series:
    df = kp_index_entities_to_df(_KP_ROWS_REGRESSION)
    df["time_tag"] = pd.to_datetime(df["time_tag"], utc=True)
    return (
        df.set_index("time_tag")
        .sort_index()
        .resample(config.model_cfg.resample_rule)
        .agg({"estimated_kp": "last"})
        .dropna()["estimated_kp"]
    )


def test_build_feature_row():
    series = _build_series()
    next_time = series.index[-1] + pd.Timedelta(config.model_cfg.resample_rule)

    X = build_feature_row(series, next_time, config.model_cfg)

    expected_cols = [f"kp_lag{l}" for l in config.model_cfg.lags] + [
        "kp_roll8_mean",
        "kp_roll8_max",
        "hour_sin",
        "hour_cos",
    ]
    assert isinstance(X, pd.DataFrame)
    assert X.shape == (1, len(expected_cols))
    assert list(X.columns) == expected_cols
    assert not X.isnull().any().any()
    assert X["hour_sin"].between(-1, 1).all()
    assert X["hour_cos"].between(-1, 1).all()


def test_forecast_step():
    series = _build_series()
    df = kp_index_entities_to_df(_KP_ROWS_REGRESSION)
    X, y = extract(df=df)
    model, _, _, _, _ = train(X=X, y=y)

    prediction, updated_series = forecast_step(
        model, series, step=1, cfg=config.model_cfg
    )

    assert set(prediction.keys()) == {"time_tag", "predicted_kp", "horizon"}
    assert prediction["horizon"] == 1
    assert 0.0 <= prediction["predicted_kp"] <= 9.0
    assert len(updated_series) == len(series) + 1
    assert updated_series.index[-1] == prediction["time_tag"]


def test_predictions_to_entities():
    from datetime import timezone

    issued_at = datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)
    predictions = [
        {
            "time_tag": datetime(2024, 1, 1, 3, 0, tzinfo=timezone.utc),
            "predicted_kp": 3.5,
            "horizon": 1,
        },
        {
            "time_tag": datetime(2024, 1, 1, 6, 0, tzinfo=timezone.utc),
            "predicted_kp": 5.1,
            "horizon": 2,
        },
    ]

    entities = predictions_to_entities(predictions, issued_at, model_version="0.1.0")

    assert len(entities) == 2
    assert all(isinstance(e, KpForecastEntity) for e in entities)
    assert entities[0].horizon == 1
    assert entities[1].horizon == 2
    assert entities[0].predicted_g == 0  # kp 3.5 < 5 → G0
    assert entities[1].predicted_g == 1  # kp 5.1 → G1
    assert entities[0].model_version == "0.1.0"


async def test_forecast():
    df = kp_index_entities_to_df(_KP_ROWS_REGRESSION)
    X, y = extract(df=df)
    model, _, _, _, _ = train(X=X, y=y)

    n_steps = 6
    df_predictions, entities = await forecast(model=model, df=df, n_steps=n_steps)

    assert isinstance(df_predictions, pd.DataFrame)
    assert len(df_predictions) == n_steps
    assert set(df_predictions.columns) == {"time_tag", "predicted_kp", "horizon"}
    assert df_predictions["predicted_kp"].between(0.0, 9.0).all()
    assert len(entities) == n_steps
    assert all(isinstance(e, KpForecastEntity) for e in entities)
